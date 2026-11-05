/*
 * kmeans_mpi.c — MPI K-Means (PDC Project, Milestone M2).
 *
 * Same Lloyd's algorithm as the serial baseline (src/kmeans_serial.c),
 * distributed over P ranks:
 *
 *   1. Rank 0 loads the full dataset and computes the initial centroids
 *      with the IDENTICAL deterministic farthest-point (maximin)
 *      initialisation as the serial version (centroid 0 = row 0, each
 *      next centroid = the point farthest from its nearest already-chosen
 *      centroid, ties broken by lowest index). The plan file's stale
 *      "first K points" line is NOT used: M1 proved it fails verification
 *      on the Small tier, and any difference here would make the
 *      serial comparison meaningless. Centroids are MPI_Bcast to all
 *      ranks.
 *   2. Data is distributed with MPI_Scatterv, NOT MPI_Scatter, because
 *      N is not always divisible by P: rank r < (N % P) gets one extra
 *      point. Counts/displacements are computed in points, then scaled
 *      by D for the element counts MPI needs.
 *   3. Each rank assigns its local slice and accumulates local K x D
 *      sums and K counts. One MPI_Allreduce(MPI_SUM) per array gives
 *      every rank the global sums/counts, so every rank recomputes the
 *      SAME new centroids itself — no extra Bcast is needed after the
 *      reduction, and the convergence check is replicated, not voted.
 *   4. Empty clusters keep their previous centroid (same guard as
 *      serial), so there is no division by zero.
 *   5. Final assignments are collected with MPI_Gatherv (same
 *      remainder-aware counts) and rank 0 writes them in the serial
 *      format: one int32 per point, in file order.
 *
 * Data format: identical to the serial version — headerless float64
 * binary, row-major, N x D; N, D come from the command line.
 *
 * Convergence: shift = max over centroids of ||new - old||_2; stop
 * when shift < epsilon or max_iters iterations are done.
 *
 * Usage:
 *   mpirun -np P bin/kmeans_mpi <datafile> <N> <D> <K> <max_iters>
 *                              <epsilon> [assignments_file]
 *
 * Build:  make mpi            (or: mpicc -O2 -Wall -Wextra -std=c11
 *                               -o bin/kmeans_mpi src/kmeans_mpi.c -lm)
 * Example: python3 scripts/gen_data.py --tier small --out data/small.bin
 *          mpirun -np 4 bin/kmeans_mpi data/small.bin 100000 16 8 100 1e-4
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

/* Abort every rank on an MPI failure; a lone rank returning early while
 * others wait at a collective would deadlock the job. */
#define MPI_CHECK(call)                                                 \
    do {                                                                \
        int _err = (call);                                              \
        if (_err != MPI_SUCCESS) {                                      \
            char _msg[MPI_MAX_ERROR_STRING];                            \
            int _len = 0;                                               \
            MPI_Error_string(_err, _msg, &_len);                        \
            fprintf(stderr, "MPI error in %s: %s\n", #call, _msg);      \
            MPI_Abort(MPI_COMM_WORLD, 1);                               \
        }                                                               \
    } while (0)

/* Parse a strictly positive integer argument; returns 0 on success. */
static int parse_long(const char *s, long *out)
{
    char *end = NULL;
    long v;

    if (s == NULL || *s == '\0') {
        return -1;
    }
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v <= 0) {
        return -1;
    }
    *out = v;
    return 0;
}

/* Parse a strictly positive finite double argument; returns 0 on success. */
static int parse_double_pos(const char *s, double *out)
{
    char *end = NULL;
    double v;

    if (s == NULL || *s == '\0') {
        return -1;
    }
    errno = 0;
    v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !(v > 0.0) || !isfinite(v)) {
        return -1;
    }
    *out = v;
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: mpirun -np P %s <datafile> <N> <D> <K> <max_iters> "
            "<epsilon> [assignments_file]\n"
            "  datafile: headerless float64 binary, row-major, N x D\n"
            "  N, D, K, max_iters: positive integers, K <= N\n"
            "  epsilon: positive convergence threshold on max centroid shift\n",
            prog);
}

/* Farthest-point initialisation, bit-identical to kmeans_serial.c.
 * Only rank 0 runs this, on the full dataset. */
static int init_centroids_farthest(const double *data, size_t n, size_t d,
                                   size_t k, double *centroids)
{
    double *mind = malloc(n * sizeof(*mind));
    size_t i, j, c;

    if (mind == NULL) {
        return -1;
    }
    for (i = 0; i < n; i++) {
        mind[i] = INFINITY;
    }
    memcpy(centroids, data, d * sizeof(*centroids));
    for (c = 1; c < k; c++) {
        const double *prev = centroids + (c - 1) * d;
        size_t far = 0;

        for (i = 0; i < n; i++) {
            const double *pt = data + i * d;
            double dist = 0.0;

            for (j = 0; j < d; j++) {
                double diff = pt[j] - prev[j];
                dist += diff * diff;
            }
            if (dist < mind[i]) {
                mind[i] = dist;
            }
            if (mind[i] > mind[far]) {
                far = i; /* strict >: lowest index wins ties */
            }
        }
        memcpy(centroids + c * d, data + far * d, d * sizeof(*centroids));
    }
    free(mind);
    return 0;
}

int main(int argc, char *argv[])
{
    const char *datafile;
    const char *assignfile = NULL;
    long n_long, d_long, k_long, max_iters_long;
    double epsilon;
    size_t n, d, k, max_iters;
    size_t i, j, c, iter;
    int rank = 0, size = 1;
    int *counts_pts = NULL;     /* points per rank, length = size */
    int *displs_pts = NULL;     /* point offsets per rank */
    int *sendcounts = NULL;     /* element counts for Scatterv */
    int *displs_elem = NULL;    /* element displacements for Scatterv */
    size_t local_n = 0;
    double *data = NULL;        /* full data, rank 0 only */
    double *local = NULL;       /* local slice: local_n x D */
    double *centroids = NULL;   /* K x D, replicated on every rank */
    double *sums = NULL;        /* local K x D sums */
    double *gsums = NULL;       /* global K x D sums after Allreduce */
    long long *counts = NULL;   /* local K counts */
    long long *gcounts = NULL;  /* global K counts after Allreduce */
    int *lassign = NULL;        /* local assignments */
    int *assign = NULL;         /* full assignments, rank 0 only */
    double sse = 0.0, gsse = 0.0;
    double shift = 0.0;
    size_t empty_events = 0;
    int converged = 0;
    double t_start = 0.0, t_end = 0.0;
    int rc = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 7 || argc > 8) {
        if (rank == 0) {
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    datafile = argv[1];
    if (parse_long(argv[2], &n_long) != 0 ||
        parse_long(argv[3], &d_long) != 0 ||
        parse_long(argv[4], &k_long) != 0 ||
        parse_long(argv[5], &max_iters_long) != 0 ||
        parse_double_pos(argv[6], &epsilon) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: invalid numeric argument.\n");
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if (argc == 8) {
        assignfile = argv[7];
    }
    n = (size_t)n_long;
    d = (size_t)d_long;
    k = (size_t)k_long;
    max_iters = (size_t)max_iters_long;
    if (k > n) {
        if (rank == 0) {
            fprintf(stderr, "Error: K (%zu) must not exceed N (%zu).\n", k, n);
        }
        MPI_Finalize();
        return 1;
    }
    if (n > 0 && d > ((size_t)-1) / n) {
        if (rank == 0) {
            fprintf(stderr, "Error: dataset dimensions too large.\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Partition: first (N % size) ranks get one extra point, so the
     * remainder case works and no point is dropped or duplicated. */
    counts_pts = malloc((size_t)size * sizeof(*counts_pts));
    displs_pts = malloc((size_t)size * sizeof(*displs_pts));
    sendcounts = malloc((size_t)size * sizeof(*sendcounts));
    displs_elem = malloc((size_t)size * sizeof(*displs_elem));
    if (counts_pts == NULL || displs_pts == NULL || sendcounts == NULL ||
        displs_elem == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Error: out of memory for partition arrays.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    {
        size_t base = n / (size_t)size;
        size_t rem = n % (size_t)size;
        size_t off = 0;
        int r;

        for (r = 0; r < size; r++) {
            size_t cnt = base + ((size_t)r < rem ? 1 : 0);

            /* MPI counts are int: refuse rather than silently truncate. */
            if (cnt > (size_t)INT_MAX / d || off > (size_t)INT_MAX / d) {
                if (rank == 0) {
                    fprintf(stderr,
                            "Error: dataset too large for MPI int counts.\n");
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            counts_pts[r] = (int)cnt;
            displs_pts[r] = (int)off;
            sendcounts[r] = (int)(cnt * d);
            displs_elem[r] = (int)(off * d);
            off += cnt;
        }
    }
    local_n = (size_t)counts_pts[rank];

    centroids = malloc(k * d * sizeof(*centroids));
    sums = malloc(k * d * sizeof(*sums));
    gsums = malloc(k * d * sizeof(*gsums));
    counts = malloc(k * sizeof(*counts));
    gcounts = malloc(k * sizeof(*gcounts));
    /* local_n may be 0 when P > N: allocate 1 element so the pointers
     * stay non-NULL and the (empty) loops below are still safe. */
    local = malloc((local_n > 0 ? local_n : 1) * d * sizeof(*local));
    lassign = malloc((local_n > 0 ? local_n : 1) * sizeof(*lassign));
    if (centroids == NULL || sums == NULL || gsums == NULL ||
        counts == NULL || gcounts == NULL || local == NULL ||
        lassign == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Error: out of memory for N=%zu, D=%zu, K=%zu.\n",
                    n, d, k);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        FILE *fp = fopen(datafile, "rb");

        if (fp == NULL) {
            fprintf(stderr, "Error: cannot open '%s': %s\n", datafile,
                    strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        data = malloc(n * d * sizeof(*data));
        if (data == NULL) {
            fprintf(stderr, "Error: out of memory for dataset.\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fread(data, sizeof(*data), n * d, fp) != n * d) {
            fprintf(stderr,
                    "Error: '%s' is shorter than N*D doubles "
                    "(expected %zu points x %zu dims).\n",
                    datafile, n, d);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(fp);

        /* Reject non-finite input before it can poison centroids/SSE. */
        for (i = 0; i < n * d; i++) {
            if (!isfinite(data[i])) {
                fprintf(stderr, "Error: non-finite value in dataset at "
                                "element %zu.\n",
                        i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        if (init_centroids_farthest(data, n, d, k, centroids) != 0) {
            fprintf(stderr, "Error: out of memory for initialisation.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Every rank starts from the same initial centroids. */
    MPI_CHECK(MPI_Bcast(centroids, (int)(k * d), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD));

    /* Distribute the data. Rank 0 sends from its full buffer; the
     * receive counts match the Scatterv partition computed above. */
    MPI_CHECK(MPI_Scatterv(data, sendcounts, displs_elem, MPI_DOUBLE,
                           local, (int)(local_n * d), MPI_DOUBLE, 0,
                           MPI_COMM_WORLD));
    /* Rank 0 no longer needs the full copy. */
    free(data);
    data = NULL;

    /* Local slices are finite by construction (rank 0 checked the full
     * file), but re-check cheaply: a corrupt slice must not silently
     * poison the global sums. */
    for (i = 0; i < local_n * d; i++) {
        if (!isfinite(local[i])) {
            fprintf(stderr, "Rank %d: non-finite value in local slice at "
                            "element %zu.\n",
                    rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    t_start = MPI_Wtime();
    for (iter = 0; iter < max_iters; iter++) {
        double iter_sse = 0.0;

        memset(sums, 0, k * d * sizeof(*sums));
        memset(counts, 0, k * sizeof(*counts));

        /* Assignment step over the local slice only. */
        for (i = 0; i < local_n; i++) {
            const double *pt = local + i * d;
            double best_dist = 0.0;
            size_t best = 0;

            for (c = 0; c < k; c++) {
                const double *cent = centroids + c * d;
                double dist = 0.0;

                for (j = 0; j < d; j++) {
                    double diff = pt[j] - cent[j];
                    dist += diff * diff;
                }
                if (c == 0 || dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            lassign[i] = (int)best;
            iter_sse += best_dist;
            for (j = 0; j < d; j++) {
                sums[best * d + j] += pt[j];
            }
            counts[best]++;
        }

        /* Global sums, counts and SSE: identical on every rank. */
        MPI_CHECK(MPI_Allreduce(sums, gsums, (int)(k * d), MPI_DOUBLE,
                                MPI_SUM, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(counts, gcounts, (int)k, MPI_LONG_LONG,
                                MPI_SUM, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(&iter_sse, &gsse, 1, MPI_DOUBLE, MPI_SUM,
                                MPI_COMM_WORLD));

        /* Update step + convergence shift, replicated on every rank
         * from the identical global sums, so no Bcast is needed. */
        shift = 0.0;
        for (c = 0; c < k; c++) {
            double move_sq = 0.0;

            if (gcounts[c] == 0) {
                /* Empty cluster: keep the previous centroid unchanged. */
                empty_events++;
                continue;
            }
            for (j = 0; j < d; j++) {
                double new_val = gsums[c * d + j] / (double)gcounts[c];
                double diff = new_val - centroids[c * d + j];
                centroids[c * d + j] = new_val;
                move_sq += diff * diff;
            }
            if (sqrt(move_sq) > shift) {
                shift = sqrt(move_sq);
            }
        }
        sse = gsse;

        if (shift < epsilon) {
            converged = 1;
            iter++; /* count the iteration just completed */
            break;
        }
    }
    t_end = MPI_Wtime();

    /* Collect final assignments on rank 0, in file order. */
    if (rank == 0) {
        assign = malloc(n * sizeof(*assign));
        if (assign == NULL) {
            fprintf(stderr, "Error: out of memory for assignments.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_CHECK(MPI_Gatherv(lassign, (int)local_n, MPI_INT,
                          assign, counts_pts, displs_pts, MPI_INT, 0,
                          MPI_COMM_WORLD));

    if (rank == 0 && assignfile != NULL) {
        FILE *af = fopen(assignfile, "wb");
        size_t w;

        if (af == NULL) {
            fprintf(stderr, "Error: cannot write '%s': %s\n", assignfile,
                    strerror(errno));
            rc = 1;
            goto done;
        }
        /* Write as fixed-width int32, same as the serial version. */
        for (i = 0, w = 0; i < n; i++) {
            int32_t v = (int32_t)assign[i];
            if (fwrite(&v, sizeof(v), 1, af) == 1) {
                w++;
            } else {
                break;
            }
        }
        {
            int close_ok = (fclose(af) == 0);
            if (w != n || !close_ok) {
                fprintf(stderr,
                        "Error: failed writing assignments to '%s'.\n",
                        assignfile);
                rc = 1;
                goto done;
            }
        }
    }

    if (rank == 0) {
        printf("K-Means MPI version\n");
        printf("  ranks: %d  data: %s  N=%zu  D=%zu  K=%zu  max_iters=%zu  "
               "epsilon=%g\n",
               size, datafile, n, d, k, max_iters, epsilon);
        printf("  iterations: %zu  converged: %s  final max centroid "
               "shift: %.6g\n",
               iter, converged ? "yes" : "no", shift);
        printf("  SSE: %.6g  empty-cluster events: %zu\n", sse, empty_events);
        printf("  time: %.3f s (wall, iteration loop incl. communication)\n",
               t_end - t_start);
        printf("  cluster counts:");
        for (c = 0; c < k; c++) {
            printf(" %lld", gcounts[c]);
        }
        printf("\n  final centroids:\n");
        for (c = 0; c < k; c++) {
            printf("    [%zu]", c);
            for (j = 0; j < d; j++) {
                printf(" %.6f", centroids[c * d + j]);
            }
            printf("\n");
        }
    }

done:
    free(counts_pts);
    free(displs_pts);
    free(sendcounts);
    free(displs_elem);
    free(data);
    free(local);
    free(centroids);
    free(sums);
    free(gsums);
    free(counts);
    free(gcounts);
    free(lassign);
    free(assign);
    MPI_Finalize();
    return rc;
}

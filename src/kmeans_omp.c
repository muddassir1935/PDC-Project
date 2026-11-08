/*
 * kmeans_omp.c — OpenMP K-Means (PDC Project, Milestone M3).
 *
 * Same Lloyd's algorithm as the serial baseline, with the assignment loop
 * parallelised:
 *   1. Assign every point to its nearest centroid
 *      (#pragma omp parallel for schedule(static)).
 *   2. Recompute each centroid as the mean of its assigned points.
 *   3. Repeat until convergence or max iterations.
 *
 * Shared conventions (MUST stay identical to kmeans_serial.c, otherwise
 * cross-version comparison is meaningless):
 *
 *   Data format: binary, headerless, little-endian float64 (C double),
 *     row-major: point i occupies elements [i*D, i*D + D). N and D come
 *     from the command line, never from the file itself.
 *
 *   Initial centroids (deterministic farthest-point / maximin): centroid 0
 *     is row 0 of the file; each next centroid is the point farthest
 *     (squared Euclidean) from its nearest already-chosen centroid, ties
 *     broken by lowest index. Kept SERIAL on purpose: the init must
 *     reproduce the serial centroids bit-for-bit, and it is O(K*N*D),
 *     negligible next to the iterations. (The plan's stale "first K
 *     points" line is wrong — that init failed verification on the Small
 *     tier in M1 and was replaced by farthest-point; see kmeans_serial.c.)
 *
 *   Convergence: shift = max over centroids k of ||new_k - old_k||_2.
 *     Stop when shift < epsilon, or when max_iters iterations are done.
 *
 *   Empty clusters: if a cluster gets zero points in an iteration, its
 *     centroid is kept unchanged (no division by zero, no NaN).
 *     Empty-cluster events are counted and reported.
 *
 * Race / false-sharing design (the point of this milestone):
 *   The assignment loop writes per-point sums into a shared K x D array
 *   in the serial version — in parallel that would be a data race (two
 *   threads adding to the same centroid sum). So every thread accumulates
 *   into its OWN private K x D sums + K counts, and the per-thread
 *   partials are merged into the global sums once, serially, after the
 *   parallel loop finishes (implicit barrier at the end of the for).
 *   Each thread's block is padded up to a multiple of 64 bytes (one
 *   cache line) so two threads' partials never share a cache line —
 *   otherwise the cores would keep invalidating each other's lines
 *   (false sharing) and the parallel loop would run slower than serial.
 *   SSE uses an OpenMP reduction (+); summation order differs from
 *   serial, so SSE/centroids are compared with a tolerance, not exactly.
 *
 * Usage:
 *   kmeans_omp <datafile> <N> <D> <K> <max_iters> <epsilon>
 *                [assignments_file]
 *   Threads: OMP_NUM_THREADS (default: all available cores).
 *   assignments_file (optional): final assignment of every point, one
 *     int32 per point in file order — same format as the serial version,
 *     so scripts/compare_assignments.py can check it against serial.
 *
 * Build:  make omp            (or: gcc -O2 -Wall -Wextra -std=c11 -fopenmp
 *                               -o bin/kmeans_omp src/kmeans_omp.c -lm)
 * Example: bin/kmeans_omp data/small.bin 100000 16 8 100 1e-4
 *          OMP_NUM_THREADS=2 bin/kmeans_omp data/small.bin 100000 16 8 100 1e-4
 */

#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cache-line size used to pad per-thread partials (see header). */
#define CACHE_LINE_BYTES 64

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
            "Usage: %s <datafile> <N> <D> <K> <max_iters> <epsilon> "
            "[assignments_file]\n"
            "  datafile: headerless float64 binary, row-major, N x D\n"
            "  N, D, K, max_iters: positive integers, K <= N\n"
            "  epsilon: positive convergence threshold on max centroid shift\n"
            "  threads: OMP_NUM_THREADS (default: all available cores)\n",
            prog);
}

int main(int argc, char *argv[])
{
    const char *datafile;
    const char *assignfile = NULL;
    long n_long, d_long, k_long, max_iters_long;
    double epsilon;
    size_t n, d, k, max_iters;
    size_t i, j, c, iter;
    FILE *fp = NULL;
    double *data = NULL;       /* N x D */
    double *centroids = NULL;  /* K x D */
    double *sums = NULL;       /* K x D (merged global sums) */
    size_t *counts = NULL;     /* K (merged global counts) */
    int *assign = NULL;        /* N */
    double *tsums = NULL;      /* nthreads x stride_d: per-thread sums */
    size_t *tcounts = NULL;    /* nthreads x stride_k: per-thread counts */
    size_t stride_d;           /* per-thread sums block, padded to 64 B */
    size_t stride_k;           /* per-thread counts block, padded to 64 B */
    int nthreads;
    double sse = 0.0;
    double shift = 0.0;
    size_t empty_events = 0;
    int converged = 0;
    double t_start, t_end;

    if (argc < 7 || argc > 8) {
        usage(argv[0]);
        return 1;
    }
    datafile = argv[1];
    if (parse_long(argv[2], &n_long) != 0 ||
        parse_long(argv[3], &d_long) != 0 ||
        parse_long(argv[4], &k_long) != 0 ||
        parse_long(argv[5], &max_iters_long) != 0 ||
        parse_double_pos(argv[6], &epsilon) != 0) {
        fprintf(stderr, "Error: invalid numeric argument.\n");
        usage(argv[0]);
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
        fprintf(stderr, "Error: K (%zu) must not exceed N (%zu).\n", k, n);
        return 1;
    }
    /* Guard against size_t overflow in the N*D allocation. */
    if (n > 0 && d > ((size_t)-1) / n) {
        fprintf(stderr, "Error: dataset dimensions too large.\n");
        return 1;
    }

    nthreads = omp_get_max_threads();
    if (nthreads < 1) {
        fprintf(stderr, "Error: OpenMP reports no threads.\n");
        return 1;
    }

    /* Pad each thread's partial block to a whole number of cache lines,
     * so no two threads ever write into the same cache line. */
    stride_d = (k * d + CACHE_LINE_BYTES / sizeof(double) - 1) /
               (CACHE_LINE_BYTES / sizeof(double)) *
               (CACHE_LINE_BYTES / sizeof(double));
    stride_k = (k + CACHE_LINE_BYTES / sizeof(size_t) - 1) /
               (CACHE_LINE_BYTES / sizeof(size_t)) *
               (CACHE_LINE_BYTES / sizeof(size_t));

    fp = fopen(datafile, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: cannot open '%s': %s\n", datafile,
                strerror(errno));
        return 1;
    }

    data = malloc(n * d * sizeof(*data));
    centroids = malloc(k * d * sizeof(*centroids));
    sums = malloc(k * d * sizeof(*sums));
    counts = malloc(k * sizeof(*counts));
    assign = malloc(n * sizeof(*assign));
    tsums = malloc((size_t)nthreads * stride_d * sizeof(*tsums));
    tcounts = malloc((size_t)nthreads * stride_k * sizeof(*tcounts));
    if (data == NULL || centroids == NULL || sums == NULL ||
        counts == NULL || assign == NULL || tsums == NULL ||
        tcounts == NULL) {
        fprintf(stderr, "Error: out of memory for N=%zu, D=%zu, K=%zu, "
                        "threads=%d.\n",
                n, d, k, nthreads);
        goto fail;
    }

    if (fread(data, sizeof(*data), n * d, fp) != n * d) {
        fprintf(stderr,
                "Error: '%s' is shorter than N*D doubles "
                "(expected %zu points x %zu dims).\n",
                datafile, n, d);
        goto fail;
    }
    fclose(fp);
    fp = NULL;

    /* Reject non-finite input before it can poison centroids/SSE. */
    for (i = 0; i < n * d; i++) {
        if (!isfinite(data[i])) {
            fprintf(stderr, "Error: non-finite value in dataset at "
                            "element %zu.\n",
                    i);
            goto fail;
        }
    }

    /* Initial centroids: deterministic farthest-point, IDENTICAL to the
     * serial baseline (see header). Kept serial on purpose. */
    {
        double *mind = malloc(n * sizeof(*mind));
        if (mind == NULL) {
            fprintf(stderr, "Error: out of memory for initialisation.\n");
            goto fail;
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
            memcpy(centroids + c * d, data + far * d,
                   d * sizeof(*centroids));
        }
        free(mind);
    }

    t_start = omp_get_wtime();
    for (iter = 0; iter < max_iters; iter++) {
        double iter_sse = 0.0;
        int t;

        memset(tsums, 0, (size_t)nthreads * stride_d * sizeof(*tsums));
        memset(tcounts, 0, (size_t)nthreads * stride_k * sizeof(*tcounts));

        /* Assignment step, parallelised. Each thread writes only its
         * own padded partials, so there is no race and no false sharing.
         * Tie-breaking matches serial exactly (strict <, lowest index). */
#pragma omp parallel default(none) shared(data, centroids, assign, tsums, tcounts, n, d, k, nthreads, stride_d, stride_k) reduction(+:iter_sse)
        {
            int tid = omp_get_thread_num();
            double *lsums = tsums + (size_t)tid * stride_d;
            size_t *lcounts = tcounts + (size_t)tid * stride_k;

#pragma omp for schedule(static)
            for (size_t ii = 0; ii < n; ii++) {
                const double *pt = data + ii * d;
                double best_dist = 0.0;
                size_t best = 0;
                size_t cc, jj;

                for (cc = 0; cc < k; cc++) {
                    const double *cent = centroids + cc * d;
                    double dist = 0.0;

                    for (jj = 0; jj < d; jj++) {
                        double diff = pt[jj] - cent[jj];
                        dist += diff * diff;
                    }
                    if (cc == 0 || dist < best_dist) {
                        best_dist = dist;
                        best = cc;
                    }
                }
                assign[ii] = (int)best;
                iter_sse += best_dist;
                for (jj = 0; jj < d; jj++) {
                    lsums[best * d + jj] += pt[jj];
                }
                lcounts[best]++;
            }
        } /* implicit barrier: all threads done before the merge */

        /* Merge per-thread partials into the global sums/counts. */
        memset(sums, 0, k * d * sizeof(*sums));
        memset(counts, 0, k * sizeof(*counts));
        for (t = 0; t < nthreads; t++) {
            const double *lsums = tsums + (size_t)t * stride_d;
            const size_t *lcounts = tcounts + (size_t)t * stride_k;

            for (c = 0; c < k; c++) {
                counts[c] += lcounts[c];
                for (j = 0; j < d; j++) {
                    sums[c * d + j] += lsums[c * d + j];
                }
            }
        }

        /* Update step + convergence shift (serial, same as baseline). */
        shift = 0.0;
        for (c = 0; c < k; c++) {
            double move_sq = 0.0;

            if (counts[c] == 0) {
                /* Empty cluster: keep the previous centroid unchanged. */
                empty_events++;
                continue;
            }
            for (j = 0; j < d; j++) {
                double new_val = sums[c * d + j] / (double)counts[c];
                double diff = new_val - centroids[c * d + j];
                centroids[c * d + j] = new_val;
                move_sq += diff * diff;
            }
            if (sqrt(move_sq) > shift) {
                shift = sqrt(move_sq);
            }
        }
        sse = iter_sse;

        if (shift < epsilon) {
            converged = 1;
            iter++; /* count the iteration just completed */
            break;
        }
    }
    t_end = omp_get_wtime();

    if (assignfile != NULL) {
        FILE *af = fopen(assignfile, "wb");
        size_t w;

        if (af == NULL) {
            fprintf(stderr, "Error: cannot write '%s': %s\n", assignfile,
                    strerror(errno));
            goto fail;
        }
        /* Write as fixed-width int32 so later versions compare exactly. */
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
                goto fail;
            }
        }
    }

    printf("K-Means OpenMP\n");
    printf("  data: %s  N=%zu  D=%zu  K=%zu  max_iters=%zu  epsilon=%g\n",
           datafile, n, d, k, max_iters, epsilon);
    printf("  threads: %d\n", nthreads);
    printf("  iterations: %zu  converged: %s  final max centroid shift: %.6g\n",
           iter, converged ? "yes" : "no", shift);
    printf("  SSE: %.6g  empty-cluster events: %zu\n", sse, empty_events);
    printf("  time (wall): %.3f s\n", t_end - t_start);
    printf("  cluster counts:");
    for (c = 0; c < k; c++) {
        printf(" %zu", counts[c]);
    }
    printf("\n  final centroids:\n");
    for (c = 0; c < k; c++) {
        printf("    [%zu]", c);
        for (j = 0; j < d; j++) {
            printf(" %.6f", centroids[c * d + j]);
        }
        printf("\n");
    }

    free(data);
    free(centroids);
    free(sums);
    free(counts);
    free(assign);
    free(tsums);
    free(tcounts);
    return 0;

fail:
    if (fp != NULL) {
        fclose(fp);
    }
    free(data);
    free(centroids);
    free(sums);
    free(counts);
    free(assign);
    free(tsums);
    free(tcounts);
    return 1;
}

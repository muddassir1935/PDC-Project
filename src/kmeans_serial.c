/*
 * kmeans_serial.c — Serial K-Means baseline (PDC Project, Milestone M1).
 *
 * Lloyd's algorithm with Euclidean distance:
 *   1. Assign every point to its nearest centroid.
 *   2. Recompute each centroid as the mean of its assigned points.
 *   3. Repeat until convergence or max iterations.
 *
 * Shared conventions (MUST stay identical in kmeans_mpi.c, kmeans_omp.c
 * and kmeans_hybrid.c, otherwise cross-version comparison is meaningless):
 *
 *   Data format (see scripts/gen_data.py):
 *     Binary, headerless, little-endian float64 (C double), row-major:
 *     point i occupies elements [i*D, i*D + D). No header — N and D
 *     come from the command line (and from the generator's sidecar
 *     <out>.meta.json / the filename, never from the file itself).
 *
 *   Initial centroids (deterministic farthest-point / maximin):
 *     Centroid 0 is row 0 of the file; each next centroid is the point
 *     farthest (squared Euclidean) from its nearest already-chosen
 *     centroid, ties broken by lowest index. This spreads the starting
 *     centroids across the data instead of bunching them in one cluster.
 *     NOTE: the first version used "first K rows" as initial centroids,
 *     but that failed verification on the Small tier: the first 8 rows
 *     covered only 3 of the 8 true clusters (five came from one cluster),
 *     leaving one centroid empty for 99 iterations, two true clusters
 *     merged and one split, with SSE 1.03e7 vs ~1.6e6 for the true
 *     clustering. A random sample of K rows from K clusters almost never
 *     covers all clusters, so farthest-point replaced it. It needs no
 *     PRNG, so every later version reproduces it exactly from the file
 *     alone. Cost O(K*N*D) via an incremental nearest-centroid distance
 *     array; no extra pass over the data per candidate.
 *
 *   Convergence:
 *     shift = max over centroids k of ||new_k - old_k||_2 (Euclidean).
 *     Stop when shift < epsilon, or when max_iters iterations are done.
 *
 *   Empty clusters:
 *     If a cluster gets zero points in an iteration, its centroid is kept
 *     unchanged (no division by zero, no NaN). Empty-cluster events are
 *     counted and reported.
 *
 * Usage:
 *   kmeans_serial <datafile> <N> <D> <K> <max_iters> <epsilon>
 *                   [assignments_file]
 *   assignments_file (optional): final assignment of every point, one
 *     int32 per point in file order — used by M2/M3/M4 to compare against
 *     the serial reference.
 *
 * Build:  make serial          (or: gcc -O2 -Wall -Wextra -std=c11
 *                               -o bin/kmeans_serial src/kmeans_serial.c -lm)
 * Example: python3 scripts/gen_data.py --tier small --out data/small.bin
 *          bin/kmeans_serial data/small.bin 100000 16 8 100 1e-4
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
            "  epsilon: positive convergence threshold on max centroid shift\n",
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
    double *sums = NULL;       /* K x D */
    size_t *counts = NULL;     /* K */
    int *assign = NULL;        /* N */
    double sse = 0.0;
    double shift = 0.0;
    size_t empty_events = 0;
    int converged = 0;
    clock_t t_start, t_end;

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
    if (data == NULL || centroids == NULL || sums == NULL ||
        counts == NULL || assign == NULL) {
        fprintf(stderr, "Error: out of memory for N=%zu, D=%zu, K=%zu.\n",
                n, d, k);
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

    /* Initial centroids: deterministic farthest-point (see header). */
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

    t_start = clock();
    for (iter = 0; iter < max_iters; iter++) {
        double iter_sse = 0.0;

        memset(sums, 0, k * d * sizeof(*sums));
        memset(counts, 0, k * sizeof(*counts));

        /* Assignment step: nearest centroid by squared Euclidean distance
         * (sqrt is monotone, so comparing squared distances is enough). */
        for (i = 0; i < n; i++) {
            const double *pt = data + i * d;
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
            assign[i] = (int)best;
            iter_sse += best_dist;
            for (j = 0; j < d; j++) {
                sums[best * d + j] += pt[j];
            }
            counts[best]++;
        }

        /* Update step + convergence shift. */
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
    t_end = clock();

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

    printf("K-Means serial baseline\n");
    printf("  data: %s  N=%zu  D=%zu  K=%zu  max_iters=%zu  epsilon=%g\n",
           datafile, n, d, k, max_iters, epsilon);
    printf("  iterations: %zu  converged: %s  final max centroid shift: %.6g\n",
           iter, converged ? "yes" : "no", shift);
    printf("  SSE: %.6g  empty-cluster events: %zu\n", sse, empty_events);
    printf("  time: %.3f s\n",
           (double)(t_end - t_start) / (double)CLOCKS_PER_SEC);
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
    return 1;
}

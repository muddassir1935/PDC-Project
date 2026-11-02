# Architecture — Hybrid MPI + OpenMP K-Means

## Why hybrid?

- **Pure OpenMP** on a multi-core machine can hit memory-bus contention when many threads stream a large dataset simultaneously, and it cannot scale beyond one node's shared memory.
- **Pure MPI** pays extra communication and memory overhead: every process holds its own copy of shared data (e.g. centroids) and all coordination goes through messages.
- **Hybrid** balances both: MPI gives coarse-grained partitioning *between* processes (each process owns only its N/P slice), OpenMP gives fine-grained parallelism *inside* a process over its local slice. Communication happens once per iteration (one `MPI_Allreduce`), while the heavy distance work (~95% of runtime) is threaded locally with no message overhead.

## Data layout

- Dataset: N points × D dimensions, stored as one flat `double`/`float` array in row-major order (point i occupies elements `i*D .. i*D+D-1`), so a contiguous block of points is one contiguous memory range — exactly what `MPI_Scatterv` sends.
- Centroids: K × D array, replicated on every rank (small: even K=64, D=128 is only 8192 values).
- Assignments: one `int` per local point, local to each rank.

## One iteration, step by step

1. Every rank holds the same current centroids (initially broadcast by Rank 0; afterwards computed identically on all ranks — see step 4).
2. **Assignment (OpenMP):** each thread takes a static share of the local points, computes the Euclidean distance to all K centroids, picks the nearest, and accumulates:
   - a **per-thread** partial sum array (K × D) and per-thread counts (K), to avoid a race on shared sums;
   - after the loop, the per-thread partials are merged into the rank's local sums.
   Per-thread arrays are padded/aligned so two threads' hot counters do not share a cache line (**false sharing**).
3. **Aggregation (MPI):** `MPI_Allreduce(local_sums, global_sums, K*D, MPI_DOUBLE, MPI_SUM)` and the same for counts. One collective gives **every** rank the global result.
4. **Update (replicated):** each rank divides global sums by global counts to get the new centroids. All ranks perform the identical computation on identical data, so all copies stay in sync — this is why the workflow diagram's "broadcast new centroids" step needs no separate `MPI_Bcast` in the implementation.
5. **Convergence (replicated):** displacement = sum of squared centroid movement (or max movement); if < ε or max iterations reached, stop. Every rank reaches the same decision from the same numbers, so no rank exits early and deadlocks a collective.

## Edge cases handled by design

- **N not divisible by P:** `MPI_Scatterv` with per-rank counts/displacements — the first `N % P` ranks get one extra point. (`MPI_Scatter` would require N % P == 0.)
- **Empty cluster:** if a cluster's global count is 0, dividing would produce NaN. Rule: keep the previous centroid unchanged for that iteration.
- **Float reproducibility:** parallel summation order differs from serial, so parallel centroids differ from the serial baseline by tiny floating-point amounts. Correctness is checked with a tolerance (and by comparing final assignments on the vast majority of points), not by bit-equality.
- **Fair comparison:** serial, MPI, OpenMP and hybrid runs all start from the **same initial centroids** (same selection rule and seed); otherwise differences in results could come from initialization, not from parallelization.

## Timing methodology

- `MPI_Wtime()` around the whole run and around the assignment/update phases; `MPI_Barrier` before starting the timer so ranks start together.
- Report computation time and communication (Allreduce) time separately — that split is what the Amdahl's Law comparison is built on.
- Every configuration is run 3–5 times; mean ± standard deviation is reported.

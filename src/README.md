# src/ — Implementations

One file per implementation, all sharing the same input format and the same initial-centroid rule so results are directly comparable:

| File | Implementation | Milestone |
|---|---|---|
| `kmeans_serial.c` | Serial baseline (correctness reference) | M1 |
| `kmeans_mpi.c` | MPI only — Scatterv + Allreduce | M2 |
| `kmeans_omp.c` | OpenMP only — parallel assignment loop | M3 |
| `kmeans_hybrid.c` | Hybrid MPI + OpenMP (final deliverable) | M4 |

Shared conventions (set in M1, kept identical in every version):

- Dataset file: binary, headerless — the generator script records N and D in the filename/CLI, programs take them as arguments.
- Arguments: `<datafile> <N> <D> <K> <max_iters> <epsilon>`
- Output: final centroids, iteration count, and phase timings (compute vs communication) on stdout.

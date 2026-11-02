# Milestones & Working Rules

**Deadline:** expected 15–20 Nov 2026. Code milestones are front-loaded so the core implementation is complete before the window opens; benchmarks polishing, report and slides use the later days.

## Working rules

- Work happens in **real time** — no backdated commits, no fake authors.
- One milestone = one branch `feature/m<n>-<name>`, merged into `main` via pull request after review.
- Every push is reviewed/approved before it lands.
- Each milestone is only "done" when its **Done when** check passes, not when code merely exists.

## Milestones

| # | Milestone | Branch | Done when |
|---|---|---|---|
| M0 | Repository setup: README, Makefile skeleton, folder structure, architecture & milestone docs | `main` (initial commit) | Structure visible on GitHub; `make` runs and reports status |
| M1 | Serial baseline (`src/kmeans_serial.c`) + dataset generator (`scripts/gen_data.py`, fixed seed) — *Lead: Muddassir* | `feature/m1-serial-baseline` | Serial run on Small tier (100k×16, K=8) produces sensible clusters and prints timing |
| M2 | MPI version (`src/kmeans_mpi.c`): Scatterv distribution, local sums/counts, Allreduce aggregation — *Lead: Faiz & Muddassir* | `feature/m2-mpi` | Matches serial assignments (within float tolerance) on same data/seed; runs on 1/2/4 ranks |
| M3 | OpenMP version (`src/kmeans_omp.c`): parallel-for assignment, per-thread sums + merge — *Lead: Qaim & Faiz* | `feature/m3-openmp` | Matches serial; scales across 1/2/4 threads |
| M4 | Hybrid integration (`src/kmeans_hybrid.c`): MPI between processes + OpenMP inside; empty-cluster guard; compute vs communication timing split — *All members* | `feature/m4-hybrid` | Hybrid matches serial; runs in mixed configs (e.g. 2 ranks × 2 threads) |
| M5 | Benchmarks: strong & weak scaling (1/2/4/8 units), 3–5 repeats, speedup/efficiency, Amdahl comparison, plots — *Lead: Qaim & Faiz* | `feature/m5-benchmarks` | CSV results + Speedup / Efficiency / Weak-scaling graphs in `benchmarks/` |
| M6 | Final report + viva slides + demo run instructions — *All members* | `feature/m6-report` | Report PDF and slides in `report/`; a fresh clone builds and runs the demo |

## Scope guards

- Benchmark tiers Small and Medium run in full. Large (1M×128, K=64) runs at least once for the headline number if time allows. Extra Large (5M) is a stretch goal and is the **first** thing cut if time gets tight — core code, correctness and graphs are never cut.
- Benchmarks run on a 4-core machine are real for 1/2/4 units; 8-unit runs on such a machine are oversubscribed and must be labelled as such in the report (or run on a machine with more cores).

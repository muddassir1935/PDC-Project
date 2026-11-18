# PDC Final Report — Draft Outline (M6, prepared 16 Nov 2026)

**Status:** DRAFT outline + verified results tables only. Not the report itself.
Figures are the 4 local PNGs in `benchmarks/` (not yet in remote main — they
need a GitHub web-UI upload or a teammate `git push` before the report cites
them from the repo; binaries cannot go through the snapshot push route).
All numbers below were recomputed from `benchmarks/results.csv` (69 rows) on
16 Nov 2026 and match `benchmarks/M5_RUN_LOG.md`.

**Team:** Muddassir Rizwan 23K-0784, Faiz Hussain 23K-2061, Qaim Mehdi 23K-2075
**Project:** Parallel K-Means Clustering on Large-Scale Datasets using Hybrid MPI + OpenMP
**Report/viva window:** Tue–Thu 18–20 Nov 2026

## 1. Proposed structure

1. Introduction & problem — why K-Means, why hybrid MPI+OpenMP, dataset tiers
2. Design — SPMD, Rank 0 farthest-point (maximin) init, MPI_Scatterv slices,
   OpenMP parallel-for assignment, local K×D sums + counts, MPI_Allreduce(MPI_SUM),
   convergence Δ centroids < ε (1e-6) or max 100 iterations
3. Implementation — serial / MPI / OpenMP / hybrid, empty-cluster guard,
   deterministic init (fixed seed 42) so every config does identical work
4. Correctness — hybrid 2×2 vs serial on medium-hard: **0 mismatches / 500,000
   assignments**, max centroid diff 0 (bit-identical), tol 1e-4
5. Methodology — machine, timing definitions, repeats (3 per config)
6. Strong scaling — table + `speedup.png` + `efficiency.png`
7. Weak scaling — table + `weak_scaling.png`
8. Scheduling — static vs dynamic + `schedule.png`
9. Communication analysis — compute vs comm split per config
10. Limitations & honest reading — host cap, variance, oversubscription
11. Conclusion & future work

## 2. Benchmark setup (verified)

- Dataset for all scaling numbers: **medium-hard** — N=500,000, D=64, K=32,
  centre range [-1, 1], noise 1.0, seed 42, **100 iterations fixed**
  (clusters overlap, so Lloyd's runs the full 100; standard Medium converges
  in 2 iterations and is useless for scaling — see run log §"Why medium-hard").
- Machine: this workspace VM — 4 vCPUs visible, ~7.7 GB RAM, **shared host CPU
  (~1.5 CPUs)**. All speedups below are **host-capped, indicative only**.
  Definitive scaling numbers belong on a dedicated machine (user's 2P+10E laptop
  or a teammate's machine).
- Timing: serial = iteration loop only; hybrid wall = Bcast + Scatterv +
  iterations + Gatherv; MPI = loop incl. per-iteration Allreduce.

## 3. Strong scaling — verified table (T1 = 152.72 s)

| config | wall (mean ± std) | S = T1/Tp | E = S/P |
|---|---|---|---|
| serial 1×1 | 152.72 ± 1.07 s | 1.00 | 1.00 |
| hybrid 1×1 | 158.11 s (1 rep) | 0.97 | 0.97 |
| hybrid 1×2 | 122.21 ± 25.66 s | 1.25 | 0.62 |
| hybrid 2×1 | 64.76 ± 0.25 s | 2.36 | 1.18 |
| hybrid 2×2 | 64.59 ± 0.31 s | 2.36 | 0.59 |
| hybrid 4×1 | 74.17 ± 1.40 s | 2.06 | 0.51 |
| hybrid 1×4 | 92.81 ± 0.15 s | 1.65 | 0.41 |
| hybrid 4×2 * | 85.11 ± 3.54 s | 1.79 | 0.22 |
| hybrid 2×4 * | 64.67 ± 0.46 s | 2.36 | 0.30 |
| hybrid 8×1 * | 101.20 ± 5.29 s | 1.51 | 0.19 |
| mpi 2×1 | 55.44 ± 0.26 s | 2.75 | 1.38 |
| mpi 4×1 | 65.31 ± 0.90 s | 2.34 | 0.58 |
| mpi 8×1 * | 82.19 ± 6.25 s | 1.86 | 0.23 |
| omp 1×2 | 56.87 ± 0.88 s | 2.69 | 1.34 |
| omp 1×4 | 62.28 ± 1.06 s | 2.45 | 0.61 |

\* oversubscribed (8 units on a 4-vCPU VM). Figure: `benchmarks/speedup.png`
(best decomposition per unit count), `benchmarks/efficiency.png`.

**Reading (put this in the report, verbatim if useful):**
- Everything plateaus at ~2.4–2.75×: the VM's shared CPU, not a code limit.
- E > 1 at 2 units (MPI 1.38, OMP 1.34) is NOT superlinear scaling — host-share
  variance across the ~1 h run plus cache effects. On a dedicated machine E ≤ 1
  is expected.
- Ranks beat threads on this host: 2×1 (64.8 s) >> 1×2 (122.2 s, noisy:
  137/137/93 s); pure-MPI 2 ranks (55.4 s) and pure-OpenMP 2 threads (56.9 s)
  both beat hybrid 2×2 (64.6 s) — hybrid pays both MPI collectives and OpenMP
  region/merge overhead per iteration.

## 4. Communication share (hybrid, max across ranks)

| config | comm share |
|---|---|
| 2×1 / 2×2 | ~2% |
| 4×1 | 11.4% (8.5 s) |
| 4×2 | 26.3% (22.4 s) |
| 8×1 | **48.6% (49.2 s)** — under oversubscription the per-iteration Allreduces dominate |

## 5. Static vs dynamic schedule (same source)

| config | dynamic (1024) | static | static faster by |
|---|---|---|---|
| 2×2 | 65.56 ± 0.58 s | 55.90 ± 0.38 s | 14.7% |
| 1×4 | 92.95 ± 0.73 s | 77.58 ± 0.22 s | 16.5% |

Static wins on this VM because its cores are equal — no load imbalance, no
dynamic dispatch overhead. **Viva point the other way:** on the user's 2P+10E
laptop dynamic is expected to win (fast cores idle at the barrier under
static). Figure: `benchmarks/schedule.png`. Re-run on the laptop if time allows.

## 6. Weak scaling (125k points per rank, time per iteration)

| P | dataset | iters | s/iter |
|---|---|---|---|
| 1 | weak-125k | 24 (converged) | 0.238 |
| 2 | weak-250k | 100 | 0.334 |
| 4 | medium-hard | 100 | 0.753 |
| 8 * | weak-1m | 100 | 2.005 |

Per-iteration time rises steeply instead of staying flat — mostly the host CPU
budget split over more ranks, not the code. Caveat: P=1 converged in 24 iters
(less work), so compare per-iteration, not wall time. Figure:
`benchmarks/weak_scaling.png`.

## 7. Do NOT quote as scaling results

- **Large headline (1M×128, K=64, 2 iters):** serial 13.23 s, hybrid 2×2
  16.97 s — hybrid is SLOWER because the tier converges in 2 iterations and
  the timed region is dominated by the one-off 1 GB distribution/gather.
  Not a scaling result.
- **8-unit runs:** oversubscribed on this VM; label them as such or re-run on
  a machine with more cores.
- **VM timings generally:** run-to-run variance is large (hybrid 1×2 repeats:
  137.1 / 137.0 / 92.6 s) — the host share is burstable, not a fixed cap.

## 8. Open decisions for the user (before Tue)

1. Plots upload: GitHub web UI → `benchmarks/` (4 PNGs), or Faiz/Qaim `git push`.
2. Re-run key configs on the laptop (2P+10E) for real scaling numbers? At minimum
   the static-vs-dynamic comparison — that is the viva's strongest laptop point.
3. Large-hard variant (like medium-hard, 1M×128 overlapping clusters) for a real
   Large scaling number, or cut Large scaling and report it as a limitation?
4. Report format: PDF via LaTeX / Typst / Word — pick one; slides after the report.

## Sources

- `benchmarks/results.csv` (69 rows, recomputed 16 Nov 2026)
- `benchmarks/M5_RUN_LOG.md` (methodology, caveats, interruption log)
- `benchmarks/plot_benchmarks.py` (regenerates all 4 PNGs from the CSV)
- `docs/milestones.md` (M6 scope: report PDF + viva slides in `report/`)

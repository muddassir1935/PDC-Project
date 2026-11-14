# M5 Benchmark Run Log — 14 Nov 2026

Branch: `feature/m5-benchmarks` (from main `36d0cc0`, M4 merged via PR #4).
Runner: `benchmarks/run_benchmarks.py` (3 repeats per config, 1 for the
two singletons). Plots: `benchmarks/plot_benchmarks.py`.
Raw results: `benchmarks/results.csv` (one row per run, written incrementally).

## Machine (this workspace VM)

- 4 vCPUs visible (`nproc` = 4), AMD EPYC 9D64, ~7.7 GB RAM.
- **Host CPU cap:** the cell shares ~1.5 CPUs (runtime `cell_shared_cpus=1.5`).
  Measured: serial Medium-hard 148.5 s vs hybrid 2 ranks x 2 threads compute
  97.3 s = 1.53x — exactly the cap, not a code limit. All VM speedups are
  therefore **host-capped** and must be labelled as such in the report;
  real scaling numbers need the user's laptop (2P+10E) or a teammate's
  machine. 8-unit runs are additionally oversubscribed (8 units > 4 vCPUs).
- Toolchain: gcc 13.3, persistent OpenMPI wrappers
  `~/workspace/tools/openmpi/bin/{mpicc,mpirun}` (OpenMPI 4.1.6, repaired
  13 Nov), `mpirun --allow-run-as-root`. Python 3.12, numpy 1.26.4,
  matplotlib 3.6.3.

## Datasets

| dataset | N | D | K | centres | noise | seed | file |
|---|---|---|---|---|---|---|---|
| medium (standard) | 500,000 | 64 | 32 | [-10, 10] | 1.0 | 42 | data/medium.bin (pre-existing) |
| **medium-hard** | 500,000 | 64 | 32 | **[-1, 1]** | 1.0 | 42 | data/medium-hard.bin |
| weak-125k | 125,000 | 64 | 32 | [-1, 1] | 1.0 | 42 | data/weak-125k.bin |
| weak-250k | 250,000 | 64 | 32 | [-1, 1] | 1.0 | 42 | data/weak-250k.bin |
| weak-1m | 1,000,000 | 64 | 32 | [-1, 1] | 1.0 | 42 | data/weak-1m.bin |
| large (standard) | 1,000,000 | 128 | 64 | [-10, 10] | 1.0 | 42 | data/large.bin |

Datasets are NOT committed (regenerable from seed + parameters);
`data/*.bin` is gitignored, sidecar `*.meta.json` records parameters.

## Why medium-hard (fixed 100 iterations)

- Standard Medium converges in **2 iterations** (final shift exactly 0):
  in 64-D, centres ~90 apart vs noise spread ~8, so farthest-point init
  already lands one centroid per true cluster. Its timed region is then
  dominated by one-off data distribution, not the iteration loop — useless
  for scaling. Tighter epsilon (1e-6) does not help: shift is exactly 0.
- Centre range 2.0 still converges in 3 iterations. Range 1.0 makes the
  clusters overlap (inter-centre distance ~6.5 vs noise spread ~8), so
  Lloyd's algorithm runs the full 100 iterations (final shift 0.0031,
  not converged at eps=1e-6). Every configuration therefore does
  IDENTICAL work: 100 iterations, same data, same deterministic init.
- `scripts/gen_data.py` gained `--centre-range` / `--noise-std` flags
  (M5 change, defaults unchanged: 10.0 / 1.0, recorded in the sidecar).
- SSE is ~3.2e7 for both Medium and medium-hard: SSE measures
  within-cluster variance (N*D*sigma^2 = 500k*64*1), not centre spread,
  so equal SSE is expected, not a data error.

## Timing definitions (differ per program — read before comparing)

- serial: iteration loop only (timer starts after farthest-point init).
- mpi: iteration loop incl. per-iteration Allreduce (Bcast/Scatterv
  distribution and Gatherv excluded).
- omp: iteration loop wall time.
- hybrid: wall = Bcast + Scatterv + iterations + Gatherv; compute and
  communication split reported (max across ranks).
- At 100 iterations the loop dominates in every program; the one-off
  distribution/gather is a few percent of hybrid wall time (measured
  2.9% communication at 2x2). `time_per_iter_s` in the CSV makes the
  fixed-iteration design explicit.

## Static vs dynamic schedule

- Same hybrid source; static variant built with `-DHYBRID_STATIC_SCHEDULE`
  (`make hybrid-static` -> `bin/kmeans_hybrid_static`). Default build is
  unchanged: `schedule(dynamic, 1024)`.
- Expectation for the viva: on the user's 2P+10E laptop dynamic should win
  (fast cores would idle at the barrier under static); on this host-capped
  VM the gap is expected to be small — the comparison is a methodology
  point, and the real numbers belong on the laptop.

## Configurations run

- Strong (medium-hard, 100 iters, eps 1e-6): serial; hybrid 1x1 (1 rep,
  sanity), 1x2, 2x1, 2x2, 4x1, 1x4, 4x2*, 2x4*, 8x1*; MPI-only 2/4/8*;
  OpenMP-only 2/4 threads. (* = oversubscribed on this VM.)
- Static vs dynamic: hybrid vs hybrid-static at 2x2 and 1x4.
- Weak (125k points per rank): hybrid 1x1, 2x1, 4x1, 8x1* on
  weak-125k / weak-250k / medium-hard / weak-1m.
- Large headline (standard tier, 1 rep each): serial + hybrid 2x2 on
  large (1M x 128, K=64).

## Results (filled in after the run)

- See `benchmarks/results.csv` and the plots (`speedup.png`,
  `efficiency.png`, `weak_scaling.png`, `schedule.png`).
- Suite completed 14 Nov ~12:55 PKT: **69 runs** (3 repeats per config,
  1 for hybrid 1x1 and the two Large runs). Serial T1 = 152.72 +- 1.07 s.
- **Correctness: PASS** — hybrid 2x2 vs serial on medium-hard:
  0 mismatches / 500,000 assignments, max centroid diff 0 (bit-identical),
  `compare_assignments.py` tol 1e-4.

### Strong scaling (medium-hard, 100 iters, T1 = 152.72 s)

| config | wall (mean +- std) | S = T1/Tp | E = S/P |
|---|---|---|---|
| serial 1x1 | 152.72 +- 1.07 s | 1.00 | 1.00 |
| hybrid 1x1 | 158.11 s (1 rep) | 0.97 | 0.97 |
| hybrid 1x2 | 122.21 +- 25.66 s | 1.25 | 0.62 |
| hybrid 2x1 | 64.76 +- 0.25 s | 2.36 | 1.18 |
| hybrid 2x2 | 64.59 +- 0.31 s | 2.36 | 0.59 |
| hybrid 4x1 | 74.17 +- 1.40 s | 2.06 | 0.51 |
| hybrid 1x4 | 92.81 +- 0.15 s | 1.65 | 0.41 |
| hybrid 4x2 * | 85.11 +- 3.54 s | 1.79 | 0.22 |
| hybrid 2x4 * | 64.67 +- 0.46 s | 2.36 | 0.30 |
| hybrid 8x1 * | 101.20 +- 5.29 s | 1.51 | 0.19 |
| mpi 2x1 | 55.44 +- 0.26 s | 2.75 | 1.38 |
| mpi 4x1 | 65.31 +- 0.90 s | 2.34 | 0.58 |
| mpi 8x1 * | 82.19 +- 6.25 s | 1.86 | 0.23 |
| omp 1x2 | 56.87 +- 0.88 s | 2.69 | 1.34 |
| omp 1x4 | 62.28 +- 1.06 s | 2.45 | 0.61 |

\* oversubscribed (8 units on a 4-vCPU VM). Hybrid curve in `speedup.png`
uses the best decomposition per unit count (2x1, 2x2, 2x4).

**Reading the numbers (all host-capped — see below):**
- Everything plateaus at ~2.4–2.75x: the VM's shared CPU, not the code.
  Do NOT quote these speedups as the project's scaling in the report
  without the host-cap label.
- E > 1 at 2 units (MPI 1.38, OMP 1.34, hybrid 2x1 1.18) is NOT real
  superlinear scaling — it is host-share variance across the ~1 h run
  (parallel configs ran when the host gave more CPU) plus cache effects.
  On a dedicated machine E <= 1 is expected.
- Ranks beat threads on this host: 2x1 (64.8 s) >> 1x2 (122.2 s, and
  noisy: 137/137/93 s); 4x1 (74.2 s) beats 1x4 (92.8 s). Pure-MPI 2
  ranks (55.4 s) and pure-OpenMP 2 threads (56.9 s) both beat hybrid
  2x2 (64.6 s) — the hybrid pays both MPI collectives and OpenMP
  region/merge overhead per iteration.
- Communication share (hybrid, max across ranks): 2x1/2x2 ~2%,
  4x1 11.4% (8.5 s), 4x2 26.3% (22.4 s), 8x1 **48.6% (49.2 s)** —
  under oversubscription the per-iteration Allreduces dominate. That is
  a real effect worth one viva line; the absolute seconds are host-bound.

### Static vs dynamic schedule (same source, 2x2 and 1x4)

| config | dynamic(1024) | static | static faster by |
|---|---|---|---|
| 2x2 | 65.56 +- 0.58 s | 55.90 +- 0.38 s | 14.7% |
| 1x4 | 92.95 +- 0.73 s | 77.58 +- 0.22 s | 16.5% |

Static wins on this VM because its cores are equal — no load imbalance
for static to suffer from, and no dynamic dispatch overhead. The viva
point stands the other way round: on the user's 2P+10E laptop dynamic
is expected to win (fast cores idle at the barrier under static).
Re-run this comparison on the laptop for the report.

### Weak scaling (time per iteration; 125k points per rank)

| P | dataset | iters | s/iter |
|---|---|---|---|
| 1 | weak-125k | 24 (converged) | 0.238 |
| 2 | weak-250k | 100 | 0.334 |
| 4 | medium-hard | 100 | 0.753 |
| 8 * | weak-1m | 100 | 2.005 |

Per-iteration time rises steeply with P instead of staying flat — again
mostly the host CPU budget (a fixed ~1.5–4 core share split over more
ranks), not the code; 8x1 is additionally oversubscribed and pays
~200 s wall (195–207 s) for 1M points. The P=1 point is work-normalised
(it converged in 24 iters; see caveat above).

### Large headline (standard tier, 1M x 128, K=64, 2 iters)

Serial 13.23 s, hybrid 2x2 16.97 s — hybrid is SLOWER, because the tier
converges in 2 iterations and the timed region is then dominated by the
one-off 1 GB distribution/gather, which serial never pays. Not a scaling
result; do not quote it as one. (A hard Large variant, like medium-hard,
would be needed for a real Large scaling number — M6/report decision.)

## Interruption 14 Nov ~11:05 PKT (runtime restart)

- The first full run (PID 18562, started ~10:50) was killed by a
  runtime/VM restart during hybrid 1x2 repeat 3/3, after ~16 min.
  No rows were lost or half-written: `results.csv` kept its 6 completed
  rows (serial x3, hybrid 1x1 x1, hybrid 1x2 x2) and the in-flight
  repeat simply never landed.
- Resume: `benchmarks/run_benchmarks.py --resume` (added after the
  restart) skips configurations already complete in the CSV and
  continues numbering from the next repeat. Logs: `raw/m5-run.log`,
  `raw/m5-run-resumed.log` (gitignored raw transcripts).
- Early numbers (host-capped, see above): serial 151.5 / 153.6 / 153.0 s;
  hybrid 1x1 158.1 s (MPI overhead over serial); hybrid 1x2 137.1 /
  137.0 s (only ~1.11x vs serial — the ~1.5-CPU cap, not the code).
- **Variance warning (resumed run):** hybrid 1x2 repeat 3 measured
  **92.6 s** vs 137.1 / 137.0 s for repeats 1-2, and hybrid 2x1 measured
  ~65 s (2.35x vs serial — above the supposed 1.5-CPU cap). The host CPU
  share is NOT a fixed cap: it varies run to run on this shared VM
  (burstable / noisy neighbours). So VM timings carry large run-to-run
  variance — the CSV's mean +/- std shows it, and the report must say
  the VM numbers are indicative only; definitive scaling numbers belong
  on a dedicated machine (user's laptop / teammate's machine).
- **Weak-scaling caveat:** weak-125k converged in **24 iterations**
  (not 100) — with fewer points the centroid shifts drop below eps=1e-6
  sooner — while weak-250k / medium-hard / weak-1m run the full 100.
  Wall-time weak scaling is therefore confounded (P=1 does less work);
  the CSV records `iterations` and `time_per_iter_s`, and the weak plot
  uses time-per-iteration so the comparison is work-normalised. For a
  future re-run on real hardware, either fix the iteration count (e.g.
  eps tiny enough that all tiers hit max_iters) or compare per-iteration.
- **mpirun slot failure (resumed run, ~25 runs in):** hybrid 8x1 refused
  to launch — Open MPI defaults slots to core count (4) and needs
  `--oversubscribe` for 8 ranks. Not a code failure. Runner fixed
  (`--oversubscribe` on all mpirun calls) and resumed with `--resume`;
  4x2/2x4 (8 units via threads) were unaffected — only 8 MPI ranks hit
  the slot limit. Same applies to MPI-only 8 ranks and weak 8x1.

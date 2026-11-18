# Parallel K-Means Clustering on Large-Scale Datasets using Hybrid MPI + OpenMP

**Status: DRAFT skeleton (Markdown) — 17 Nov 2026.** Sections 1–11 follow the
M6 outline; every results table is verified against `benchmarks/results.csv`
(69 rows, 14 Nov 2026). Prose is drafted where it does not depend on open
decisions; anything that does is marked **[PENDING DECISION]**.

**Final format: [PENDING DECISION]** — LaTeX / Typst / Word, to be chosen by
the user. This file converts to whichever is picked; nothing here assumes one.

**Figures: [PENDING]** — the 4 PNGs (`speedup.png`, `efficiency.png`,
`weak_scaling.png`, `schedule.png`) exist only locally in `benchmarks/`.
They are NOT in remote main (binaries cannot go through the snapshot push
route). Before the report cites them from the repo, upload them via the
GitHub web UI into `benchmarks/` or a teammate `git push`. Figure slots
below use relative paths that resolve once the PNGs are in the repo.

**Team:** Muddassir Rizwan (23K-0784), Faiz Hussain (23K-2061),
Qaim Mehdi (23K-2075)
**Course:** Parallel & Distributed Computing (PDC), FAST-NUCES Karachi
**Report / viva window:** 18–20 Nov 2026

---

## 1. Introduction

K-Means is one of the most widely used clustering algorithms: given N points
in D dimensions, partition them into K clusters by repeatedly assigning each
point to its nearest centroid and recomputing centroids as cluster means,
until the centroids stop moving. The assignment step is embarrassingly
parallel — each point's nearest centroid can be found independently — which
makes K-Means a natural fit for parallelisation. The interesting question is
not whether it parallelises, but how the two standard models combine: MPI
between processes (distributed memory, explicit communication) and OpenMP
inside a process (shared memory, thread-level parallelism).

This project implements K-Means four ways — serial, MPI-only, OpenMP-only,
and hybrid MPI + OpenMP — on the same deterministic workload, verifies the
parallel versions against the serial baseline, and measures strong scaling,
weak scaling, scheduling policy, and communication cost. The headline result
is honest rather than flattering: on the shared virtual machine used for
benchmarking, every configuration plateaus at roughly 2.4–2.75× speedup,
because the host, not the code, is the bottleneck (§6, §11). The report says
so plainly, and says which numbers would change on dedicated hardware.

## 2. Problem and datasets

### 2.1 Datasets

All scaling numbers use the **medium-hard** tier unless stated otherwise:
N = 500,000 points, D = 64, K = 32, centres drawn from [-1, 1], noise
σ = 1.0, fixed seed 42. The narrow centre range makes clusters overlap, so
Lloyd's algorithm runs the full 100 iterations instead of converging early —
every configuration therefore does identical work, which is what makes the
timing comparison fair.

| dataset | N | D | K | centres | noise | seed |
|---|---:|---:|---:|---|---:|---:|
| small | 100,000 | 16 | 8 | [-10, 10] | 1.0 | 42 |
| medium (standard) | 500,000 | 64 | 32 | [-10, 10] | 1.0 | 42 |
| **medium-hard** | **500,000** | **64** | **32** | **[-1, 1]** | **1.0** | **42** |
| weak-125k | 125,000 | 64 | 32 | [-1, 1] | 1.0 | 42 |
| weak-250k | 250,000 | 64 | 32 | [-1, 1] | 1.0 | 42 |
| weak-1m | 1,000,000 | 64 | 32 | [-1, 1] | 1.0 | 42 |
| large (standard) | 1,000,000 | 128 | 64 | [-10, 10] | 1.0 | 42 |

Datasets are regenerable from seed + parameters (`scripts/gen_data.py`) and
are not committed; sidecar `*.meta.json` files record the parameters.

Why not standard Medium for scaling: it converges in **2 iterations** (in
64-D, centres ~90 apart vs noise spread ~8, so farthest-point initialisation
already lands one centroid per true cluster). Its timed region is then
dominated by one-off data distribution, not the iteration loop — useless for
scaling. The same 2-iteration artefact affects the Large tier (§11).

### 2.2 Algorithm

Lloyd's algorithm with deterministic farthest-point (maximin)
initialisation by Rank 0, fixed seed 42. Convergence when the maximum
centroid shift Δ < ε = 1e-6, or after max 100 iterations. Medium-hard runs
the full 100 iterations (final shift 0.0031, not converged at ε = 1e-6).

## 3. Design

Single-program-multiple-data (SPMD):

1. Rank 0 selects the K initial centroids (farthest-point) and broadcasts
   them.
2. Data is split into N/P slices with `MPI_Scatterv` (uneven slices allowed).
3. Each rank assigns its local points in parallel with an OpenMP
   `parallel for` over points, accumulating local K×D sums and counts.
4. Local sums and counts are combined across ranks with
   `MPI_Allreduce(MPI_SUM)`; every rank recomputes the same new centroids.
5. Repeat until Δ centroids < ε or 100 iterations; results gathered with
   `MPI_Gatherv`.

The per-iteration communication is exactly one Allreduce of K×D doubles
plus K counts — O(K·D) per iteration, independent of N — so communication
cost grows with the number of ranks, not with the data size.

## 4. Implementation

Four programs share the same initialisation and convergence logic:

- `src/kmeans_serial.c` — baseline; iteration loop only is timed.
- `src/kmeans_mpi.c` — MPI distribution and Allreduce aggregation.
- `src/kmeans_omp.c` — OpenMP parallel-for assignment, per-thread sums
  with cache-line padding, merged after the loop.
- `src/kmeans_hybrid.c` — MPI between ranks, OpenMP within a rank; reports
  a compute-vs-communication timing split (max across ranks).

Guards and determinism: an empty-cluster guard prevents division by zero
when a cluster loses all its points; fixed seed 42 and identical initial
centroids mean every configuration performs the same arithmetic in the same
order per point, so correctness comparisons are exact, not statistical.

Build: `make` (GCC 13.3, OpenMPI 4.1.6). The static-schedule hybrid variant
is built with `-DHYBRID_STATIC_SCHEDULE` (`make hybrid-static`); the
default build uses `schedule(dynamic, 1024)`.

## 5. Correctness

Hybrid 2×2 vs serial on medium-hard, compared with
`compare_assignments.py` (tolerance 1e-4):

- **0 mismatches / 500,000 assignments**
- Maximum centroid difference **0 (bit-identical)**

The parallel versions reproduce the serial result exactly on the same
data and seed. Correctness was verified before any scaling claim was made.

## 6. Methodology

### 6.1 Machine

Benchmarks ran on this workspace VM: 4 vCPUs visible (AMD EPYC 9D64),
~7.7 GB RAM, **shared host CPU of ~1.5 CPUs**. Toolchain: GCC 13.3,
OpenMPI 4.1.6, Python 3.12, NumPy 1.26.4, Matplotlib 3.6.3.

**Host cap — read this before the tables.** All speedups in §7 are
host-capped and indicative only. Measured: serial medium-hard 148.5 s vs
hybrid 2 ranks × 2 threads compute 97.3 s = 1.53× — exactly the host cap,
not a code limit. Definitive scaling numbers belong on a dedicated machine
(the user's 2P+10E laptop or a teammate's machine). **[PENDING DECISION:
laptop re-run — at minimum the static-vs-dynamic comparison.]**

### 6.2 Timing definitions (differ per program)

- serial: iteration loop only (timer starts after initialisation).
- mpi: iteration loop including the per-iteration Allreduce (Bcast/Scatterv
  distribution and Gatherv excluded).
- omp: iteration loop wall time.
- hybrid: wall = Bcast + Scatterv + iterations + Gatherv; compute and
  communication split reported (max across ranks).

At 100 iterations the loop dominates in every program; the one-off
distribution/gather is a few percent of hybrid wall time (measured 2.9%
communication at 2×2).

### 6.3 Repeats

3 repeats per configuration (1 repeat for hybrid 1×1 and for the two Large
runs). 69 runs total, 14 Nov 2026, recorded row-by-row in
`benchmarks/results.csv`; runner `benchmarks/run_benchmarks.py`
(`--resume` supported), plots `benchmarks/plot_benchmarks.py`.

**Variance warning.** The host CPU share is not a fixed cap — it varies run
to run (burstable / noisy neighbours). Hybrid 1×2 measured 137.1 / 137.0 /
92.6 s across its three repeats. Treat every VM timing as indicative; the
mean ± std in the tables shows the spread, and no conclusion in this report
rests on a single run.

## 7. Strong scaling

Dataset: medium-hard, 100 iterations, ε = 1e-6. T1 (serial) = 152.72 s.

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

\* oversubscribed (8 units on a 4-vCPU VM). The hybrid curve in the speedup
figure uses the best decomposition per unit count (2×1, 2×2, 2×4).

![Speedup](../benchmarks/speedup.png)
![Efficiency](../benchmarks/efficiency.png)

**Reading the numbers (all host-capped — §6.1):**

- Everything plateaus at ~2.4–2.75×: the VM's shared CPU, not a code limit.
- E > 1 at 2 units (MPI 1.38, OpenMP 1.34, hybrid 2×1 1.18) is NOT
  superlinear scaling — host-share variance across the ~1 h run plus cache
  effects. On a dedicated machine E ≤ 1 is expected.
- Ranks beat threads on this host: 2×1 (64.8 s) is far faster than 1×2
  (122.2 s, and noisy: 137/137/93 s); 4×1 (74.2 s) beats 1×4 (92.8 s).
  Pure-MPI 2 ranks (55.4 s) and pure-OpenMP 2 threads (56.9 s) both beat
  hybrid 2×2 (64.6 s) — the hybrid pays both MPI collectives and OpenMP
  region/merge overhead per iteration.
- Best wall time overall: MPI-only, 2 ranks, 55.44 s (S = 2.75).

## 8. Weak scaling

125,000 points per rank; time per iteration (work-normalised — see caveat).

| P | dataset | iters | s/iter |
|---|---|---|---|
| 1 | weak-125k | 24 (converged) | 0.238 |
| 2 | weak-250k | 100 | 0.334 |
| 4 | medium-hard | 100 | 0.753 |
| 8 * | weak-1m | 100 | 2.005 |

\* oversubscribed (8 units on a 4-vCPU VM).

![Weak scaling](../benchmarks/weak_scaling.png)

Per-iteration time rises steeply with P instead of staying flat — mostly
the host CPU budget (a fixed ~1.5–4 core share split over more ranks), not
the code. The 8×1 point additionally pays oversubscription (~200 s wall
for 1M points).

**Caveat:** P = 1 converged in 24 iterations while the other tiers ran the
full 100, so wall-time weak scaling is confounded (P = 1 does less work).
The table compares per-iteration time for that reason. On a re-run, either
fix the iteration count or compare per-iteration, as done here.

## 9. Scheduling: static vs dynamic

Same hybrid source; static variant built with `-DHYBRID_STATIC_SCHEDULE`.

| config | dynamic (1024) | static | static faster by |
|---|---|---|---|
| 2×2 | 65.56 ± 0.58 s | 55.90 ± 0.38 s | 14.7% |
| 1×4 | 92.95 ± 0.73 s | 77.58 ± 0.22 s | 16.5% |

![Static vs dynamic](../benchmarks/schedule.png)

Static wins on this VM because its cores are equal — no load imbalance for
static to suffer from, and no dynamic dispatch overhead. **The viva point
runs the other way:** on the user's 2P+10E laptop (fast P-cores + slower
E-cores) dynamic is expected to win, because under static the fast cores
idle at the barrier waiting for the slow ones. **[PENDING DECISION: re-run
this comparison on the laptop — it is the strongest viva point in the
report, and the VM numbers alone cannot show it.]**

## 10. Communication analysis

Hybrid communication share (max across ranks), from the compute/comm split:

| config | comm share |
|---|---|
| 2×1 / 2×2 | ~2% |
| 4×1 | 11.4% (8.5 s) |
| 4×2 | 26.3% (22.4 s) |
| 8×1 | **48.6% (49.2 s)** — under oversubscription the per-iteration Allreduces dominate |

Communication is negligible at 2 units and grows with rank count; at 8
oversubscribed units it is half the wall time. That growth is a real
effect worth one viva line; the absolute seconds are host-bound (§6.1).

## 11. Limitations and honest reading

1. **Host-capped VM.** Every speedup in §7 is capped by the shared ~1.5-CPU
   host, not by the code. Quote them only with the host-cap label.
2. **8-unit runs are oversubscribed.** 8 units on a 4-vCPU VM; label them
   as such or re-run on a machine with more cores. (Also: Open MPI defaults
   slots to core count, so 8 ranks needed `--oversubscribe` to launch at
   all — a launcher detail, not a code failure.)
3. **Run-to-run variance is large.** Hybrid 1×2 repeats: 137.1 / 137.0 /
   92.6 s. The host share is burstable, not a fixed cap.
4. **Weak-scaling P = 1 converged early** (24 iterations) — compare
   per-iteration, not wall time (§8).
5. **Do NOT quote as scaling results:**
   - Large headline (1M×128, K=64, 2 iterations): serial 13.23 s, hybrid
     2×2 16.97 s — hybrid is SLOWER because the tier converges in 2
     iterations and the timed region is dominated by the one-off 1 GB
     distribution/gather. Not a scaling result.
   - 8-unit runs (oversubscribed — see 2).
   - VM timings generally (variance — see 3).
6. **Large scaling: [PENDING DECISION]** — either build a Large-hard
   variant (like medium-hard: overlapping clusters, 1M×128) for a real
   Large scaling number, or cut Large scaling and report it as a
   limitation. The current Large number must not be quoted either way.
7. **E > 1 is not superlinear scaling** (§7) — host-share variance plus
   cache effects.

## 12. Conclusion and future work

The hybrid MPI + OpenMP K-Means is correct (0/500,000 mismatches,
bit-identical centroids), and on the benchmark host its best configuration
(MPI-only, 2 ranks) reaches 2.75× over serial, with hybrid 2×2 at 2.36×.
The plateau at ~2.4–2.75× is a property of the shared VM, not of the
implementation: communication is ~2% at 2 units, static scheduling beats
dynamic by ~15% on equal cores, and the design's per-iteration
communication is O(K·D), independent of N.

Future work: **[PENDING DECISIONS]** — (a) re-run key configurations on
the 2P+10E laptop for dedicated-hardware scaling numbers, above all the
static-vs-dynamic comparison; (b) Large-hard variant or Large-as-limitation
(§11.6); (c) larger-scale runs (more ranks, XL 5M tier) if hardware allows.

---

## Appendix A. Reproducibility

- Generate data: `python3 scripts/gen_data.py` (seed 42; `--centre-range`
  / `--noise-std` for the hard variants).
- Build: `make` (serial, MPI, OpenMP, hybrid, hybrid-static).
- Benchmarks: `python3 benchmarks/run_benchmarks.py` (3 repeats per
  config; `--resume` after an interruption).
- Plots: `python3 benchmarks/plot_benchmarks.py` (regenerates all 4 PNGs
  from `benchmarks/results.csv`).
- Correctness: `compare_assignments.py` (tol 1e-4).

## Appendix B. Sources

- `benchmarks/results.csv` (69 rows, 14 Nov 2026)
- `benchmarks/M5_RUN_LOG.md` (methodology, caveats, interruption log)
- `benchmarks/plot_benchmarks.py`
- `report/M6_REPORT_DRAFT.md` (outline + verified tables, 16 Nov 2026)
- `docs/milestones.md` (M6 scope: report PDF + viva slides in `report/`)

## Open decisions (do not guess — ask the user)

1. Plots upload route: GitHub web UI → `benchmarks/` (4 PNGs), or
   Faiz/Qaim `git push`.
2. Laptop re-run of key configs (at minimum static-vs-dynamic).
3. Large-hard variant vs cutting Large scaling as a limitation.
4. Report format: PDF via LaTeX / Typst / Word.

Slides come after the report, in the chosen format's companion form.

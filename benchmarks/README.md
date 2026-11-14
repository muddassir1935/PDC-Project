# benchmarks/ — Results & Scaling Plots

Milestone 5 results (14 Nov 2026, branch `feature/m5-benchmarks`):

- `results.csv` — one row per run configuration: implementation, MPI ranks, OpenMP threads, dataset tier, mean time, std deviation, speedup, efficiency.
- `speedup.png` — Speedup vs processing units (strong scaling), with the Amdahl's Law theoretical curve.
- `efficiency.png` — Parallel efficiency vs processing units.
- `weak_scaling.png` — Execution time vs ranks at proportionally growing dataset size.
- `schedule.png` — Static vs dynamic OpenMP schedule (hybrid 2x2, 1x4).
- `run_benchmarks.py` — the runner (3 repeats per config; `--resume` skips finished configs after an interrupted run).
- `plot_benchmarks.py` — regenerates all four PNGs from `results.csv`.
- `M5_RUN_LOG.md` — full methodology, machine, datasets, timing definitions, and the results tables with the honest reading (host-capped VM numbers, what needs a re-run on real hardware).

Methodology: every configuration runs 3–5 times; reported times are mean ± standard deviation. Compute time and MPI communication time are recorded separately. Runs that oversubscribe the machine's physical cores are labelled as oversubscribed.

Headline caveat: this VM shares ~1.5 CPUs, so all speedups plateau at ~2.4–2.75x — a machine artefact, not a code limit. See `M5_RUN_LOG.md` before quoting any number in the report.

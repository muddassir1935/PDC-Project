# benchmarks/ — Results & Scaling Plots

Filled in Milestone 5:

- `results.csv` — one row per run configuration: implementation, MPI ranks, OpenMP threads, dataset tier, mean time, std deviation, speedup, efficiency.
- `speedup.png` — Speedup vs processing units (strong scaling), with the Amdahl's Law theoretical curve.
- `efficiency.png` — Parallel efficiency vs processing units.
- `weak_scaling.png` — Execution time vs ranks at proportionally growing dataset size.

Methodology: every configuration runs 3–5 times; reported times are mean ± standard deviation. Compute time and MPI communication time are recorded separately. Runs that oversubscribe the machine's physical cores are labelled as oversubscribed.

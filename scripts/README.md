# scripts/ — Dataset Generation & Plotting

| Script | Purpose | Milestone |
|---|---|---|
| `gen_data.py` | Generate synthetic clustered datasets (fixed seed → reproducible). Tiers: Small 100k×16, Medium 500k×64, Large 1M×128, XL 5M×32 | M1 |
| `plot_results.py` | Read benchmark CSVs and produce Speedup / Efficiency / Weak-scaling plots | M5 |

Datasets are generated locally and are **not** committed (see `data/README.md`); the same seed regenerates the identical dataset on any machine.

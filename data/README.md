# data/ — Generated Datasets

Dataset files (`.bin` / `.dat` / `.csv`) are **not committed** — they are large and fully reproducible:

```bash
python3 scripts/gen_data.py --points 100000 --dims 16 --clusters 8 --seed 42 --out data/small.bin
```

Same seed + same parameters = identical dataset, on any machine. Only this README lives here so the folder structure is visible in the repository.

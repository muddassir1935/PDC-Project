# data/ — Generated Datasets

Dataset files (`.bin` / `.dat` / `.csv`) are **not committed** — they are large and fully reproducible:

```bash
python3 scripts/gen_data.py --points 100000 --dims 16 --clusters 8 --seed 42 --out data/small.bin
```

Same seed + same parameters = identical dataset, on any machine. Only this README lives here so the folder structure is visible in the repository.

## Data format (M1, shared by all versions)

Written by `scripts/gen_data.py`, read by `src/kmeans_serial.c` (and the
MPI/OpenMP/hybrid versions in M2–M4):

- Binary, **headerless**, little-endian float64 (C `double`), row-major:
  point `i` occupies elements `[i*D, i*D + D)`. File size is `N * D * 8`
  bytes. There is no header — pass `N` and `D` to the C program as
  arguments (they are also recorded in the `<out>.meta.json` sidecar,
  which the C program does not read).
- Initial centroids (identical rule in every version): deterministic
  **farthest-point** selection — centroid 0 is row 0, each next centroid
  is the point farthest from its nearest already-chosen centroid (ties:
  lowest index). A first version used the first K rows, but verification
  showed those covered only 3 of 8 true clusters on the Small tier, so
  it was replaced; farthest-point needs no PRNG and reproduces exactly
  from the file alone in every version.
- Convergence: stop when the largest centroid movement in an iteration
  is `< epsilon`, or when `max_iters` is reached. A cluster with zero
  points keeps its previous centroid (no division by zero).

## M1 example

```bash
python3 scripts/gen_data.py --tier small --out data/small.bin
make serial
bin/kmeans_serial data/small.bin 100000 16 8 100 1e-4
```

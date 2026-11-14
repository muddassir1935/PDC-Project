#!/usr/bin/env python3
"""Generate synthetic clustered datasets for the PDC K-Means project (M1).

Data format (written by this script, read by src/kmeans_serial.c and all
later versions — keep them in sync):

  * File: binary, headerless, little-endian float64 (C ``double``),
    row-major: point ``i`` occupies elements ``[i*D, i*D + D)``.
    File size in bytes is exactly ``N * D * 8``. There is NO header in
    the file itself; N and D are passed to the C program on its command
    line.
  * Sidecar: ``<out>.meta.json`` records ``points`` (N), ``dims`` (D),
    ``clusters`` (K), ``seed`` and the byte size, so a dataset can be
    re-run without remembering its parameters. The C program does NOT
    read the sidecar; it is documentation/reproducibility only.
  * Datasets are NOT committed (see data/README.md): the same seed +
    parameters regenerate the identical file on any machine.

Generation (fixed seed -> reproducible):
  1. K true centres drawn uniform in [-10, 10] per dimension.
  2. Each point is assigned a true cluster (round-robin, so clusters are
     near-equal in size), then drawn as centre + N(0, 1) noise.
  3. All points are SHUFFLED with the same seeded RNG, so file order
     carries no cluster structure (the C programs' farthest-point
     initialisation starts from row 0, and must not be biased by
     generation order).

Benchmark tiers (from the project plan):

  ==========  =========  =====  ====  =================
  tier        points     dims   K     notes
  ==========  =========  =====  ====  =================
  small       100,000    16     8     M1 smoke tier
  medium      500,000    64     32    strong-scaling tier
  large       1,000,000  128    64    headline tier
  xl          5,000,000  32     16    stretch / stress only
  ==========  =========  =====  ====  =================

Usage:
  python3 scripts/gen_data.py --tier small --out data/small.bin
  python3 scripts/gen_data.py --points 100000 --dims 16 --clusters 8 \\
      --seed 42 --out data/small.bin
  bin/kmeans_serial data/small.bin 100000 16 8 100 1e-4
"""

import argparse
import json
import sys

import numpy as np

TIERS = {
    "small": {"points": 100_000, "dims": 16, "clusters": 8},
    "medium": {"points": 500_000, "dims": 64, "clusters": 32},
    "large": {"points": 1_000_000, "dims": 128, "clusters": 64},
    "xl": {"points": 5_000_000, "dims": 32, "clusters": 16},
}

CENTRE_RANGE = 10.0  # true centres uniform in [-10, 10]
NOISE_STD = 1.0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tier", choices=sorted(TIERS),
                   help="benchmark tier preset (explicit flags override it)")
    p.add_argument("--points", type=int, help="number of points N")
    p.add_argument("--dims", type=int, help="dimensions per point D")
    p.add_argument("--clusters", type=int, help="true clusters K")
    p.add_argument("--seed", type=int, default=42,
                   help="fixed RNG seed for reproducibility (default: 42)")
    p.add_argument("--centre-range", type=float, default=CENTRE_RANGE,
                   help="true centres uniform in [-R, R] (default: 10.0); "
                        "smaller R with the same noise makes clusters "
                        "overlap, so K-Means needs more iterations — "
                        "the M5 'hard' datasets use R=3.0")
    p.add_argument("--noise-std", type=float, default=NOISE_STD,
                   help="per-dimension Gaussian noise std (default: 1.0)")
    p.add_argument("--out", required=True, help="output .bin path")
    args = p.parse_args(argv)

    params = dict(TIERS[args.tier]) if args.tier else {}
    for name in ("points", "dims", "clusters"):
        value = getattr(args, name)
        if value is not None:
            params[name] = value
    missing = [n for n in ("points", "dims", "clusters") if n not in params]
    if missing:
        p.error(f"missing parameters {missing}: give --tier or explicit flags")
    for name, value in params.items():
        if value <= 0:
            p.error(f"--{name} must be positive, got {value}")
    if params["clusters"] > params["points"]:
        p.error("clusters K must not exceed points N")
    args.params = params
    return args


def main(argv=None):
    args = parse_args(argv)
    n, d, k = (args.params["points"], args.params["dims"],
               args.params["clusters"])
    rng = np.random.default_rng(args.seed)

    centres = rng.uniform(-args.centre_range, args.centre_range,
                          size=(k, d))
    # Round-robin true labels: near-equal cluster sizes.
    labels = np.arange(n) % k
    data = centres[labels] + rng.normal(0.0, args.noise_std, size=(n, d))
    # Shuffle so file order carries no cluster structure (see docstring).
    data = data[rng.permutation(n)]

    data.astype("<f8").tofile(args.out)
    meta = {
        "points": n,
        "dims": d,
        "clusters": k,
        "seed": args.seed,
        "centre_range": args.centre_range,
        "noise_std": args.noise_std,
        "dtype": "float64 little-endian, headerless, row-major",
        "bytes": n * d * 8,
        "format_note": "point i = elements [i*D, i*D+D); "
                       "no header; pass N and D to the C program",
    }
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out}: N={n} D={d} K={k} seed={args.seed} "
          f"({n * d * 8} bytes) + sidecar {args.out}.meta.json")
    print("Run it with:")
    print(f"  bin/kmeans_serial {args.out} {n} {d} {k} 100 1e-4")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare K-Means outputs between two versions (M2/M3/M4 vs serial).

Assignment files are int32, one per point, in file order (written by
src/kmeans_serial.c, src/kmeans_mpi.c and later versions).

Parallel summation order differs from serial, so tiny numeric differences
are expected: assignments may differ on a few borderline points and
centroids may differ in the last digits. The comparison therefore reports
mismatch counts/fractions and max centroid difference against a tolerance,
instead of demanding byte equality.

Usage:
  python3 scripts/compare_assignments.py --ref ref.assign --new new.assign
  python3 scripts/compare_assignments.py --ref ref.assign --new new.assign \
      --ref-log ref.log --new-log new.log --centroid-tol 1e-4
"""

import argparse
import re
import sys

import numpy as np

CENTROID_RE = re.compile(r"^\s+\[\d+\]((?:\s+-?\d+\.?\d*(?:e-?\d+)?)+)\s*$")


def parse_centroids(log_path):
    cents = []
    with open(log_path) as fh:
        for line in fh:
            m = CENTROID_RE.match(line)
            if m:
                cents.append([float(v) for v in m.group(1).split()])
    if not cents:
        raise ValueError(f"no centroids found in {log_path}")
    return np.array(cents)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="reference assignments (serial)")
    ap.add_argument("--new", required=True, help="assignments to check")
    ap.add_argument("--ref-log", help="reference program stdout (for centroids)")
    ap.add_argument("--new-log", help="program stdout to check (for centroids)")
    ap.add_argument("--centroid-tol", type=float, default=1e-4,
                    help="max allowed abs centroid difference (default 1e-4)")
    ap.add_argument("--mismatch-frac", type=float, default=0.001,
                    help="max allowed assignment mismatch fraction (default 0.1%%)")
    args = ap.parse_args()

    ref = np.fromfile(args.ref, dtype=np.int32)
    new = np.fromfile(args.new, dtype=np.int32)
    if ref.shape != new.shape:
        print(f"FAIL: assignment lengths differ: ref={ref.size} new={new.size}")
        return 1
    mism = int(np.sum(ref != new))
    frac = mism / ref.size if ref.size else 0.0
    print(f"assignments: N={ref.size} mismatches={mism} ({100 * frac:.4f}%)")
    ok = frac <= args.mismatch_frac

    if args.ref_log and args.new_log:
        rc = parse_centroids(args.ref_log)
        nc = parse_centroids(args.new_log)
        if rc.shape != nc.shape:
            print(f"FAIL: centroid shapes differ: ref={rc.shape} new={nc.shape}")
            return 1
        maxdiff = float(np.max(np.abs(rc - nc)))
        print(f"centroids: shape={rc.shape} max abs diff={maxdiff:.3g} "
              f"(tol {args.centroid_tol:g})")
        ok = ok and maxdiff <= args.centroid_tol

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

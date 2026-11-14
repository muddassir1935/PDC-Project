#!/usr/bin/env python3
"""M5 scaling benchmarks — PDC Project (Milestone M5, 14 Nov 2026).

Runs the strong-scaling, weak-scaling and static-vs-dynamic experiments
on the merged M4 hybrid (branch feature/m5-benchmarks, from main 36d0cc0),
3 repeats per configuration (1 repeat for the two slowest singletons:
hybrid 1x1 sanity check and the Large-tier headline runs).

Workload (why fixed 100 iterations):
  The standard Medium tier (500k x 64, K=32, centres in [-10, 10])
  converges in 2 iterations, so its timed region is dominated by the
  one-off data distribution instead of the iteration loop — useless for
  scaling. The M5 dataset "medium-hard" uses the same N/D/K/seed but
  centres in [-1, 1] (scripts/gen_data.py --centre-range 1.0), so the
  clusters overlap and Lloyd's algorithm runs the full 100 iterations
  (final max centroid shift 0.0031, not yet converged at eps=1e-6).
  Every configuration therefore does IDENTICAL work: 100 iterations over
  the same data with the same deterministic farthest-point init. Timing
  is the reported program wall time; time-per-iteration is also recorded
  so the fixed-iteration design is visible in the CSV.

Machine (this workspace VM): 4 vCPUs, ~7.7 GB RAM (see M5_RUN_LOG.md).
  8-unit configurations oversubscribe the 4 physical cores and are
  labelled oversubscribed in the CSV. 2->4 thread scaling was flat in
  this VM in M4 (host cap, not code) — the numbers below are labelled
  host-capped where that repeats.

Timing definitions (differ per program — see M5_RUN_LOG.md):
  serial: iteration loop only (init excluded, t_start after init).
  mpi:    iteration loop incl. per-iteration Allreduce communication
          (Bcast/Scatterv distribution and Gatherv excluded).
  omp:    iteration loop wall time.
  hybrid: wall = Bcast + Scatterv + iterations + Gatherv; compute and
          communication split reported (max across ranks).

Usage:
  python3 benchmarks/run_benchmarks.py [--repeats N] [--only STRONG|WEAK|LARGE|STATIC]
  Results append to benchmarks/results.csv (one row per run, written
  after each run so an interrupted session keeps partial results).
  Plots: python3 benchmarks/plot_benchmarks.py
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin")
DATA = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(ROOT, "benchmarks", "results.csv")
MPIRUN = os.path.join(os.path.expanduser("~"), "workspace", "tools",
                      "openmpi", "bin", "mpirun")

# Persistent OpenMPI wrappers only (system mpicc does not exist here).
ENV = dict(os.environ)
ENV["PATH"] = (os.path.join(os.path.expanduser("~"), "workspace", "tools",
                           "openmpi", "bin") + os.pathsep + ENV["PATH"])

MAX_ITERS = 100
EPSILON = "1e-6"

DATASETS = {
    # name: (file, N, D, K)
    "medium-hard": ("medium-hard.bin", 500_000, 64, 32),
    "weak-125k": ("weak-125k.bin", 125_000, 64, 32),
    "weak-250k": ("weak-250k.bin", 250_000, 64, 32),
    "weak-1m": ("weak-1m.bin", 1_000_000, 64, 32),
    "large": ("large.bin", 1_000_000, 128, 64),
}

TIME_RE = re.compile(r"time(?: \(wall\))?:\s+([0-9.]+)\s*s")
ITER_RE = re.compile(r"iterations:\s+(\d+)\s+converged:\s+(yes|no)")
COMPUTE_RE = re.compile(r"compute:\s+([0-9.]+)\s*s")
COMM_RE = re.compile(r"communication:\s+([0-9.]+)\s*s")

FIELDNAMES = [
    "experiment", "implementation", "ranks", "threads", "units",
    "dataset", "n", "d", "k", "max_iters", "epsilon", "repeat",
    "iterations", "converged", "wall_s", "compute_s", "comm_s",
    "time_per_iter_s", "oversubscribed", "schedule", "note",
]


def parse_output(text):
    wall = None
    for line in text.splitlines():
        m = TIME_RE.search(line)
        if m:
            wall = float(m.group(1))
    iters, conv = None, None
    m = ITER_RE.search(text)
    if m:
        iters, conv = int(m.group(1)), m.group(2)
    compute = comm = None
    m = COMPUTE_RE.search(text)
    if m:
        compute = float(m.group(1))
    m = COMM_RE.search(text)
    if m:
        comm = float(m.group(1))
    return wall, iters, conv, compute, comm


def run_config(cfg, repeat):
    """Run one configuration once; return the CSV row dict."""
    dataset = DATASETS[cfg["dataset"]]
    fname, n, d, k = dataset
    datafile = os.path.join(DATA, fname)
    impl = cfg["implementation"]
    env = dict(ENV)
    if impl == "serial":
        cmd = [os.path.join(BIN, "kmeans_serial"), datafile,
               str(n), str(d), str(k), str(MAX_ITERS), EPSILON]
    elif impl == "mpi":
        # --oversubscribe: 8-rank configs exceed this 4-slot host's slots;
        # that IS the oversubscription under test, not an error.
        cmd = [MPIRUN, "--allow-run-as-root", "--oversubscribe",
               "-np", str(cfg["ranks"]),
               os.path.join(BIN, "kmeans_mpi"), datafile,
               str(n), str(d), str(k), str(MAX_ITERS), EPSILON]
    elif impl == "omp":
        env["OMP_NUM_THREADS"] = str(cfg["threads"])
        cmd = [os.path.join(BIN, "kmeans_omp"), datafile,
               str(n), str(d), str(k), str(MAX_ITERS), EPSILON]
    elif impl in ("hybrid", "hybrid-static"):
        env["OMP_NUM_THREADS"] = str(cfg["threads"])
        binary = ("kmeans_hybrid_static" if impl == "hybrid-static"
                  else "kmeans_hybrid")
        cmd = [MPIRUN, "--allow-run-as-root", "--oversubscribe",
               "-np", str(cfg["ranks"]),
               os.path.join(BIN, binary), datafile,
               str(n), str(d), str(k), str(MAX_ITERS), EPSILON]
    else:
        raise ValueError(f"unknown implementation {impl!r}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=env, timeout=1800)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  !! FAILED rc={proc.returncode}: {' '.join(cmd)}", flush=True)
        print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
        raise RuntimeError(f"benchmark run failed: {cfg} repeat {repeat}")
    wall, iters, conv, compute, comm = parse_output(proc.stdout)
    if wall is None or iters is None:
        print("  !! UNPARSEABLE OUTPUT", flush=True)
        print(proc.stdout[-2000:], flush=True)
        raise RuntimeError(f"could not parse output: {cfg} repeat {repeat}")
    return {
        "experiment": cfg["experiment"],
        "implementation": impl,
        "ranks": cfg["ranks"],
        "threads": cfg["threads"],
        "units": cfg["ranks"] * cfg["threads"],
        "dataset": cfg["dataset"],
        "n": n, "d": d, "k": k,
        "max_iters": MAX_ITERS, "epsilon": EPSILON,
        "repeat": repeat,
        "iterations": iters, "converged": conv,
        "wall_s": wall, "compute_s": compute or "",
        "comm_s": comm or "",
        "time_per_iter_s": round(wall / iters, 4) if iters else "",
        "oversubscribed": "yes" if cfg["ranks"] * cfg["threads"] > 4
                          else "no",
        "schedule": cfg.get("schedule", ""),
        "note": cfg.get("note", ""),
    }


def strong_configs():
    cfgs = []
    cfgs.append({"experiment": "strong", "implementation": "serial",
                 "ranks": 1, "threads": 1, "dataset": "medium-hard"})
    # Hybrid 1x1: sanity check (MPI overhead over serial), 1 repeat.
    cfgs.append({"experiment": "strong", "implementation": "hybrid",
                 "ranks": 1, "threads": 1, "dataset": "medium-hard",
                 "repeats": 1, "note": "sanity: MPI overhead vs serial"})
    for ranks, threads in [(1, 2), (2, 1), (2, 2), (4, 1), (1, 4),
                           (4, 2), (2, 4), (8, 1)]:
        note = ""
        if ranks * threads > 4:
            note = "oversubscribed: 8 units on 4-core VM"
        cfgs.append({"experiment": "strong", "implementation": "hybrid",
                     "ranks": ranks, "threads": threads,
                     "dataset": "medium-hard", "note": note})
    for ranks in (2, 4, 8):
        cfgs.append({"experiment": "strong", "implementation": "mpi",
                     "ranks": ranks, "threads": 1,
                     "dataset": "medium-hard",
                     "note": "oversubscribed: 8 units on 4-core VM"
                             if ranks > 4 else ""})
    for threads in (2, 4):
        cfgs.append({"experiment": "strong", "implementation": "omp",
                     "ranks": 1, "threads": threads,
                     "dataset": "medium-hard"})
    return cfgs


def static_configs():
    # Static vs dynamic OpenMP schedule, same hybrid source (M5 variant).
    return [
        {"experiment": "static-vs-dynamic", "implementation": "hybrid",
         "ranks": 2, "threads": 2, "dataset": "medium-hard",
         "schedule": "dynamic(1024)"},
        {"experiment": "static-vs-dynamic", "implementation": "hybrid-static",
         "ranks": 2, "threads": 2, "dataset": "medium-hard",
         "schedule": "static"},
        {"experiment": "static-vs-dynamic", "implementation": "hybrid",
         "ranks": 1, "threads": 4, "dataset": "medium-hard",
         "schedule": "dynamic(1024)"},
        {"experiment": "static-vs-dynamic", "implementation": "hybrid-static",
         "ranks": 1, "threads": 4, "dataset": "medium-hard",
         "schedule": "static"},
    ]


def weak_configs():
    # Work per unit fixed: N grows with P (125k points per unit).
    cfgs = []
    for ranks, ds in [(1, "weak-125k"), (2, "weak-250k"),
                      (4, "medium-hard"), (8, "weak-1m")]:
        cfgs.append({"experiment": "weak", "implementation": "hybrid",
                     "ranks": ranks, "threads": 1, "dataset": ds,
                     "note": "oversubscribed: 8 units on 4-core VM"
                             if ranks > 4 else ""})
    return cfgs


def large_configs():
    return [
        {"experiment": "large-headline", "implementation": "serial",
         "ranks": 1, "threads": 1, "dataset": "large", "repeats": 1,
         "note": "standard Large tier (R=10), headline number"},
        {"experiment": "large-headline", "implementation": "hybrid",
         "ranks": 2, "threads": 2, "dataset": "large", "repeats": 1,
         "note": "standard Large tier (R=10), headline number"},
    ]


def load_completed():
    """(experiment, implementation, ranks, threads, dataset) -> #rows done.

    Lets a run interrupted (e.g. by a VM/runtime restart) resume without
    redoing finished configurations. Rows are keyed on the full config so
    a re-run with different parameters never reuses stale rows.
    """
    done = {}
    if not os.path.exists(CSV_PATH):
        return done
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["experiment"], row["implementation"],
                   int(row["ranks"]), int(row["threads"]), row["dataset"])
            done[key] = done.get(key, 0) + 1
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", choices=["STRONG", "WEAK", "LARGE", "STATIC",
                                       "ALL"], default="ALL")
    ap.add_argument("--resume", action="store_true",
                    help="skip configurations already completed in "
                         "results.csv (e.g. after an interrupted run)")
    args = ap.parse_args(argv)
    completed = load_completed() if args.resume else {}
    if args.resume:
        print(f"resume: {sum(completed.values())} completed run(s) in "
              f"{CSV_PATH}", flush=True)

    groups = []
    if args.only in ("STRONG", "ALL"):
        groups.append(("strong scaling", strong_configs()))
    if args.only in ("STATIC", "ALL"):
        groups.append(("static vs dynamic", static_configs()))
    if args.only in ("WEAK", "ALL"):
        groups.append(("weak scaling", weak_configs()))
    if args.only in ("LARGE", "ALL"):
        groups.append(("Large headline", large_configs()))

    need_header = not os.path.exists(CSV_PATH)
    n_done = 0
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if need_header:
            w.writeheader()
        for gname, cfgs in groups:
            print(f"== {gname}: {len(cfgs)} configurations ==", flush=True)
            for cfg in cfgs:
                reps = cfg.get("repeats", args.repeats)
                key = (cfg["experiment"], cfg["implementation"],
                       cfg["ranks"], cfg["threads"], cfg["dataset"])
                have = completed.get(key, 0) if args.resume else 0
                if have >= reps:
                    print(f"  {cfg['implementation']} "
                          f"{cfg['ranks']}x{cfg['threads']} "
                          f"[{cfg['dataset']}] — already done "
                          f"({have}/{reps}), skipping", flush=True)
                    continue
                label = (f"{cfg['implementation']} "
                         f"{cfg['ranks']}x{cfg['threads']} "
                         f"[{cfg['dataset']}]")
                for r in range(have + 1, reps + 1):
                    print(f"  {label} repeat {r}/{reps} ...",
                          end=" ", flush=True)
                    row = run_config(cfg, r)
                    w.writerow(row)
                    fh.flush()
                    print(f"{row['wall_s']:.3f}s "
                          f"({row['iterations']} iters)", flush=True)
                    n_done += 1
    print(f"done: {n_done} runs, results in {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

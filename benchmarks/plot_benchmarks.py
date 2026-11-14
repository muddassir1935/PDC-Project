#!/usr/bin/env python3
"""Plot M5 benchmark results (benchmarks/results.csv -> PNGs).

Figures (all mean +/- std over repeats):
  speedup.png       — strong scaling: speedup S = T1/Tp vs units, with the
                      ideal line and Amdahl curves (serial fraction
                      s = 0.02 and s = 0.05). Units > 4 are shaded: this
                      4-vCPU VM oversubscribes there AND is host-capped
                      (see M5_RUN_LOG.md), so the plateau is a machine
                      artefact, not a code limit.
  efficiency.png    — parallel efficiency E = S/P vs units.
  weak_scaling.png  — weak scaling: wall time vs ranks at fixed work per
                      unit (125k points per rank); ideal = flat line.
  schedule.png      — static vs dynamic OpenMP schedule (hybrid 2x2, 1x4).

Usage: python3 benchmarks/plot_benchmarks.py
"""

import csv
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "benchmarks", "results.csv")
OUT = os.path.join(ROOT, "benchmarks")

HOST_CAP_NOTE = ("VM is host-capped (~1.5 CPU shares): "
                 "plateau is a machine artefact, not a code limit.")
OVERSUB_NOTE = "shaded: >4 units oversubscribe this 4-vCPU VM"


def load():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            row["wall_s"] = float(row["wall_s"])
            row["time_per_iter_s"] = float(row["time_per_iter_s"])
            row["iterations"] = int(row["iterations"])
            row["ranks"] = int(row["ranks"])
            row["threads"] = int(row["threads"])
            row["units"] = int(row["units"])
            rows.append(row)
    return rows


def agg(rows, experiment, implementation, units=None):
    """Mean +/- std wall time for one (experiment, implementation[, units])."""
    sel = [r for r in rows
           if r["experiment"] == experiment
           and r["implementation"] == implementation
           and (units is None or r["units"] == units)]
    if not sel:
        return None
    vals = [r["wall_s"] for r in sel]
    return statistics.mean(vals), (statistics.stdev(vals)
                                  if len(vals) > 1 else 0.0)


def series(rows, experiment, implementation):
    """Sorted [(units, mean, std)] for one implementation.

    For hybrid, several (ranks x threads) decompositions share one unit
    count (e.g. 2x1 vs 1x2 at 2 units) and differ a lot (2x1 ~65 s vs
    1x2 ~122 s) — averaging them would hide the structure, so the
    hybrid curve uses the BEST decomposition per unit count (the one a
    practitioner would choose). MPI/OMP have one config per unit count.
    """
    units = sorted({r["units"] for r in rows
                    if r["experiment"] == experiment
                    and r["implementation"] == implementation})
    out = []
    for u in units:
        if implementation == "hybrid":
            decomps = sorted({(r["ranks"], r["threads"]) for r in rows
                              if r["experiment"] == experiment
                              and r["implementation"] == implementation
                              and r["units"] == u})
            best = None
            for ranks, threads in decomps:
                sel = [r["wall_s"] for r in rows
                       if r["experiment"] == experiment
                       and r["implementation"] == implementation
                       and r["ranks"] == ranks
                       and r["threads"] == threads]
                m = statistics.mean(sel)
                if best is None or m < best[0]:
                    best = (m, statistics.stdev(sel) if len(sel) > 1
                            else 0.0, ranks, threads)
            out.append((u, best[0], best[1]))
        else:
            a = agg(rows, experiment, implementation, u)
            if a:
                out.append((u, a[0], a[1]))
    return out


def shade_oversubscribed(ax, xmax):
    ax.axvspan(4.5, xmax, color="#f0f0f0", zorder=0)
    ax.text(4.6, 0.98, "oversubscribed\n(>4 units, 4-vCPU VM)",
            transform=ax.get_xaxis_transform(), va="top", ha="left",
            fontsize=8, color="#666666")


def main():
    rows = load()
    t1 = agg(rows, "strong", "serial", 1)
    assert t1, "serial baseline missing from results.csv"
    t1_mean, t1_std = t1
    print(f"serial T1 = {t1_mean:.3f} +- {t1_std:.3f} s")

    impls = [("hybrid", "Hybrid MPI+OpenMP (best decomposition)", "o-"),
             ("mpi", "MPI only", "s-"),
             ("omp", "OpenMP only", "^-")]
    colors = {"hybrid": "#1f77b4", "mpi": "#d62728", "omp": "#2ca02c"}

    # ---- speedup -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    max_units = 1
    for impl, label, marker in impls:
        s = series(rows, "strong", impl)
        if not s:
            continue
        us = [p[0] for p in s]
        sp = [t1_mean / p[1] for p in s]
        err = [t1_mean * p[2] / p[1] ** 2 for p in s]
        ax.errorbar(us, sp, yerr=err, marker=marker[0], linestyle="-",
                    color=colors[impl], label=label, capsize=3)
        max_units = max(max_units, max(us))
    xs = [1, 2, 4, 8]
    ax.plot(xs, xs, "k--", lw=1, label="ideal")
    for s_frac, ls in [(0.02, ":"), (0.05, ":")]:
        ax.plot(xs, [1.0 / (s_frac + (1 - s_frac) / p) for p in xs],
                color="#888888", linestyle=ls,
                label=f"Amdahl s={s_frac}")
    ax.set_xlabel("Processing units (ranks x threads)")
    ax.set_ylabel("Speedup S = T1 / Tp")
    ax.set_title("Strong scaling — Medium-hard (500k x 64, K=32, 100 iters)")
    ax.set_xticks([1, 2, 4, 8])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 8.5)
    shade_oversubscribed(ax, 8.5)
    ax.text(0.01, -0.18, HOST_CAP_NOTE, transform=ax.transAxes, fontsize=8,
            color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speedup.png"), dpi=150)
    plt.close(fig)

    # ---- efficiency ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for impl, label, marker in impls:
        s = series(rows, "strong", impl)
        if not s:
            continue
        us = [p[0] for p in s]
        eff = [t1_mean / p[1] / p[0] for p in s]
        err = [t1_mean * p[2] / p[1] ** 2 / p[0] for p in s]
        ax.errorbar(us, eff, yerr=err, marker=marker[0], linestyle="-",
                    color=colors[impl], label=label, capsize=3)
    ax.axhline(1.0, color="k", linestyle="--", lw=1, label="ideal (E=1)")
    ax.set_xlabel("Processing units (ranks x threads)")
    ax.set_ylabel("Efficiency E = S / P")
    ax.set_title("Parallel efficiency — Medium-hard (500k x 64, K=32)")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 8.5)
    shade_oversubscribed(ax, 8.5)
    ax.text(0.01, -0.18, HOST_CAP_NOTE, transform=ax.transAxes, fontsize=8,
            color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficiency.png"), dpi=150)
    plt.close(fig)

    # ---- weak scaling --------------------------------------------------
    # Work-normalised: weak-125k converged in 24 iterations while the
    # larger weak tiers ran the full 100, so wall time is confounded;
    # plot time-per-iteration instead (see M5_RUN_LOG.md).
    weak = sorted({r["ranks"] for r in rows if r["experiment"] == "weak"})
    if weak:
        fig, ax = plt.subplots(figsize=(8, 5))
        means, stds, iters = [], [], []
        for p in weak:
            sel = [r for r in rows if r["experiment"] == "weak"
                   and r["ranks"] == p]
            vals = [r["time_per_iter_s"] for r in sel]
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            iters.append(sel[0]["iterations"])
        ax.errorbar(weak, means, yerr=stds, marker="o", linestyle="-",
                    color="#1f77b4", capsize=3, label="hybrid (measured)")
        ax.axhline(means[0], color="k", linestyle="--", lw=1,
                   label="ideal (flat)")
        for p, m, it in zip(weak, means, iters):
            ax.annotate(f"{it} iters", (p, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
        ax.set_xlabel("MPI ranks P (N = 125k x P points, D=64, K=32)")
        ax.set_ylabel("Time per iteration (s)")
        ax.set_title("Weak scaling — fixed 125k points per rank "
                     "(time per iteration; ideal = flat)")
        ax.set_xticks(weak)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.text(0.01, -0.18, HOST_CAP_NOTE, transform=ax.transAxes,
                fontsize=8, color="#666666")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "weak_scaling.png"), dpi=150)
        plt.close(fig)

    # ---- static vs dynamic ---------------------------------------------
    stat = [r for r in rows if r["experiment"] == "static-vs-dynamic"]
    if stat:
        fig, ax = plt.subplots(figsize=(8, 5))
        configs = sorted({(r["ranks"], r["threads"]) for r in stat})
        labels, dyn, dyn_e, sta, sta_e = [], [], [], [], []
        for ranks, threads in configs:
            labels.append(f"{ranks} ranks x {threads} threads")
            for sched, dst, err in (("dynamic(1024)", dyn, dyn_e),
                                    ("static", sta, sta_e)):
                sel = [r["wall_s"] for r in stat
                       if r["ranks"] == ranks and r["threads"] == threads
                       and r["schedule"] == sched]
                dst.append(statistics.mean(sel))
                err.append(statistics.stdev(sel) if len(sel) > 1 else 0.0)
        x = range(len(labels))
        w = 0.35
        ax.bar([i - w / 2 for i in x], dyn, w, yerr=dyn_e, capsize=3,
               label="dynamic(1024)", color="#1f77b4")
        ax.bar([i + w / 2 for i in x], sta, w, yerr=sta_e, capsize=3,
               label="static", color="#ff7f0e")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Wall time (s, mean +- std)")
        ax.set_title("OpenMP schedule: dynamic vs static — Medium-hard")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0.01, -0.18,
                "Same hybrid source; static via -DHYBRID_STATIC_SCHEDULE. "
                "On the user's 2P+10E laptop dynamic is expected to win; "
                "on this host-capped VM the gap is small.",
                transform=ax.transAxes, fontsize=8, color="#666666")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "schedule.png"), dpi=150)
        plt.close(fig)

    print("plots written to benchmarks/*.png")


if __name__ == "__main__":
    main()

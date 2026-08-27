#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for ADE/FDE differences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_errors(path: Path):
    data = np.load(path, allow_pickle=False)
    trip_ids = data["trip_ids"] if "trip_ids" in data.files else None
    return data["ade"].astype(np.float64), data["fde"].astype(np.float64), trip_ids


def bootstrap_delta(target, baseline, rng, n_bootstrap, trip_ids=None):
    """Paired bootstrap of the mean per-window delta.

    With ``trip_ids`` the resampling unit is the trip, as described in the
    paper: whole trips are drawn with replacement and the delta is aggregated
    only within the sampled trips, so windows from one trip stay together and
    within-trip correlation is not treated as independent evidence. Without it
    the unit is the individual window, which understates the variance.
    """
    n = min(len(target), len(baseline))
    target = target[:n]
    baseline = baseline[:n]
    diff = target - baseline
    observed = float(diff.mean())
    boots = np.empty(n_bootstrap, dtype=np.float64)

    if trip_ids is not None:
        # Group windows by trip once, then resample trip blocks.
        order = np.argsort(trip_ids[:n], kind="stable")
        diff_sorted = diff[order]
        starts = np.flatnonzero(np.r_[True, np.diff(trip_ids[:n][order]) != 0])
        counts = np.diff(np.r_[starts, n])
        sums = np.add.reduceat(diff_sorted, starts)
        n_trips = len(starts)
        for i in range(n_bootstrap):
            pick = rng.integers(0, n_trips, size=n_trips)
            boots[i] = sums[pick].sum() / counts[pick].sum()
    else:
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boots[i] = diff[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if observed <= 0:
        opposite_tail_count = int((boots >= 0).sum())
        p = 2.0 * min(float((boots >= 0).mean()), float((boots <= 0).mean()))
    else:
        opposite_tail_count = int((boots <= 0).sum())
        p = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return observed, float(lo), float(hi), min(max(p, 0.0), 1.0), opposite_tail_count, n


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def fmt_p(value, n_bootstrap):
    if value <= 0:
        return f"<{2.0 / n_bootstrap:.4f}"
    return f"{value:.4f}"


def write_md(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset", "pred_len", "target", "baseline", "samples",
        "delta_ade_m", "ade_ci95", "ade_tail_resamples", "ade_p",
        "delta_fde_m", "fde_ci95", "fde_tail_resamples", "fde_p",
    ]
    with path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser("Make paired bootstrap table")
    parser.add_argument("--target", required=True, help="Target error npz")
    parser.add_argument("--baseline", nargs="+", required=True,
                        help="name=path pairs for baseline error npz files")
    parser.add_argument("--dataset", default="Porto-15s-s12")
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--target_name", default="PriMoTraj")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    target_ade, target_fde, target_trips = load_errors(Path(args.target))
    if target_trips is None:
        print("WARNING: no trip_ids in the target errors file; falling back to "
              "window-level resampling, which understates the variance.")
    rng = np.random.default_rng(args.seed)
    rows = []

    for item in args.baseline:
        if "=" not in item:
            raise ValueError(f"Baseline must be name=path, got {item}")
        name, path_text = item.split("=", 1)
        base_ade, base_fde, base_trips = load_errors(Path(path_text))
        # Trip ids depend only on the evaluated window set, so the target's
        # array applies to every baseline evaluated on the same subset.
        trips = target_trips if target_trips is not None else base_trips
        if (target_trips is not None and base_trips is not None
                and not np.array_equal(target_trips, base_trips)):
            raise ValueError(f"trip_ids of {name} do not match the target's; "
                             "the two files describe different window sets.")
        ade_obs, ade_lo, ade_hi, ade_p, ade_tail, n = bootstrap_delta(
            target_ade, base_ade, rng, args.n_bootstrap, trips)
        fde_obs, fde_lo, fde_hi, fde_p, fde_tail, _ = bootstrap_delta(
            target_fde, base_fde, rng, args.n_bootstrap, trips)
        rows.append({
            "dataset": args.dataset,
            "pred_len": args.pred_len,
            "target": args.target_name,
            "baseline": name,
            "samples": n,
            "delta_ade_m": fmt(ade_obs),
            "ade_ci95": f"[{fmt(ade_lo)}, {fmt(ade_hi)}]",
            "ade_tail_resamples": f"{ade_tail}/{args.n_bootstrap}",
            "ade_p": fmt_p(ade_p, args.n_bootstrap),
            "delta_fde_m": fmt(fde_obs),
            "fde_ci95": f"[{fmt(fde_lo)}, {fmt(fde_hi)}]",
            "fde_tail_resamples": f"{fde_tail}/{args.n_bootstrap}",
            "fde_p": fmt_p(fde_p, args.n_bootstrap),
        })

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_md(output_md, rows)
    print(f"rows={len(rows)}")
    print(f"csv={output_csv}")
    print(f"md={output_md}")


if __name__ == "__main__":
    main()

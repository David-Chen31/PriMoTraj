# coding=utf-8
"""Collect BENCHMARK lines from run logs into a mean +/- std table.

    python collect_results.py logs
    python collect_results.py logs --csv results.csv
    python collect_results.py "logs/*_pred6.log"

Every training entry point in this package prints a single line starting with
``BENCHMARK|`` carrying the model name, horizon, parameter count, batch
inference time and the test metrics. This script parses those lines and
aggregates them over seeds, which is how the paper's Porto table is built.
"""

import argparse
import csv
import glob
import statistics
from collections import defaultdict
from pathlib import Path


def parse_log(path):
    """Return the fields of the last BENCHMARK line in a log, or None."""
    record = None
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("BENCHMARK|"):
            fields = {}
            for chunk in line[len("BENCHMARK|"):].split("|"):
                if "=" in chunk:
                    key, value = chunk.split("=", 1)
                    fields[key.strip()] = value.strip()
            record = fields
    return record


def collect(paths):
    logs = []
    for pattern in paths:
        p = Path(pattern)
        if p.is_dir():
            logs.extend(sorted(str(f) for f in p.glob("*.log")))
        else:
            logs.extend(sorted(glob.glob(pattern)))

    groups = defaultdict(list)
    for log in logs:
        record = parse_log(log)
        if record is None:
            print(f"  (no BENCHMARK line) {log}")
            continue
        key = (record.get("model", "?"),
               record.get("seq_len", "?"),
               record.get("pred_len", "?"))
        groups[key].append(record)
    return groups


def agg(values):
    numbers = []
    for v in values:
        try:
            numbers.append(float(v))
        except (TypeError, ValueError):
            pass
    if not numbers:
        return None, None
    mean = statistics.fmean(numbers)
    std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser("Aggregate BENCHMARK lines over seeds")
    parser.add_argument("paths", nargs="+", help="log files, globs, or directories")
    parser.add_argument("--csv", default="", help="also write the table to this CSV")
    args = parser.parse_args()

    groups = collect(args.paths)
    if not groups:
        print("No BENCHMARK lines found.")
        return 1

    rows = []
    for (model, seq_len, pred_len), records in sorted(
            groups.items(), key=lambda kv: (int(kv[0][2] or 0), kv[0][0])):
        ade_mean, ade_std = agg([r.get("test_ade_m") for r in records])
        fde_mean, fde_std = agg([r.get("test_fde_m") for r in records])
        mse_mean, _ = agg([r.get("test_mse") for r in records])
        ms_mean, _ = agg([r.get("infer_ms") for r in records])
        rows.append(dict(
            model=model, seq_len=seq_len, pred_len=pred_len, n_seeds=len(records),
            params=records[0].get("params", ""),
            ade_mean=ade_mean, ade_std=ade_std,
            fde_mean=fde_mean, fde_std=fde_std,
            mse_mean=mse_mean, infer_ms=ms_mean,
        ))

    # Plain ASCII, generously spaced, so the table survives any console encoding.
    header = (f"{'model':<16}{'T':>4}{'H':>4}{'seeds':>6}{'params':>10}"
              f"{'ADE(m)':>22}{'FDE(m)':>22}{'ms/batch':>11}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ade = f"{r['ade_mean']:.3f}"
        fde = f"{r['fde_mean']:.3f}"
        if r["n_seeds"] > 1:
            ade += f" +/- {r['ade_std']:.3f}"
            fde += f" +/- {r['fde_std']:.3f}"
        print(f"{r['model']:<16}{r['seq_len']:>4}{r['pred_len']:>4}{r['n_seeds']:>6}"
              f"{r['params']:>10}{ade:>22}{fde:>22}{r['infer_ms']:>11.3f}")

    if args.csv:
        out = Path(args.csv)
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

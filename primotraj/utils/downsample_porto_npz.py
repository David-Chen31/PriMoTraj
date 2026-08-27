"""Build downsampled Porto npz variants with preserved trip-level splits.

Input npz is expected to contain the standard repo fields:
    all_data: [N, 9]
    trip_lengths: [num_trips]
    mean/std: optional, ignored because stats are recomputed per variant

The output keeps the same feature layout:
    [lat, lon, speed, heading_sin, heading_cos, hour_sin, hour_cos, dow_sin, dow_cos]

Speed and heading are recomputed after downsampling.  Split ids preserve the
original chronological 70/10/20 trip assignment so variants differ only by
sampling interval and retained-trip filtering, not by re-splitting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FEATURE_NAMES = np.array(
    [
        "lat",
        "lon",
        "speed",
        "heading_sin",
        "heading_cos",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def recompute_motion_features(feat, interval_sec):
    out = feat.copy()
    lats = out[:, 0].astype(np.float64)
    lons = out[:, 1].astype(np.float64)
    if len(out) <= 1:
        out[:, 2] = 0.0
        out[:, 3] = 0.0
        out[:, 4] = 1.0
        return out

    dist = haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    speed = np.concatenate([[0.0], dist / float(interval_sec)])
    heading = np.concatenate([[0.0], np.arctan2(np.diff(lons), np.diff(lats))])
    out[:, 2] = speed.astype(np.float32)
    out[:, 3] = np.sin(heading).astype(np.float32)
    out[:, 4] = np.cos(heading).astype(np.float32)
    return out


def split_id_for_trip(trip_idx, n_trips):
    n_train = int(n_trips * 0.7)
    n_val = int(n_trips * 0.1)
    if trip_idx < n_train:
        return 0
    if trip_idx < n_train + n_val:
        return 1
    return 2


def build_windows_count(lengths, split_ids, seq_len, pred_len):
    total = 0
    by_split = {0: 0, 1: 0, 2: 0}
    need = seq_len + pred_len
    for length, split_id in zip(lengths, split_ids):
        n_win = max(0, int(length) - need + 1)
        total += n_win
        by_split[int(split_id)] += n_win
    return total, by_split


def summarize_variant(all_data, trip_lengths, split_ids, original_n_trips, report_seq_len, report_pred_lens):
    speed = all_data[:, 2] if len(all_data) else np.array([], dtype=np.float32)
    stop_ratio = float((speed < 0.5).mean()) if len(speed) else 0.0
    high_speed_ratio = float((speed > 15.0).mean()) if len(speed) else 0.0
    row = {
        "trips_kept": int(len(trip_lengths)),
        "points": int(len(all_data)),
        "mean_trip_length": float(np.mean(trip_lengths)) if len(trip_lengths) else 0.0,
        "removed_trips": int(original_n_trips - len(trip_lengths)),
        "removed_ratio": float((original_n_trips - len(trip_lengths)) / max(1, original_n_trips)),
        "mean_speed": float(speed.mean()) if len(speed) else 0.0,
        "median_speed": float(np.median(speed)) if len(speed) else 0.0,
        "stop_ratio": stop_ratio,
        "high_speed_ratio": high_speed_ratio,
        "train_trips": int((split_ids == 0).sum()),
        "val_trips": int((split_ids == 1).sum()),
        "test_trips": int((split_ids == 2).sum()),
    }
    for pred_len in report_pred_lens:
        total, by_split = build_windows_count(trip_lengths, split_ids, report_seq_len, pred_len)
        row[f"windows_s{report_seq_len}_p{pred_len}"] = int(total)
        row[f"train_windows_s{report_seq_len}_p{pred_len}"] = int(by_split[0])
        row[f"val_windows_s{report_seq_len}_p{pred_len}"] = int(by_split[1])
        row[f"test_windows_s{report_seq_len}_p{pred_len}"] = int(by_split[2])
    return row


def write_stats(rows, csv_path, md_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser("Downsample Porto npz while preserving original trip splits")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 2, 4, 8, 20])
    parser.add_argument("--base_interval_sec", type=int, default=15)
    parser.add_argument("--min_len", type=int, default=18)
    parser.add_argument("--report_seq_len", type=int, default=12)
    parser.add_argument("--report_pred_lens", nargs="+", type=int, default=[3, 6, 12, 24])
    parser.add_argument("--stats_csv", default="/root/AMD2/benchmarks/results/porto_downsample_stats_v2.csv")
    parser.add_argument("--stats_md", default="/root/AMD2/benchmarks/results/porto_downsample_stats_v2.md")
    args = parser.parse_args()

    raw = np.load(args.input)
    all_data = raw["all_data"].astype(np.float32, copy=False)
    trip_lengths = raw["trip_lengths"].astype(np.int64)
    offsets = np.zeros(len(trip_lengths) + 1, dtype=np.int64)
    np.cumsum(trip_lengths, out=offsets[1:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    original_n_trips = len(trip_lengths)
    for factor in args.factors:
        interval = int(args.base_interval_sec * factor)
        chunks = []
        lengths = []
        split_ids = []
        original_trip_ids = []
        retained_flags = np.zeros(original_n_trips, dtype=np.int8)

        for trip_idx, length in enumerate(trip_lengths):
            start = int(offsets[trip_idx])
            end = int(offsets[trip_idx + 1])
            feat = all_data[start:end:factor]
            if len(feat) < args.min_len:
                continue
            feat = recompute_motion_features(feat, interval)
            chunks.append(feat)
            lengths.append(len(feat))
            split_ids.append(split_id_for_trip(trip_idx, original_n_trips))
            original_trip_ids.append(trip_idx)
            retained_flags[trip_idx] = 1

        if not chunks:
            raise RuntimeError(f"No trips retained for factor={factor}")

        variant_data = np.concatenate(chunks, axis=0).astype(np.float32)
        variant_lengths = np.asarray(lengths, dtype=np.int32)
        variant_split_ids = np.asarray(split_ids, dtype=np.int8)
        variant_original_trip_ids = np.asarray(original_trip_ids, dtype=np.int64)

        train_data = variant_data[variant_split_ids.repeat(variant_lengths) == 0]
        if len(train_data) == 0:
            raise RuntimeError(f"No train points retained for factor={factor}")
        mean = train_data.astype(np.float64, copy=False).mean(axis=0).astype(np.float32)
        std = train_data.astype(np.float64, copy=False).std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0

        out_path = out_dir / f"porto_ds{interval}.npz"
        np.savez_compressed(
            out_path,
            all_data=variant_data,
            trip_lengths=variant_lengths,
            mean=mean,
            std=std,
            split_ids=variant_split_ids,
            original_trip_ids=variant_original_trip_ids,
            original_retained_flags=retained_flags,
            dataset_name=np.array(f"Porto-{interval}s"),
            sample_interval_sec=np.array(interval, dtype=np.int32),
            downsample_factor=np.array(factor, dtype=np.int32),
            split_strategy=np.array("preserve_original_trip_70_10_20"),
            coordinate_order=np.array("lat_lon"),
            feature_names=FEATURE_NAMES,
        )

        row = {
            "variant": f"Porto-{interval}s",
            "factor": factor,
            "sample_interval_sec": interval,
            "path": str(out_path),
        }
        row.update(
            summarize_variant(
                variant_data,
                variant_lengths,
                variant_split_ids,
                original_n_trips,
                args.report_seq_len,
                args.report_pred_lens,
            )
        )
        stats_rows.append(row)
        print(
            f"{row['variant']}: trips={row['trips_kept']:,d} points={row['points']:,d} "
            f"removed={row['removed_ratio']:.2%} mean_speed={row['mean_speed']:.3f}"
        )

    write_stats(stats_rows, Path(args.stats_csv), Path(args.stats_md))
    print(f"stats_csv={args.stats_csv}")
    print(f"stats_md={args.stats_md}")


if __name__ == "__main__":
    main()


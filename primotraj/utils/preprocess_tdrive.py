"""
Preprocess T-Drive Taxi trajectories for AMD trajectory prediction.

The raw T-Drive release contains irregularly sampled GPS logs. For trajectory
forecasting we build fixed-interval segments by keeping consecutive points whose
time delta matches `sample_interval_sec` within `tolerance_sec`, then convert
each segment into the same 9-feature representation used by Porto:

    [lat, lon, speed(m/s), heading_sin, heading_cos,
     hour_sin, hour_cos, dow_sin, dow_cos]

Usage:
    python utils/preprocess_tdrive.py
    python utils/preprocess_tdrive.py --sample_interval_sec 300 --min_points 72
    python utils/preprocess_tdrive.py --max_files 200 --output data/tdrive_smoke.npz
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm


FEATURE_NAMES = [
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


def haversine_m(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def parse_total_seconds(ts_text: str) -> int:
    day = int(ts_text[8:10])
    hour = int(ts_text[11:13])
    minute = int(ts_text[14:16])
    second = int(ts_text[17:19])
    return ((day * 24 + hour) * 60 + minute) * 60 + second


def build_features(lat_lon, start_time: datetime, sample_interval_sec: int):
    coords = np.asarray(lat_lon, dtype=np.float64)
    lats = coords[:, 0]
    lons = coords[:, 1]
    n = len(coords)

    if n < 2:
        return None

    dist = haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    speed = np.concatenate([[0.0], dist / float(sample_interval_sec)])
    heading = np.concatenate([[0.0], np.arctan2(np.diff(lons), np.diff(lats))])

    timestamps = [start_time + i * timedelta(seconds=sample_interval_sec) for i in range(n)]
    hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in timestamps], dtype=np.float64)
    dows = np.array([t.weekday() for t in timestamps], dtype=np.float64)

    return np.column_stack(
        [
            lats,
            lons,
            speed,
            np.sin(heading),
            np.cos(heading),
            np.sin(2.0 * math.pi * hours / 24.0),
            np.cos(2.0 * math.pi * hours / 24.0),
            np.sin(2.0 * math.pi * dows / 7.0),
            np.cos(2.0 * math.pi * dows / 7.0),
        ]
    ).astype(np.float32)


def iter_segments(file_path: Path, sample_interval_sec: int, tolerance_sec: int):
    prev_t = None
    cur_points = []
    cur_start_text = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(",", 3)
            if len(parts) < 4:
                continue

            ts_text = parts[1]
            lon = float(parts[2])
            lat = float(parts[3])
            t = parse_total_seconds(ts_text)

            if prev_t is None:
                cur_points = [(lat, lon)]
                cur_start_text = ts_text
                prev_t = t
                continue

            dt = t - prev_t
            prev_t = t

            if dt == 0:
                continue

            if abs(dt - sample_interval_sec) <= tolerance_sec:
                cur_points.append((lat, lon))
            else:
                if cur_points:
                    yield cur_start_text, cur_points
                cur_points = [(lat, lon)]
                cur_start_text = ts_text

    if cur_points:
        yield cur_start_text, cur_points


def main():
    parser = argparse.ArgumentParser(description="Preprocess T-Drive trajectories for AMD")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/T-drive Taxi Trajectories/release/taxi_log_2008_by_id",
        help="Directory containing raw T-Drive taxi *.txt files",
    )
    parser.add_argument("--output", type=str, default="data/tdrive_processed_5min.npz")
    parser.add_argument("--sample_interval_sec", type=int, default=300, help="Target fixed sampling interval")
    parser.add_argument("--tolerance_sec", type=int, default=2, help="Allowed delta deviation from target interval")
    parser.add_argument("--min_points", type=int, default=60, help="Minimum kept segment length in points")
    parser.add_argument("--max_speed", type=float, default=55.0, help="Drop segments whose max speed exceeds this")
    parser.add_argument("--max_files", type=int, default=0, help="Only scan first N taxi files (0=all)")
    parser.add_argument("--max_segments", type=int, default=0, help="Only keep first N valid segments after filtering")
    parser.add_argument("--report_seq_len", type=int, default=48, help="Report usable windows for this seq_len")
    parser.add_argument("--report_pred_len", type=int, default=12, help="Report usable windows for this pred_len")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.txt"))
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Scanning {len(files):,d} taxi files from {input_dir}")
    print(
        f"Task definition: fixed_dt={args.sample_interval_sec}s, "
        f"tolerance={args.tolerance_sec}s, min_points={args.min_points}"
    )

    segments = []
    skipped_short = 0
    skipped_speed = 0
    raw_segments = 0

    for file_path in tqdm(files, desc="Reading taxis"):
        for start_text, lat_lon in iter_segments(file_path, args.sample_interval_sec, args.tolerance_sec):
            raw_segments += 1
            if len(lat_lon) < args.min_points:
                skipped_short += 1
                continue

            start_time = datetime.fromisoformat(start_text)
            features = build_features(lat_lon, start_time, args.sample_interval_sec)
            if features is None:
                skipped_short += 1
                continue

            if float(features[:, 2].max()) > args.max_speed:
                skipped_speed += 1
                continue

            segments.append((parse_total_seconds(start_text), features))
            if args.max_segments > 0 and len(segments) >= args.max_segments:
                break
        if args.max_segments > 0 and len(segments) >= args.max_segments:
            break

    if not segments:
        raise RuntimeError("No valid fixed-interval segments were produced.")

    segments.sort(key=lambda x: x[0])
    trip_lengths = np.array([seg.shape[0] for _, seg in segments], dtype=np.int32)
    all_data = np.concatenate([seg for _, seg in segments], axis=0).astype(np.float32)

    n_train_segments = max(1, int(len(trip_lengths) * 0.7))
    n_train_points = int(trip_lengths[:n_train_segments].sum())
    train_data = all_data[:n_train_points]

    mean = train_data.mean(axis=0).astype(np.float32)
    std = train_data.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        all_data=all_data,
        trip_lengths=trip_lengths,
        mean=mean,
        std=std,
        dataset_name=np.array("tdrive"),
        sample_interval_sec=np.array(args.sample_interval_sec, dtype=np.int32),
        split_strategy=np.array("chronological_segments"),
        coordinate_order=np.array("lat_lon"),
        feature_names=np.array(FEATURE_NAMES),
    )

    need = args.report_seq_len + args.report_pred_len
    usable_windows = int(sum(max(0, int(length) - need + 1) for length in trip_lengths))
    file_mb = out_path.stat().st_size / 1024.0 / 1024.0

    print("")
    print(f"Saved to {out_path} ({file_mb:.1f} MB)")
    print(f"  Valid segments : {len(trip_lengths):,d}")
    print(f"  Total points   : {len(all_data):,d}")
    print(f"  Raw segments   : {raw_segments:,d}")
    print(f"  Skipped short  : {skipped_short:,d}")
    print(f"  Skipped speed  : {skipped_speed:,d}")
    print(f"  Feature names  : {FEATURE_NAMES}")
    print(
        f"  Windows for seq_len={args.report_seq_len}, "
        f"pred_len={args.report_pred_len}: {usable_windows:,d}"
    )
    print(f"  Train mean     : {mean}")
    print(f"  Train std      : {std}")


if __name__ == "__main__":
    main()

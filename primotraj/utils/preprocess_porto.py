"""
Preprocess Porto Taxi dataset for AMD trajectory prediction.

Converts raw GPS trajectories into feature arrays:
    [lat, lon, speed(m/s), heading_sin, heading_cos, hour_sin, hour_cos, dow_sin, dow_cos]

Usage:
    python utils/preprocess_porto.py
    python utils/preprocess_porto.py --input data/train.csv --output data/porto_processed.npz
    python utils/preprocess_porto.py --max_trips 50000   # quick test with subset
    python utils/preprocess_porto.py --sample_ratio 0.25 --output data/porto_quarter.npz
"""

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Porto metropolitan area bounding box
PORTO_BOUNDS = (-8.75, -8.50, 41.05, 41.25)  # lon_min, lon_max, lat_min, lat_max

FEATURE_NAMES = [
    'lat', 'lon', 'speed', 'heading_sin', 'heading_cos',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]


def process_trip(polyline, start_timestamp, dt=15):
    """Convert one trip's polyline + timestamp into a (T, 9) feature array."""
    coords = np.array(polyline, dtype=np.float64)  # (T, 2): [lon, lat]
    T = len(coords)
    lons, lats = coords[:, 0], coords[:, 1]

    # --- Speed via vectorized Haversine ---
    dlat_r = np.radians(np.diff(lats))
    dlon_r = np.radians(np.diff(lons))
    lat1_r, lat2_r = np.radians(lats[:-1]), np.radians(lats[1:])
    a = np.sin(dlat_r / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon_r / 2) ** 2
    dist = 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    speed = np.concatenate([[0.0], dist / dt])

    # --- Heading (bearing angle) ---
    heading = np.concatenate([[0.0], np.arctan2(np.diff(lons), np.diff(lats))])

    # --- Time features ---
    timestamps = start_timestamp + np.arange(T) * dt
    dts = pd.to_datetime(timestamps, unit='s', utc=True)
    hours = dts.hour + dts.minute / 60.0
    dows = dts.dayofweek  # 0=Mon, 6=Sun

    features = np.column_stack([
        lats, lons, speed,
        np.sin(heading), np.cos(heading),
        np.sin(2 * np.pi * hours / 24.0), np.cos(2 * np.pi * hours / 24.0),
        np.sin(2 * np.pi * dows / 7.0),  np.cos(2 * np.pi * dows / 7.0),
    ]).astype(np.float32)

    return features


def in_bounds(polyline, bounds=PORTO_BOUNDS):
    """Check whether all trajectory points fall inside Porto bounding box."""
    coords = np.array(polyline)
    return (coords[:, 0].min() >= bounds[0] and coords[:, 0].max() <= bounds[1] and
            coords[:, 1].min() >= bounds[2] and coords[:, 1].max() <= bounds[3])


def main():
    parser = argparse.ArgumentParser(description='Preprocess Porto Taxi data for AMD')
    parser.add_argument('--input',     type=str,   default='data/train.csv')
    parser.add_argument('--output',    type=str,   default='data/porto_processed.npz')
    parser.add_argument('--min_len',   type=int,   default=20,   help='Min trajectory length (points)')
    parser.add_argument('--max_trips', type=int,   default=0,    help='Max trips to process (0=all)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Fraction of rows to keep after filtering (0<ratio<=1)')
    parser.add_argument('--max_speed', type=float, default=50.0, help='Max speed (m/s) to keep a trip')
    args = parser.parse_args()

    if not (0 < args.sample_ratio <= 1.0):
        raise ValueError('--sample_ratio must satisfy 0 < sample_ratio <= 1.0')

    # --- Read raw CSV ---
    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Filter rows with missing GPS
    df = df[~df['MISSING_DATA'].astype(bool)].reset_index(drop=True)
    print(f"  After MISSING_DATA filter: {len(df)}")

    if args.sample_ratio < 1.0:
        n_keep = max(1, int(len(df) * args.sample_ratio))
        df = df.head(n_keep)
        print(f"  Using first {len(df)} rows ({args.sample_ratio:.2%})")

    if args.max_trips > 0:
        df = df.head(args.max_trips)
        print(f"  Using first {len(df)} rows")

    # --- Process each trip ---
    chunks = []
    trip_lengths = []
    skipped = {'short': 0, 'bounds': 0, 'speed': 0, 'parse': 0}

    for idx in tqdm(range(len(df)), desc="Processing trips"):
        row = df.iloc[idx]
        try:
            polyline = json.loads(row['POLYLINE'])
        except Exception:
            skipped['parse'] += 1
            continue

        if len(polyline) < args.min_len:
            skipped['short'] += 1
            continue

        if not in_bounds(polyline):
            skipped['bounds'] += 1
            continue

        feat = process_trip(polyline, int(row['TIMESTAMP']))

        if feat[:, 2].max() > args.max_speed:
            skipped['speed'] += 1
            continue

        chunks.append(feat)
        trip_lengths.append(len(feat))

    print(f"\nValid trips: {len(chunks)}")
    print(f"Skipped:     {skipped}")

    # --- Concatenate and compute normalization stats ---
    all_data = np.concatenate(chunks, axis=0)
    trip_lengths = np.array(trip_lengths, dtype=np.int32)

    # Stats from training portion (first 70% of trips, chronological)
    n_train_trips = int(len(trip_lengths) * 0.7)
    n_train_pts = int(trip_lengths[:n_train_trips].sum())
    train_data = all_data[:n_train_pts]
    mean = train_data.mean(axis=0).astype(np.float32)
    std  = train_data.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0  # avoid division by zero

    # --- Save ---
    np.savez_compressed(
        args.output,
        all_data=all_data,
        trip_lengths=trip_lengths,
        mean=mean,
        std=std,
    )

    total_pts = len(all_data)
    file_mb = __import__('os').path.getsize(args.output) / 1024 / 1024

    print(f"\nSaved to {args.output} ({file_mb:.1f} MB)")
    print(f"  Trips:    {len(trip_lengths)}")
    print(f"  Points:   {total_pts}")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Train mean: {mean}")
    print(f"  Train std:  {std}")


if __name__ == '__main__':
    main()

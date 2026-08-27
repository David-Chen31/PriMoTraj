# coding=utf-8
"""Generate a small synthetic dataset in the Porto .npz format.

    python make_demo_data.py --output data/demo.npz

This exists so the package can be checked end to end without downloading the
real datasets. The trajectories are smooth constant-turn-rate curves, not real
GPS traces, so the resulting numbers are a smoke test only -- never a result.

Array layout (identical to primotraj/utils/preprocess_porto.py):
    all_data     (N, 9) float32  lat, lon, speed, sin(h), cos(h), 4 time features
    trip_lengths (n_trips,) int32
    mean, std    (9,) float32    per-channel statistics of all_data
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser("Synthetic Porto-format trajectories")
    parser.add_argument("--output", default="data/demo.npz")
    parser.add_argument("--n_trips", type=int, default=200)
    parser.add_argument("--trip_len", type=int, default=40)
    parser.add_argument("--interval_sec", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trips = []
    for _ in range(args.n_trips):
        lat = 41.15 + rng.normal(0, 0.01)
        lon = -8.61 + rng.normal(0, 0.01)
        heading = rng.uniform(0, 2 * np.pi)
        turn_rate = rng.normal(0, 0.05)
        speed = abs(rng.normal(8.0, 2.0))
        trip = []
        for t in range(args.trip_len):
            heading += turn_rate
            speed = max(0.0, speed + rng.normal(0, 0.2))
            dlat = speed * np.cos(heading) * args.interval_sec / 111320.0
            dlon = (speed * np.sin(heading) * args.interval_sec
                    / (111320.0 * np.cos(np.radians(lat))))
            lat, lon = lat + dlat, lon + dlon
            trip.append([lat, lon, speed, np.sin(heading), np.cos(heading),
                         np.sin(t / 24), np.cos(t / 24),
                         np.sin(t / 7), np.cos(t / 7)])
        trips.append(np.array(trip, dtype=np.float32))

    all_data = np.concatenate(trips, axis=0).astype(np.float32)
    trip_lengths = np.full(len(trips), args.trip_len, dtype=np.int32)
    mean = all_data.mean(axis=0).astype(np.float32)
    std = all_data.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, all_data=all_data, trip_lengths=trip_lengths, mean=mean, std=std)
    print(f"wrote {out}: {all_data.shape[0]:,d} points from {len(trips)} trips")
    print("Smoke-test data only -- do not report numbers computed on it.")


if __name__ == "__main__":
    main()

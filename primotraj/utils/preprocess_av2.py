"""
Preprocess Argoverse 2 motion-forecasting test split into AMD/LiteMoTraj npz format.

Output format is compatible with existing trajectory loaders in this repo:
    all_data:     (N_total, 9) float32
    trip_lengths: (n_trips,) int32
    mean/std:     (9,) float32 (computed on first 70% trips)

Feature layout:
    [lat_proxy, lon_proxy, speed, heading_sin, heading_cos,
     hour_sin, hour_cos, dow_sin, dow_cos]

Notes:
- AV2 uses local x/y in metres. Existing training scripts compute ADE/FDE with
  Haversine on the first two channels (assumed lat/lon). To keep scripts
  unchanged, we convert x/y metres to "proxy degrees" with a fixed scale:
      deg = metres / 111320
  This preserves local distance approximately in metres for small displacements.
- This script uses focal-track annotations (gt_trajectory_x/y) as prediction
  targets and observed focal-track points from scenario parquet as history.

Usage:
    python LiteMoTraj/utils/preprocess_av2.py \
        --input_root /root/autodl-tmp/data/av2_data/motion_forecasting \
        --output /root/AMD2/data/av2_focal_processed.npz
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURE_NAMES = [
    "lat_proxy",
    "lon_proxy",
    "speed",
    "heading_sin",
    "heading_cos",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

M_PER_DEG = 111320.0


def _to_datetime_utc_ns(ts_ns: int) -> datetime:
    # AV2 timestamp fields are nanoseconds in UTC-like epoch scale.
    return datetime.fromtimestamp(float(ts_ns) / 1e9, tz=timezone.utc)


def _build_features_from_xy(
    x_m: np.ndarray,
    y_m: np.ndarray,
    start_ts_ns: int,
    dt_sec: float,
) -> np.ndarray:
    x_m = np.asarray(x_m, dtype=np.float64)
    y_m = np.asarray(y_m, dtype=np.float64)
    n = x_m.shape[0]

    lat = y_m / M_PER_DEG
    lon = x_m / M_PER_DEG

    if n >= 2:
        dx = np.diff(x_m)
        dy = np.diff(y_m)
        speed = np.concatenate([[0.0], np.sqrt(dx * dx + dy * dy) / max(dt_sec, 1e-6)])
        heading = np.concatenate([[0.0], np.arctan2(dy, dx)])
    else:
        speed = np.zeros((n,), dtype=np.float64)
        heading = np.zeros((n,), dtype=np.float64)

    t0 = _to_datetime_utc_ns(int(start_ts_ns))
    times = [t0 + i * timedelta(seconds=dt_sec) for i in range(n)]
    hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in times], dtype=np.float64)
    dows = np.array([t.weekday() for t in times], dtype=np.float64)

    feats = np.column_stack(
        [
            lat,
            lon,
            speed,
            np.sin(heading),
            np.cos(heading),
            np.sin(2.0 * math.pi * hours / 24.0),
            np.cos(2.0 * math.pi * hours / 24.0),
            np.sin(2.0 * math.pi * dows / 7.0),
            np.cos(2.0 * math.pi * dows / 7.0),
        ]
    ).astype(np.float32)

    return feats


def _list_scenario_files(test_dir: Path) -> dict[str, Path]:
    files = sorted(test_dir.glob("*/scenario_*.parquet"))
    out: dict[str, Path] = {}
    for fp in files:
        stem = fp.stem
        if stem.startswith("scenario_"):
            sid = stem[len("scenario_") :]
            out[sid] = fp
    return out


def _load_focal_annotations(path: Path, pred_len: int) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    ann = pd.read_parquet(path)
    ann = ann[ann["is_focal_track"] == True].copy()  # noqa: E712

    by_sid: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for row in ann.itertuples(index=False):
        sid = str(row.scenario_id)
        track_id = str(row.track_id)
        fx = np.asarray(row.gt_trajectory_x, dtype=np.float64)
        fy = np.asarray(row.gt_trajectory_y, dtype=np.float64)
        if fx.shape[0] < pred_len or fy.shape[0] < pred_len:
            continue
        by_sid[sid] = (track_id, fx[:pred_len], fy[:pred_len])

    return by_sid


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AV2 into AMD/LiteMoTraj npz format")
    parser.add_argument(
        "--input_root",
        type=str,
        default="/root/autodl-tmp/data/av2_data/motion_forecasting",
        help="AV2 motion_forecasting root directory",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="",
        help="Path to focal annotation parquet. Empty means auto-detect under input_root.",
    )
    parser.add_argument("--output", type=str, default="/root/AMD2/data/av2_focal_processed.npz")
    parser.add_argument("--obs_len", type=int, default=50, help="Observed history length")
    parser.add_argument("--pred_len", type=int, default=60, help="Future length from annotation")
    parser.add_argument("--dt_sec", type=float, default=0.1, help="AV2 sampling interval in seconds")
    parser.add_argument("--max_scenarios", type=int, default=0, help="For smoke test only; 0 means all")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    test_dir = input_root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"AV2 test directory not found: {test_dir}")

    ann_path = Path(args.annotation) if args.annotation else (input_root / "av2_mf_focal_test_annotations.parquet")
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    print(f"Input root : {input_root}")
    print(f"Annotation : {ann_path}")
    print(f"obs_len={args.obs_len}, pred_len={args.pred_len}, dt={args.dt_sec}s")

    ann_by_sid = _load_focal_annotations(ann_path, pred_len=args.pred_len)
    scen_files = _list_scenario_files(test_dir)

    scenario_ids = sorted(set(ann_by_sid.keys()) & set(scen_files.keys()))
    if args.max_scenarios > 0:
        scenario_ids = scenario_ids[: args.max_scenarios]

    if not scenario_ids:
        raise RuntimeError("No overlapping scenarios between annotations and scenario parquet files.")

    segments: list[tuple[str, np.ndarray]] = []
    skipped_missing_track = 0
    skipped_short_obs = 0
    skipped_bad_future = 0

    for sid in tqdm(scenario_ids, desc="Processing AV2 scenarios"):
        track_id, fut_x, fut_y = ann_by_sid[sid]
        sp = scen_files[sid]

        try:
            sdf = pd.read_parquet(
                sp,
                columns=[
                    "track_id",
                    "observed",
                    "timestep",
                    "position_x",
                    "position_y",
                    "start_timestamp",
                ],
            )
        except Exception:
            continue

        focal = sdf[sdf["track_id"].astype(str) == track_id]
        if focal.empty:
            skipped_missing_track += 1
            continue

        obs = focal[focal["observed"] == True].sort_values("timestep")  # noqa: E712
        if len(obs) < args.obs_len:
            skipped_short_obs += 1
            continue

        obs = obs.tail(args.obs_len)
        obs_x = obs["position_x"].to_numpy(dtype=np.float64)
        obs_y = obs["position_y"].to_numpy(dtype=np.float64)

        if fut_x.shape[0] != args.pred_len or fut_y.shape[0] != args.pred_len:
            skipped_bad_future += 1
            continue

        full_x = np.concatenate([obs_x, fut_x], axis=0)
        full_y = np.concatenate([obs_y, fut_y], axis=0)

        start_ts_ns = int(float(obs["start_timestamp"].iloc[0]))
        feats = _build_features_from_xy(full_x, full_y, start_ts_ns=start_ts_ns, dt_sec=args.dt_sec)
        segments.append((sid, feats))

    if not segments:
        raise RuntimeError("No valid AV2 segments were produced.")

    # Stable ordering ensures deterministic split in current dataloaders.
    segments.sort(key=lambda kv: kv[0])

    trip_lengths = np.array([seg.shape[0] for _, seg in segments], dtype=np.int32)
    all_data = np.concatenate([seg for _, seg in segments], axis=0).astype(np.float32)

    n_trips = len(trip_lengths)
    n_train_trips = max(1, int(n_trips * 0.7))
    n_train_pts = int(trip_lengths[:n_train_trips].sum())
    train_data = all_data[:n_train_pts]

    # Keep float64 stats for numerical consistency with some baseline loaders.
    mean64 = train_data.astype(np.float64, copy=False).mean(axis=0, dtype=np.float64)
    std64 = train_data.astype(np.float64, copy=False).std(axis=0, dtype=np.float64)
    std64[std64 < 1e-8] = 1.0
    mean = mean64.astype(np.float32)
    std = std64.astype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        all_data=all_data,
        trip_lengths=trip_lengths,
        mean=mean,
        std=std,
        stats_dtype=np.array("float64"),
        dataset_name=np.array("av2_mf_focal_test"),
        split_strategy=np.array("scenario_sorted_70_10_20"),
        coordinate_order=np.array("lat_proxy_lon_proxy"),
        coordinate_source=np.array("xy_m_to_proxy_deg"),
        feature_names=np.array(FEATURE_NAMES),
        obs_len=np.array(args.obs_len, dtype=np.int32),
        pred_len=np.array(args.pred_len, dtype=np.int32),
        sample_interval_sec=np.array(args.dt_sec, dtype=np.float32),
    )

    need = args.obs_len + min(args.pred_len, 12)
    usable_windows = int(sum(max(0, int(tl) - need + 1) for tl in trip_lengths))
    print("")
    print(f"Saved: {out_path}")
    print(f"  segments: {len(segments):,d}")
    print(f"  total points: {len(all_data):,d}")
    print(f"  skipped missing focal track: {skipped_missing_track:,d}")
    print(f"  skipped short observed: {skipped_short_obs:,d}")
    print(f"  skipped bad future len: {skipped_bad_future:,d}")
    print(f"  feature names: {FEATURE_NAMES}")
    print(f"  windows(need={need}) ~ {usable_windows:,d}")
    print(f"  train mean[:3]: {mean[:3]}")
    print(f"  train std[:3] : {std[:3]}")


if __name__ == "__main__":
    main()

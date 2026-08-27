# coding=utf-8
"""Shared trajectory data loading, evaluation and timing helpers.

RECONSTRUCTED FILE. The original ``baselines/traj_utils.py`` was lost together
with the training server. It has been rebuilt from the surviving call sites
(``GRU/main_traj.py``, ``DLinear/main_traj.py``,
``benchmarks/scripts/compute_sample_errors_v2.py``,
``benchmarks/scripts/profile_online_latency_v2.py``,
``benchmarks/scripts/run_extended_sota_adapters.py``) and from the equivalent
implementations that survived inside ``primotraj/utils/traj_dataloader.py``,
``primotraj/main_traj.py`` and ``PatchTST/PatchTST_supervised/main_traj.py``.

Convention (fixed by the call sites): this loader yields **raw, un-normalized**
windows, and callers normalize explicitly with ``(batch - mean) / std``. That is
why ``mean`` / ``std`` are passed into ``evaluate_model`` and
``benchmark_inference_ms``. It is the one deliberate difference from
``primotraj/utils/traj_dataloader.py``, which normalizes inside ``_load``.
"""

import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def haversine_torch(pred, true):
    """Vectorised Haversine distance in metres.

    Args:
        pred, true: (B, T, 2) with [..., 0]=lat, [..., 1]=lon in degrees.
    Returns:
        (B, T) distances in metres.
    """
    R = 6371000.0
    lat1, lon1 = torch.deg2rad(true[..., 0]), torch.deg2rad(true[..., 1])
    lat2, lon2 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, min=1e-12, max=1.0 - 1e-7)
    return R * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


def denorm_latlon(x_norm, mean, std):
    mean2 = mean[:2].view(1, 1, 2)
    std2 = std[:2].view(1, 1, 2)
    return x_norm[..., :2] * std2 + mean2


class TrajectoryDataset(Dataset):
    """Map-style dataset returning sliding-window trajectory samples."""

    def __init__(self, data, windows, seq_len, pred_len):
        self.data = data
        self.windows = windows
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s = int(self.windows[idx])
        x = self.data[s: s + self.seq_len]
        y = self.data[s + self.seq_len: s + self.seq_len + self.pred_len]
        return x, y


class TrajectoryDataLoader:
    """Load preprocessed trajectory data and provide train/val/test DataLoaders.

    Yields raw (un-normalized) windows; ``.mean`` / ``.std`` expose the
    statistics stored in the .npz so callers can normalize themselves.
    """

    def __init__(self, data_path, batch_size, seq_len, pred_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_len = seq_len + pred_len
        self.target_slice = slice(0, None)
        self._load(data_path)

    def _load(self, path):
        raw = np.load(path)
        all_data = raw['all_data'].astype(np.float32)
        trip_lengths = raw['trip_lengths']
        self.mean = raw['mean']
        self.std = raw['std']
        self.n_feature = all_data.shape[1]

        offsets = np.zeros(len(trip_lengths) + 1, dtype=np.int64)
        np.cumsum(trip_lengths, out=offsets[1:])

        # Chronological trip-level split: 70% train, 10% val, 20% test.
        # Downsampled Porto variants store explicit split_ids so the original
        # trip-level assignment survives the removal of short trips.
        n = len(trip_lengths)
        if 'split_ids' in raw.files:
            split_ids = raw['split_ids'].astype(np.int64)
            split_trip_indices = {
                'train': np.where(split_ids == 0)[0],
                'val': np.where(split_ids == 1)[0],
                'test': np.where(split_ids == 2)[0],
            }
        else:
            n_train = int(n * 0.7)
            n_val = int(n * 0.1)
            split_trip_indices = {
                'train': np.arange(0, n_train, dtype=np.int64),
                'val': np.arange(n_train, n_train + n_val, dtype=np.int64),
                'test': np.arange(n_train + n_val, n, dtype=np.int64),
            }

        data_tensor = torch.from_numpy(all_data)

        self._splits = {}
        for name, trip_indices in split_trip_indices.items():
            windows = self._build_windows(offsets, trip_lengths, trip_indices)
            print(f"  {name:5s}: {len(windows):>8,d} windows from {len(trip_indices):,d} trips")
            self._splits[name] = (data_tensor, windows)

    def _build_windows(self, offsets, trip_lengths, trip_indices):
        windows = []
        for i in trip_indices:
            n_win = int(trip_lengths[i]) - self.window_len + 1
            if n_win > 0:
                base = int(offsets[i])
                windows.extend(range(base, base + n_win))
        return np.array(windows, dtype=np.int64)

    def _make(self, name, shuffle, drop_last):
        data, windows = self._splits[name]
        ds = TrajectoryDataset(data, windows, self.seq_len, self.pred_len)
        nw_env = os.environ.get("TRAJ_NUM_WORKERS")
        if nw_env is not None:
            try:
                nw = max(0, int(nw_env))
            except ValueError:
                nw = min(4, os.cpu_count() or 1)
        else:
            nw = min(4, os.cpu_count() or 1)
        pin = torch.cuda.is_available()
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=(nw > 0),
        )

    def get_train(self, shuffle=True):
        return self._make('train', shuffle, drop_last=True)

    def get_val(self, shuffle=True):
        return self._make('val', shuffle, drop_last=True)

    def get_test(self):
        return self._make('test', shuffle=False, drop_last=False)


@torch.no_grad()
def evaluate_model(model, loader, criterion, mean, std, device, split_name="val"):
    """Normalized-space MSE/MAE plus meter-level ADE/FDE after inverse transform."""
    model.eval()
    mse_list, mae_list, ade_list, fde_list = [], [], [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x = (batch_x - mean) / std
        batch_y = (batch_y - mean) / std

        outputs = model(batch_x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        mse = criterion(outputs, batch_y)
        mae = torch.mean(torch.abs(outputs - batch_y))

        pred_latlon = denorm_latlon(outputs, mean, std)
        true_latlon = denorm_latlon(batch_y, mean, std)
        dist = haversine_torch(pred_latlon, true_latlon)

        mse_list.append(mse.item())
        mae_list.append(mae.item())
        ade_list.append(dist.mean().item())
        fde_list.append(dist[:, -1].mean().item())

    mse_avg = float(np.mean(mse_list)) if mse_list else float("inf")
    mae_avg = float(np.mean(mae_list)) if mae_list else float("inf")
    ade_avg = float(np.mean(ade_list)) if ade_list else float("inf")
    fde_avg = float(np.mean(fde_list)) if fde_list else float("inf")

    print(
        f"[{split_name.upper()}] MSE: {mse_avg:.6f} | MAE: {mae_avg:.6f} | "
        f"ADE(m): {ade_avg:.6f} | FDE(m): {fde_avg:.6f}"
    )
    return mse_avg, mae_avg, ade_avg, fde_avg


@torch.no_grad()
def benchmark_inference_ms(model, loader, device, warmup=20, steps=100, mean=None, std=None):
    """Mean forward-pass wall time per batch in milliseconds, measured after warmup."""
    model.eval()
    durations = []
    all_durations = []
    n_seen = 0
    max_iters = warmup + steps

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        if mean is not None and std is not None:
            batch_x = (batch_x - mean) / std
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        all_durations.append(dt_ms)

        if n_seen >= warmup:
            durations.append(dt_ms)
        n_seen += 1
        if n_seen >= max_iters:
            break

    # A split smaller than the warmup budget leaves no post-warmup samples;
    # fall back to every timing rather than reporting nan.
    if not durations:
        durations = all_durations
    if not durations:
        return float("nan")
    return float(np.mean(durations))

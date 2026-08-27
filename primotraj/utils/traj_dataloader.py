"""
Trajectory DataLoader for AMD model.

Loads preprocessed Porto .npz data (from preprocess_porto.py) and provides
train / val / test DataLoaders with sliding-window trajectory samples.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataLoader:
    """Load preprocessed trajectory data and provide train/val/test DataLoaders.

    Interface is compatible with the original CustomDataLoader so AMD model code
    can be used with minimal changes (exposes .n_feature, .target_slice,
    get_train/val/test).
    """

    def __init__(self, data_path, batch_size, seq_len, pred_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_len = seq_len + pred_len
        # Predict all features (M mode) — model handles slicing if needed
        self.target_slice = slice(0, None)

        self._load(data_path)

    def _load(self, path):
        raw = np.load(path)
        all_data = raw['all_data']           # (N_total, 9) float32
        trip_lengths = raw['trip_lengths']    # (n_trips,)   int32
        self.mean = raw['mean']              # (9,) float32
        self.std  = raw['std']               # (9,) float32
        self.n_feature = all_data.shape[1]   # 9

        # Normalize
        all_data = ((all_data - self.mean) / self.std).astype(np.float32)

        # Cumulative offsets for indexing into concatenated array
        offsets = np.zeros(len(trip_lengths) + 1, dtype=np.int64)
        np.cumsum(trip_lengths, out=offsets[1:])

        # Chronological split by trips: 70% train, 10% val, 20% test.
        # Downsampled Porto variants can store explicit split_ids to preserve
        # the original trip-level assignment after short trips are removed.
        n = len(trip_lengths)
        if 'split_ids' in raw.files:
            split_ids = raw['split_ids'].astype(np.int64)
            split_trip_indices = {
                'train': np.where(split_ids == 0)[0],
                'val':   np.where(split_ids == 1)[0],
                'test':  np.where(split_ids == 2)[0],
            }
        else:
            n_train = int(n * 0.7)
            n_val   = int(n * 0.1)
            split_trip_indices = {
                'train': np.arange(0, n_train, dtype=np.int64),
                'val':   np.arange(n_train, n_train + n_val, dtype=np.int64),
                'test':  np.arange(n_train + n_val, n, dtype=np.int64),
            }

        # Shared tensor — efficient for DataLoader workers
        data_tensor = torch.from_numpy(all_data)

        self._splits = {}
        for name, trip_indices in split_trip_indices.items():
            windows = self._build_windows(offsets, trip_lengths, trip_indices)
            print(f"  {name:5s}: {len(windows):>8,d} windows from {len(trip_indices):,d} trips")
            self._splits[name] = (data_tensor, windows)

    def _build_windows(self, offsets, trip_lengths, trip_indices):
        """Return int64 array of data-start indices for all valid sliding windows."""
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
        # Use multiple workers to overlap data preparation with GPU compute.
        # pin_memory speeds up CPU→GPU transfers when CUDA is available.
        import os
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

    def get_val(self, shuffle=False, drop_last=False):
        """Return the complete deterministic validation split by default."""
        return self._make('val', shuffle=shuffle, drop_last=drop_last)

    def get_test(self):
        return self._make('test', shuffle=False, drop_last=False)


class TrajectoryDataset(Dataset):
    """Map-style dataset that returns sliding-window trajectory samples.

    Each sample:
        x: (seq_len, n_feature)  — input trajectory window
        y: (pred_len, n_feature) — ground-truth future window
    """

    def __init__(self, data, windows, seq_len, pred_len):
        self.data = data          # (N_total, 9) torch.Tensor
        self.windows = windows    # (n_windows,) np.int64
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s = int(self.windows[idx])
        x = self.data[s : s + self.seq_len]                         # (seq_len, 9)
        y = self.data[s + self.seq_len : s + self.seq_len + self.pred_len]  # (pred_len, 9)
        return x, y

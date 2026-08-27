import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataLoader:
    def __init__(self, data_path, batch_size, seq_len, pred_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_len = seq_len + pred_len
        self.target_slice = slice(0, None)
        self._load(data_path)

    def _load(self, path):
        raw = np.load(path)
        all_data = raw['all_data']
        trip_lengths = raw['trip_lengths']
        self.mean = raw['mean']
        self.std = raw['std']
        self.n_feature = all_data.shape[1]
        all_data = ((all_data - self.mean) / self.std).astype(np.float32)
        offsets = np.zeros(len(trip_lengths) + 1, dtype=np.int64)
        np.cumsum(trip_lengths, out=offsets[1:])
        n = len(trip_lengths)
        if 'split_ids' in raw.files:
            split_ids = raw['split_ids'].astype(np.int64)
            split_trip_indices = {
                'train': np.where(split_ids == 0)[0],
                'val': np.where(split_ids == 1)[0],
                'test': np.where(split_ids == 2)[0],
            }
        else:
            n_train, n_val = int(n * 0.7), int(n * 0.1)
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
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)

    def get_train(self):
        return self._make('train', True, True)

    def get_val(self):
        return self._make('val', True, True)

    def get_test(self):
        return self._make('test', False, False)


class TrajectoryDataset(Dataset):
    def __init__(self, data, windows, seq_len, pred_len):
        self.data, self.windows = data, windows
        self.seq_len, self.pred_len = seq_len, pred_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s = int(self.windows[idx])
        return self.data[s:s + self.seq_len], self.data[s + self.seq_len:s + self.seq_len + self.pred_len]

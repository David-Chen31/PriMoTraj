import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Porto trajectory dataset for sliding-window forecasting.

    Expected npz keys:
    - all_data: float32 array, shape (N_total_points, 9)
    - trip_lengths: int array, shape (n_trips,)
    - mean: float array, shape (9,)
    - std: float array, shape (9,)

    This dataset builds windows within each trip only (no cross-trip windows).
    Split is performed by trip index using ratio 7:1:2.
    """

    def __init__(
            self,
            root_path,
            data_path='porto_processed.npz',
            flag='train',
            size=None,
            features='M',
            target='lat',
            scale=False,
            timeenc=0,
            freq='15s'):
        if size is None:
            self.seq_len = 48
            self.label_len = 0
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Keep these arguments for compatibility with the existing data_provider API.
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self._read_data()

    def _read_data(self):
        file_path = os.path.join(self.root_path, self.data_path)
        npz = np.load(file_path, allow_pickle=False)

        self.all_data = npz['all_data'].astype(np.float32)
        self.trip_lengths = npz['trip_lengths'].astype(np.int64)
        self.mean = npz['mean'].astype(np.float32)
        self.std = npz['std'].astype(np.float32)

        if self.all_data.ndim != 2 or self.all_data.shape[1] != 9:
            raise ValueError('all_data must have shape (N, 9).')
        if self.trip_lengths.ndim != 1:
            raise ValueError('trip_lengths must have shape (n_trips,).')
        if self.trip_lengths.sum() != self.all_data.shape[0]:
            raise ValueError('Sum of trip_lengths must equal all_data.shape[0].')

        n_trips = len(self.trip_lengths)
        n_train = int(n_trips * 0.7)
        n_val = int(n_trips * 0.1)

        if self.set_type == 0:
            trip_start_idx = 0
            trip_end_idx = n_train
        elif self.set_type == 1:
            trip_start_idx = n_train
            trip_end_idx = n_train + n_val
        else:
            trip_start_idx = n_train + n_val
            trip_end_idx = n_trips

        trip_offsets = np.concatenate(([0], np.cumsum(self.trip_lengths)))
        self.window_start_indices = []

        need_len = self.seq_len + self.pred_len
        for trip_id in range(trip_start_idx, trip_end_idx):
            trip_global_start = int(trip_offsets[trip_id])
            trip_len = int(self.trip_lengths[trip_id])
            max_start = trip_len - need_len
            if max_start < 0:
                continue

            starts = trip_global_start + np.arange(max_start + 1, dtype=np.int64)
            self.window_start_indices.append(starts)

        if self.window_start_indices:
            self.window_start_indices = np.concatenate(self.window_start_indices, axis=0)
        else:
            self.window_start_indices = np.array([], dtype=np.int64)

    def __getitem__(self, index):
        s_begin = int(self.window_start_indices[index])
        s_end = s_begin + self.seq_len
        y_begin = s_end
        y_end = y_begin + self.pred_len

        seq_x = self.all_data[s_begin:s_end]            # (seq_len, 9)
        seq_y = self.all_data[y_begin:y_end]            # (pred_len, 9)

        # Keep placeholders for compatibility with existing iTransformer dataloader pipeline.
        seq_x_mark = torch.zeros((self.seq_len, 1), dtype=torch.float32)
        seq_y_mark = torch.zeros((self.pred_len, 1), dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.window_start_indices)

    def inverse_transform(self, data):
        """Inverse transform normalized features using npz-provided global mean/std."""
        return data * self.std + self.mean

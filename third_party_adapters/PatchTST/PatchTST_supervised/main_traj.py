import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.PatchTST import Model as PatchTST


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def haversine_torch(pred, true):
    r = 6371000.0
    lat1, lon1 = torch.deg2rad(true[..., 0]), torch.deg2rad(true[..., 1])
    lat2, lon2 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return r * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


def denorm_latlon(x_norm, mean, std):
    mean2 = mean[:2].view(1, 1, 2)
    std2 = std[:2].view(1, 1, 2)
    return x_norm[..., :2] * std2 + mean2


class TrajectoryDataset(Dataset):
    def __init__(self, data, windows, seq_len, pred_len):
        self.data = data
        self.windows = windows
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s = int(self.windows[idx])
        x = self.data[s:s + self.seq_len]
        y = self.data[s + self.seq_len:s + self.seq_len + self.pred_len]
        return x, y


class TrajectoryDataLoader:
    def __init__(self, data_path, batch_size, seq_len, pred_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_len = seq_len + pred_len
        self._load(data_path)

    @staticmethod
    def _compute_train_stats(all_data, trip_lengths):
        n_train_trips = int(len(trip_lengths) * 0.7)
        n_train_pts = int(trip_lengths[:n_train_trips].sum())
        train_data = all_data[:n_train_pts].astype(np.float64, copy=False)
        mean = train_data.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = train_data.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-8] = 1.0
        return mean, std

    def _load(self, path):
        raw = np.load(path)
        all_data = raw["all_data"]
        trip_lengths = raw["trip_lengths"]
        if 'stats_dtype' in raw.files and str(raw['stats_dtype']) == 'float64':
            self.mean = raw['mean'].astype(np.float32)
            self.std = raw['std'].astype(np.float32)
        else:
            self.mean, self.std = self._compute_train_stats(all_data, trip_lengths)
        self.n_feature = int(all_data.shape[1])

        all_data = ((all_data - self.mean) / self.std).astype(np.float32)
        offsets = np.zeros(len(trip_lengths) + 1, dtype=np.int64)
        np.cumsum(trip_lengths, out=offsets[1:])

        n = len(trip_lengths)
        if "split_ids" in raw.files:
            split_ids = raw["split_ids"].astype(np.int64)
            split_trip_indices = {
                "train": np.where(split_ids == 0)[0],
                "val": np.where(split_ids == 1)[0],
                "test": np.where(split_ids == 2)[0],
            }
        else:
            n_train = int(n * 0.7)
            n_val = int(n * 0.1)
            split_trip_indices = {
                "train": np.arange(0, n_train, dtype=np.int64),
                "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
                "test": np.arange(n_train + n_val, n, dtype=np.int64),
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
        return self._make("train", True, True)

    def get_val(self):
        return self._make("val", True, True)

    def get_test(self):
        return self._make("test", False, False)


@torch.no_grad()
def benchmark_inference_ms(model, loader, device, warmup=20, steps=100):
    model.eval()
    durations = []
    all_durations = []
    n_seen = 0
    max_iters = warmup + steps

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
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

    if durations:
        return float(sum(durations) / len(durations))
    if all_durations:
        return float(sum(all_durations) / len(all_durations))
    return float("nan")


@torch.no_grad()
def evaluate(model, loader, criterion, mean, std, device, split_name="val"):
    model.eval()
    mse_list, mae_list, ade_list, fde_list = [], [], [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        mse = criterion(outputs, batch_y)
        mae = torch.mean(torch.abs(outputs - batch_y))

        pred_latlon = denorm_latlon(outputs, mean, std)
        true_latlon = denorm_latlon(batch_y, mean, std)
        dist = haversine_torch(pred_latlon, true_latlon)
        ade = dist.mean()
        fde = dist[:, -1].mean()

        mse_list.append(mse.item())
        mae_list.append(mae.item())
        ade_list.append(ade.item())
        fde_list.append(fde.item())

    mse_avg = float(np.mean(mse_list)) if mse_list else float("inf")
    mae_avg = float(np.mean(mae_list)) if mae_list else float("inf")
    ade_avg = float(np.mean(ade_list)) if ade_list else float("inf")
    fde_avg = float(np.mean(fde_list)) if fde_list else float("inf")

    print(
        f"[{split_name.upper()}] MSE: {mse_avg:.6f} | MAE: {mae_avg:.6f} | "
        f"ADE(m): {ade_avg:.6f} | FDE(m): {fde_avg:.6f}"
    )

    return mse_avg, mae_avg, ade_avg, fde_avg


def main():
    parser = argparse.ArgumentParser("PatchTST trajectory baseline")
    parser.add_argument("--data_path", type=str, default="./dataset/porto_processed.npz")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc_dropout", type=float, default=0.05)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--bench_warmup", type=int, default=20)
    parser.add_argument("--bench_steps", type=int, default=100)
    parser.add_argument("--save_path", type=str, default=None,
                        help="if set, save model checkpoint to this .pt file after training")
    parser.add_argument("--load_path", type=str, default=None,
                        help="if set, load checkpoint and skip training")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = TrajectoryDataLoader(args.data_path, args.batch_size, args.seq_len, args.pred_len)
    train_loader = data_loader.get_train()
    val_loader = data_loader.get_val()
    test_loader = data_loader.get_test()

    mean = torch.tensor(data_loader.mean, dtype=torch.float32, device=device)
    std = torch.tensor(data_loader.std, dtype=torch.float32, device=device)

    cfg = argparse.Namespace(
        enc_in=data_loader.n_feature,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        head_dropout=args.head_dropout,
        patch_len=args.patch_len,
        stride=args.stride,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
    )

    model = PatchTST(cfg).to(device)
    n_params = count_parameters(model)
    print(f"Parameter Count: {n_params:,d}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path} ...")
        ckpt = torch.load(args.load_path, map_location=device, weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded — skipping training.")
        args.epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        train_mse = float(np.mean(epoch_loss)) if epoch_loss else float("inf")
        print(f"Epoch [{epoch}/{args.epochs}] Train MSE: {train_mse:.6f}")
        evaluate(model, val_loader, criterion, mean, std, device, split_name="val")

    print("Final evaluation on test set:")
    test_mse, test_mae, test_ade, test_fde = evaluate(model, test_loader, criterion, mean, std, device, split_name="test")
    infer_ms = benchmark_inference_ms(model, test_loader, device, warmup=args.bench_warmup, steps=args.bench_steps)

    print(f"Inference Time: {infer_ms:.3f} ms/iter")
    print(
        "BENCHMARK|"
        f"model=PatchTST|seq_len={args.seq_len}|pred_len={args.pred_len}|"
        f"params={n_params}|infer_ms={infer_ms:.6f}|"
        f"test_mse={test_mse:.6f}|test_mae={test_mae:.6f}|"
        f"test_ade_m={test_ade:.6f}|test_fde_m={test_fde:.6f}"
    )

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "args": vars(args),
            "mean": data_loader.mean,
            "std": data_loader.std,
        }, args.save_path)
        print(f"Checkpoint saved → {args.save_path}")


if __name__ == "__main__":
    main()

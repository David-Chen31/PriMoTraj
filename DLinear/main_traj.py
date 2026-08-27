import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.traj_utils import (
    TrajectoryDataLoader,
    benchmark_inference_ms,
    count_parameters,
    evaluate_model,
    set_seed,
)


class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x
        left = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        right = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_pad = torch.cat([left, x, right], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, kernel_size=25, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
            for seasonal, trend in zip(self.linear_seasonal, self.linear_trend):
                seasonal.weight.data.fill_(1.0 / seq_len)
                trend.weight.data.fill_(1.0 / seq_len)
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal.weight.data.fill_(1.0 / seq_len)
            self.linear_trend.weight.data.fill_(1.0 / seq_len)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(
                seasonal.size(0), self.enc_in, self.pred_len,
                dtype=seasonal.dtype, device=seasonal.device,
            )
            trend_out = torch.zeros_like(seasonal_out)
            for idx in range(self.enc_in):
                seasonal_out[:, idx, :] = self.linear_seasonal[idx](seasonal[:, idx, :])
                trend_out[:, idx, :] = self.linear_trend[idx](trend[:, idx, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        return (seasonal_out + trend_out).permute(0, 2, 1)


def main():
    parser = argparse.ArgumentParser("DLinear trajectory baseline")
    parser.add_argument("--data_path", type=str, default="./data/porto_processed.npz")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--bench_warmup", type=int, default=20)
    parser.add_argument("--bench_steps", type=int, default=100)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
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

    model = DLinear(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=data_loader.n_feature,
        kernel_size=args.kernel_size,
        individual=args.individual,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Parameter Count: {n_params:,d}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.load_path and os.path.isfile(args.load_path):
        print(f"Loading checkpoint from {args.load_path} ...")
        ckpt = torch.load(args.load_path, map_location=device, weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        print("Checkpoint loaded - skipping training.")
        args.epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x = (batch_x - mean) / std
            batch_y = (batch_y - mean) / std

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        train_mse = sum(epoch_loss) / len(epoch_loss) if epoch_loss else float("inf")
        print(f"Epoch [{epoch}/{args.epochs}] Train MSE: {train_mse:.6f}")
        evaluate_model(model, val_loader, criterion, mean, std, device, split_name="val")

    print("Final evaluation on test set:")
    test_mse, test_mae, test_ade, test_fde = evaluate_model(
        model, test_loader, criterion, mean, std, device, split_name="test"
    )
    infer_ms = benchmark_inference_ms(
        model, test_loader, device, warmup=args.bench_warmup, steps=args.bench_steps, mean=mean, std=std
    )

    print(f"Inference Time: {infer_ms:.3f} ms/iter")
    print(
        "BENCHMARK|"
        f"model=DLinear|seq_len={args.seq_len}|pred_len={args.pred_len}|"
        f"params={n_params}|infer_ms={infer_ms:.6f}|"
        f"test_mse={test_mse:.6f}|test_mae={test_mae:.6f}|"
        f"test_ade_m={test_ade:.6f}|test_fde_m={test_fde:.6f}|"
        f"kernel_size={args.kernel_size}|individual={int(args.individual)}"
    )

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "args": vars(args),
            "mean": data_loader.mean,
            "std": data_loader.std,
        }, args.save_path)
        print(f"Checkpoint saved -> {args.save_path}")


if __name__ == "__main__":
    main()

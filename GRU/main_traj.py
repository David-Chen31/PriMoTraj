import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.traj_utils import (  # noqa: E402
    TrajectoryDataLoader,
    benchmark_inference_ms,
    count_parameters,
    evaluate_model,
    set_seed,
)


class RNNForecaster(nn.Module):
    def __init__(self, input_size, pred_len, hidden_size=64, num_layers=1, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.pred_len = pred_len
        self.input_size = input_size
        self.rnn_type = rnn_type.upper()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported rnn_type={rnn_type}")
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, pred_len * input_size),
        )

    def forward(self, x):
        if self.rnn_type == "LSTM":
            _, (hidden, _) = self.rnn(x)
        else:
            _, hidden = self.rnn(x)
        out = self.head(hidden[-1])
        return out.view(x.size(0), self.pred_len, self.input_size)


def main():
    parser = argparse.ArgumentParser("GRU/LSTM trajectory baseline")
    parser.add_argument("--data_path", type=str, default="./data/porto_processed.npz")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rnn_type", choices=["GRU", "LSTM"], default="GRU")
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

    model = RNNForecaster(
        input_size=data_loader.n_feature,
        pred_len=args.pred_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        f"model={args.rnn_type}|seq_len={args.seq_len}|pred_len={args.pred_len}|"
        f"params={n_params}|infer_ms={infer_ms:.6f}|"
        f"test_mse={test_mse:.6f}|test_mae={test_mae:.6f}|"
        f"test_ade_m={test_ade:.6f}|test_fde_m={test_fde:.6f}|"
        f"hidden_size={args.hidden_size}|num_layers={args.num_layers}|dropout={args.dropout}"
    )

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "mean": data_loader.mean,
                "std": data_loader.std,
            },
            args.save_path,
        )
        print(f"Checkpoint saved -> {args.save_path}")


if __name__ == "__main__":
    main()

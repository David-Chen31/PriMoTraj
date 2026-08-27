import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from models.TimeMixer import Model as TimeMixer
from utils.traj_dataloader import TrajectoryDataLoader


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


@torch.no_grad()
def benchmark_inference_ms(model, loader, device, warmup=20, steps=100):
    model.eval()
    durations = []
    all_durations = []
    n_seen = 0
    max_iters = warmup + steps

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch_x, None, None, None)
        if device.type == 'cuda':
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
    return float('nan')


@torch.no_grad()
def evaluate(model, loader, criterion, mean, std, device, split_name='val'):
    model.eval()

    mse_list = []
    mae_list = []
    ade_list = []
    fde_list = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x, None, None, None)

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

    mse_avg = float(np.mean(mse_list)) if mse_list else float('inf')
    mae_avg = float(np.mean(mae_list)) if mae_list else float('inf')
    ade_avg = float(np.mean(ade_list)) if ade_list else float('inf')
    fde_avg = float(np.mean(fde_list)) if fde_list else float('inf')

    print(
        f'[{split_name.upper()}] MSE: {mse_avg:.6f} | MAE: {mae_avg:.6f} | '
        f'ADE(m): {ade_avg:.6f} | FDE(m): {fde_avg:.6f}'
    )

    return mse_avg, mae_avg, ade_avg, fde_avg


def main():
    parser = argparse.ArgumentParser('TimeMixer trajectory baseline (strict dataloader control)')
    parser.add_argument('--data_path', type=str, default='./dataset/porto_processed.npz')
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2026)

    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--down_sampling_layers', type=int, default=1)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=5)

    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--bench_warmup', type=int, default=20)
    parser.add_argument('--bench_steps', type=int, default=100)
    parser.add_argument('--save_path', type=str, default=None,
                        help='if set, save model checkpoint to this .pt file after training')
    parser.add_argument('--load_path', type=str, default=None,
                        help='if set, load checkpoint and skip training')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_loader = TrajectoryDataLoader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    train_loader = data_loader.get_train()
    val_loader = data_loader.get_val()
    test_loader = data_loader.get_test()

    mean = torch.tensor(data_loader.mean, dtype=torch.float32, device=device)
    std = torch.tensor(data_loader.std, dtype=torch.float32, device=device)

    cfg = argparse.Namespace(
        task_name='long_term_forecast',
        model='TimeMixer',
        seq_len=args.seq_len,
        label_len=0,
        pred_len=args.pred_len,
        down_sampling_window=args.down_sampling_window,
        down_sampling_layers=args.down_sampling_layers,
        down_sampling_method='avg',
        channel_independence=1,
        e_layers=args.e_layers,
        moving_avg=args.moving_avg,
        decomp_method='moving_avg',
        top_k=args.top_k,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        enc_in=9,
        c_out=9,
        embed='fixed',
        freq='s',
        use_norm=1,
        use_future_temporal_feature=0,
    )

    model = TimeMixer(cfg).to(device)
    n_params = count_parameters(model)
    print(f'Parameter Count: {n_params:,d}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.load_path and os.path.isfile(args.load_path):
        print(f'Loading checkpoint from {args.load_path} ...')
        ckpt = torch.load(args.load_path, map_location=device, weights_only=False)
        state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print('Checkpoint loaded — skipping training.')
        args.epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, None, None, None)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        train_mse = float(np.mean(epoch_loss)) if epoch_loss else float('inf')
        print(f'Epoch [{epoch}/{args.epochs}] Train MSE: {train_mse:.6f}')
        evaluate(model, val_loader, criterion, mean, std, device, split_name='val')

    print('Final evaluation on test set:')
    test_mse, test_mae, test_ade, test_fde = evaluate(model, test_loader, criterion, mean, std, device, split_name='test')
    infer_ms = benchmark_inference_ms(model, test_loader, device, warmup=args.bench_warmup, steps=args.bench_steps)

    print(f'Inference Time: {infer_ms:.3f} ms/iter')
    print(
        'BENCHMARK|'
        f'model=TimeMixer|seq_len={args.seq_len}|pred_len={args.pred_len}|'
        f'params={n_params}|infer_ms={infer_ms:.6f}|'
        f'test_mse={test_mse:.6f}|test_mae={test_mae:.6f}|'
        f'test_ade_m={test_ade:.6f}|test_fde_m={test_fde:.6f}'
    )

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'args': vars(args),
            'mean': data_loader.mean,
            'std': data_loader.std,
        }, args.save_path)
        print(f'Checkpoint saved → {args.save_path}')


if __name__ == '__main__':
    main()

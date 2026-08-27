# coding=utf-8
"""Load and evaluate a PriMoTraj checkpoint.

Version-2 checkpoints contain the complete model/training configuration. Raw
legacy state dictionaries remain supported when all architecture flags are
provided explicitly together with ``--allow_partial_load`` when necessary.

Usage:
    python load_best.py --checkpoint checkpoints/PriMoTraj/seed2024_pred6/best.pt \
        --data data/porto_ds15.npz
"""

import argparse
import sys
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from tqdm import tqdm

from utils.traj_dataloader import TrajectoryDataLoader
from models.tsAMD import PriMoTraj


def haversine_torch(pred, true):
    R = 6371000.0
    lat1, lon1 = torch.deg2rad(true[..., 0]), torch.deg2rad(true[..., 1])
    lat2, lon2 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (torch.sin(dlat / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
    a = torch.clamp(a, min=1e-12, max=1.0 - 1e-7)
    return R * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


@torch.no_grad()
def benchmark_inference_ms(model, batch_x, device, warmup=20, steps=100):
    """Measure pure model forward-pass latency on a pre-fetched GPU batch.
    DataLoader overhead is excluded intentionally.
    """
    model.eval()
    batch_x = batch_x.to(device)
    # Warmup
    for _ in range(warmup):
        model(batch_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Measure
    durations = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(durations) / len(durations))


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    # ---- Load weights first so we can auto-detect pred_len ----
    print(f"Loading weights from {args.checkpoint} ...")
    raw_checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = None
    if isinstance(raw_checkpoint, dict) and 'model_state' in raw_checkpoint:
        state_dict = raw_checkpoint['model_state']
        saved_args = raw_checkpoint.get('args')
        if isinstance(saved_args, dict):
            for key in (
                'seq_len', 'pred_len', 'n_block', 'alpha', 'mix_layer_num',
                'mix_layer_scale', 'patch', 'norm', 'layernorm', 'dropout',
                'use_mdm', 'use_moe', 'moe_num_experts', 'moe_top_k',
                'moe_ff_dim',
                'pm_input', 'motion_prior', 'motion_prior_weight',
                'motion_prior_recent_weight', 'motion_prior_mode',
                'motion_prior_gate_hidden', 'motion_prior_residual_scale',
                'motion_prior_damping', 'motion_prior_n_priors',
                'motion_prior_gate_horizon', 'motion_prior_residual_head',
                'motion_prior_residual_hidden', 'motion_prior_residual_init',
            ):
                if key in saved_args:
                    setattr(args, key, saved_args[key])
    else:
        state_dict = raw_checkpoint

    # Strip _orig_mod. prefix if saved from torch.compile()
    if any(k.startswith("_orig_mod.") for k in state_dict):
        print("  Detected torch.compile() checkpoint — stripping '_orig_mod.' prefix")
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # Drop buffers that aren't part of the model architecture
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in ("data_mean", "data_std")}

    # Auto-detect pred_len from direct_head weight shape
    if "direct_head.weight" in state_dict:
        ckpt_pred_len = state_dict["direct_head.weight"].shape[0]
        if ckpt_pred_len != args.pred_len:
            print(f"  Auto-detected pred_len={ckpt_pred_len} from checkpoint "
                  f"(CLI had {args.pred_len}), using {ckpt_pred_len}")
            args.pred_len = ckpt_pred_len

    # ---- Data (built after pred_len is finalised) ----
    print(f"Loading data from {args.data} ...")
    data_loader = TrajectoryDataLoader(
        args.data, args.batch_size, args.seq_len, args.pred_len,
    )
    test_data = data_loader.get_test()
    geo_mean = torch.tensor(data_loader.mean[:2], device=device)
    geo_std  = torch.tensor(data_loader.std[:2],  device=device)

    # ---- Build model ----
    model = PriMoTraj(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        dropout=args.dropout,
        n_block=args.n_block,
        patch=args.patch,
        k=args.mix_layer_num,
        c=args.mix_layer_scale,
        alpha=args.alpha,
        target_slice=data_loader.target_slice,
        norm=args.norm,
        layernorm=args.layernorm,
        use_mdm=args.use_mdm,
        use_moe=args.use_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_ff_dim=args.moe_ff_dim,
        pm_input=args.pm_input,
        motion_prior=args.motion_prior,
        motion_prior_weight=args.motion_prior_weight,
        motion_prior_recent_weight=args.motion_prior_recent_weight,
        motion_prior_mode=args.motion_prior_mode,
        motion_prior_gate_hidden=args.motion_prior_gate_hidden,
        motion_prior_residual_scale=args.motion_prior_residual_scale,
        motion_prior_damping=args.motion_prior_damping,
        motion_prior_n_priors=args.motion_prior_n_priors,
        motion_prior_gate_horizon=args.motion_prior_gate_horizon,
        motion_prior_residual_head=args.motion_prior_residual_head,
        motion_prior_residual_hidden=args.motion_prior_residual_hidden,
        motion_prior_residual_init=args.motion_prior_residual_init,
    ).to(device)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=not args.allow_partial_load,
    )
    if missing:
        print(f"  WARNING: missing keys  : {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---- Evaluate ----
    criterion = torch.nn.MSELoss()
    metric_sums = {key: 0.0 for key in ('loss', 'mse', 'mae', 'ade', 'fde')}
    test_count = 0

    print("\nTest evaluation ...")
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            mse = ((outputs - batch_y) ** 2).mean()
            mae = torch.abs(outputs - batch_y).mean()
            pred_geo = outputs[:, :, :2] * geo_std + geo_mean
            true_geo = batch_y[:, :, :2]  * geo_std + geo_mean
            dists = haversine_torch(pred_geo, true_geo)
            batch_count = batch_x.size(0)
            for key, value in (
                ('loss', loss), ('mse', mse), ('mae', mae),
                ('ade', dists.mean()), ('fde', dists[:, -1].mean()),
            ):
                metric_sums[key] += value.item() * batch_count
            test_count += batch_count
            pbar.set_description(f"  loss={metric_sums['loss'] / test_count:.6f}")

    metrics = {key: value / max(1, test_count) for key, value in metric_sums.items()}

    # Pre-fetch fixed batches once — benchmark excludes DataLoader overhead
    bench_bs = args.bench_batch_size
    bench_batch = next(iter(test_data))[0][:bench_bs]
    b1_batch    = bench_batch[:1]

    infer_ms = benchmark_inference_ms(
        model, bench_batch, device,
        warmup=args.bench_warmup, steps=args.bench_steps,
    )
    infer_ms_b1 = benchmark_inference_ms(
        model, b1_batch, device,
        warmup=args.bench_warmup, steps=args.bench_steps,
    )

    print(f"\n{'=' * 50}")
    print(f"  Test MSE : {metrics['mse']:.6f}")
    print(f"  Test MAE : {metrics['mae']:.6f}")
    print(f"  Test ADE : {metrics['ade']:.1f} m")
    print(f"  Test FDE : {metrics['fde']:.1f} m")
    print(f"  Inference (batch={bench_bs}) : {infer_ms:.3f} ms/iter")
    print(f"  Inference (batch=1)          : {infer_ms_b1:.3f} ms/sample")
    print(f"  Params   : {n_params:,}")
    print(
        "BENCHMARK|"
        f"model=PriMoTraj|seq_len={args.seq_len}|pred_len={args.pred_len}|"
        f"params={n_params}|infer_ms={infer_ms:.6f}|infer_ms_b1={infer_ms_b1:.6f}|"
        f"bench_batch_size={bench_bs}|"
        f"test_mse={metrics['mse']:.6f}|test_mae={metrics['mae']:.6f}|"
        f"test_ade_m={metrics['ade']:.6f}|test_fde_m={metrics['fde']:.6f}|"
        f"use_mdm={int(args.use_mdm)}|use_moe={int(args.use_moe)}|"
        f"motion_prior={args.motion_prior}|motion_prior_weight={args.motion_prior_weight}|"
        f"motion_prior_recent_weight={args.motion_prior_recent_weight}|"
        f"motion_prior_mode={args.motion_prior_mode}|"
        f"motion_prior_gate_hidden={args.motion_prior_gate_hidden}|"
        f"motion_prior_residual_scale={args.motion_prior_residual_scale}|"
        f"motion_prior_damping={args.motion_prior_damping}|"
        f"motion_prior_n_priors={args.motion_prior_n_priors}|"
        f"motion_prior_gate_horizon={args.motion_prior_gate_horizon}|"
        f"motion_prior_residual_head={args.motion_prior_residual_head}|"
        f"motion_prior_residual_hidden={args.motion_prior_residual_hidden}|"
        f"motion_prior_residual_init={args.motion_prior_residual_init}"
    )
    print(f"{'=' * 50}")


def parse_args():
    parser = argparse.ArgumentParser(description='Load and evaluate a raw best.pt checkpoint')
    # required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best.pt (raw state_dict)')
    parser.add_argument('--data',       type=str,
                        default=str(ROOT / 'data/porto_processed.npz'))
    # must match training config exactly
    parser.add_argument('--seq_len',          type=int,   default=12)
    parser.add_argument('--pred_len',         type=int,   default=6)
    parser.add_argument('--n_block',          type=int,   default=1)
    parser.add_argument('--alpha',            type=float, default=0.5)
    parser.add_argument('--mix_layer_num',    type=int,   default=3)
    parser.add_argument('--mix_layer_scale',  type=int,   default=2)
    parser.add_argument('--patch',            type=int,   default=8)
    parser.add_argument('--norm',     type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--layernorm',type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--dropout',          type=float, default=0.1)
    parser.add_argument('--use_mdm',  type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--use_moe',  type=lambda x: x.lower() != 'false', default=False)
    parser.add_argument('--pm_input', type=str, default='x', choices=['x', 'mdm'])
    parser.add_argument('--motion_prior', type=str, default='cvmix',
                        choices=['none', 'last', 'cv', 'cvmix'])
    parser.add_argument('--motion_prior_weight', type=float, default=1.0)
    parser.add_argument('--motion_prior_recent_weight', type=float, default=0.5)
    parser.add_argument('--motion_prior_mode', type=str, default='gate',
                        choices=['blend', 'residual', 'gate'])
    parser.add_argument('--motion_prior_gate_hidden', type=int, default=64)
    parser.add_argument('--motion_prior_residual_scale', type=float, default=0.0)
    parser.add_argument('--motion_prior_damping', type=float, default=0.8)
    parser.add_argument('--motion_prior_n_priors', type=int, default=7, choices=[5, 7])
    parser.add_argument('--motion_prior_gate_horizon', type=int, default=3)
    parser.add_argument('--motion_prior_residual_head', type=str, default='full_window',
                        choices=['none', 'full_window'])
    parser.add_argument('--motion_prior_residual_hidden', type=int, default=64)
    parser.add_argument('--motion_prior_residual_init', type=float, default=0.15)
    parser.add_argument('--moe_num_experts',  type=int,   default=8)
    parser.add_argument('--moe_top_k',        type=int,   default=2)
    parser.add_argument('--moe_ff_dim',       type=int,   default=2048)
    # eval
    parser.add_argument('--batch_size',      type=int, default=512,
                        help='Batch size used for test evaluation throughput')
    parser.add_argument('--bench_batch_size', type=int, default=128,
                        help='Batch size for inference latency benchmark (default 128, matches training)')
    parser.add_argument('--bench_warmup', type=int, default=20)
    parser.add_argument('--bench_steps',  type=int, default=100)
    parser.add_argument('--allow_partial_load', action='store_true',
                        help='allow missing/unexpected keys for explicit legacy recovery')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

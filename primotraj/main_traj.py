# coding=utf-8
"""PriMoTraj training and evaluation on GPS-derived trajectories.

Paper-aligned defaults use T=12, H=6, seven priors, the full-window bounded
residual, and normalized MSE plus Haversine-FDE. See run_primotraj_deployed.sh.
"""

import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import torch
from tqdm import tqdm
from copy import deepcopy
import time

from utils.general import set_seed
from utils.traj_dataloader import TrajectoryDataLoader
from models.tsAMD import PriMoTraj


def haversine_torch(pred, true):
    """
    Vectorised Haversine distance in metres.

    Args:
        pred, true: (B, T, 2)  with [..., 0]=lat, [..., 1]=lon  (degrees)
    Returns:
        (B, T) distances in metres
    """
    R = 6371000.0
    lat1, lon1 = torch.deg2rad(true[..., 0]), torch.deg2rad(true[..., 1])
    lat2, lon2 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, min=1e-12, max=1.0 - 1e-7)
    return R * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


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


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    set_seed(args.seed)
    show_progress = bool(args.progress and sys.stdout.isatty())

    # ---- Data ----
    print(f"Loading data from {args.data} ...")
    data_loader = TrajectoryDataLoader(
        args.data, args.batch_size, args.seq_len, args.pred_len,
    )
    train_data = data_loader.get_train()
    val_data   = data_loader.get_val(shuffle=False, drop_last=False)
    test_data  = data_loader.get_test()

    # Denormalization constants for lat/lon (first 2 features)
    geo_mean = torch.tensor(data_loader.mean[:2], device=device)
    geo_std  = torch.tensor(data_loader.std[:2],  device=device)

    # ---- Model ----
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
        tpg_speed_idx=args.tpg_speed_idx,
        tpg_heading_sin_idx=args.tpg_heading_sin_idx,
        tpg_heading_cos_idx=args.tpg_heading_cos_idx,
        tpg_pool_t=args.tpg_pool_t,
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.train_epochs),
        eta_min=args.min_learning_rate,
    )

    # ---- Checkpoint directory ----
    save_directory = os.path.join(args.checkpoint_dir, args.name)
    if os.path.exists(save_directory):
        import glob
        import re
        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        save_directory = f"{path}{n}"
    os.makedirs(save_directory)
    print(f"Checkpoints → {save_directory}")

    best_val_ade = float('inf')
    best_model = None

    # ==== Training ====
    for epoch in range(args.train_epochs):
        # --- Train ---
        model.train()
        train_mloss = torch.zeros(1, device=device)
        iter_time = 0.0
        print(f"\nepoch {epoch + 1}/{args.train_epochs}")
        print("Train")
        pbar = tqdm(enumerate(train_data), total=len(train_data), disable=not show_progress)
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            t0 = time.time()
            outputs, moe_loss = model(batch_x)
            optimizer.zero_grad()
            loss = args.mse_loss_weight * criterion(outputs, batch_y) + moe_loss
            if args.fde_loss_weight > 0 or args.ade_loss_weight > 0:
                pred_geo = outputs[:, :, :2] * geo_std + geo_mean
                true_geo = batch_y[:, :, :2] * geo_std + geo_mean
                geo_dist_km = haversine_torch(pred_geo, true_geo) / 1000.0
                if args.fde_loss_weight > 0:
                    loss = loss + args.fde_loss_weight * geo_dist_km[:, -1].mean()
                if args.ade_loss_weight > 0:
                    # Legacy ablations only; the paper configuration uses FDE.
                    loss = loss + args.ade_loss_weight * geo_dist_km.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            t1 = time.time()
            train_mloss = (train_mloss * i + loss.detach()) / (i + 1)
            iter_time   = (iter_time * i + (t1 - t0) * 1000) / (i + 1)
            pbar.set_description(f"  loss={train_mloss.item():.6f}")
        print(f"  train loss: {train_mloss.item():.6f}  iter: {iter_time:.1f}ms")

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_ade_sum = 0.0
        val_count = 0
        print("Val")
        pbar = tqdm(enumerate(val_data), total=len(val_data), disable=not show_progress)
        with torch.no_grad():
            for i, (batch_x, batch_y) in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                batch_count = batch_x.size(0)
                val_loss_sum += loss.item() * batch_count
                # ADE in metres
                pred_geo = outputs[:, :, :2] * geo_std + geo_mean
                true_geo = batch_y[:, :, :2] * geo_std + geo_mean
                ade = haversine_torch(pred_geo, true_geo).mean()
                val_ade_sum += ade.item() * batch_count
                val_count += batch_count
                pbar.set_description(f"  loss={val_loss_sum / val_count:.6f}")

            val_mloss = val_loss_sum / max(1, val_count)
            val_ade = val_ade_sum / max(1, val_count)
            if val_ade < best_val_ade:
                best_val_ade = val_ade
                best_model = deepcopy(model.state_dict())
                torch.save(
                    {
                        'format_version': 2,
                        'model_name': 'PriMoTraj',
                        'model_state': best_model,
                        'args': vars(args),
                        'best_val_ade_m': best_val_ade,
                        'val_nmse_at_best_ade': val_mloss,
                        'epoch': epoch + 1,
                    },
                    os.path.join(save_directory, "best.pt"),
                )

        scheduler.step()
        print(f"  val loss: {val_mloss:.6f}  ADE: {val_ade:.1f}m  lr: {scheduler.get_last_lr()[0]:.3e}")

    # ==== Test ====
    if best_model is None:
        # Selection is strictly "lowest validation ADE", so a run whose every
        # epoch produced a non-finite ADE never records a checkpoint. Fail with
        # the actual cause instead of load_state_dict(None) further down.
        raise RuntimeError(
            f"No checkpoint was selected across {args.train_epochs} epoch(s): "
            "validation ADE was never finite. This usually means the run "
            "diverged (NaN/inf loss); check the training log above."
        )
    model.load_state_dict(best_model)
    model.eval()

    metric_sums = {key: 0.0 for key in ('loss', 'mse', 'mae', 'ade', 'fde')}
    test_count = 0

    print("\nFinal Test")
    pbar = tqdm(enumerate(test_data), total=len(test_data), disable=not show_progress)

    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            mse = ((outputs - batch_y) ** 2).mean()
            mae = torch.abs(outputs - batch_y).mean()
            # Geographic metrics on lat/lon
            pred_geo = outputs[:, :, :2] * geo_std + geo_mean
            true_geo = batch_y[:, :, :2] * geo_std + geo_mean
            dists = haversine_torch(pred_geo, true_geo)   # (B, pred_len)
            ade = dists.mean()
            fde = dists[:, -1].mean()
            batch_count = batch_x.size(0)
            for key, value in (
                ('loss', loss), ('mse', mse), ('mae', mae),
                ('ade', ade), ('fde', fde),
            ):
                metric_sums[key] += value.item() * batch_count
            test_count += batch_count
            pbar.set_description(f"  loss={metric_sums['loss'] / test_count:.6f}")

    metrics = {key: value / max(1, test_count) for key, value in metric_sums.items()}

    infer_ms = benchmark_inference_ms(model, test_data, device, warmup=args.bench_warmup, steps=args.bench_steps)

    print(f"\n{'=' * 50}")
    print(f"  Test MSE : {metrics['mse']:.6f}")
    print(f"  Test MAE : {metrics['mae']:.6f}")
    print(f"  Test ADE : {metrics['ade']:.1f} m")
    print(f"  Test FDE : {metrics['fde']:.1f} m")
    print(f"  Inference Time : {infer_ms:.3f} ms/iter")
    print(
        "BENCHMARK|"
        f"model=PriMoTraj|seq_len={args.seq_len}|pred_len={args.pred_len}|"
        f"params={n_params}|infer_ms={infer_ms:.6f}|"
        f"test_mse={metrics['mse']:.6f}|test_mae={metrics['mae']:.6f}|"
        f"test_ade_m={metrics['ade']:.6f}|test_fde_m={metrics['fde']:.6f}|"
        f"pm_input={args.pm_input}|use_mdm={int(args.use_mdm)}|use_moe={int(args.use_moe)}|"
        f"moe_num_experts={args.moe_num_experts}|moe_top_k={args.moe_top_k}|"
        f"moe_ff_dim={args.moe_ff_dim}|"
        f"tpg_speed_idx={args.tpg_speed_idx}|"
        f"tpg_heading_sin_idx={args.tpg_heading_sin_idx}|"
        f"tpg_heading_cos_idx={args.tpg_heading_cos_idx}|"
        f"tpg_pool_t={args.tpg_pool_t}|"
        f"mse_loss_weight={args.mse_loss_weight}|"
        f"fde_loss_weight={args.fde_loss_weight}|ade_loss_weight={args.ade_loss_weight}|"
        f"optimizer=AdamW|checkpoint_metric=val_ade_m|"
        f"weight_decay={args.weight_decay}|grad_clip_norm={args.grad_clip_norm}|"
        f"min_learning_rate={args.min_learning_rate}|"
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
    parser = argparse.ArgumentParser(description='PriMoTraj trajectory prediction')

    # basic
    parser.add_argument('--seed',           type=int,   default=2024)
    parser.add_argument('--data',           type=str,   default=str(ROOT / 'data/porto_processed.npz'))
    parser.add_argument('--checkpoint_dir', type=str,   default=str(ROOT / 'checkpoints'))
    parser.add_argument('--name',           type=str,   default='primotraj_porto')

    # forecasting
    parser.add_argument('--seq_len',  type=int, default=12, help='input window')
    parser.add_argument('--pred_len', type=int, default=6, help='prediction window')

    # model
    parser.add_argument('--n_block',        type=int,   default=1)
    parser.add_argument('--alpha',          type=float, default=0.5,  help='cross-feature interaction weight')
    parser.add_argument('--mix_layer_num',  type=int,   default=3,    help='MDM depth k')
    parser.add_argument('--mix_layer_scale',type=int,   default=2,    help='MDM scale factor c')
    parser.add_argument('--patch',          type=int,   default=8,    help='DDI patch size')
    parser.add_argument('--norm',           type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--layernorm',      type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--use_mdm',        type=lambda x: x.lower() != 'false', default=True,
                        help='enable MDM multi-scale temporal decomposition')
    parser.add_argument('--use_moe',        type=lambda x: x.lower() != 'false', default=False,
                        help='enable MoE destination-aware predictor')
    parser.add_argument('--pm_input',        type=str,   default='x',  choices=['x', 'mdm'],
                        help="PM input: 'x' = original (gated MDM residual), 'mdm' = MDM output feeds PM directly")
    parser.add_argument('--motion_prior',    type=str,   default='cvmix',
                        choices=['none', 'last', 'cv', 'cvmix'],
                        help='lat/lon motion prior applied after the neural head')
    parser.add_argument('--motion_prior_weight', type=float, default=1.0,
                        help='blend weight for motion prior on lat/lon (0 disables, 1 replaces)')
    parser.add_argument('--motion_prior_recent_weight', type=float, default=0.5,
                        help='cvmix velocity blend: recent one-step weight vs short-window mean')
    parser.add_argument('--motion_prior_mode', type=str, default='gate',
                        choices=['blend', 'residual', 'gate'],
                        help='how to combine neural head and motion priors')
    parser.add_argument('--motion_prior_gate_hidden', type=int, default=64,
                        help='hidden size of tiny dynamic prior gate')
    parser.add_argument('--motion_prior_residual_scale', type=float, default=0.0,
                        help='bounded residual scale for residual/gate prior modes')
    parser.add_argument('--motion_prior_damping', type=float, default=0.8,
                        help='velocity damping factor used by gate prior bank')
    # Deployed motion-prior gate configuration.
    parser.add_argument('--motion_prior_n_priors', type=int, default=7, choices=[5, 7],
                        help='size of the prior bank: 5 = compact bank (P1-P5), '
                             '7 = deployed bank (adds constant-acceleration and '
                             'long-window smoothed-velocity priors)')
    parser.add_argument('--motion_prior_gate_horizon', type=int, default=3,
                        help='K_g: number of trailing input steps the gate MLP reads')
    parser.add_argument('--motion_prior_residual_head', type=str, default='full_window',
                        choices=['none', 'full_window'],
                        help="'none' = scalar-bounded correction of the neural head "
                             "(compact configuration); 'full_window' = independent "
                             "two-layer residual MLP over the whole input window with "
                             "a learned per-step bound (deployed configuration)")
    parser.add_argument('--motion_prior_residual_hidden', type=int, default=64,
                        help='hidden width r of the full-window residual head')
    parser.add_argument('--motion_prior_residual_init', type=float, default=0.15,
                        help='initial value of the learned per-step residual bound')
    parser.add_argument('--moe_num_experts',type=int,   default=8,    help='number of experts in MoE')
    parser.add_argument('--moe_top_k',      type=int,   default=2,    help='top-k experts selected by gate')
    parser.add_argument('--moe_ff_dim',     type=int,   default=2048, help='hidden dim of each MoE expert MLP')

    # optimisation
    parser.add_argument('--train_epochs',   type=int,   default=5)
    parser.add_argument('--batch_size',     type=int,   default=2048)
    parser.add_argument('--learning_rate',  type=float, default=5e-5)
    parser.add_argument('--min_learning_rate', type=float, default=1e-6,
                        help='non-zero floor for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--bench_warmup',   type=int,   default=20,   help='warmup iterations for inference benchmark')
    parser.add_argument('--bench_steps',    type=int,   default=100,  help='measured iterations for inference benchmark')
    parser.add_argument('--progress',       type=lambda x: x.lower() != 'false', default=True,
                        help='show tqdm progress bars (auto-disabled when stdout is not a TTY)')

    # Turning-Point Gate (TPG) — feature channel indices in the input tensor
    # Porto layout: lat=0 lon=1 speed=2 heading_sin=3 heading_cos=4 hour_sin=5 ...
    parser.add_argument('--tpg_speed_idx',        type=int, default=-1,
                        help='channel index for speed (TPG); -1 disables speed signal')
    parser.add_argument('--tpg_heading_sin_idx',  type=int, default=-1,
                        help='channel index for sin(heading) (TPG); -1 disables heading signal')
    parser.add_argument('--tpg_heading_cos_idx',  type=int, default=-1,
                        help='channel index for cos(heading) (TPG); -1 disables heading signal')
    parser.add_argument('--tpg_pool_t',           type=int, default=8,
                        help='temporal pooling bins in TurningPointGate MLP (default 8 ≈ 2 min @ 15 s)')
    parser.add_argument('--mse_loss_weight', type=float, default=0.1,
                        help='beta_mse for normalized MSE over all target channels')
    parser.add_argument('--fde_loss_weight', type=float, default=1.0,
                        help='weight for the paper Haversine-FDE term; metres are scaled by /1000')
    parser.add_argument('--ade_loss_weight', type=float, default=0.0,
                        help='legacy-only Haversine-ADE term; keep 0 for the paper configuration')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

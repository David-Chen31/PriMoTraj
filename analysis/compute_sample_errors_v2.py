#!/usr/bin/env python3
"""Export sample-level ADE/FDE errors for paired bootstrap tests."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
# Release layout ships the model under primotraj/; an older layout used src/.
MODEL_ROOT = ROOT / "primotraj" if (ROOT / "primotraj").is_dir() else ROOT / "src"


def add_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


add_path(ROOT)

from baselines.traj_utils import TrajectoryDataLoader, denorm_latlon, haversine_torch  # noqa: E402
from baselines.physics_baselines import PhysicsBaseline  # noqa: E402


def state_from_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("args") or {}
    return ckpt, {}


def namespace_from(defaults: dict, args_dict: dict):
    merged = dict(defaults)
    merged.update({k: v for k, v in args_dict.items() if v is not None})
    return argparse.Namespace(**merged)


def load_primotraj(checkpoint: Path, input_size: int, seq_len: int, pred_len: int, device):
    add_path(MODEL_ROOT)
    from models.tsAMD import AMD

    state, saved_args = state_from_checkpoint(checkpoint, device)
    cfg = namespace_from(
        {
            "n_block": 1,
            "alpha": 0.5,
            "mix_layer_num": 3,
            "mix_layer_scale": 2,
            "patch": 8,
            "norm": True,
            "layernorm": True,
            "dropout": 0.1,
            "use_mdm": True,
            "use_moe": False,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_ff_dim": 2048,
            "pm_input": "x",
            "motion_prior": "cvmix",
            "motion_prior_weight": 1.0,
            "motion_prior_recent_weight": 0.5,
            "motion_prior_mode": "gate",
            "motion_prior_gate_hidden": 16,
            "motion_prior_residual_scale": 0.02,
            "motion_prior_damping": 0.8,
        },
        saved_args,
    )
    model = AMD(
        input_shape=(seq_len, input_size),
        pred_len=pred_len,
        dropout=cfg.dropout,
        n_block=cfg.n_block,
        patch=cfg.patch,
        k=cfg.mix_layer_num,
        c=cfg.mix_layer_scale,
        alpha=cfg.alpha,
        target_slice=slice(0, None),
        norm=cfg.norm,
        layernorm=cfg.layernorm,
        use_mdm=cfg.use_mdm,
        use_moe=cfg.use_moe,
        moe_num_experts=getattr(cfg, "moe_num_experts", 8),
        moe_top_k=getattr(cfg, "moe_top_k", 2),
        moe_ff_dim=getattr(cfg, "moe_ff_dim", 2048),
        pm_input=getattr(cfg, "pm_input", "x"),
        motion_prior=getattr(cfg, "motion_prior", "cvmix"),
        motion_prior_weight=getattr(cfg, "motion_prior_weight", 1.0),
        motion_prior_recent_weight=getattr(cfg, "motion_prior_recent_weight", 0.5),
        motion_prior_mode=getattr(cfg, "motion_prior_mode", "gate"),
        motion_prior_gate_hidden=getattr(cfg, "motion_prior_gate_hidden", 16),
        motion_prior_residual_scale=getattr(cfg, "motion_prior_residual_scale", 0.02),
        motion_prior_damping=getattr(cfg, "motion_prior_damping", 0.8),
        # Deployed-configuration knobs. Without these the model is rebuilt with
        # the legacy compact defaults (5 priors, no residual head) and
        # strict=False then silently drops the weights that do not fit.
        motion_prior_n_priors=getattr(cfg, "motion_prior_n_priors", 5),
        motion_prior_gate_horizon=getattr(cfg, "motion_prior_gate_horizon", 3),
        motion_prior_residual_head=getattr(cfg, "motion_prior_residual_head", "none"),
        motion_prior_residual_hidden=getattr(cfg, "motion_prior_residual_hidden", 64),
        motion_prior_residual_init=getattr(cfg, "motion_prior_residual_init", 0.15),
        motion_prior_experts=getattr(cfg, "motion_prior_experts", "closed_form"),
        motion_prior_expert_hidden=getattr(cfg, "motion_prior_expert_hidden", 40),
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the rebuilt model: missing={list(missing)} "
            f"unexpected={list(unexpected)}")
    model.eval()
    return model


def load_patchtst(checkpoint: Path, input_size: int, seq_len: int, pred_len: int, device):
    add_path(ROOT / "PatchTST" / "PatchTST_supervised")
    module = importlib.import_module("models.PatchTST")
    state, saved_args = state_from_checkpoint(checkpoint, device)
    cfg_args = namespace_from(
        {
            "d_model": 128,
            "d_ff": 256,
            "n_heads": 4,
            "e_layers": 2,
            "dropout": 0.1,
            "fc_dropout": 0.05,
            "head_dropout": 0.0,
            "patch_len": 6,
            "stride": 3,
        },
        saved_args,
    )
    cfg = argparse.Namespace(
        enc_in=input_size,
        seq_len=seq_len,
        pred_len=pred_len,
        e_layers=cfg_args.e_layers,
        n_heads=cfg_args.n_heads,
        d_model=cfg_args.d_model,
        d_ff=cfg_args.d_ff,
        dropout=cfg_args.dropout,
        fc_dropout=cfg_args.fc_dropout,
        head_dropout=cfg_args.head_dropout,
        patch_len=cfg_args.patch_len,
        stride=cfg_args.stride,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
    )
    model = module.Model(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_timemixer(checkpoint: Path, input_size: int, seq_len: int, pred_len: int, device):
    add_path(ROOT / "TimeMixer")
    module = importlib.import_module("models.TimeMixer")
    state, saved_args = state_from_checkpoint(checkpoint, device)
    cfg_args = namespace_from(
        {
            "d_model": 64,
            "d_ff": 128,
            "e_layers": 2,
            "dropout": 0.1,
            "moving_avg": 25,
            "down_sampling_layers": 1,
            "down_sampling_window": 2,
            "top_k": 5,
        },
        saved_args,
    )
    cfg = argparse.Namespace(
        task_name="long_term_forecast",
        model="TimeMixer",
        seq_len=seq_len,
        label_len=0,
        pred_len=pred_len,
        down_sampling_window=cfg_args.down_sampling_window,
        down_sampling_layers=cfg_args.down_sampling_layers,
        down_sampling_method="avg",
        channel_independence=1,
        e_layers=cfg_args.e_layers,
        moving_avg=cfg_args.moving_avg,
        decomp_method="moving_avg",
        top_k=cfg_args.top_k,
        d_model=cfg_args.d_model,
        d_ff=cfg_args.d_ff,
        dropout=cfg_args.dropout,
        enc_in=input_size,
        c_out=input_size,
        embed="fixed",
        freq="s",
        use_norm=1,
        use_future_temporal_feature=0,
    )
    model = module.Model(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_lstm(checkpoint: Path, input_size: int, pred_len: int, device):
    from GRU.main_traj import RNNForecaster

    state, saved_args = state_from_checkpoint(checkpoint, device)
    cfg = namespace_from(
        {"hidden_size": 64, "num_layers": 1, "dropout": 0.0, "rnn_type": "LSTM"},
        saved_args,
    )
    model = RNNForecaster(
        input_size=input_size,
        pred_len=pred_len,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.rnn_type,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_physics(model_name: str, pred_len: int, device, kf_q=1e-4, kf_r=1e-2, kf_damping=1.0):
    if model_name == "last":
        return PhysicsBaseline(pred_len, mode="last").to(device)
    if model_name == "cv":
        return PhysicsBaseline(pred_len, mode="cv").to(device)
    if model_name == "cvmix":
        return PhysicsBaseline(pred_len, mode="cvmix", recent_steps=3, mix_weight=0.5).to(device)
    if model_name == "cv_kf":
        return PhysicsBaseline(pred_len, mode="cv_kf", kf_q=kf_q, kf_r=kf_r, kf_damping=kf_damping).to(device)
    raise ValueError(f"Unknown physics model {model_name}")


def forward_model(model, model_type: str, x):
    if model_type == "timemixer":
        return model(x, None, None, None)
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Compute sample-level trajectory errors")
    parser.add_argument("--model_type", required=True,
                        choices=["primotraj", "patchtst", "timemixer", "lstm", "last", "cv", "cvmix", "cv_kf"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="Porto-15s-s12")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=131072)
    parser.add_argument("--sample_strategy", choices=["sequential", "even"], default="even")
    parser.add_argument("--kf_q", type=float, default=1e-4)
    parser.add_argument("--kf_r", type=float, default=1e-2)
    parser.add_argument("--kf_damping", type=float, default=1.0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    loader_owner = TrajectoryDataLoader(args.data_path, args.batch_size, args.seq_len, args.pred_len)
    if args.max_samples > 0 and args.sample_strategy == "even":
        data_tensor, windows = loader_owner._splits["test"]
        if len(windows) > args.max_samples:
            keep = np.linspace(0, len(windows) - 1, num=args.max_samples, dtype=np.int64)
            loader_owner._splits["test"] = (data_tensor, windows[keep])
    # Trip id per retained window. Windows are data-start offsets into the
    # concatenated array, so the owning trip is a searchsorted lookup over the
    # trip boundaries -- it depends only on the split, not on the model, and is
    # what the paper's trip-level paired bootstrap resamples over.
    _raw = np.load(args.data_path)
    _bounds = np.zeros(len(_raw["trip_lengths"]) + 1, dtype=np.int64)
    np.cumsum(_raw["trip_lengths"], out=_bounds[1:])
    trip_ids = np.searchsorted(_bounds, loader_owner._splits["test"][1], side="right") - 1
    trip_ids = trip_ids.astype(np.int32)
    test_loader = loader_owner.get_test()
    mean = torch.tensor(loader_owner.mean, dtype=torch.float32, device=device)
    std = torch.tensor(loader_owner.std, dtype=torch.float32, device=device)
    input_size = int(loader_owner.n_feature)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if args.model_type == "primotraj":
        model = load_primotraj(ckpt, input_size, args.seq_len, args.pred_len, device)
    elif args.model_type == "patchtst":
        model = load_patchtst(ckpt, input_size, args.seq_len, args.pred_len, device)
    elif args.model_type == "timemixer":
        model = load_timemixer(ckpt, input_size, args.seq_len, args.pred_len, device)
    elif args.model_type == "lstm":
        model = load_lstm(ckpt, input_size, args.pred_len, device)
    else:
        model = load_physics(args.model_type, args.pred_len, device, args.kf_q, args.kf_r, args.kf_damping)

    ade_parts = []
    fde_parts = []
    n_seen = 0
    for raw_x, raw_y in test_loader:
        raw_x = raw_x.to(device)
        raw_y = raw_y.to(device)
        x = (raw_x - mean) / std
        y = (raw_y - mean) / std
        pred = forward_model(model, args.model_type, x)
        pred_latlon = denorm_latlon(pred, mean, std)
        true_latlon = denorm_latlon(y, mean, std)
        dist = haversine_torch(pred_latlon, true_latlon)
        ade_parts.append(dist.mean(dim=1).detach().cpu())
        fde_parts.append(dist[:, -1].detach().cpu())
        n_seen += raw_x.size(0)
        if args.max_samples > 0 and n_seen >= args.max_samples:
            break

    ade = torch.cat(ade_parts).numpy()
    fde = torch.cat(fde_parts).numpy()
    if args.max_samples > 0:
        ade = ade[:args.max_samples]
        fde = fde[:args.max_samples]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ade=ade.astype(np.float32),
        fde=fde.astype(np.float32),
        trip_ids=trip_ids[:len(ade)],
        dataset=args.dataset,
        model_type=args.model_type,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        sample_strategy=args.sample_strategy,
    )
    print(f"model={args.model_type} pred_len={args.pred_len} samples={len(ade)} ade={ade.mean():.6f} fde={fde.mean():.6f}")
    print(f"output={out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze PriMoTraj gate allocation and empirical prior agreement."""

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# Release layout ships the model under primotraj/; the development tree used
# LiteMoTraj/. Accept either so the script runs from both.
LITE_ROOT = ROOT / "primotraj" if (ROOT / "primotraj").is_dir() else ROOT / "LiteMoTraj"
for item in (str(ROOT), str(LITE_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

# LITE_ROOT is on sys.path, so import the modules directly rather than through
# a package name that differs between the release and development layouts.
from utils.traj_dataloader import TrajectoryDataLoader  # noqa: E402
from models.tsAMD import PriMoTraj  # noqa: E402



def haversine_torch(pred, true):
    radius = 6371000.0
    lat1, lon1 = torch.deg2rad(true[..., 0]), torch.deg2rad(true[..., 1])
    lat2, lon2 = torch.deg2rad(pred[..., 0]), torch.deg2rad(pred[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, min=1e-12, max=1.0 - 1e-7)
    return radius * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


def load_gate_model(checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    args = state.get("args", {}) if isinstance(state, dict) else {}
    pred_len = int(args.get("pred_len", 6))
    seq_len = int(args.get("seq_len", 12))
    model = PriMoTraj(
        input_shape=(seq_len, 9),
        pred_len=pred_len,
        dropout=args.get("dropout", 0.1),
        n_block=args.get("n_block", 1),
        patch=args.get("patch", 8),
        k=args.get("mix_layer_num", 3),
        c=args.get("mix_layer_scale", 2),
        alpha=args.get("alpha", 0.5),
        target_slice=slice(0, None),
        norm=args.get("norm", True),
        layernorm=args.get("layernorm", True),
        use_mdm=args.get("use_mdm", True),
        use_moe=args.get("use_moe", False),
        moe_num_experts=args.get("moe_num_experts", 8),
        moe_top_k=args.get("moe_top_k", 2),
        moe_ff_dim=args.get("moe_ff_dim", 2048),
        tpg_speed_idx=args.get("tpg_speed_idx", -1),
        tpg_heading_sin_idx=args.get("tpg_heading_sin_idx", -1),
        tpg_heading_cos_idx=args.get("tpg_heading_cos_idx", -1),
        tpg_pool_t=args.get("tpg_pool_t", 8),
        pm_input=args.get("pm_input", "x"),
        motion_prior=args.get("motion_prior", "cvmix"),
        motion_prior_weight=args.get("motion_prior_weight", 1.0),
        motion_prior_recent_weight=args.get("motion_prior_recent_weight", 0.5),
        motion_prior_mode=args.get("motion_prior_mode", "gate"),
        motion_prior_gate_hidden=args.get("motion_prior_gate_hidden", 64),
        motion_prior_residual_scale=args.get("motion_prior_residual_scale", 0.0),
        motion_prior_damping=args.get("motion_prior_damping", 0.8),
        motion_prior_n_priors=args.get("motion_prior_n_priors", 7),
        motion_prior_gate_horizon=args.get("motion_prior_gate_horizon", 3),
        motion_prior_residual_head=args.get("motion_prior_residual_head", "full_window"),
        motion_prior_residual_hidden=args.get("motion_prior_residual_hidden", 64),
        motion_prior_residual_init=args.get("motion_prior_residual_init", 0.15),
    ).to(device)
    sd = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, args


def denorm_latlon(x_norm, mean, std):
    return x_norm[..., :2] * std[:2].view(1, 1, 2) + mean[:2].view(1, 1, 2)


def motion_state(batch_x, mean, std):
    latlon = denorm_latlon(batch_x, mean, std)
    if latlon.size(1) < 2:
        return torch.zeros(latlon.size(0), dtype=torch.long, device=latlon.device)
    step_dist = haversine_torch(latlon[:, 1:, :], latlon[:, :-1, :])
    mean_step = step_dist.mean(dim=1)
    return torch.bucketize(mean_step, torch.tensor([5.0, 30.0, 80.0], device=latlon.device))


@torch.no_grad()
def analyze(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_gate_model(args.checkpoint, device)
    prior_names = list(model.PRIOR_NAMES[:model.motion_prior_n_priors])
    n_priors = len(prior_names)
    pred_len = int(ckpt_args.get("pred_len", args.pred_len))
    seq_len = int(ckpt_args.get("seq_len", args.seq_len))
    loader_wrap = TrajectoryDataLoader(args.data_path, args.batch_size, seq_len, pred_len)
    loader = loader_wrap.get_test()
    mean = torch.tensor(loader_wrap.mean, dtype=torch.float32, device=device)
    std = torch.tensor(loader_wrap.std, dtype=torch.float32, device=device)

    totals = {
        "n": 0,
        "entropy": 0.0,
        "top_agree": 0.0,
        "uniform_ade": 0.0,
        "weighted_prior_ade": 0.0,
        "final_model_ade": 0.0,
    }
    gate_sum = torch.zeros(n_priors, device=device)
    oracle_sum = torch.zeros(n_priors, device=device)
    top_sum = torch.zeros(n_priors, device=device)
    state_rows = {}

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        n_tail = min(model.motion_prior_gate_horizon, batch_x.size(1))
        weights = torch.softmax(model.motion_gate(batch_x[:, -n_tail:, :].reshape(batch_x.size(0), -1)), dim=-1)
        pred, _ = model(batch_x)
        priors = model._motion_prior_bank(batch_x, pred[:, :, :2])
        weighted = (weights[:, :, None, None] * priors).sum(dim=1)
        uniform = priors.mean(dim=1)

        y_latlon = denorm_latlon(batch_y, mean, std)
        prior_latlon = priors * std[:2].view(1, 1, 1, 2) + mean[:2].view(1, 1, 1, 2)
        weighted_latlon = denorm_latlon(weighted, mean, std)
        uniform_latlon = denorm_latlon(uniform, mean, std)
        final_latlon = denorm_latlon(pred, mean, std)

        prior_ade = haversine_torch(
            prior_latlon.reshape(-1, pred_len, 2),
            y_latlon[:, None, :, :].expand(-1, n_priors, -1, -1).reshape(-1, pred_len, 2),
        ).mean(dim=1).view(batch_x.size(0), n_priors)
        oracle = prior_ade.argmin(dim=1)
        top_gate = weights.argmax(dim=1)
        entropy = -(weights * (weights.clamp_min(1e-12).log())).sum(dim=1)
        uniform_ade = haversine_torch(uniform_latlon, y_latlon).mean(dim=1)
        weighted_ade = haversine_torch(weighted_latlon, y_latlon).mean(dim=1)
        final_ade = haversine_torch(final_latlon, y_latlon).mean(dim=1)

        batch_n = batch_x.size(0)
        totals["n"] += int(batch_n)
        totals["entropy"] += float(entropy.sum().item())
        totals["top_agree"] += float((top_gate == oracle).float().sum().item())
        totals["uniform_ade"] += float(uniform_ade.sum().item())
        totals["weighted_prior_ade"] += float(weighted_ade.sum().item())
        totals["final_model_ade"] += float(final_ade.sum().item())
        gate_sum += weights.sum(dim=0)
        oracle_sum += torch.bincount(oracle, minlength=n_priors).to(device)
        top_sum += torch.bincount(top_gate, minlength=n_priors).to(device)

        states = motion_state(batch_x, mean, std)
        for state_id in states.unique().tolist():
            mask = states == int(state_id)
            if not mask.any():
                continue
            row = state_rows.setdefault(int(state_id), {
                "n": 0,
                "entropy": 0.0,
                "agree": 0.0,
                "final_ade": 0.0,
                "gate": torch.zeros(n_priors, device=device),
            })
            row_n = int(mask.sum().item())
            row["n"] += row_n
            row["entropy"] += float(entropy[mask].sum().item())
            row["agree"] += float((top_gate[mask] == oracle[mask]).float().sum().item())
            row["final_ade"] += float(final_ade[mask].sum().item())
            row["gate"] += weights[mask].sum(dim=0)

    if totals["n"] == 0:
        raise RuntimeError("No batches analyzed")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "max_batches": args.max_batches,
        "samples": totals["n"],
        "gate_entropy": totals["entropy"] / totals["n"],
        "top_prior_oracle_agreement": totals["top_agree"] / totals["n"],
        "uniform_gate_ade_m": totals["uniform_ade"] / totals["n"],
        "weighted_prior_ade_m": totals["weighted_prior_ade"] / totals["n"],
        "final_model_ade_m": totals["final_model_ade"] / totals["n"],
    }
    for idx, name in enumerate(prior_names):
        row[f"gate_mean_{name}"] = float(gate_sum[idx].item() / totals["n"])
        row[f"oracle_frac_{name}"] = float(oracle_sum[idx].item() / totals["n"])
        row[f"top_gate_frac_{name}"] = float(top_sum[idx].item() / totals["n"])

    fields = list(row.keys())
    write_header = not out_csv.exists() or args.overwrite
    mode = "w" if args.overwrite else "a"
    with out_csv.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    if args.state_output_csv:
        state_csv = Path(args.state_output_csv)
        state_csv.parent.mkdir(parents=True, exist_ok=True)
        fields_state = ["dataset", "checkpoint", "seq_len", "pred_len", "state_id", "samples", "gate_entropy", "top_prior_oracle_agreement", "final_model_ade_m"] + [
            f"gate_mean_{name}" for name in prior_names
        ]
        with state_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields_state)
            writer.writeheader()
            for state_id in sorted(state_rows):
                item = state_rows[state_id]
                out = {
                    "dataset": args.dataset,
                    "checkpoint": str(args.checkpoint),
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "state_id": state_id,
                    "samples": item["n"],
                    "gate_entropy": item["entropy"] / item["n"],
                    "top_prior_oracle_agreement": item["agree"] / item["n"],
                    "final_model_ade_m": item["final_ade"] / item["n"],
                }
                for idx, name in enumerate(prior_names):
                    out[f"gate_mean_{name}"] = float(item["gate"][idx].item() / item["n"])
                writer.writerow(out)

    print(f"samples={totals['n']}")
    print(f"gate_entropy={row['gate_entropy']:.6f}")
    print(f"top_prior_oracle_agreement={row['top_prior_oracle_agreement']:.6f}")
    print(f"uniform_gate_ade_m={row['uniform_gate_ade_m']:.6f}")
    print(f"weighted_prior_ade_m={row['weighted_prior_ade_m']:.6f}")
    print(f"final_model_ade_m={row['final_model_ade_m']:.6f}")
    print(f"csv={out_csv}")


def main():
    parser = argparse.ArgumentParser("Analyze PriMoTraj motion-prior weights")
    parser.add_argument("--dataset", default="Porto-15s-s12")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--output_csv", default="benchmarks/results/gate_sanity_v2.csv")
    parser.add_argument("--state_output_csv", default="benchmarks/results/gate_sanity_by_state_v2.csv")
    parser.add_argument("--overwrite", action="store_true", default=False)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()

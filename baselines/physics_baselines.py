# coding=utf-8
"""Deterministic (zero-parameter) physics baselines: Last / CV / CVMix / CV-KF.

RECONSTRUCTED FILE. The original ``baselines/physics_baselines.py`` was lost
together with the training server. It has been rebuilt from:

* the constructor and call signature preserved in
  ``benchmarks/scripts/compute_sample_errors_v2.py`` and
  ``benchmarks/scripts/profile_online_latency_v2.py``
  (``PhysicsBaseline(pred_len, mode=..., recent_steps=3, mix_weight=0.5,
  kf_q=..., kf_r=..., kf_damping=...)``, an ``nn.Module`` called as
  ``model(x)`` on normalized input);
* the exact command lines recorded in ``benchmarks/logs_*/Physics*.log``
  (``--mode {last,cv,cvmix,cv_kf} --recent_steps --mix_weight --tune_kf
  --tune_max_batches``);
* the prior formulas that survived in ``primotraj/models/tsAMD.py``
  (``_motion_prior_bank``), which the paper's prior bank is built from;
* the stdout/BENCHMARK line format of the surviving physics logs, so
  ``benchmarks/scripts/update_formal_registry.py`` can parse new runs.

The numbers already recorded in ``benchmarks/results/`` were produced by the
original file. Re-running this module should reproduce Last / CV /
CVMix closely (they are closed-form), while CV-KF depends on filter details
that the logs pin down only through its tuned ``q`` / ``r`` / ``damping``
values, so treat re-run CV-KF numbers as a re-derivation rather than a bitwise
reproduction. All modes operate on the normalized coordinate state, matching
``kf_coord=normalized`` in the original logs.
"""

import argparse
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


class PhysicsBaseline(nn.Module):
    """Closed-form coordinate extrapolation with no trainable parameters.

    Input:  (B, T, F) normalized window, channels 0 and 1 are lat/lon.
    Output: (B, pred_len, F); channels 0:2 hold the extrapolation and the
            remaining channels repeat the last observed frame.

    Modes:
        last   - constant position: repeat the last observed point.
        cv     - constant velocity from the last one-step difference.
        cvmix  - blend of the last one-step velocity and the mean velocity over
                 the last ``recent_steps`` steps, weighted by ``mix_weight``.
        cv_kf  - constant-velocity Kalman filter over the observed window in
                 normalized coordinates, rolled out with per-step ``damping``.
    """

    def __init__(self, pred_len, mode="cv", recent_steps=3, mix_weight=0.5,
                 kf_q=1e-4, kf_r=1e-2, kf_damping=1.0):
        super().__init__()
        mode = mode.lower()
        if mode not in {"last", "cv", "cvmix", "cv_kf"}:
            raise ValueError(f"Unknown physics mode: {mode}")
        self.pred_len = int(pred_len)
        self.mode = mode
        self.recent_steps = max(1, int(recent_steps))
        self.mix_weight = float(mix_weight)
        self.kf_q = float(kf_q)
        self.kf_r = float(kf_r)
        self.kf_damping = float(kf_damping)

    # ---- velocity estimates -------------------------------------------------

    def _vel_recent(self, pos):
        if pos.size(1) >= 2:
            return pos[:, -1:, :] - pos[:, -2:-1, :]
        return torch.zeros_like(pos[:, -1:, :])

    def _vel_window(self, pos):
        k = min(self.recent_steps, pos.size(1) - 1)
        if k < 1:
            return torch.zeros_like(pos[:, -1:, :])
        return (pos[:, -1:, :] - pos[:, -1 - k: -k, :]) / float(k)

    # ---- CV Kalman filter ---------------------------------------------------

    def _cv_kalman(self, pos):
        """Filter the observed positions, then roll the state forward.

        State s = [px, py, vx, vy] with unit time step; process noise q*I,
        measurement noise r*I, position-only observation model.
        """
        B, T, _ = pos.shape
        device, dtype = pos.device, pos.dtype

        state = torch.zeros(B, 4, device=device, dtype=dtype)
        state[:, 0:2] = pos[:, 0, :]
        if T >= 2:
            state[:, 2:4] = pos[:, 1, :] - pos[:, 0, :]

        cov = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4).clone()
        F = torch.eye(4, device=device, dtype=dtype)
        F[0, 2] = 1.0
        F[1, 3] = 1.0
        F = F.expand(B, 4, 4)
        Q = self.kf_q * torch.eye(4, device=device, dtype=dtype)
        R = self.kf_r * torch.eye(2, device=device, dtype=dtype)
        H = torch.zeros(2, 4, device=device, dtype=dtype)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        Hb = H.expand(B, 2, 4)

        for t in range(1, T):
            state = torch.bmm(F, state.unsqueeze(-1)).squeeze(-1)
            cov = torch.bmm(torch.bmm(F, cov), F.transpose(1, 2)) + Q

            innovation = pos[:, t, :] - torch.bmm(Hb, state.unsqueeze(-1)).squeeze(-1)
            S = torch.bmm(torch.bmm(Hb, cov), Hb.transpose(1, 2)) + R
            K = torch.bmm(torch.bmm(cov, Hb.transpose(1, 2)), torch.linalg.inv(S))
            state = state + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
            cov = cov - torch.bmm(torch.bmm(K, Hb), cov)

        pos_h = state[:, 0:2]
        vel_h = state[:, 2:4]
        outs = []
        for _ in range(self.pred_len):
            vel_h = vel_h * self.kf_damping
            pos_h = pos_h + vel_h
            outs.append(pos_h)
        return torch.stack(outs, dim=1)

    # ---- forward ------------------------------------------------------------

    def forward(self, x):
        pos = x[..., :2]
        last = pos[:, -1:, :]
        steps = torch.arange(
            1, self.pred_len + 1, device=x.device, dtype=x.dtype
        ).view(1, self.pred_len, 1)

        if self.mode == "last":
            latlon = last.expand(-1, self.pred_len, -1)
        elif self.mode == "cv":
            latlon = last + steps * self._vel_recent(pos)
        elif self.mode == "cvmix":
            w = self.mix_weight
            vel = w * self._vel_recent(pos) + (1.0 - w) * self._vel_window(pos)
            latlon = last + steps * vel
        else:
            latlon = self._cv_kalman(pos)

        out = x[:, -1:, :].expand(-1, self.pred_len, -1).clone()
        out[..., :2] = latlon
        return out


def _parse_values(text, cast=float):
    return [cast(v) for v in str(text).replace(",", " ").split() if v]


@torch.no_grad()
def _tune_kf(args, loader, mean, std, device, criterion):
    """Grid-search q / r / damping on the validation split only."""
    best = None
    # get_val() drops the last partial batch, which can empty a small validation
    # split entirely; tune over every validation window instead.
    val_loader = loader._make("val", shuffle=False, drop_last=False)
    for q in _parse_values(args.kf_q_values):
        for r in _parse_values(args.kf_r_values):
            for damping in _parse_values(args.kf_damping_values):
                model = PhysicsBaseline(
                    args.pred_len, mode="cv_kf",
                    recent_steps=args.recent_steps, mix_weight=args.mix_weight,
                    kf_q=q, kf_r=r, kf_damping=damping,
                ).to(device)
                mse_l, mae_l, ade_l, fde_l = [], [], [], []
                for i, (batch_x, batch_y) in enumerate(val_loader):
                    if args.tune_max_batches and i >= args.tune_max_batches:
                        break
                    batch_x = (batch_x.to(device) - mean) / std
                    batch_y = (batch_y.to(device) - mean) / std
                    pred = model(batch_x)
                    from baselines.traj_utils import denorm_latlon, haversine_torch
                    dist = haversine_torch(
                        denorm_latlon(pred, mean, std), denorm_latlon(batch_y, mean, std)
                    )
                    mse_l.append(criterion(pred, batch_y).item())
                    mae_l.append(torch.mean(torch.abs(pred - batch_y)).item())
                    ade_l.append(dist.mean().item())
                    fde_l.append(dist[:, -1].mean().item())
                if not ade_l:
                    continue
                score = sum(ade_l) / len(ade_l)
                cand = (score, q, r, damping,
                        sum(mse_l) / len(mse_l), sum(mae_l) / len(mae_l),
                        score, sum(fde_l) / len(fde_l))
                if best is None or cand[0] < best[0]:
                    best = cand
    return best


def main():
    parser = argparse.ArgumentParser("Physics trajectory baselines")
    parser.add_argument("--data_path", type=str, default="./data/porto_processed.npz")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mode", choices=["last", "cv", "cvmix", "cv_kf"], default="cv")
    parser.add_argument("--recent_steps", type=int, default=3)
    parser.add_argument("--mix_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--bench_warmup", type=int, default=20)
    parser.add_argument("--bench_steps", type=int, default=100)
    # CV-KF hyperparameters; tuned on validation only.
    parser.add_argument("--kf_q", type=float, default=1e-4)
    parser.add_argument("--kf_r", type=float, default=1e-2)
    parser.add_argument("--kf_damping", type=float, default=1.0)
    parser.add_argument("--tune_kf", action="store_true", default=False)
    parser.add_argument("--tune_max_batches", type=int, default=160,
                        help="0 = use the whole validation split")
    parser.add_argument("--kf_q_values", type=str, default="1e-5 1e-4 1e-3 1e-2")
    parser.add_argument("--kf_r_values", type=str, default="1e-4 1e-3 1e-2 1e-1")
    parser.add_argument("--kf_damping_values", type=str, default="1.0 0.95 0.9")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = TrajectoryDataLoader(args.data_path, args.batch_size, args.seq_len, args.pred_len)
    mean = torch.tensor(loader.mean, dtype=torch.float32, device=device)
    std = torch.tensor(loader.std, dtype=torch.float32, device=device)
    criterion = nn.MSELoss()

    model = PhysicsBaseline(
        args.pred_len, mode=args.mode,
        recent_steps=args.recent_steps, mix_weight=args.mix_weight,
        kf_q=args.kf_q, kf_r=args.kf_r, kf_damping=args.kf_damping,
    ).to(device)
    print(f"Parameter Count: {count_parameters(model):,d}")

    if args.mode == "cv_kf" and args.tune_kf:
        best = _tune_kf(args, loader, mean, std, device, criterion)
        if best is None:
            print("WARNING: CV-KF tuning evaluated no validation batch; "
                  "keeping the command-line q / r / damping.")
        if best is not None:
            _, q, r, damping, val_mse, val_mae, val_ade, val_fde = best
            args.kf_q, args.kf_r, args.kf_damping = q, r, damping
            model = PhysicsBaseline(
                args.pred_len, mode="cv_kf",
                recent_steps=args.recent_steps, mix_weight=args.mix_weight,
                kf_q=q, kf_r=r, kf_damping=damping,
            ).to(device)
            print(
                f"Best CV-KF validation setting: q={q:g} r={r:g} damping={damping:g} "
                f"val_mse={val_mse:.6f} val_mae={val_mae:.6f} "
                f"val_ade_m={val_ade:.6f} val_fde_m={val_fde:.6f}"
            )

    print("Final evaluation on test set:")
    test_loader = loader.get_test()
    test_mse, test_mae, test_ade, test_fde = evaluate_model(
        model, test_loader, criterion, mean, std, device, split_name="test"
    )
    infer_ms = benchmark_inference_ms(
        model, test_loader, device,
        warmup=args.bench_warmup, steps=args.bench_steps, mean=mean, std=std,
    )

    print(f"Inference Time: {infer_ms:.3f} ms/iter")
    print(
        "BENCHMARK|"
        f"model=Physics_{args.mode}|seq_len={args.seq_len}|pred_len={args.pred_len}|"
        f"params=0|infer_ms={infer_ms:.6f}|"
        f"test_mse={test_mse:.6f}|test_mae={test_mae:.6f}|"
        f"test_ade_m={test_ade:.6f}|test_fde_m={test_fde:.6f}|"
        f"recent_steps={args.recent_steps}|mix_weight={args.mix_weight:.6f}|"
        f"kf_q={args.kf_q:g}|kf_r={args.kf_r:g}|kf_damping={args.kf_damping:g}|"
        "kf_coord=normalized"
    )


if __name__ == "__main__":
    main()

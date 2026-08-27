# coding=utf-8
"""Check that the two PriMoTraj gate configurations have the expected size.

    python verify_model_config.py

compact  - the earlier compact bank (P1-P5, scalar residual bound), kept so the
           rebuild of the deployed path is checkable against a known-good
           reference: 1,653 parameters at H=6 and 1,731 at H=12.
deployed - the configuration described in the paper: seven priors, gate hidden
           width h_g=64 with K_g=3, and a full-window residual head with a
           learned per-step bound. Expected 11,129 (11.1K) at H=6 and
           11,993 (12.0K) at H=12, of which 2,247 (2.2K) are gate parameters.

Also checks the prior bank itself: on a constant-velocity window the
constant-velocity, smoothed, mixed, constant-acceleration and long-window
priors must all reproduce the future exactly.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "primotraj"))

from models.tsAMD import PriMoTraj  # noqa: E402

BASE = dict(
    n_block=1, dropout=0.1, patch=8, k=3, c=2, alpha=0.5,
    target_slice=slice(0, None), norm=True, layernorm=True,
    use_mdm=True, use_moe=False,
    motion_prior="cvmix", motion_prior_weight=1.0,
    motion_prior_recent_weight=0.5, motion_prior_mode="gate",
    motion_prior_damping=0.8,
)

COMPACT = dict(
    motion_prior_gate_hidden=16, motion_prior_residual_scale=0.02,
    motion_prior_n_priors=5, motion_prior_residual_head="none",
)
DEPLOYED = dict(
    motion_prior_gate_hidden=64, motion_prior_residual_scale=0.0,
    motion_prior_n_priors=7, motion_prior_gate_horizon=3,
    motion_prior_residual_head="full_window",
    motion_prior_residual_hidden=64, motion_prior_residual_init=0.15,
)

EXPECTED = {
    ("compact", 6): 1653, ("compact", 12): 1731,
    ("deployed", 6): 11129, ("deployed", 12): 11993,
}

T, F = 12, 9


def main() -> int:
    ok = True

    for name, cfg in (("compact", COMPACT), ("deployed", DEPLOYED)):
        for H in (6, 12):
            torch.manual_seed(0)
            model = PriMoTraj(input_shape=(T, F), pred_len=H, **BASE, **cfg)
            total = sum(p.numel() for p in model.parameters())
            gate = sum(p.numel() for p in model.motion_gate.parameters())
            residual = 0
            if hasattr(model, "residual_head"):
                residual = (sum(p.numel() for p in model.residual_head.parameters())
                            + model.residual_log_scale.numel())

            with torch.no_grad():
                y, _ = model(torch.randn(4, T, F))
            shape_ok = tuple(y.shape) == (4, H, F)

            expected = EXPECTED[(name, H)]
            match = total == expected
            ok &= match and shape_ok
            print(f"{name:9s} H={H:<3d} params={total:>6,d} (expected {expected:>6,d}) "
                  f"{'OK' if match else 'MISMATCH':8s} gate={gate:>5,d} "
                  f"residual={residual:>5,d} forward={tuple(y.shape)}"
                  f"{'' if shape_ok else '  BAD SHAPE'}")

    torch.manual_seed(0)
    model = PriMoTraj(input_shape=(T, F), pred_len=6, **BASE, **DEPLOYED)
    bank = model._motion_prior_bank(torch.randn(2, T, F), torch.zeros(2, 6, 2))
    bank_ok = tuple(bank.shape) == (2, 7, 6, 2)
    ok &= bank_ok
    print(f"\nprior bank shape: {tuple(bank.shape)} (expected (2, 7, 6, 2))")

    velocity = torch.tensor([0.3, -0.2]).view(1, 1, 2)
    idx = torch.arange(T).view(1, T, 1).float()
    window = torch.zeros(1, T, F)
    window[..., :2] = idx * velocity
    bank = model._motion_prior_bank(window, torch.zeros(1, 6, 2))
    future = window[:, -1:, :2] + torch.arange(1, 7).view(1, 6, 1).float() * velocity
    for i, label in ((1, "P2 constant-velocity"), (2, "P3 smoothed"),
                     (3, "P4 mixed"), (5, "P6 constant-acceleration"),
                     (6, "P7 long-window")):
        err = (bank[:, i] - future).abs().max().item()
        ok &= err < 1e-5
        print(f"  {label:26s} max|err| on a constant-velocity window: {err:.2e}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

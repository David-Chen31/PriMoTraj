import math

import torch
import torch.nn as nn

from models.common import RevIN
from models.common import PM
from models.common import MDM
from models.tsmoe import AMS


class PriMoTraj(nn.Module):
    """Feed-forward gating over deterministic motion priors.

    ``AMD`` remains an alias at the end of this module so legacy checkpoints
    and experiment scripts keep importing successfully.
    """

    PRIOR_NAMES = (
        "P1_constant_position",
        "P2_constant_velocity_recent",
        "P3_smoothed_velocity_2step",
        "P4_mixed_velocity_1step_2step",
        "P5_damped_velocity",
        "P6_constant_acceleration_componentwise_clipped",
        "P7_long_window_velocity_4step",
    )

    def __init__(
        self,
        input_shape,
        pred_len,
        n_block,
        dropout,
        patch,
        k,
        c,
        alpha,
        target_slice,
        norm=True,
        layernorm=True,
        use_mdm=True,
        use_moe=True,
        moe_num_experts=8,
        moe_top_k=2,
        moe_ff_dim=2048,
        tpg_speed_idx=-1,
        tpg_heading_sin_idx=-1,
        tpg_heading_cos_idx=-1,
        tpg_pool_t=8,
        pm_input='x',
        motion_prior='none',
        motion_prior_weight=0.0,
        motion_prior_recent_weight=0.5,
        motion_prior_mode='blend',
        motion_prior_gate_hidden=16,
        motion_prior_residual_scale=0.0,
        motion_prior_damping=0.8,
        motion_prior_n_priors=5,
        motion_prior_gate_horizon=3,
        motion_prior_residual_head='none',
        motion_prior_residual_hidden=64,
        motion_prior_residual_init=0.15,
    ):
        super().__init__()

        self.target_slice = target_slice
        self.norm = norm
        self.use_mdm = use_mdm
        self.use_moe = use_moe
        self.pm_input = pm_input  # 'x' or 'mdm'
        self.motion_prior = motion_prior
        self.motion_prior_weight = float(motion_prior_weight)
        self.motion_prior_recent_weight = float(motion_prior_recent_weight)
        self.motion_prior_mode = motion_prior_mode
        self.motion_prior_residual_scale = float(motion_prior_residual_scale)
        self.motion_prior_damping = float(motion_prior_damping)
        # Deployed configuration knobs (see Sec. "Motion-Prior Gate").
        # n_priors=5 keeps the earlier compact bank (P1-P5); n_priors=7 adds the
        # constant-acceleration and long-window smoothed-velocity priors.
        self.motion_prior_n_priors = int(motion_prior_n_priors)
        if self.motion_prior_n_priors not in (5, 7):
            raise ValueError('motion_prior_n_priors must be 5 or 7')
        self.motion_prior_gate_horizon = max(1, int(motion_prior_gate_horizon))
        self.motion_prior_residual_head = str(motion_prior_residual_head)
        self.register_buffer(
            '_motion_steps',
            torch.arange(1, pred_len + 1, dtype=torch.float32).view(1, pred_len, 1),
            persistent=False,
        )

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=layernorm) if self.use_mdm else None

        self.fc_blocks = nn.ModuleList([PM(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
                                        for _ in range(n_block)])

        self.moe = AMS(
            input_shape,
            pred_len,
            ff_dim=moe_ff_dim,
            dropout=dropout,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
        ) if self.use_moe else None
        self.direct_head = nn.Linear(input_shape[0], pred_len)

        if self.motion_prior_mode == 'gate':
            # Tiny sample-wise router over cheap priors; latency cost is negligible
            # compared with the existing temporal mixer.
            k_g = min(self.motion_prior_gate_horizon, input_shape[0])
            gate_in = k_g * input_shape[-1]
            hidden = max(4, int(motion_prior_gate_hidden))
            self.motion_gate = nn.Sequential(
                nn.Linear(gate_in, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.motion_prior_n_priors),
            )

            if self.motion_prior_residual_head == 'full_window':
                # Independent two-layer MLP over the whole input window,
                # R^{TF} -> R^{r} -> R^{2H}. The output layer is zero-initialised
                # so training starts from pure prior weighting.
                r_hidden = max(4, int(motion_prior_residual_hidden))
                out_layer = nn.Linear(r_hidden, 2 * pred_len)
                nn.init.zeros_(out_layer.weight)
                nn.init.zeros_(out_layer.bias)
                self.residual_head = nn.Sequential(
                    nn.Linear(input_shape[0] * input_shape[-1], r_hidden),
                    nn.GELU(),
                    out_layer,
                )
                # Per-step positive bound lambda_r = softplus(lambda_tilde),
                # initialised so that softplus(lambda_tilde) = residual_init.
                init = max(1e-6, float(motion_prior_residual_init))
                inv_softplus = math.log(math.expm1(init))
                self.residual_log_scale = nn.Parameter(
                    torch.full((pred_len,), inv_softplus, dtype=torch.float32)
                )

        # Learnable scalar to blend MDM residual (only for pm_input='x' path).
        # pm_input='mdm': PM processes MDM output directly — no separate residual.
        if self.use_mdm and not self.use_moe and self.pm_input == 'x':
            self.mdm_gate = nn.Parameter(torch.zeros(1))

    def _motion_prior_latlon(self, raw_x, pred_len):
        """Short-horizon lat/lon prior in the same normalised space as output."""
        last = raw_x[:, -1:, :2]
        if self.motion_prior == 'last' or raw_x.size(1) < 2:
            return last.expand(-1, pred_len, -1)

        steps = self._motion_steps[:, :pred_len]
        if steps.dtype != raw_x.dtype:
            steps = steps.to(dtype=raw_x.dtype)

        vel_recent = raw_x[:, -1:, :2] - raw_x[:, -2:-1, :2]
        if self.motion_prior == 'cv':
            vel = vel_recent
        elif self.motion_prior == 'cvmix':
            if raw_x.size(1) >= 3:
                vel_smooth = 0.5 * (raw_x[:, -1:, :2] - raw_x[:, -3:-2, :2])
            else:
                vel_smooth = vel_recent
            w = self.motion_prior_recent_weight
            vel = w * vel_recent + (1.0 - w) * vel_smooth
        else:
            raise ValueError(f"Unsupported motion_prior: {self.motion_prior}")

        return last + steps * vel

    def _motion_prior_bank(self, raw_x, pred_latlon):
        pred_len = pred_latlon.size(1)
        last = raw_x[:, -1:, :2]
        steps = self._motion_steps[:, :pred_len]
        if steps.dtype != raw_x.dtype:
            steps = steps.to(dtype=raw_x.dtype)

        if raw_x.size(1) >= 2:
            vel_recent = raw_x[:, -1:, :2] - raw_x[:, -2:-1, :2]
        else:
            vel_recent = raw_x.new_zeros(raw_x.size(0), 1, 2)

        if raw_x.size(1) >= 3:
            vel_smooth = 0.5 * (raw_x[:, -1:, :2] - raw_x[:, -3:-2, :2])
        else:
            vel_smooth = vel_recent

        w = self.motion_prior_recent_weight
        vel_cvmix = w * vel_recent + (1.0 - w) * vel_smooth

        damping = max(0.0, min(0.99, self.motion_prior_damping))
        if damping == 0.0:
            damped_steps = torch.ones_like(steps)
        else:
            damped_steps = (1.0 - damping ** steps) / (1.0 - damping)

        priors = [
            last.expand(-1, pred_len, -1),
            last + steps * vel_recent,
            last + steps * vel_smooth,
            last + steps * vel_cvmix,
            last + damped_steps * vel_recent,
        ]

        if self.motion_prior_n_priors == 7:
            # P6: component-wise clipping in dataset-normalized coordinates,
            # after the internal RevIN inversion and before conversion back to
            # latitude/longitude degrees:
            # a_j* = clamp((v_r-v_p)_j, -|v_r,j|, |v_r,j|).
            if raw_x.size(1) >= 3:
                vel_prev = raw_x[:, -2:-1, :2] - raw_x[:, -3:-2, :2]
            else:
                vel_prev = vel_recent
            accel = vel_recent - vel_prev
            bound = vel_recent.abs()
            accel = torch.max(torch.min(accel, bound), -bound)
            priors.append(last + steps * vel_recent
                          + 0.5 * steps * (steps - 1.0) * accel)

            # P7: long-window smoothed velocity v_l = (p_T - p_{T-4}) / 4.
            if raw_x.size(1) >= 5:
                vel_long = (raw_x[:, -1:, :2] - raw_x[:, -5:-4, :2]) / 4.0
            else:
                vel_long = vel_smooth
            priors.append(last + steps * vel_long)

        return torch.stack(priors, dim=1)

    def _apply_motion_prior(self, pred, raw_x):
        if self.motion_prior == 'none' or self.motion_prior_weight <= 0.0:
            return pred
        if pred.size(-1) < 2 or raw_x.size(-1) < 2:
            return pred

        if self.motion_prior_mode == 'gate':
            n_tail = min(self.motion_prior_gate_horizon, raw_x.size(1))
            gate_x = raw_x[:, -n_tail:, :].reshape(raw_x.size(0), -1)
            weights = torch.softmax(self.motion_gate(gate_x), dim=-1)
            priors = self._motion_prior_bank(raw_x, pred[:, :, :2])
            base = (weights[:, :, None, None] * priors).sum(dim=1)
            if self.motion_prior_residual_head == 'full_window':
                # Bounded full-window residual: lambda_r(h) * tanh(R_theta(X)(h)).
                delta = self.residual_head(raw_x.reshape(raw_x.size(0), -1))
                delta = delta.view(raw_x.size(0), pred.size(1), 2)
                lam = nn.functional.softplus(self.residual_log_scale).view(1, -1, 1)
                base = base + lam * torch.tanh(delta)
            elif self.motion_prior_residual_scale > 0.0:
                base = base + self.motion_prior_residual_scale * torch.tanh(pred[:, :, :2] - base)
            pred[:, :, :2] = base
            return pred

        prior = self._motion_prior_latlon(raw_x, pred.size(1))
        if self.motion_prior_mode == 'residual':
            pred[:, :, :2] = prior + self.motion_prior_residual_scale * torch.tanh(pred[:, :, :2] - prior)
            return pred

        w = max(0.0, min(1.0, self.motion_prior_weight))
        if w >= 1.0:
            pred[:, :, :2] = prior
        else:
            pred[:, :, :2] = (1.0 - w) * pred[:, :, :2] + w * prior
        return pred

    def forward(self, x):
        # [batch_size, seq_len, feature_num]
        raw_x = x

        if self.norm:
            x = self.rev_norm(x, 'norm')

        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]

        if self.use_mdm:
            time_embedding = self.pastmixing(x)
        else:
            time_embedding = x

        # enc selection: 'x' preserves spatial precision (gated MDM residual),
        # 'mdm' feeds MDM output into PM directly (no separate residual needed).
        enc = time_embedding if self.pm_input == 'mdm' else x
        for fc_block in self.fc_blocks:
            enc = fc_block(enc)

        if self.use_moe:
            x, moe_loss = self.moe(enc, time_embedding)
        else:
            if self.use_mdm and self.pm_input == 'x':
                # Gated additive residual: sigmoid(gate) starts at 0.5, learns to
                # let MDM info through gradually as training stabilises.
                beta = torch.sigmoid(self.mdm_gate)
                x = self.direct_head(enc + beta * time_embedding)
            else:
                # pm_input='mdm': PM already processed MDM output; direct predict.
                x = self.direct_head(enc)
            moe_loss = x.new_zeros(())

        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)

        x = self._apply_motion_prior(x, raw_x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss


# Backward-compatible import name used by legacy scripts/checkpoints.
AMD = PriMoTraj

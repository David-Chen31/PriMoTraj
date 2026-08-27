import torch
import torch.nn as nn
import math


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x, alpha=10):
        # x [batch_size, seq_len]
        output = torch.zeros_like(x)
        # [batch_size]
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        # [batch_size, num_expert]
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        # Ablation Spare MoE
        # output[mask] = 0
        # [batch_size, seq_len]
        return output

    def forward(self, x):
        # [batch_size, seq_len]

        x = self.gate(x)
        clean_logits = x
        # [batch_size, num_experts]

        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)

        # random order
        # indices = torch.randperm(gates.size(0))
        # shuffled_gates = gates[indices]

        # average
        # value = 1.0 / x.shape[1]
        # gates = torch.full(x.shape, value, device=x.device)

        return gates


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        super(AMS, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len
        self.seq_len = input_shape[0]
        self.ff_dim = ff_dim

        self.gating = TopKGating(input_shape[0], num_experts, top_k)
        self.dropout = nn.Dropout(dropout)

        # Fused experts: one tensorized MLP bank instead of Python loops over experts.
        self.w1 = nn.Parameter(torch.empty(num_experts, self.seq_len, ff_dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, ff_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, ff_dim, pred_len))
        self.b2 = nn.Parameter(torch.empty(num_experts, pred_len))

        self.reset_parameters()
        self.loss_coef = loss_coef
        assert (self.top_k <= self.num_experts)

    def reset_parameters(self):
        # Match linear-layer style initialization for each expert bank.
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in1 = self.seq_len
        bound1 = 1.0 / math.sqrt(fan_in1)
        nn.init.uniform_(self.b1, -bound1, bound1)

        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in2 = self.ff_dim
        bound2 = 1.0 / math.sqrt(fan_in2)
        nn.init.uniform_(self.b2, -bound2, bound2)

    def _expert_forward(self, x_flat):
        # x_flat: [N, seq_len]
        hidden = torch.einsum('ns,esh->neh', x_flat, self.w1) + self.b1
        hidden = torch.nn.functional.gelu(hidden)
        hidden = self.dropout(hidden)
        out = torch.einsum('neh,ehp->nep', hidden, self.w2) + self.b2
        return out

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        # [batch_size, feature_num, seq_len]
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        seq_len = x.shape[2]

        # Flatten feature dimension into batch for one-shot expert/gating computation.
        x_flat = x.reshape(batch_size * feature_num, seq_len)
        time_flat = time_embedding.reshape(batch_size * feature_num, seq_len)

        # [B*F, E]
        gates = self.gating(time_flat)

        # [B*F, E, P]
        expert_outputs = self._expert_forward(x_flat)

        # [B*F, P]
        mixed_output = (gates.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # [B, F, P]
        output = mixed_output.reshape(batch_size, feature_num, self.pred_len)

        # Load-balancing loss: aggregate gate importance per feature over batch.
        importance = gates.reshape(batch_size, feature_num, self.num_experts).sum(dim=0)  # [F, E]
        if self.num_experts == 1:
            loss = gates.new_zeros(())
        else:
            eps = 1e-10
            imp = importance.float()
            cv = imp.var(dim=1, unbiased=True) / (imp.mean(dim=1) ** 2 + eps)
            loss = self.loss_coef * cv.sum().to(gates.dtype)

        return output, loss
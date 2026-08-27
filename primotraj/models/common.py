import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x
        # x [batch_size, feature_num, seq_len]
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        # [batch_size, feature_num, seq_len]
        return sample_x[n - 1]


class PM(nn.Module):
    """Parallel Mixer — fully parallel replacement for DDI.

    Two stages with residual connections:
      1. Depthwise temporal convolution — local pattern extraction
      2. Pointwise feature MLP — cross-channel interaction

    Eliminates DDI's sequential recurrent loop, reducing CUDA kernel
    launches from ~50 to ~10 for large-batch inference.
    """

    def __init__(self, input_shape, dropout=0.2, patch=8, alpha=0.5, layernorm=True):
        super(PM, self).__init__()
        seq_len, feat = input_shape
        self.alpha = alpha

        self.layernorm = layernorm
        if layernorm:
            self.norm = nn.BatchNorm1d(seq_len * feat)

        # Depthwise temporal conv (one filter per feature channel)
        ks = patch - 1 if patch % 2 == 0 else patch   # ensure odd kernel
        self.dw_norm = nn.BatchNorm1d(feat)
        self.dw_conv = nn.Conv1d(feat, feat, kernel_size=ks,
                                 padding=ks // 2, groups=feat)
        self.dw_drop = nn.Dropout(dropout)

        # Pointwise feature mixing
        if alpha > 0.0:
            ff_dim = 2 ** math.ceil(math.log2(feat))
            self.pw_norm = nn.BatchNorm1d(feat)
            self.pw_mix = nn.Sequential(
                nn.Linear(feat, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, feat),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        # x: [B, F, T]
        if self.layernorm:
            x = self.norm(x.reshape(x.size(0), -1)).reshape(x.shape)

        # Stage 1: temporal convolution (fully parallel, single CUDA kernel)
        res = x
        h = self.dw_norm(x)
        h = F.gelu(self.dw_conv(h))
        h = self.dw_drop(h)
        x = res + h

        # Stage 2: feature mixing
        if self.alpha > 0.0:
            res = x
            h = self.pw_norm(x)
            h = h.transpose(1, 2)    # [B, T, F]
            h = self.pw_mix(h)       # [B, T, F]
            h = h.transpose(1, 2)    # [B, F, T]
            x = res + self.alpha * h

        return x


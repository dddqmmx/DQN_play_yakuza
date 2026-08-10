# -*- coding: utf-8 -*-
"""
Continuous Thought Machine 核心组件（vendored）。

改编自 Sakana AI 的官方实现：
    https://github.com/SakanaAI/continuous-thought-machines
    models/modules.py (SynapseUNET, SuperLinear, Squeeze)
    models/utils.py   (compute_decay, compute_normalized_entropy)
    技术报告: https://arxiv.org/abs/2505.05522
License: Apache License 2.0 (Sakana AI)

为什么复制而不是 import：上游 `models/ctm.py` 顶层就 `import huggingface_hub`，
且整个 repo 假定以 `models.*` 为包根；直接依赖会把训练节点绑死在它的目录布局上。
这里只取真正用到的四个东西，并做了两处本地修改：

  1. `SynapseUNET` 的首层由 `nn.LazyLinear` 改成显式 `nn.Linear(in_dims, ...)`。
     Lazy 模块在 `load_state_dict` 之前必须先跑一次 dummy forward 才有形状，
     决策节点是按 checkpoint 冷启动的，显式维度省掉这个坑。
  2. 加了中文注释说明每个张量的形状，便于对着 `ctm_planner.py` 读。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze(nn.Module):
    """nn.Sequential 里没法直接写 lambda，用它来 squeeze 指定维度。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SuperLinear(nn.Module):
    """
    神经元级模型（NLM，论文里的 g_theta_d）：给每个神经元一套**私有**权重。

    输入 (B, N, in_dims) —— N 个神经元，每个带一段长度 in_dims 的历史；
    输出 (B, N, out_dims)。einsum 等价于对每个神经元 n 单独做一次
    `x[:, n, :] @ w1[:, :, n] + b1[:, n, :]`，只是并行做完。

    这是 CTM 与普通 RNN 最本质的区别：时间维度上的非线性由每个神经元自己的
    小模型决定，而不是全网共享一个激活函数。
    """

    def __init__(self, in_dims, out_dims, N, T=1.0, do_norm=False, dropout=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else nn.Identity()
        self.do_norm = do_norm

        # w1: (memory_length, out_dims, d_model)
        self.register_parameter(
            "w1",
            nn.Parameter(
                torch.empty((in_dims, out_dims, N)).uniform_(
                    -1 / math.sqrt(in_dims + out_dims),
                    1 / math.sqrt(in_dims + out_dims),
                ),
                requires_grad=True,
            ),
        )
        # b1: (1, d_model, out_dims)
        self.register_parameter("b1", nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        # 可学习温度
        self.register_parameter("T", nn.Parameter(torch.Tensor([T])))

    def forward(self, x):
        out = self.dropout(x)
        out = self.layernorm(out)
        out = torch.einsum("BDM,MHD->BDH", out, self.w1) + self.b1
        return out.squeeze(-1) / self.T


class SynapseUNET(nn.Module):
    """
    synapse 模型（论文里的 f_theta1）：把 [注意力输出, 上一 tick 的后激活] 混成
    下一 tick 的前激活。U-Net 结构（带跳连）比单层 MLP 效果好，因为它能在多个
    尺度上混合神经元之间的信息。

    `depth` 个宽度点 → `depth-1` 组下采样/上采样块。
    """

    def __init__(self, in_dims, out_dims, depth, minimum_width=16, dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth

        widths = np.linspace(out_dims, minimum_width, depth)

        self.first_projection = nn.Sequential(
            nn.Linear(in_dims, int(widths[0])),
            nn.LayerNorm(int(widths[0])),
            nn.SiLU(),
        )

        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()

        for i in range(len(widths) - 1):
            self.down_projections.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(int(widths[i]), int(widths[i + 1])),
                    nn.LayerNorm(int(widths[i + 1])),
                    nn.SiLU(),
                )
            )
            self.up_projections.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(int(widths[i + 1]), int(widths[i])),
                    nn.LayerNorm(int(widths[i])),
                    nn.SiLU(),
                )
            )
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        outs_down = [self.first_projection(x)]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))

        outs_up = outs_down[-1]
        num_blocks = len(self.up_projections)
        for i in range(num_blocks):
            idx = num_blocks - 1 - i
            out_up = self.up_projections[idx](outs_up)
            outs_up = self.skip_lns[idx](out_up + outs_down[idx])
        return outs_up


def build_synapses(in_dims, d_model, synapse_depth, dropout=0.0):
    """depth==1 时退化成单层 GLU + LayerNorm，否则用 U-Net。"""
    if synapse_depth <= 1:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dims, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model),
        )
    return SynapseUNET(in_dims, d_model, synapse_depth, minimum_width=16, dropout=dropout)


def build_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length,
                              memory_hidden_dims, d_model, dropout=0.0):
    """
    构造 trace_processor：把 (B, d_model, memory_length) 的前激活历史
    压成 (B, d_model) 的后激活。GLU 是上游实测好用的非线性。
    """
    if deep_nlms:
        return nn.Sequential(
            SuperLinear(memory_length, 2 * memory_hidden_dims, d_model,
                        do_norm=do_layernorm_nlm, dropout=dropout),
            nn.GLU(),
            SuperLinear(memory_hidden_dims, 2, d_model,
                        do_norm=do_layernorm_nlm, dropout=dropout),
            nn.GLU(),
            Squeeze(-1),
        )
    return nn.Sequential(
        SuperLinear(memory_length, 2, d_model, do_norm=do_layernorm_nlm, dropout=dropout),
        nn.GLU(),
        Squeeze(-1),
    )


def compute_decay(T, params, clamp_lims=(0, 15)):
    """同步性计算里那条可学习的指数衰减：越久远的 tick 权重越低。"""
    assert isinstance(clamp_lims, tuple) and len(clamp_lims) == 2, "clamp_lims 必须是长度 2 的 tuple"
    indices = torch.arange(T - 1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    return torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))


def compute_normalized_entropy(logits, reduction="mean"):
    """
    最后一维 softmax 的归一化熵，取值 [0, 1]。CTM 用 `1 - 熵` 当 certainty：
    分布越尖锐说明这一 tick "想得越清楚"。
    """
    preds = F.softmax(logits, dim=-1)
    log_preds = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(preds * log_preds, dim=-1)
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32, device=logits.device))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape) > 2 and reduction == "mean":
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)
    return normalized_entropy

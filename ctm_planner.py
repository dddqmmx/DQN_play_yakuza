# -*- coding: utf-8 -*-
"""
CTMPlannerNet：输出**动作序列**的 Continuous Thought Machine。

与旧 ProNet 的区别不只是骨干换了，而是决策的形状变了：

    ProNet:        state -> Q(s, a)                (B, A, Q)      每步一个动作
    CTMPlannerNet: state -> Q_j(s, a), j=0..L-1    (B, L, A, Q)   一次一条计划

`j` 是"从现在起第 j 步"，所以 argmax 出来的 (a_0, a_1, ..., a_{L-1}) 就是模型
自己规划的连招。计划不是承诺：下一次决策会重新规划，可以推翻上一次的主意，
提交多少步由**逐槽位**的决断程度决定（见 `plan_slot_confidence` / `commit_length`）。
`plan_confidence`（整条计划取均值）只作日志用 —— 拿它去定提交长度会让 commit
恒等于 1，整条计划只有第 0 格被执行，原委见 `commit_length` 的注释。

CTM 的四个要素都在 `forward` 的那个 tick 循环里：
  1. 内部递归      —— 同一帧画面上"想" iterations 次
  2. 神经元级模型  —— 每个神经元用私有 MLP 处理自己的前激活历史
  3. 同步性表征    —— 神经元两两同步度既用来查询画面（action），也用来出计划（out）
  4. certainty     —— 每个 tick 自评确定度，用来挑"想清楚了"的那个 tick
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from config import CTM_CONFIG
from ctm_components import (
    build_neuron_level_models,
    build_synapses,
    compute_normalized_entropy,
)
from network_components import NoisyLinear, ResidualBlock


class CTMPlannerNet(nn.Module):
    """
    forward(x, boss_health, self_health)
        -> q_dist     (B, L, A, Q, T)
           certainties (B, 2, T)   # [归一化熵, 1-归一化熵]，与上游 CTM 的约定一致
    """

    def __init__(self, num_actions: int, input_channels: int = 4, config: dict = None):
        super().__init__()
        cfg = dict(CTM_CONFIG)
        cfg.update(config or {})
        self.config = cfg

        self.num_actions = num_actions
        self.plan_length = int(cfg["plan_length"])
        self.num_quantiles = int(cfg["num_quantiles"])
        self.iterations = int(cfg["iterations"])
        self.d_model = int(cfg["d_model"])
        self.d_input = int(cfg["d_input"])
        self.memory_length = int(cfg["memory_length"])
        self.n_synch_out = int(cfg["n_synch_out"])
        self.n_synch_action = int(cfg["n_synch_action"])
        self.neuron_select_type = cfg["neuron_select_type"]
        self.token_grid = int(cfg["token_grid"])
        self.use_noisy = bool(cfg["use_noisy"])
        dropout = float(cfg.get("dropout", 0.0))

        self._verify_args()

        # ---------------------------------------------------------- 视觉 token 化
        # (B, 4, 160, 160) -> (B, 1, 4, 160, 160) -> Conv3D 抽帧间时空演变
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.input_channels = input_channels
        self.layer1 = ResidualBlock(32 * input_channels, 64, stride=2, config=cfg)
        self.layer2 = ResidualBlock(64, 128, stride=2, config=cfg)
        # 不管输入分辨率是多少，都归一到 token_grid x token_grid 个 token
        self.token_pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))

        n_visual_tokens = self.token_grid * self.token_grid
        self.kv_proj = nn.Sequential(nn.Linear(128, self.d_input), nn.LayerNorm(self.d_input))
        self.positional_embedding = nn.Parameter(torch.zeros(1, n_visual_tokens, self.d_input))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        # 血量单独做一个 token，和画面 token 一起进注意力，而不是最后才拼上去
        self.health_token = nn.Sequential(nn.Linear(2, self.d_input), nn.LayerNorm(self.d_input))

        # ------------------------------------------------------------ CTM 核心
        self.q_proj = nn.Linear(self._synch_size(self.n_synch_action), self.d_input)
        self.attention = nn.MultiheadAttention(
            self.d_input, int(cfg["heads"]), dropout=dropout, batch_first=True
        )
        self.synapses = build_synapses(
            in_dims=self.d_input + self.d_model,
            d_model=self.d_model,
            synapse_depth=int(cfg["synapse_depth"]),
            dropout=dropout,
        )
        self.trace_processor = build_neuron_level_models(
            deep_nlms=bool(cfg["deep_nlms"]),
            do_layernorm_nlm=bool(cfg["do_layernorm_nlm"]),
            memory_length=self.memory_length,
            memory_hidden_dims=int(cfg["memory_hidden_dims"]),
            d_model=self.d_model,
            dropout=dropout,
        )

        # 起始状态本身是学出来的（"还没看画面时脑子里是什么样"）
        self.register_parameter(
            "start_activated_state",
            nn.Parameter(
                torch.zeros(self.d_model).uniform_(
                    -math.sqrt(1 / self.d_model), math.sqrt(1 / self.d_model)
                )
            ),
        )
        self.register_parameter(
            "start_trace",
            nn.Parameter(
                torch.zeros(self.d_model, self.memory_length).uniform_(
                    -math.sqrt(1 / (self.d_model + self.memory_length)),
                    math.sqrt(1 / (self.d_model + self.memory_length)),
                )
            ),
        )

        # ---------------------------------------------------------- 同步性参数
        self.synch_size_action = self._synch_size(self.n_synch_action)
        self.synch_size_out = self._synch_size(self.n_synch_out)
        self._set_synch_parameters("action", self.n_synch_action, int(cfg["n_random_pairing_self"]))
        self._set_synch_parameters("out", self.n_synch_out, int(cfg["n_random_pairing_self"]))

        # ------------------------------------------------------------ 计划头
        # 每个槽位一套 dueling：value 管"这一步大概值多少"，advantage 管"选哪个动作"
        #
        # 注意 value/advantage 用**普通 Linear**，探索噪声单独走 explore_head：
        # 把 NoisyLinear 直接铺在 (A × quantiles) 上是无效的 —— 动作排序看的是
        # `q.mean(-1)`，而各分位数上的噪声互相独立，一平均就按 sqrt(num_quantiles)
        # 衰减掉。实测衰减 4.7x（sqrt(21)=4.6），噪声 std 只剩 0.011，而最优/次优
        # 动作的 Q 差距是 0.213 —— 差 19 倍，探索等于没有，策略必然锁死在一个动作上。
        # explore_head 只输出 (L, A)，广播到所有分位数，因此不会被平均抵消。
        self.value_head = nn.Linear(self.synch_size_out, self.plan_length * self.num_quantiles)
        self.advantage_head = nn.Linear(
            self.synch_size_out, self.plan_length * num_actions * self.num_quantiles
        )
        self.explore_head = (
            NoisyLinear(
                self.synch_size_out,
                self.plan_length * num_actions,
                std_init=float(cfg.get("noisy_std", 0.5)),
            )
            if self.use_noisy
            else None
        )

    # ------------------------------------------------------------------ 校验
    def _verify_args(self):
        assert self.neuron_select_type in ("first-last", "random", "random-pairing"), \
            f"未知的 neuron_select_type: {self.neuron_select_type}"
        if self.neuron_select_type == "first-last":
            assert self.d_model >= self.n_synch_out + self.n_synch_action, \
                "first-last 需要 d_model >= n_synch_out + n_synch_action"
        else:
            assert self.d_model >= max(self.n_synch_out, self.n_synch_action), \
                "d_model 必须不小于 n_synch_out / n_synch_action"

    def _synch_size(self, n_synch: int) -> int:
        if self.neuron_select_type == "random-pairing":
            return n_synch
        return (n_synch * (n_synch + 1)) // 2

    def _set_synch_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int):
        d_model = self.d_model
        if self.neuron_select_type == "first-last":
            if synch_type == "out":
                left = right = torch.arange(0, n_synch)
            else:
                left = right = torch.arange(d_model - n_synch, d_model)
        elif self.neuron_select_type == "random":
            left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
        else:  # random-pairing
            assert n_synch > n_random_pairing_self, \
                f"random-pairing 需要 n_synch > n_random_pairing_self（{n_synch} vs {n_random_pairing_self}）"
            left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            right = torch.cat((
                left[:n_random_pairing_self],
                torch.from_numpy(
                    np.random.choice(np.arange(d_model), size=n_synch - n_random_pairing_self)
                ),
            ))
        self.register_buffer(f"{synch_type}_neuron_indices_left", left.long())
        self.register_buffer(f"{synch_type}_neuron_indices_right", right.long())
        self.register_parameter(
            f"decay_params_{synch_type}",
            nn.Parameter(torch.zeros(self._synch_size(n_synch)), requires_grad=True),
        )

    # -------------------------------------------------------------- CTM 内部
    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """
        同步性 = 一对神经元后激活时间序列的点积。因为是线性递推，不需要真的存整段
        历史，用 (decay_alpha, decay_beta) 两个累加量就能在每个 tick 上增量更新。
        """
        if synch_type == "action":
            n_synch = self.n_synch_action
            idx_left = self.action_neuron_indices_left
            idx_right = self.action_neuron_indices_right
        else:
            n_synch = self.n_synch_out
            idx_left = self.out_neuron_indices_left
            idx_right = self.out_neuron_indices_right

        if self.neuron_select_type == "random-pairing":
            pairwise_product = activated_state[:, idx_left] * activated_state[:, idx_right]
        else:
            if self.neuron_select_type == "first-last":
                if synch_type == "action":
                    selected_left = selected_right = activated_state[:, -n_synch:]
                else:
                    selected_left = selected_right = activated_state[:, :n_synch]
            else:
                selected_left = activated_state[:, idx_left]
                selected_right = activated_state[:, idx_right]
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            i, j = torch.triu_indices(n_synch, n_synch, device=activated_state.device)
            pairwise_product = outer[:, i, j]

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        return decay_alpha / torch.sqrt(decay_beta), decay_alpha, decay_beta

    def compute_features(self, x, boss_health, self_health):
        """画面 + 血量 -> 注意力的 key/value token 序列 (B, N+1, d_input)。"""
        # uint8 存进 replay buffer 省 4 倍内存，归一化放在这里做
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        b = x.size(0)
        feat_3d = self.conv3d(x.unsqueeze(1))
        feat_2d = feat_3d.reshape(b, 32 * self.input_channels, feat_3d.size(3), feat_3d.size(4))
        feat = self.token_pool(self.layer2(self.layer1(feat_2d)))  # (B, 128, G, G)

        tokens = feat.flatten(2).transpose(1, 2)                   # (B, G*G, 128)
        tokens = self.kv_proj(tokens) + self.positional_embedding
        health = self.health_token(
            torch.cat([boss_health, self_health], dim=1).float()
        ).unsqueeze(1)                                             # (B, 1, d_input)
        return torch.cat([tokens, health], dim=1)

    def plan_head(self, synchronisation_out):
        """同步性表征 -> 整条计划的分位数 Q 值 (B, L, A, Q)。"""
        b = synchronisation_out.size(0)
        value = self.value_head(synchronisation_out).view(b, self.plan_length, 1, self.num_quantiles)
        advantage = self.advantage_head(synchronisation_out).view(
            b, self.plan_length, self.num_actions, self.num_quantiles
        )
        q = value + advantage - advantage.mean(dim=2, keepdim=True)
        if self.explore_head is not None:
            # (B, L, A) -> 广播到所有分位数：整体平移某个动作的分布，
            # 这样 q.mean(-1) 拿到的是完整噪声而不是被 sqrt(Q) 稀释过的
            q = q + self.explore_head(synchronisation_out).view(
                b, self.plan_length, self.num_actions, 1
            )
        return q

    @staticmethod
    def compute_certainty(q_dist):
        """
        (B, L, A, Q) -> (B, 2)，与 CTM 的 [熵, 1-熵] 约定一致。

        **只用来在同一次前向的各个 tick 之间做比较**（挑 tick、以及 loss 里的
        tick 聚合）。同一次前向内 Q 的尺度是一致的，所以比较有意义。
        它**不能**跨状态/跨训练阶段比较：熵由 Q 的绝对尺度主导，而 Q 的尺度是
        奖励量纲、会随训练不断变大——实测 σ=2.0 的纯随机 Q 能得到 0.42 的
        "certainty"，而 σ=0.3 下真正决断的 Q 只有 0.03。提交长度因此不能用它，
        见 `plan_slot_confidence`。
        """
        entropy = compute_normalized_entropy(q_dist.mean(-1), reduction="mean")
        return torch.stack([entropy, 1.0 - entropy], dim=-1)

    @staticmethod
    def plan_slot_confidence(q_dist):
        """
        (B, L, A, Q) -> (B, L)：**逐槽位**的决断程度。提交几步就看它。

        用**相对动作间隔** = (最优 Q - 次优 Q) / (最优 Q - 最差 Q)。它衡量的是
        "最优动作比替代方案好多少"，对 Q 的绝对尺度免疫，所以固定阈值在整个训练期
        都成立。单个槽位、13 个动作下的实测标定：毫无偏好 ≈0.145，最优高出 2σ
        ≈0.199，3σ ≈0.298，10σ ≈0.719。

        **不要在这里取均值再判阈值**（原来的做法），有两个致命后果：
          1. 槽位 3 决断不决断，和"能不能安全提交 2 步"毫无关系，取均值却让它一票
             同权。而越靠后的槽位只由自举一致性训练、天生更平滑：6145 步的
             ctm_planner_small.pth 在 378 帧真实画面上实测逐槽位均值
             0.287 / 0.116 / 0.111 / 0.083 —— 后三格甚至低于"毫无偏好"的 0.145，
             于是恰恰是最没学好的槽位在否决连招。
          2. 取均值会把动态范围压没：无偏好时单槽位 p99 能到 0.467，4 槽取均值后
             p99.9 只剩 0.347，而阈值是按单槽位尺度标的。
        """
        q = q_dist.mean(-1)                                   # (B, L, A)
        top2 = q.topk(2, dim=-1).values
        gap = top2[..., 0] - top2[..., 1]
        spread = q.max(dim=-1).values - q.min(dim=-1).values
        return gap / (spread + 1e-6)                          # (B, L)

    @staticmethod
    def plan_confidence(q_dist):
        """
        (B, L, A, Q) -> (B,)：整条计划的平均决断程度，**只用于日志/监控**
        （tensorboard 的 `Plan/Confidence`）。

        提交长度不要用它 —— 用 `plan_slot_confidence` + `commit_length`，
        原因见前者的注释。
        """
        return CTMPlannerNet.plan_slot_confidence(q_dist).mean(dim=-1)

    def forward(self, x, boss_health, self_health, track=False):
        b = x.size(0)
        device = x.device if torch.is_tensor(x) else self.start_trace.device

        kv = self.compute_features(x, boss_health, self_health)

        state_trace = self.start_trace.unsqueeze(0).expand(b, -1, -1)          # (B, D, M)
        activated_state = self.start_activated_state.unsqueeze(0).expand(b, -1)  # (B, D)

        q_dist_all = torch.empty(
            b, self.plan_length, self.num_actions, self.num_quantiles, self.iterations,
            device=device, dtype=torch.float32,
        )
        certainties = torch.empty(b, 2, self.iterations, device=device, dtype=torch.float32)

        decay_alpha_action = decay_beta_action = None
        # 上游的修复：不 clamp 的话 exp(-decay) 会跑飞
        self.decay_params_action.data.clamp_(0, 15)
        self.decay_params_out.data.clamp_(0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(b, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(b, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type="out"
        )

        attention_tracking = []
        for tick in range(self.iterations):
            # 1) 用 action 同步性去"看"画面
            synch_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type="action"
            )
            q = self.q_proj(synch_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=track
            )
            attn_out = attn_out.squeeze(1)

            # 2) synapse 混合信息 -> 前激活；写进历史
            state = self.synapses(torch.cat((attn_out, activated_state), dim=-1))
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # 3) 每个神经元用自己的 NLM 把历史压成后激活
            activated_state = self.trace_processor(state_trace)

            # 4) 用 out 同步性出这一 tick 的计划
            synch_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type="out"
            )
            q_dist = self.plan_head(synch_out)
            q_dist_all[..., tick] = q_dist
            certainties[..., tick] = self.compute_certainty(q_dist)

            if track:
                attention_tracking.append(attn_weights.detach().cpu().numpy())

        if track:
            return q_dist_all, certainties, np.array(attention_tracking)
        return q_dist_all, certainties

    # -------------------------------------------------------------- 计划选取
    @staticmethod
    def pick_most_certain(q_dist_all, certainties):
        """
        按 CTM 的做法挑"自认为想清楚了"的那个 tick。
        (B,L,A,Q,T) + (B,2,T) -> (B,L,A,Q), (B,), (B,)
        """
        tick = certainties[:, 1].argmax(dim=-1)                       # (B,)
        idx = tick.view(-1, 1, 1, 1, 1).expand(
            -1, q_dist_all.size(1), q_dist_all.size(2), q_dist_all.size(3), 1
        )
        picked = q_dist_all.gather(-1, idx).squeeze(-1)               # (B,L,A,Q)
        certainty = certainties[:, 1].gather(-1, tick.unsqueeze(-1)).squeeze(-1)
        return picked, certainty, tick

    def commit_length(self, slot_confidence):
        """
        逐槽位决断程度 (B, L) -> 提交几步 (B,)。

        规则：从槽位 0 往后数，**数到第一个不够决断的槽位为止**。也就是
        "对下一步有把握就接着打，哪一步没底就停下来重新看画面"。至少提交 1 步，
        否则一帧都推进不了。

        原来的做法是先把 L 个槽位的决断度平均成一个数，再线性映射到 [1, L]：

            k = round(1 + clamp((conf - lo) / (hi - lo), 0, 1) * (L - 1))

        三个毛病叠起来让 commit **恒等于 1**，整条计划只有第 0 格被执行过
        （1600 步训练日志里 32/32 条 Step 行都是"提交1 均提交1.0 中断0"；
        6145 步的 checkpoint 在 378 帧真实画面上重放，378/378 也都只提交 1 步）：

          1. 平均把最没学好的后段槽位与槽位 0 等同看待（见 plan_slot_confidence）；
          2. 阈值按单槽位尺度标定却作用在均值上，两者动态范围不匹配；
          3. round() 让标称的 lo 形同虚设 —— L=4 时要 conf ≥ lo + (hi-lo)/6 = 0.267
             才够 2 步；而 hi=0.60 对应单槽位约 7σ 的间隔，训练里根本到不了，
             所以 commit=L 是死代码。

        现在阈值直接作用在它被标定的那个尺度上，也没有 round 带来的死区。
        """
        if slot_confidence.dim() != 2:
            raise ValueError(
                f"commit_length 要 (B, L) 的逐槽位决断度，收到 {tuple(slot_confidence.shape)}；"
                "整条计划的均值（plan_confidence）只能用于日志"
            )
        decisive = slot_confidence >= float(self.config["commit_confidence_lo"])
        # 前缀长度 = 从左起连续 True 的个数：cumprod 一碰到 False 就永久归零
        k = decisive.long().cumprod(dim=-1).sum(dim=-1)
        return k.clamp(1, self.plan_length).long()

    def reset_noise(self):
        if self.explore_head is not None:
            self.explore_head.reset_noise()

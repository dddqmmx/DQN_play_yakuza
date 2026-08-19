# -*- coding: utf-8 -*-
"""
CTMPlannerAgent：训练 `CTMPlannerNet` 的智能体。

一次决策产出一条长度 L 的计划，但只有**槽位 0** 有真正的环境反馈可用（它是实际
执行并拿到 reward 的那一步）。槽位 1..L-1 靠"自举一致性"学：

    Q_0(s,·)  <-  r + γ^n · max_a Q_0^target(s',·)        标准 QR-DQN（double + n-step）
    Q_j(s,·)  <-  Q_{j-1}^target(s',·)                     计划自举，j = 1..L-1

第二条读作："我这一帧计划的第 j 件事，应该等于我下一帧计划的第 j-1 件事。"
一致性链的末端是槽位 0，而槽位 0 是有奖励信号的，所以整条计划都被价值锚住，
不会退化成自说自话。同时它天然给出"改主意"的度量：计划漂移越小，说明模型越
说到做到（见 decision_node 的 Plan/Drift）。

两个 loss 都按 CTM 的做法跨内部 tick 聚合：取"损失最小的那次思考"和"自认为最
确定的那次思考"各一半。这一项是让 certainty 变得可信的关键 —— 没有它，
`pick_most_certain` 挑出来的 tick 就只是个装饰。
（注意 certainty 只用于挑 tick；提交几步看的是逐槽位的
`CTMPlannerNet.plan_slot_confidence`，不是它取均值后的 `plan_confidence`。）
"""
from __future__ import annotations

import copy
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from config import CTM_CONFIG, DEVICE, FILE_PATHS, TRAINING_CONFIG
from ctm_planner import CTMPlannerNet
from replay_buffer import PlanRun, PrioritizedReplayBuffer, Transition, UniformReplayBuffer

ARCH_NAME = "ctm"


class CTMPlannerAgent:
    """接口刻意与 `dqn_agent.DQNAgent` 对齐，decision_node 只需一层工厂分支。"""

    use_recurrent = False  # CTM 的"记忆"在内部 tick 里，不跨环境步携带隐状态

    def __init__(self, num_actions, network_config=None, checkpoint_file=None, log_dir=None):
        cfg = dict(CTM_CONFIG)
        cfg.update(network_config or {})
        self.network_config = cfg

        self.num_actions = num_actions
        self.plan_length = int(cfg["plan_length"])
        self.num_quantiles = int(cfg["num_quantiles"])
        self.checkpoint_file = checkpoint_file or FILE_PATHS["ctm_checkpoint"]

        self.policy_net = CTMPlannerNet(num_actions, config=cfg).to(DEVICE)
        self.target_net = CTMPlannerNet(num_actions, config=cfg).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000)
        self.scaler = torch.amp.GradScaler(
            device=DEVICE.type if DEVICE.type != "cpu" else "cpu",
            enabled=(DEVICE.type != "cpu"),
        )

        # 两个 buffer：PER 存 n-step（喂槽位 0 的 TD），均匀 buffer 存 1 步（喂计划一致性）
        self.memory = PrioritizedReplayBuffer(TRAINING_CONFIG["memory_capacity"])
        self.plan_memory = UniformReplayBuffer(int(cfg["plan_memory_capacity"]))
        # 第三个 buffer：把"一次 commit 里连续执行掉的那一串"整条存下来，
        # 用来给槽位 j 安一个**真实回报**的锚（见 _flush_plan_run / update_model）
        self.plan_run_memory = UniformReplayBuffer(int(cfg.get("plan_run_capacity", 2048)))
        self.plan_run_batch_size = int(cfg.get("plan_run_batch_size", 8))
        self.plan_return_weight = float(cfg.get("plan_return_weight", 0.5))
        self._run_pending = []

        self.batch_size = int(cfg.get("batch_size", TRAINING_CONFIG["batch_size"]))
        self.plan_batch_size = int(cfg["plan_batch_size"])
        self.gamma = TRAINING_CONFIG["gamma"]
        self.n_step = TRAINING_CONFIG.get("n_step", 3)
        self.n_step_buffer = deque(maxlen=self.n_step)
        # 收到过多少条环境样本。decision_node 用它把训练速率压到数据速率附近
        # （replay ratio），否则训练线程会占满 GPU 把决策拖慢。
        self.transitions_seen = 0

        self.plan_loss_weight = float(cfg["plan_loss_weight"])
        self.plan_curriculum_lookahead = int(cfg["plan_curriculum_lookahead"])
        self.explore_epsilon = float(cfg.get("explore_epsilon", 0.0))

        # QR-DQN 的固定分位数中点 [1/(2N), 3/(2N), ..., (2N-1)/(2N)]
        self.quantiles = (
            torch.linspace(0.0, 1.0, self.num_quantiles + 1, device=DEVICE)[1:]
            - 0.5 / self.num_quantiles
        )

        self.tau = TRAINING_CONFIG.get("tau", 0.005)
        self.reward_scale = TRAINING_CONFIG.get("reward_scale", 0.1)
        self.epsilon = 0.0  # Noisy Nets 负责探索；保留字段是为了兼容旧的日志/开关
        self.steps = 0
        self.writer = SummaryWriter(log_dir or FILE_PATHS["tensorboard_logs"])

        self.net_lock = threading.Lock()
        self.save_queue = queue.Queue(maxsize=1)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

        # 给日志用的滑动统计
        self._recent_commit = deque(maxlen=200)
        self._recent_confidence = deque(maxlen=200)
        self.last_losses = {"td": 0.0, "plan": 0.0, "plan_return": 0.0}

    # ------------------------------------------------------------------ 决策
    def select_plan(self, state, boss_health, self_health):
        """
        返回 (plan: list[int], commit: int, confidence: float)。

        plan 是"从现在起打算做的 L 个动作"，commit 是这次真正提交给游戏的步数。
        certainty（CTM 的熵）只用于挑 tick；提交长度看的是**逐槽位**的
        `plan_slot_confidence`，原因见 CTMPlannerNet 里那两个方法的注释。
        """
        with torch.no_grad():
            state_t = self._state_tensor(np.asarray(state)[None, ...])
            boss_t = torch.as_tensor([[boss_health]], dtype=torch.float32, device=DEVICE)
            self_t = torch.as_tensor([[self_health]], dtype=torch.float32, device=DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type if DEVICE.type != "cpu" else "cpu",
                dtype=torch.float16,
                enabled=(DEVICE.type != "cpu"),
            ):
                with self.net_lock:
                    q_dist_all, certainties = self.policy_net(state_t, boss_t, self_t)

            picked, _, _ = CTMPlannerNet.pick_most_certain(q_dist_all, certainties)
            plan = picked.mean(-1).argmax(-1)[0].tolist()            # (L,)
            slot_conf = CTMPlannerNet.plan_slot_confidence(picked)   # (1, L)
            commit = int(self.policy_net.commit_length(slot_conf)[0].item())
            # 报给日志的是整条计划的平均决断度；提交长度用的是逐槽位那个
            confidence = float(slot_conf.mean(dim=-1)[0].item())

        # ε 兜底：NoisyNets 的 sigma 会随训练自己收缩，探索可能悄悄归零。
        # 逐槽位替换成随机动作，保证策略永远不会彻底锁死在一个招式上。
        if self.explore_epsilon > 0:
            first_random = None
            for i in range(len(plan)):
                if random.random() < self.explore_epsilon:
                    plan[i] = random.randrange(self.num_actions)
                    if first_random is None:
                        first_random = i
                else:
                    plan[i] = int(plan[i])
            # 被 ε 改写过的槽位，它的决断度已经不作数了 —— 那一步照样执行
            # （探索的意义就在于执行），但**不能**再拿它后面的槽位继续开环连招，
            # 所以提交长度截到这一步为止，下一帧重新看画面再规划。
            # commit 恒为 1 的年代这个交互不存在，改成真会连招后才需要管。
            if first_random is not None:
                commit = min(commit, first_random + 1)

        self._recent_commit.append(commit)
        self._recent_confidence.append(confidence)
        return [int(a) for a in plan], commit, confidence

    def select_action(self, state, boss_health, self_health, hidden=None):
        """兼容旧接口：只取计划的第一步。"""
        plan, _, _ = self.select_plan(state, boss_health, self_health)
        return plan[0]

    # ------------------------------------------------------------------ 存样本
    def store_transition(self, state, boss_health, self_health, action,
                         reward, next_state, next_boss_health, next_self_health, done,
                         plan_slot=0):
        transition = Transition(
            self._as_state(state), boss_health, self_health, action, reward,
            self._as_state(next_state), next_boss_health, next_self_health, done,
        )
        self.transitions_seen += 1

        # 只有"计划第 0 格"的 1 步样本能进计划 buffer。槽位一致性的目标是
        #     slot_j(s) ≈ slot_{j-1}^tgt(s')
        # 这个恒等式要求 s' 是**执行槽位 0** 到达的下一帧。commit > 1 时，中段那些
        # 样本的 s' 是执行槽位 j 到达的，喂进去等于拿错位的目标去教计划头。
        # commit 恒为 1 的年代每条样本都来自槽位 0，所以这个字段以前只当诊断信息传；
        # 现在计划真的会被连着执行，就必须筛。
        # （n-step 的 PER buffer 不用筛：那边是 Q-learning 的 TD，本来就是 off-policy，
        #   任何动作的样本都合法。）
        if plan_slot == 0:
            self.plan_memory.add(transition)

        self._accumulate_plan_run(transition, plan_slot)

        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) == self.n_step:
            reward_sum = 0.0
            last = self.n_step_buffer[0]
            for i, t in enumerate(self.n_step_buffer):
                reward_sum += t.reward * (self.gamma ** i)
                last = t
                if t.done:
                    break
            head = self.n_step_buffer[0]
            self.memory.add(Transition(
                head.state, head.boss_health, head.self_health, head.action,
                reward_sum, last.next_state, last.next_boss_health,
                last.next_self_health, last.done,
            ))

    def _accumulate_plan_run(self, transition, plan_slot):
        """
        把"一次 commit 里连续执行掉的那几步"拼回一条 run。

        客户端每步都带上 `plan_slot`（这个动作来自计划的第几格），所以 slot 归 0
        就是一条新计划开始。对不上号（乱序、丢包、中途切换计划）时直接把手里这条
        丢掉 —— 宁可少喂，也不能拼错：拼错等于给槽位 j 安上别的槽位的回报。
        """
        if plan_slot == 0:
            self._flush_plan_run()
            self._run_pending = [transition]
        elif plan_slot == len(self._run_pending):
            self._run_pending.append(transition)
        else:
            self._run_pending = []
            return
        if transition.done or len(self._run_pending) >= self.plan_length:
            self._flush_plan_run()

    def _flush_plan_run(self):
        """把攒着的那条 run 收进 buffer。只连着走了 1 步的没有意义，丢掉。"""
        run, self._run_pending = self._run_pending, []
        # 长度 1 意味着 commit==1（或计划刚开头就被打断），此时没有任何
        # "槽位 j>0 真的被执行过"的证据可用，锚不了
        if len(run) < 2:
            return
        head, tail = run[0], run[-1]
        self.plan_run_memory.add(PlanRun(
            head.state, head.boss_health, head.self_health,
            tuple(int(t.action) for t in run),
            tuple(float(t.reward) for t in run),
            tail.next_state, tail.next_boss_health, tail.next_self_health, tail.done,
        ))

    @staticmethod
    def _as_state(state):
        arr = np.asarray(state)
        # 观测端已经是 uint8；万一拿到 float 就地转回去，省 4 倍 buffer 内存
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
        return arr.copy()

    @staticmethod
    def _state_tensor(batch_states):
        return torch.as_tensor(np.asarray(batch_states), dtype=torch.uint8, device=DEVICE)

    # ------------------------------------------------------------------ 训练
    def _quantile_huber(self, current, target):
        """
        current (N, Q) 预测分位数，target (N, Q) 目标分位数 -> (N,) 每样本 loss。

        权重 |τ_i - 1{u<0}| 里的 τ_i 必须索引**预测**那一维（下面的 dim=1），
        索引到目标维上是错的——那样等于把分位数回归退化成对称 Huber。
        """
        td = target.unsqueeze(1) - current.unsqueeze(2)              # (N, Q_cur, Q_tgt)
        huber = F.smooth_l1_loss(
            current.unsqueeze(2).expand_as(td), target.unsqueeze(1).expand_as(td),
            reduction="none",
        )
        weight = torch.abs(self.quantiles.view(1, -1, 1) - (td.detach() < 0).float())
        return (weight * huber).mean(dim=2).sum(dim=1)

    @staticmethod
    def _ctm_tick_aggregate(losses_per_tick, certainties):
        """
        CTM 的 tick 聚合（上游 utils/losses.py）：把"损失最小的那次思考"和
        "自认为最确定的那次思考"各取一半。前者提供学习信号，后者逼着 certainty
        向真正好的 tick 靠拢——不这么做的话 `pick_most_certain` 就只是个装饰。
        """
        idx_best = losses_per_tick.argmin(dim=1)
        idx_certain = certainties[:, 1].argmax(dim=-1)
        rows = torch.arange(losses_per_tick.size(0), device=losses_per_tick.device)
        return (losses_per_tick[rows, idx_best] + losses_per_tick[rows, idx_certain]) / 2.0

    def _plan_slot_mask(self, policy_plan_argmax, target_plan_argmax):
        """
        课程掩码（借 CTM maze 任务的 auto-curriculum）：只训练"已经对齐的前缀 +
        lookahead 步"。训练早期没必要把容量砸在第 6 步上——前几步都还没稳。
        返回 (B, L-1) 的 0/1 掩码。
        """
        lookahead = self.plan_curriculum_lookahead
        n_slots = self.plan_length - 1
        if lookahead < 0 or n_slots <= 0:
            return None
        agree = (policy_plan_argmax[:, 1:] == target_plan_argmax[:, :-1]).long()  # (B, L-1)
        steps = torch.arange(1, n_slots + 1, device=agree.device).unsqueeze(0)
        prefix_len = (agree.cumsum(dim=1) == steps).sum(dim=1)                    # 连续对齐的前缀长度
        upto = torch.clamp(prefix_len + lookahead, max=n_slots)
        cols = torch.arange(n_slots, device=agree.device).unsqueeze(0)
        return (cols < upto.unsqueeze(1)).float()

    def _plan_run_targets(self, runs):
        """
        把一批 run 摊成逐槽位的 (折扣回报, 实际动作, 到 tail 的步数, 掩码)。

        槽位 j 的回报锚 = Σ_{i>=j} γ^(i-j)·r_i + γ^(k-j)·V^tgt(tail)，
        也就是"从计划的第 j 格起，实际走完这条 run 拿到的东西"。k 是这条 run
        真正连续执行掉的步数。

        掩码只在 1 <= j < k 上为 1：
          - j == 0 不用这里锚，它有 PER 的 n-step TD（现成的、带优先级的）；
          - j >= k 那几格根本没被执行过，没有真实回报可言，仍然交给一致性 loss。
        """
        L = self.plan_length
        n = len(runs)
        ret = torch.zeros(n, L, dtype=torch.float32)
        act = torch.zeros(n, L, dtype=torch.long)
        steps = torch.ones(n, L, dtype=torch.float32)
        mask = torch.zeros(n, L, dtype=torch.float32)
        for b, run in enumerate(runs):
            k = min(len(run.actions), L)
            acc = 0.0
            for j in range(k - 1, -1, -1):                     # 从后往前累折扣回报
                acc = run.rewards[j] * self.reward_scale + self.gamma * acc
                ret[b, j] = acc
                act[b, j] = run.actions[j]
                steps[b, j] = k - j
                if j >= 1:
                    mask[b, j] = 1.0
        return (ret.to(DEVICE), act.to(DEVICE), steps.to(DEVICE), mask.to(DEVICE))

    def update_model(self):
        if not self.memory.is_ready(self.batch_size):
            return

        transitions, indices, is_weights = self.memory.sample(self.batch_size)
        if not transitions:
            return
        batch = Transition(*zip(*transitions))

        state = self._state_tensor(batch.state)
        boss_hp = torch.as_tensor(batch.boss_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        self_hp = torch.as_tensor(batch.self_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        action = torch.as_tensor(batch.action, dtype=torch.long, device=DEVICE)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=DEVICE) * self.reward_scale
        next_state = self._state_tensor(batch.next_state)
        next_boss = torch.as_tensor(batch.next_boss_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_self = torch.as_tensor(batch.next_self_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        done = torch.as_tensor(batch.done, dtype=torch.float32, device=DEVICE)
        is_weights = torch.as_tensor(is_weights, dtype=torch.float32, device=DEVICE)

        # 计划一致性用的 1 步样本（可能还没攒够，攒够之前只训 TD）
        use_plan = (
            self.plan_length > 1
            and self.plan_batch_size > 0
            and self.plan_memory.is_ready(self.plan_batch_size)
        )
        if use_plan:
            p_batch = Transition(*zip(*self.plan_memory.sample(self.plan_batch_size)))
            p_state = self._state_tensor(p_batch.state)
            p_boss = torch.as_tensor(p_batch.boss_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            p_self = torch.as_tensor(p_batch.self_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            p_next_state = self._state_tensor(p_batch.next_state)
            p_next_boss = torch.as_tensor(p_batch.next_boss_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            p_next_self = torch.as_tensor(p_batch.next_self_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            p_done = torch.as_tensor(p_batch.done, dtype=torch.float32, device=DEVICE)
            n_td, n_plan = state.size(0), p_state.size(0)
        else:
            n_td, n_plan = state.size(0), 0

        # 回报锚用的 run 样本（commit 恒为 1 时这个 buffer 永远是空的，loss 为 0）
        use_runs = (
            self.plan_length > 1
            and self.plan_run_batch_size > 0
            and self.plan_run_memory.is_ready(self.plan_run_batch_size)
        )
        if use_runs:
            runs = self.plan_run_memory.sample(self.plan_run_batch_size)
            r_state = self._state_tensor([r.state for r in runs])
            r_boss = torch.as_tensor([[r.boss_health] for r in runs], dtype=torch.float32, device=DEVICE)
            r_self = torch.as_tensor([[r.self_health] for r in runs], dtype=torch.float32, device=DEVICE)
            r_tail_state = self._state_tensor([r.tail_state for r in runs])
            r_tail_boss = torch.as_tensor([[r.tail_boss_health] for r in runs], dtype=torch.float32, device=DEVICE)
            r_tail_self = torch.as_tensor([[r.tail_self_health] for r in runs], dtype=torch.float32, device=DEVICE)
            r_done = torch.as_tensor([float(r.done) for r in runs], dtype=torch.float32, device=DEVICE)
            n_run = r_state.size(0)
            # 逐槽位的"从第 j 步起到 run 结束"的折扣回报、该步实际动作、到 tail 还差几步
            r_ret, r_act, r_steps, r_mask = self._plan_run_targets(runs)
        else:
            n_run = 0

        amp_kwargs = dict(
            device_type=DEVICE.type if DEVICE.type != "cpu" else "cpu",
            dtype=torch.float16,
            enabled=(DEVICE.type != "cpu"),
        )

        with self.net_lock:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

            with torch.amp.autocast(**amp_kwargs):
                # --- 把几个 batch 拼成一次前向，省掉重复的 CTM tick 循环 ---
                def _split(t, sizes):
                    out, i = [], 0
                    for n in sizes:
                        out.append(t[i:i + n] if n else None)
                        i += n
                    return out

                pol_sizes = [n_td, n_plan, n_run]
                pol_states = [state] + ([p_state] if use_plan else []) + ([r_state] if use_runs else [])
                pol_boss = [boss_hp] + ([p_boss] if use_plan else []) + ([r_boss] if use_runs else [])
                pol_self = [self_hp] + ([p_self] if use_plan else []) + ([r_self] if use_runs else [])
                all_q, all_cert = self.policy_net(
                    torch.cat(pol_states), torch.cat(pol_boss), torch.cat(pol_self)
                )
                cur_q, plan_q, run_q = _split(all_q, pol_sizes)
                cur_cert, plan_cert, _ = _split(all_cert, pol_sizes)

                with torch.no_grad():
                    # target 网络：TD 的 s'、一致性的 s'、以及 run 的尾部状态
                    tgt_sizes = [n_td, n_plan, n_run]
                    tgt_states = [next_state] + ([p_next_state] if use_plan else []) + \
                                 ([r_tail_state] if use_runs else [])
                    tgt_boss = [next_boss] + ([p_next_boss] if use_plan else []) + \
                               ([r_tail_boss] if use_runs else [])
                    tgt_self = [next_self] + ([p_next_self] if use_plan else []) + \
                               ([r_tail_self] if use_runs else [])
                    tgt_all_q, tgt_all_cert = self.target_net(
                        torch.cat(tgt_states), torch.cat(tgt_boss), torch.cat(tgt_self)
                    )
                    tgt_next, _, _ = CTMPlannerNet.pick_most_certain(tgt_all_q, tgt_all_cert)
                    tgt_next_td, tgt_next_plan, tgt_tail = _split(tgt_next, tgt_sizes)

                    # Double DQN：动作由 policy 选、价值由 target 给
                    pn_sizes = [n_td, n_run]
                    pn_states = [next_state] + ([r_tail_state] if use_runs else [])
                    pn_boss = [next_boss] + ([r_tail_boss] if use_runs else [])
                    pn_self = [next_self] + ([r_tail_self] if use_runs else [])
                    pol_next_q, pol_next_cert = self.policy_net(
                        torch.cat(pn_states), torch.cat(pn_boss), torch.cat(pn_self)
                    )
                    pol_next, _, _ = CTMPlannerNet.pick_most_certain(pol_next_q, pol_next_cert)
                    pol_next_td, pol_next_tail = _split(pol_next, pn_sizes)
                    next_action = pol_next_td[:, 0].mean(-1).argmax(-1)                # (B,)

            # ---------------- 槽位 0：标准 QR-DQN TD ----------------
            next_quantiles = tgt_next_td[:, 0].gather(
                1, next_action.view(-1, 1, 1).expand(-1, 1, self.num_quantiles)
            ).squeeze(1)                                                              # (B, Q)
            target_quantiles = (
                reward.unsqueeze(1)
                + (1 - done.unsqueeze(1)) * (self.gamma ** self.n_step) * next_quantiles
            )

            n_ticks = cur_q.size(-1)
            act_idx = action.view(-1, 1, 1, 1).expand(-1, 1, self.num_quantiles, n_ticks)
            cur_slot0 = cur_q[:, 0].gather(1, act_idx).squeeze(1)                     # (B, Q, T)

            td_per_tick = torch.stack([
                self._quantile_huber(cur_slot0[..., t], target_quantiles) for t in range(n_ticks)
            ], dim=1)                                                                 # (B, T)
            td_loss_per_sample = self._ctm_tick_aggregate(td_per_tick, cur_cert)
            td_loss = (is_weights * td_loss_per_sample).mean()

            # PER 优先级用"最确定 tick"上的 TD 误差绝对值：那才是实际会被执行的判断
            with torch.no_grad():
                certain_tick = cur_cert[:, 1].argmax(dim=-1)
                rows = torch.arange(n_td, device=DEVICE)
                q_at_certain = cur_slot0[rows, :, certain_tick]                       # (B, Q)
                priorities = (target_quantiles.mean(1) - q_at_certain.mean(1)).abs()
            self.memory.update_priorities(indices, priorities.float().cpu().numpy())

            # ---------------- 槽位 1..L-1：计划自举一致性 ----------------
            plan_loss = torch.zeros((), device=DEVICE)
            if use_plan:
                n_slots = self.plan_length - 1
                # 目标：target 网络在 s' 上的槽位 j-1（整条动作分布一起回归，
                # 只对齐 argmax 的话非贪心动作永远拿不到梯度）
                tgt_slots = tgt_next_plan[:, :-1]                                     # (B, L-1, A, Q)
                tgt_flat = tgt_slots.reshape(-1, self.num_quantiles)

                slot_losses = []
                for t in range(n_ticks):
                    cur_slots = plan_q[:, 1:, :, :, t]                                # (B, L-1, A, Q)
                    per_pair = self._quantile_huber(
                        cur_slots.reshape(-1, self.num_quantiles), tgt_flat
                    ).view(n_plan, n_slots, self.num_actions).mean(dim=2)             # (B, L-1)
                    slot_losses.append(per_pair)
                slot_losses = torch.stack(slot_losses, dim=-1)                        # (B, L-1, T)

                with torch.no_grad():
                    certain_tick_p = plan_cert[:, 1].argmax(dim=-1)
                    rows_p = torch.arange(n_plan, device=DEVICE)
                    pol_argmax = plan_q[rows_p, :, :, :, certain_tick_p].mean(-1).argmax(-1)
                    tgt_argmax = tgt_next_plan.mean(-1).argmax(-1)
                    slot_mask = self._plan_slot_mask(pol_argmax, tgt_argmax)

                if slot_mask is not None:
                    weighted = (slot_losses * slot_mask.unsqueeze(-1)).sum(dim=1)
                    plan_per_tick = weighted / slot_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                else:
                    plan_per_tick = slot_losses.mean(dim=1)                           # (B, T)

                plan_per_sample = self._ctm_tick_aggregate(plan_per_tick, plan_cert)
                # 终止步没有"下一帧的计划"，屏蔽掉
                keep = 1.0 - p_done
                plan_loss = (plan_per_sample * keep).sum() / keep.sum().clamp(min=1.0)

            # ------------- 槽位 1..k-1：真实回报锚（"照计划走"才有收益的来源） -------------
            # 一致性 loss 只保证槽位之间自洽，没有奖励信号进得去；这一项把**真的连续
            # 执行掉的**那几步的折扣回报直接安到对应槽位上，于是"计划值不值得照着走"
            # 由真实回报说了算，而不是自指。
            plan_return_loss = torch.zeros((), device=DEVICE)
            if use_runs:
                # 尾部自举（Double DQN，和槽位 0 的 TD 同构）
                tail_act = pol_next_tail[:, 0].mean(-1).argmax(-1)                     # (B,)
                tail_quantiles = tgt_tail[:, 0].gather(
                    1, tail_act.view(-1, 1, 1).expand(-1, 1, self.num_quantiles)
                ).squeeze(1)                                                          # (B, Q)
                run_losses = []
                for j in range(1, self.plan_length):
                    if float(r_mask[:, j].sum()) == 0:
                        continue
                    disc = (1.0 - r_done) * torch.pow(self.gamma, r_steps[:, j])       # (B,)
                    tgt_j = r_ret[:, j].unsqueeze(1) + disc.unsqueeze(1) * tail_quantiles
                    a_idx = r_act[:, j].view(-1, 1, 1, 1).expand(
                        -1, 1, self.num_quantiles, n_ticks
                    )
                    cur_j = run_q[:, j].gather(1, a_idx).squeeze(1)                    # (B, Q, T)
                    per_tick = torch.stack([
                        self._quantile_huber(cur_j[..., t], tgt_j) for t in range(n_ticks)
                    ], dim=1)                                                          # (B, T)
                    run_losses.append(per_tick * r_mask[:, j].unsqueeze(1))
                if run_losses:
                    # 每个样本按它真正执行掉的槽位数取平均
                    stacked = torch.stack(run_losses, dim=1).sum(dim=1)                # (B, T)
                    denom = r_mask[:, 1:].sum(dim=1, keepdim=True).clamp(min=1.0)
                    run_cert = all_cert[n_td + n_plan:]
                    per_sample = self._ctm_tick_aggregate(stacked / denom, run_cert)
                    plan_return_loss = per_sample.mean()

            loss = (
                td_loss
                + self.plan_loss_weight * plan_loss
                + self.plan_return_weight * plan_return_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.steps % 10 == 0:
                self.scheduler.step()

            with torch.no_grad():
                for tgt_p, pol_p in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tgt_p.data.copy_(self.tau * pol_p.data + (1.0 - self.tau) * tgt_p.data)
                # buffer 不同步：神经元配对索引在 __init__ 时已经从 policy 拷过来且永不变，
                # BN 统计量在 train 模式下用不到，NoisyLinear 的 epsilon 每步都会重采。

            self.steps += 1
            self.last_losses = {
                "td": float(td_loss.detach()),
                "plan": float(plan_loss.detach()),
                "plan_return": float(plan_return_loss.detach()),
            }
            if self.steps % 100 == 0:
                self.writer.add_scalar("Loss/TD", self.last_losses["td"], self.steps)
                self.writer.add_scalar("Loss/Plan", self.last_losses["plan"], self.steps)
                self.writer.add_scalar(
                    "Loss/PlanReturn", self.last_losses["plan_return"], self.steps
                )
                # 这条是回报锚有没有在工作的**总开关**：commit 恒为 1 时它永远是 0，
                # 说明计划从没被连着执行过，锚也就无从谈起。
                self.writer.add_scalar(
                    "Plan/RunBuffer", len(self.plan_run_memory), self.steps
                )
                self.writer.add_scalar(
                    "Plan/CertainTick", cur_cert[:, 1].argmax(-1).float().mean().item(), self.steps
                )
                if self._recent_commit:
                    self.writer.add_scalar(
                        "Plan/CommitLen", float(np.mean(self._recent_commit)), self.steps
                    )
                if self._recent_confidence:
                    self.writer.add_scalar(
                        "Plan/Confidence", float(np.mean(self._recent_confidence)), self.steps
                    )

    # ------------------------------------------------------------------ 存取
    def _save_worker(self):
        while True:
            try:
                task = self.save_queue.get()
                if task is None:
                    break
                filename, checkpoint = task
                torch.save(checkpoint, filename)
                self.save_queue.task_done()
            except Exception as exc:
                print(f"异步保存模型失败: {exc}")
            time.sleep(0.1)

    def save_checkpoint(self, filename=None):
        filename = filename or self.checkpoint_file
        with self.net_lock:
            checkpoint = {
                "arch": ARCH_NAME,
                "policy_state": {k: v.cpu().clone() for k, v in self.policy_net.state_dict().items()},
                "target_state": {k: v.cpu().clone() for k, v in self.target_net.state_dict().items()},
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "steps": self.steps,
                "num_actions": self.num_actions,
                "plan_length": self.plan_length,
                "network_config": self.network_config,
            }
        if self.save_queue.full():
            try:
                self.save_queue.get_nowait()
                self.save_queue.task_done()
            except Exception:
                pass
        self.save_queue.put((filename, checkpoint))

    def load_checkpoint(self, filename=None):
        """
        严格加载。旧版 DQNAgent 用 `strict=False` 兜底，结果形状不匹配会被静默吞掉，
        跑出来是个半随机初始化的网络还没人知道——这里宁可明确报错。
        """
        import os

        filename = filename or self.checkpoint_file
        if not os.path.exists(filename):
            return False
        try:
            checkpoint = torch.load(filename, map_location=DEVICE, weights_only=False)
        except Exception as exc:
            print(f">> 读取 checkpoint 失败: {filename} | {exc}")
            return False

        arch = checkpoint.get("arch")
        if arch != ARCH_NAME:
            print(f">> checkpoint 架构不符: 期望 {ARCH_NAME}，实际 {arch!r}（{filename}）")
            print(">> 旧的 ProNet 权重请用 --arch pro 加载；不会自动转换。")
            return False
        for key, mine in (("num_actions", self.num_actions), ("plan_length", self.plan_length)):
            saved = checkpoint.get(key)
            if saved is not None and saved != mine:
                print(f">> checkpoint {key} 不符: 保存的是 {saved}，当前是 {mine}（{filename}）")
                return False

        try:
            with self.net_lock:
                self.policy_net.load_state_dict(checkpoint["policy_state"], strict=True)
                self.target_net.load_state_dict(checkpoint["target_state"], strict=True)
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception:
                    print(">> 优化器状态不匹配，已重置")
                self.steps = checkpoint.get("steps", 0)
        except Exception as exc:
            print(f">> 加载权重失败（可能是 CTM 结构配置变了）: {exc}")
            return False
        return True

    def close(self):
        print(">> 正在等待后台保存任务完成...")
        self.save_queue.put(None)
        self.save_thread.join()
        self.writer.close()
        print(">> 资源已释放")

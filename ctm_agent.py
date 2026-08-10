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
（注意 certainty 只用于挑 tick；提交几步看的是 `CTMPlannerNet.plan_confidence`。）
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
from replay_buffer import PrioritizedReplayBuffer, Transition, UniformReplayBuffer

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
        self.last_losses = {"td": 0.0, "plan": 0.0}

    # ------------------------------------------------------------------ 决策
    def select_plan(self, state, boss_health, self_health):
        """
        返回 (plan: list[int], commit: int, confidence: float)。

        plan 是"从现在起打算做的 L 个动作"，commit 是这次真正提交给游戏的步数。
        certainty（CTM 的熵）只用于挑 tick；提交长度看的是 `plan_confidence`，
        原因见 CTMPlannerNet 里那两个方法的注释。
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
            confidence = CTMPlannerNet.plan_confidence(picked)
            commit = int(self.policy_net.commit_length(confidence)[0].item())
            confidence = float(confidence[0].item())

        # ε 兜底：NoisyNets 的 sigma 会随训练自己收缩，探索可能悄悄归零。
        # 逐槽位替换成随机动作，保证策略永远不会彻底锁死在一个招式上。
        if self.explore_epsilon > 0:
            plan = [
                random.randrange(self.num_actions) if random.random() < self.explore_epsilon
                else int(a)
                for a in plan
            ]

        self._recent_commit.append(commit)
        self._recent_confidence.append(confidence)
        return [int(a) for a in plan], commit, confidence

    def select_action(self, state, boss_health, self_health, hidden=None):
        """兼容旧接口：只取计划的第一步。"""
        plan, _, _ = self.select_plan(state, boss_health, self_health)
        return plan[0]

    # ------------------------------------------------------------------ 存样本
    def store_transition(self, state, boss_health, self_health, action,
                         reward, next_state, next_boss_health, next_self_health, done):
        transition = Transition(
            self._as_state(state), boss_health, self_health, action, reward,
            self._as_state(next_state), next_boss_health, next_self_health, done,
        )
        self.transitions_seen += 1

        # 1 步样本原样进计划 buffer：槽位一致性要的就是"紧邻的下一帧"
        self.plan_memory.add(transition)

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

        amp_kwargs = dict(
            device_type=DEVICE.type if DEVICE.type != "cpu" else "cpu",
            dtype=torch.float16,
            enabled=(DEVICE.type != "cpu"),
        )

        with self.net_lock:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

            with torch.amp.autocast(**amp_kwargs):
                # --- 一次前向算完两个 batch，省掉重复的 CTM tick 循环 ---
                if use_plan:
                    all_q, all_cert = self.policy_net(
                        torch.cat([state, p_state]),
                        torch.cat([boss_hp, p_boss]),
                        torch.cat([self_hp, p_self]),
                    )
                    cur_q, cur_cert = all_q[:n_td], all_cert[:n_td]
                    plan_q, plan_cert = all_q[n_td:], all_cert[n_td:]
                else:
                    cur_q, cur_cert = self.policy_net(state, boss_hp, self_hp)
                    plan_q = plan_cert = None

                with torch.no_grad():
                    if use_plan:
                        tgt_all_q, tgt_all_cert = self.target_net(
                            torch.cat([next_state, p_next_state]),
                            torch.cat([next_boss, p_next_boss]),
                            torch.cat([next_self, p_next_self]),
                        )
                        tgt_next, _, _ = CTMPlannerNet.pick_most_certain(tgt_all_q, tgt_all_cert)
                        tgt_next_td, tgt_next_plan = tgt_next[:n_td], tgt_next[n_td:]
                    else:
                        tgt_all_q, tgt_all_cert = self.target_net(next_state, next_boss, next_self)
                        tgt_next_td, _, _ = CTMPlannerNet.pick_most_certain(tgt_all_q, tgt_all_cert)
                        tgt_next_plan = None

                    # Double DQN：动作由 policy 选、价值由 target 给
                    pol_next_q, pol_next_cert = self.policy_net(next_state, next_boss, next_self)
                    pol_next, _, _ = CTMPlannerNet.pick_most_certain(pol_next_q, pol_next_cert)
                    next_action = pol_next[:, 0].mean(-1).argmax(-1)                  # (B,)

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

            loss = td_loss + self.plan_loss_weight * plan_loss

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
            }
            if self.steps % 100 == 0:
                self.writer.add_scalar("Loss/TD", self.last_losses["td"], self.steps)
                self.writer.add_scalar("Loss/Plan", self.last_losses["plan"], self.steps)
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

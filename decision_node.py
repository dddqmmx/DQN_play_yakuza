# -*- coding: utf-8 -*-
"""
AI 节点：决策 + 训练。

通信收发/编解码由 core.CommandServer 负责；本模块只实现命令处理与训练逻辑。
"""
from __future__ import annotations

import threading
import time

from config import CTM_PROFILES, DEVICE, FILE_PATHS, GAME_CONFIG, MODEL_PROFILES, TRAINING_CONFIG
from core.command_server import CommandServer


class DecisionNode:
    """AI 节点：同一进程内完成决策和训练。"""

    def __init__(self, host="0.0.0.0", port=15001, model_profile="large",
                 checkpoint=None, arch="ctm"):
        self.host = host
        self.port = port or 15001
        self.arch = arch if arch in ("ctm", "pro") else "ctm"
        profiles = self._profiles()
        self.model_profile = model_profile if model_profile in profiles else "large"
        self.checkpoint_file = checkpoint or self._default_checkpoint(self.model_profile)
        self.agent = self._create_agent(self.model_profile, self.checkpoint_file)
        self.hidden_by_client = {}
        # 上一条计划 + 自上次规划以来实际执行了几步，用来算"改主意"的程度（plan drift）
        self.last_plan_by_client = {}
        self.executed_since_plan = {}
        self.is_running = True
        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.console_thread = threading.Thread(target=self._console_worker, daemon=True)
        self.agent_lock = threading.RLock()
        self.server = CommandServer(self.host, self.port)
        self._bind_commands()

    def _profiles(self):
        return CTM_PROFILES if self.arch == "ctm" else MODEL_PROFILES

    def _bind_commands(self):
        self.server.on("action_request", self._cmd_action_request)
        self.server.on("transition", self._cmd_transition)
        self.server.on("reset_hidden", self._cmd_reset_hidden)
        self.server.on("save", self._cmd_save)
        self.server.on("client_disconnect", self._cmd_client_disconnect)

    def _default_checkpoint(self, profile):
        base = FILE_PATHS["ctm_checkpoint"] if self.arch == "ctm" else FILE_PATHS["checkpoint"]
        if profile == "large":
            return base
        stem, dot, suffix = base.rpartition(".")
        if not dot:
            return f"{base}_{profile}"
        return f"{stem}_{profile}.{suffix}"

    def _create_agent(self, profile, checkpoint_file):
        log_dir = f"{FILE_PATHS['tensorboard_logs']}_{self.arch}_{profile}"
        if self.arch == "ctm":
            from ctm_agent import CTMPlannerAgent
            return CTMPlannerAgent(
                GAME_CONFIG["num_actions"],
                network_config=CTM_PROFILES[profile],
                checkpoint_file=checkpoint_file,
                log_dir=log_dir,
            )
        from dqn_agent import DQNAgent
        return DQNAgent(
            GAME_CONFIG["num_actions"],
            network_config=MODEL_PROFILES[profile],
            checkpoint_file=checkpoint_file,
            log_dir=log_dir,
        )

    def _train_worker(self):
        """
        后台训练。两条纪律：

        1. **`agent_lock` 只用来取 agent 引用，不能罩住整个 update。**
           它存在的意义只是防止 `model`/`checkpoint` 控制台命令换掉 agent 对象
           （极罕见）；网络参数本身由 agent 自己的 `net_lock` 保护。早先把整个
           `update_model()` 圈进去，导致决策和存样本都得排队等一整个训练步 ——
           实测 CTM 一步 238ms，而每个环境步要抢两次锁，游戏侧掉到 1.8 步/s。
        2. **训练速率跟着数据速率走**（replay ratio）。在线 RL 里更新数比环境步数
           多太多就是在小 buffer 上反复过拟合，还白占 GPU 让决策变慢。
           比值必须按**本次会话**算：`agent.steps` 是从 checkpoint 载入的累计值，
           而 `transitions_seen` 每次启动归零，直接相比会让载入老 checkpoint 后
           训练被永久卡死（实测载入 1000 步的存档后，训练线程一步没跑）。
        """
        replay_ratio = float(TRAINING_CONFIG.get("replay_ratio", 0.25))
        baseline_agent = None
        baseline_steps = 0
        while self.is_running:
            try:
                with self.agent_lock:
                    agent = self.agent

                # agent 被 model/checkpoint 命令换掉时重新取基线
                if agent is not baseline_agent:
                    baseline_agent = agent
                    baseline_steps = agent.steps

                if not agent.memory.is_ready(agent.batch_size):
                    time.sleep(0.02)
                    continue

                seen = getattr(agent, "transitions_seen", None)
                if (
                    replay_ratio > 0
                    and seen is not None
                    and (agent.steps - baseline_steps) >= replay_ratio * seen
                ):
                    # 已经训得比数据来得快，让出 GPU 给决策
                    time.sleep(0.01)
                    continue

                agent.update_model()
                time.sleep(0.001)
            except Exception as exc:
                print(f">> AI 节点训练线程错误: {exc}")
                time.sleep(1.0)

    def _select_plan(self, client_id, msg):
        """返回 (plan, commit, confidence)。旧 ProNet 退化成单步计划。"""
        # 只在取引用时持锁，推理本身由 agent 的 net_lock 保护 —— 见 _train_worker 的注释
        with self.agent_lock:
            agent = self.agent
        if hasattr(agent, "select_plan"):
            return agent.select_plan(
                msg["state"], msg["boss_health"], msg["self_health"]
            )
        hidden = self.hidden_by_client.get(client_id)
        if agent.use_recurrent:
            action, next_hidden = agent.select_action(
                msg["state"], msg["boss_health"], msg["self_health"], hidden
            )
            self.hidden_by_client[client_id] = next_hidden
        else:
            action = agent.select_action(
                msg["state"], msg["boss_health"], msg["self_health"]
            )
        return [int(action)], 1, 0.0

    def _plan_drift(self, client_id, plan):
        """
        改主意的程度：上一条计划里"本该轮到现在"的那一段，和新计划对不上的比例。

        错位量取**实际执行步数**（自上次 action_request 以来收到的 transition 条数），
        不是 commit —— 客户端挨打会中途丢弃剩余计划，按 commit 错位会算出虚高的漂移。

        0 表示完全说到做到，1 表示每次都推翻重来。这个数**不应该**趋近 0 ——
        环境在变，该改主意就得改；但它长期居高不下说明计划头没学到东西。
        """
        prev = self.last_plan_by_client.get(client_id)
        executed = self.executed_since_plan.pop(client_id, 0)
        self.last_plan_by_client[client_id] = list(plan)
        if not prev or executed <= 0:
            return None
        n = min(len(prev) - executed, len(plan))
        if n <= 0:
            return None
        mismatch = sum(1 for i in range(n) if prev[executed + i] != plan[i])
        return mismatch / n

    def _store_transition(self, msg):
        # 同样只持锁取引用：存样本是纯 numpy 拷贝，不该排在训练步后面
        with self.agent_lock:
            agent = self.agent
        agent.store_transition(
            msg["state"], msg["boss_health"], msg["self_health"], msg["action"],
            msg["reward"], msg["next_state"], msg["next_boss_health"],
            msg["next_self_health"], msg["done"],
        )

    def _save(self, filename=None):
        with self.agent_lock:
            self.agent.save_checkpoint(filename or self.checkpoint_file)

    def _switch_checkpoint(self, filename):
        with self.agent_lock:
            self.checkpoint_file = filename
            loaded = self.agent.load_checkpoint(filename)
        print(f">> 保存点已切换到: {filename} | load={'ok' if loaded else 'new'}")

    def _switch_model(self, profile, checkpoint=None):
        profiles = self._profiles()
        if profile not in profiles:
            print(f">> 未知模型档位: {profile}，可用: {', '.join(profiles)}")
            return
        checkpoint = checkpoint or self._default_checkpoint(profile)
        with self.agent_lock:
            old_agent = self.agent
            old_agent.save_checkpoint(self.checkpoint_file)
            self.model_profile = profile
            self.checkpoint_file = checkpoint
            self.hidden_by_client.clear()
            self.last_plan_by_client.clear()
            self.executed_since_plan.clear()
            self.agent = self._create_agent(profile, checkpoint)
            loaded = self.agent.load_checkpoint(checkpoint)
            old_agent.close()
        print(
            f">> 已切换模型: {self.arch}/{profile} | checkpoint={checkpoint} | "
            f"load={'ok' if loaded else 'new'}"
        )

    def _print_status(self):
        with self.agent_lock:
            agent = self.agent
            plan_len = getattr(agent, "plan_length", 1)
            plan_mem = len(getattr(agent, "plan_memory", ()))
            print(
                f">> status | arch={self.arch} | model={self.model_profile} | "
                f"checkpoint={self.checkpoint_file} | plan_length={plan_len} | "
                f"steps={agent.steps} | memory={len(agent.memory)} | "
                f"plan_memory={plan_mem} | device={DEVICE}"
            )

    # ---- CommandServer handlers ----
    def _cmd_action_request(self, client_id, msg):
        if msg.get("episode_done"):
            self.hidden_by_client.pop(client_id, None)
            self.last_plan_by_client.pop(client_id, None)
            self.executed_since_plan.pop(client_id, None)
        started = time.perf_counter()
        plan, commit, confidence = self._select_plan(client_id, msg)
        drift = self._plan_drift(client_id, plan)
        with self.agent_lock:
            train_steps = self.agent.steps
            if drift is not None and train_steps % 50 == 0:
                self.agent.writer.add_scalar("Plan/Drift", drift, train_steps)
        return {
            "type": "plan",
            "plan": [int(a) for a in plan],
            "commit": int(commit),
            "confidence": float(confidence),
            "decision_ms": (time.perf_counter() - started) * 1000.0,
            "train_steps": train_steps,
        }

    def _cmd_transition(self, client_id, msg):
        self._store_transition(msg)
        self.executed_since_plan[client_id] = self.executed_since_plan.get(client_id, 0) + 1
        return {"type": "ack"}

    def _cmd_reset_hidden(self, client_id, msg):
        self.hidden_by_client.pop(client_id, None)
        self.last_plan_by_client.pop(client_id, None)
        self.executed_since_plan.pop(client_id, None)
        return {"type": "ack"}

    def _cmd_save(self, client_id, msg):
        self._save()
        return {"type": "ack"}

    def _cmd_client_disconnect(self, client_id, msg):
        self.hidden_by_client.pop(client_id, None)
        self.last_plan_by_client.pop(client_id, None)
        self.executed_since_plan.pop(client_id, None)
        return None

    def _console_worker(self):
        print(
            ">> AI 控制台命令: help, status, save [path], checkpoint <path>, "
            "model <small|medium|large> [path], exit"
        )
        while self.is_running:
            try:
                raw = input("ai> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()
            try:
                if cmd in ("help", "?"):
                    print(
                        "命令: status | save [path] | checkpoint <path> | "
                        "model <small|medium|large> [path] | exit"
                    )
                elif cmd == "status":
                    self._print_status()
                elif cmd == "save":
                    self._save(parts[1] if len(parts) > 1 else None)
                    print(">> 保存任务已提交")
                elif cmd in ("checkpoint", "ckpt"):
                    if len(parts) < 2:
                        print("用法: checkpoint <path>")
                    else:
                        self._switch_checkpoint(parts[1])
                elif cmd == "model":
                    if len(parts) < 2:
                        print(
                            f"当前模型: {self.arch}/{self.model_profile}，可用档位: "
                            f"{', '.join(self._profiles())}"
                        )
                    else:
                        self._switch_model(
                            parts[1].lower(),
                            parts[2] if len(parts) > 2 else None,
                        )
                elif cmd in ("exit", "quit", "q"):
                    print(">> 正在退出 AI 节点...")
                    self.is_running = False
                    self.server.stop()
                    self._save()
                    break
                else:
                    print(f">> 未知命令: {cmd}")
            except Exception as exc:
                print(f">> 控制台命令失败: {exc}")

    def serve_forever(self):
        if self.agent.load_checkpoint(self.checkpoint_file):
            print(">> AI 节点已加载本地模型")
        self.train_thread.start()
        self.console_thread.start()
        try:
            self.server.serve_forever(
                ready_callback=lambda: print(
                    f">> AI 节点 arch={self.arch} profile={self.model_profile} device={DEVICE}"
                )
            )
        finally:
            self.shutdown()

    def shutdown(self):
        self.is_running = False
        self.server.stop()
        with self.agent_lock:
            self.agent.save_checkpoint(self.checkpoint_file)
            self.agent.close()

# -*- coding: utf-8 -*-
"""
DecisionClient：与 AI 节点通信。

负责连接、消息编解码与命令收发，不碰截屏/按键。
消息类型:
  发出: action_request | transition | reset_hidden | save
  收到: plan | ack | error

AI 节点一律回**计划**（一串动作 + 提交步数），客户端因此不需要知道对面跑的是
哪种模型：旧的 ProNet 只是回一条 `plan=[a], commit=1` 的退化计划。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.protocol import connect_with_retry, recv_message, send_message


class DecisionClient:
    def __init__(self, host: str, port: int = 15001):
        self.host = host
        self.port = int(port)
        self.sock = None

    def connect(self, retry_delay: float = 1.0):
        if self.sock is None:
            self.sock = connect_with_retry(self.host, self.port, retry_delay=retry_delay)
        return self.sock

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _ensure(self):
        if self.sock is None:
            self.connect()

    def request_plan(
        self,
        state,
        boss_health: float,
        self_health: float,
        episode_done: bool = False,
    ) -> Tuple[List[int], int, float, float, Optional[int]]:
        """
        请求一条计划。返回 (plan, commit, confidence, decision_ms, train_steps|None)。

        `commit` 是本次真正要执行的步数（由模型对这条计划的决断程度决定），
        永远 <= len(plan)。`confidence` 只作日志/诊断用。
        """
        self._ensure()
        send_message(self.sock, {
            "type": "action_request",
            "state": state,
            "boss_health": boss_health,
            "self_health": self_health,
            "episode_done": episode_done,
        })
        reply = recv_message(self.sock)
        if reply.get("type") not in ("plan", "action"):
            raise RuntimeError(f"Decision 返回异常: {reply}")

        plan = reply.get("plan")
        if plan is None:  # 兼容只回单个 action 的旧节点
            plan = [int(reply["action"])]
        plan = [int(a) for a in plan]
        commit = int(reply.get("commit", 1))
        commit = max(1, min(commit, len(plan)))
        return (
            plan,
            commit,
            float(reply.get("confidence", 0.0)),
            float(reply.get("decision_ms", 0.0)),
            reply.get("train_steps"),
        )

    def request_action(
        self,
        state,
        boss_health: float,
        self_health: float,
        episode_done: bool = False,
    ) -> Tuple[int, float, Optional[int]]:
        """兼容旧接口：只取计划的第一步。"""
        plan, _, _, decision_ms, train_steps = self.request_plan(
            state, boss_health, self_health, episode_done
        )
        return plan[0], decision_ms, train_steps

    def send_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        """上传一条经验转移，等待 ack。"""
        self._ensure()
        send_message(self.sock, {"type": "transition", **transition})
        return recv_message(self.sock)

    def reset_hidden(self) -> Dict[str, Any]:
        self._ensure()
        send_message(self.sock, {"type": "reset_hidden"})
        return recv_message(self.sock)

    def save(self) -> Dict[str, Any]:
        self._ensure()
        send_message(self.sock, {"type": "save"})
        return recv_message(self.sock)

    def send_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """通用命令：编码发送并解码响应。"""
        self._ensure()
        send_message(self.sock, message)
        return recv_message(self.sock)

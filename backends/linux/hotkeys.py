# -*- coding: utf-8 -*-
"""
Linux 控制台指令（仅 stdin，不监听物理键盘 / evdev）。

在运行 game-node / train 的终端输入命令后回车即可。
支持短键名与语义别名，例如:
  0 / start / pause / toggle
  f8 / calibrate / locate
  f9 / quit / exit / stop
  f10 / freeze
  f5 / save
  help / status / ?
"""
from __future__ import annotations

import select
import sys
import threading
from typing import Callable, Dict, List, Optional, Set


# 主命令 → 可接受的别名（输入任一即可）
_ALIASES: Dict[str, List[str]] = {
    "0": ["0", "start", "pause", "toggle", "p", "go"],
    "f5": ["f5", "save", "s"],
    "f8": ["f8", "calibrate", "calib", "locate", "loc", "auto"],
    "f9": ["f9", "quit", "exit", "q", "stop", "bye"],
    "f10": ["f10", "freeze", "frz"],
    # resume: 聚焦游戏窗口 + 虚拟点击 + ESC
    "resume": ["resume", "r", "focus", "unpause-game", "esc"],
}

_BUILTIN = {"help", "h", "?", "status", "cmds", "commands"}


class CommandConsole:
    """从 stdin 读行并分发到已注册回调。"""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}  # canonical key -> callback
        self._alias_to_key: Dict[str, str] = {}  # alias -> canonical
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()

    def add_hotkey(self, key: str, callback: Callable):
        canonical = key.strip().lower()
        with self._lock:
            self._handlers[canonical] = callback
            aliases = _ALIASES.get(canonical, [canonical])
            for alias in aliases:
                self._alias_to_key[alias.lower()] = canonical
            # 也允许直接用 canonical
            self._alias_to_key[canonical] = canonical
        if not self._started:
            self._start()

    def _start(self):
        if self._started:
            return
        self._started = True
        self._print_banner()
        self._thread = threading.Thread(target=self._stdin_loop, daemon=True, name="cli-cmds")
        self._thread.start()

    def _print_banner(self):
        print("\n" + "-" * 42)
        print(" Linux 控制台指令（输入后回车，不监听全局键盘）")
        print(self._help_text())
        print("-" * 42)
        print("cmd> ", end="", flush=True)

    def _help_text(self) -> str:
        lines = []
        with self._lock:
            registered = set(self._handlers.keys())
        order = ["0", "resume", "f5", "f8", "f10", "f9"]
        labels = {
            "0": "启动/暂停 AI",
            "resume": "聚焦游戏+点击+ESC",
            "f5": "保存模型",
            "f8": "自动定位血条",
            "f9": "安全退出",
            "f10": "切换进程冻结",
        }
        for key in order:
            if key not in registered:
                continue
            aliases = [a for a in _ALIASES.get(key, [key]) if a != key]
            alias_s = f"  别名: {', '.join(aliases)}" if aliases else ""
            lines.append(f"  {key:4}  {labels.get(key, key)}{alias_s}")
        lines.append("  help  显示本帮助")
        lines.append("  status 已注册指令列表")
        return "\n".join(lines) if lines else "  (尚无注册指令)"

    def _resolve(self, raw: str) -> Optional[str]:
        token = raw.strip().lower()
        if not token:
            return None
        # 只取第一个词
        token = token.split()[0]
        with self._lock:
            return self._alias_to_key.get(token)

    def _dispatch(self, raw: str):
        token = raw.strip().lower()
        if not token:
            return
        head = token.split()[0]
        if head in _BUILTIN:
            if head in ("help", "h", "?"):
                print(self._help_text())
            elif head in ("status", "cmds", "commands"):
                with self._lock:
                    keys = sorted(self._handlers.keys())
                print(f">> 已注册: {', '.join(keys)}")
                print(self._help_text())
            return

        canonical = self._resolve(token)
        if canonical is None:
            print(f">> 未知指令: {token!r}  (输入 help 查看)")
            return
        with self._lock:
            cb = self._handlers.get(canonical)
        if cb is None:
            print(f">> 指令未绑定: {canonical}")
            return
        try:
            cb()
        except Exception as e:
            print(f">> 指令执行错误 [{canonical}]: {e}")

    def _stdin_loop(self):
        # 无 TTY 时（nohup/pipe）不阻塞空转刷屏
        if not sys.stdin or not hasattr(sys.stdin, "fileno"):
            print(">> 无 stdin，控制台指令不可用（前台运行终端以输入命令）")
            return
        try:
            fd = sys.stdin.fileno()
        except (OSError, ValueError):
            print(">> stdin 不可用，控制台指令已禁用")
            return

        while not self._stop.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.5)
            except (OSError, ValueError):
                break
            if not r:
                continue
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                # EOF
                break
            self._dispatch(line)
            if not self._stop.is_set():
                print("cmd> ", end="", flush=True)

    def stop(self):
        self._stop.set()


_manager = CommandConsole()


def add_hotkey(key: str, callback: Callable):
    _manager.add_hotkey(key, callback)


def stop():
    _manager.stop()

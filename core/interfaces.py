# -*- coding: utf-8 -*-
"""平台后端抽象接口。core 只依赖这些接口，不依赖 Win/Linux 细节。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np


class CaptureBackend(ABC):
    """窗口/流观测：返回 BGR uint8 帧。"""

    @abstractmethod
    def grab(self, process_name: str, region=None) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self):
        pass


class InputBackend(ABC):
    """键盘注入（不强制鼠标）。扫描码常量与动作函数由实现提供。"""

    # 扫描码（与历史 Windows 版一致）
    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20
    E = 0x12
    J = 0x24
    K = 0x25
    M = 0x32
    L = 0x26
    Q = 0x10
    I = 0x17
    O = 0x18
    ENTER = 0x1C
    ESC = 0x01

    @abstractmethod
    def press_key(self, hex_key_code: int):
        raise NotImplementedError

    @abstractmethod
    def release_key(self, hex_key_code: int):
        raise NotImplementedError

    def tap_key(self, hex_key_code: int, duration: float = 0.05):
        import time
        self.press_key(hex_key_code)
        time.sleep(duration)
        self.release_key(hex_key_code)

    def hold_key(self, hex_key_code: int, duration: float = 0.4):
        import time
        self.press_key(hex_key_code)
        time.sleep(duration)
        self.release_key(hex_key_code)

    def weak_attack(self):
        self.tap_key(self.J, 0.05)

    def strong_attack(self):
        self.tap_key(self.K, 0.05)

    def start_forward(self):
        self.press_key(self.W)

    def stop_forward(self):
        self.release_key(self.W)

    def start_back(self):
        self.press_key(self.S)

    def stop_back(self):
        self.release_key(self.S)

    def start_left(self):
        self.press_key(self.A)

    def stop_left(self):
        self.release_key(self.A)

    def start_right(self):
        self.press_key(self.D)

    def stop_right(self):
        self.release_key(self.D)

    def start_defense(self):
        self.press_key(self.I)

    def stop_defense(self):
        self.release_key(self.I)

    def dodge(self):
        self.tap_key(self.L, 0.1)

    def grab(self):
        self.tap_key(self.E, 0.05)

    def striking_pose_start(self):
        self.press_key(self.O)

    def striking_pose_cancel(self):
        self.release_key(self.O)

    def extrem_heat_mode(self):
        self.hold_key(self.Q, 0.4)

    def press_enter(self):
        self.tap_key(self.ENTER, 0.05)

    def press_esc(self):
        self.hold_key(self.ESC, 0.3)

    def resume_game(self, process_name: str = None, press_escape: bool = True):
        """
        聚焦游戏窗口 + 可选虚拟点击 + ESC。
        平台可覆盖；默认仅按 ESC。
        """
        self.press_esc()
        return {"ok": True, "method": "esc-only", "esc": press_escape}

    def shutdown(self):
        pass


class ProcessControlBackend(ABC):
    """可选进程冻结/单帧放行。"""

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def suspend(self):
        raise NotImplementedError

    @abstractmethod
    def resume(self):
        raise NotImplementedError

    @abstractmethod
    def run_one_frame(self):
        raise NotImplementedError

    @abstractmethod
    def close(self, resume: bool = True):
        raise NotImplementedError


class HotkeyBackend(ABC):
    @abstractmethod
    def add_hotkey(self, key: str, callback: Callable):
        raise NotImplementedError

    def stop(self):
        pass


class PlatformBundle:
    """一次工厂调用返回的平台能力集合。"""

    def __init__(
        self,
        name: str,
        capture: CaptureBackend,
        input: InputBackend,
        hotkeys: HotkeyBackend,
        process_factory: Callable[..., ProcessControlBackend],
        notes: str = "",
    ):
        self.name = name
        self.capture = capture
        self.input = input
        self.hotkeys = hotkeys
        self.process_factory = process_factory
        self.notes = notes

# -*- coding: utf-8 -*-
"""平台后端工厂：按 OS 加载 windows / linux 实现。"""
from __future__ import annotations

import sys

from core.interfaces import PlatformBundle


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_wayland() -> bool:
    import os
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def default_capture_method() -> str:
    import os
    env = os.environ.get("DQN_CAPTURE_METHOD")
    if env:
        return env.lower()
    if is_windows():
        return "mss"
    if is_wayland():
        return "pipewire"
    return "x11shm"


def load_platform(capture_method: str = None) -> PlatformBundle:
    """加载当前 OS 的 PlatformBundle。"""
    method = capture_method or default_capture_method()
    if is_windows():
        from backends.windows import create_bundle
        return create_bundle(capture_method=method)
    if is_linux():
        from backends.linux import create_bundle
        return create_bundle(capture_method=method)
    raise RuntimeError(f"不支持的平台: {sys.platform}")

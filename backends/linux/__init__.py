# -*- coding: utf-8 -*-
"""Linux / Proton / Wayland 平台后端：uinput + PipeWire/XShm + SIGSTOP。"""
from __future__ import annotations

from core.interfaces import (
    CaptureBackend,
    HotkeyBackend,
    InputBackend,
    PlatformBundle,
    ProcessControlBackend,
)


class LinuxCapture(CaptureBackend):
    def __init__(self, method: str = "auto"):
        from backends.linux import capture as cap
        self._grab = cap.grab_screen
        self.method = method or "auto"

    def grab(self, process_name, region=None):
        return self._grab(process_name, method=self.method, region=region)

    def release(self):
        try:
            from backends.linux import capture as cap
            if hasattr(cap, "_grabber"):
                cap._grabber.release()
        except Exception:
            pass


class LinuxInput(InputBackend):
    """uinput 虚拟键盘包装：不捕获鼠标。"""

    def __init__(self, process_name: str = None):
        from backends.linux import input_keys as keys
        self._keys = keys
        if process_name:
            keys.set_process_name(process_name)
        print(
            f">> 输入安全守卫: {'开启' if keys._guard_enabled else '关闭'} "
            f"(DQN_INPUT_GUARD) 目标={process_name or '未设置'}"
        )

    def press_key(self, hex_key_code: int):
        self._keys.press_key(hex_key_code)

    def release_key(self, hex_key_code: int):
        self._keys.release_key(hex_key_code)

    def weak_attack(self):
        self._keys.weak_attack()

    def strong_attack(self):
        self._keys.strong_attack()

    def start_forward(self):
        self._keys.start_forward()

    def stop_forward(self):
        self._keys.stop_forward()

    def start_back(self):
        self._keys.start_back()

    def stop_back(self):
        self._keys.stop_back()

    def start_left(self):
        self._keys.start_left()

    def stop_left(self):
        self._keys.stop_left()

    def start_right(self):
        self._keys.start_right()

    def stop_right(self):
        self._keys.stop_right()

    def start_defense(self):
        self._keys.start_defense()

    def stop_defense(self):
        self._keys.stop_defense()

    def dodge(self):
        self._keys.dodge()

    def grab(self):
        self._keys.grab()

    def striking_pose_start(self):
        self._keys.striking_pose_start()

    def striking_pose_cancel(self):
        self._keys.striking_pose_cancel()

    def extrem_heat_mode(self):
        self._keys.extrem_heat_mode()

    def press_enter(self):
        self._keys.press_enter()

    def press_esc(self):
        self._keys.press_esc()

    def resume_game(self, process_name: str = None, press_escape: bool = True):
        from config import GAME_CONFIG
        name = process_name or GAME_CONFIG.get("process_name", "Yakuza6.exe")
        if hasattr(self._keys, "resume_game"):
            return self._keys.resume_game(process_name=name, press_escape=press_escape)
        self.press_esc()
        return {"ok": True, "method": "esc-only", "esc": press_escape}

    def shutdown(self):
        if hasattr(self._keys, "shutdown"):
            self._keys.shutdown()


class LinuxHotkeys(HotkeyBackend):
    """仅命令行指令，不监听物理键盘。"""

    def __init__(self):
        from backends.linux import hotkeys as hk
        self._hk = hk

    def add_hotkey(self, key, callback):
        self._hk.add_hotkey(key, callback)

    def stop(self):
        if hasattr(self._hk, "stop"):
            self._hk.stop()
        elif hasattr(self._hk, "_manager"):
            self._hk._manager.stop()


class LinuxProcessControl(ProcessControlBackend):
    def __init__(self, process_name, target_fps=20.0, strict_passes=2):
        from backends.linux.process import GameProcessFreezer
        self._impl = GameProcessFreezer(process_name, target_fps, strict_passes)

    def open(self):
        self._impl.open()

    def suspend(self):
        self._impl.suspend()

    def resume(self):
        self._impl.resume()

    def run_one_frame(self):
        self._impl.run_one_frame()

    def close(self, resume=True):
        self._impl.close(resume=resume)


def create_bundle(capture_method: str = "auto") -> PlatformBundle:
    from config import GAME_CONFIG
    return PlatformBundle(
        name="linux",
        capture=LinuxCapture(method=capture_method),
        input=LinuxInput(process_name=GAME_CONFIG.get("process_name")),
        hotkeys=LinuxHotkeys(),
        process_factory=lambda process_name, target_fps=20.0, **kw: LinuxProcessControl(
            process_name, target_fps, **kw
        ),
        notes=(
            "uinput 键盘 / 不捕获鼠标\n"
            "观测: PipeWire portal 或 XWayland SHM\n"
            "输入守卫: 游戏失焦即终止（DQN_INPUT_GUARD=0 关闭）"
        ),
    )

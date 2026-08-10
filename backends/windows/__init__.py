# -*- coding: utf-8 -*-
"""Windows 平台后端：SendInput + MSS/BitBlt/DXCAM + NtSuspend。"""
from __future__ import annotations

from core.interfaces import (
    CaptureBackend,
    HotkeyBackend,
    InputBackend,
    PlatformBundle,
    ProcessControlBackend,
)


class WindowsCapture(CaptureBackend):
    def __init__(self, method: str = "mss"):
        from backends.windows import capture as cap
        self._grab = cap.grab_screen
        self.method = method if method in ("mss", "bitblt", "dxcam") else "mss"

    def grab(self, process_name, region=None):
        return self._grab(process_name, method=self.method, region=region)

    def release(self):
        try:
            from backends.windows import capture as cap
            if hasattr(cap, "_grabber"):
                cap._grabber.release()
        except Exception:
            pass


class WindowsInput(InputBackend):
    def __init__(self):
        from backends.windows import input_keys as keys
        self._keys = keys

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


class WindowsHotkeys(HotkeyBackend):
    def add_hotkey(self, key, callback):
        import keyboard
        keyboard.add_hotkey(key, callback)


class WindowsProcessControl(ProcessControlBackend):
    def __init__(self, process_name, target_fps=20.0, strict_passes=2):
        from backends.windows.process import GameProcessFreezer
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


def create_bundle(capture_method: str = "mss") -> PlatformBundle:
    return PlatformBundle(
        name="windows",
        capture=WindowsCapture(method=capture_method),
        input=WindowsInput(),
        hotkeys=WindowsHotkeys(),
        process_factory=lambda process_name, target_fps=20.0, **kw: WindowsProcessControl(
            process_name, target_fps, **kw
        ),
        notes="SendInput 键盘 | MSS/BitBlt/DXCAM 捕获",
    )

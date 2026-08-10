# -*- coding: utf-8 -*-
"""
Linux 窗口聚焦（合成器 API，不碰真实鼠标/键盘）。

niri: `niri msg action focus-window --id` — 仅切换键盘焦点，不移动实体指针。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple


def _run(cmd: List[str], timeout: float = 3.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _match_game(win: dict, process_name: str) -> bool:
    """严格匹配游戏窗口，避免把终端路径 DQN_play_yakuza 当成游戏。"""
    target = (process_name or "Yakuza6.exe").lower()
    target_base = target.replace(".exe", "")
    app_id = (win.get("app_id") or "").lower()
    title = (win.get("title") or "").lower()

    # 排除开发相关窗口
    deny = (
        "dqn_play", "opencode", "ghostty", "terminal", "chrome", "firefox",
        "code", "nvim", "vim",
    )
    if any(d in app_id for d in deny):
        return False
    if any(d in title for d in ("dqn_play", "opencode", "~/project")):
        return False

    # app_id 精确/后缀（Proton 常见 yakuza6.exe）
    if app_id == target or app_id == target_base:
        return True
    if app_id.endswith(".exe") and target_base in app_id and "yakuza" in app_id:
        return True
    if app_id == "yakuza6.exe" or app_id.endswith("yakuza6.exe"):
        return True

    # 标题：游戏正式名，排除仅含路径片段
    if "yakuza 6" in title or "song of life" in title:
        return True
    if title == "yakuza6.exe" or title.startswith("yakuza6"):
        return True

    return False


def list_niri_windows() -> List[dict]:
    if not shutil.which("niri"):
        return []
    code, out, _ = _run(["niri", "msg", "--json", "windows"])
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("windows") or []
    return []


def find_game_window_niri(process_name: str = "Yakuza6.exe") -> Optional[dict]:
    wins = list_niri_windows()
    matched = [w for w in wins if _match_game(w, process_name)]
    if not matched:
        return None
    focused = [w for w in matched if w.get("is_focused")]
    if focused:
        return focused[0]

    def area(w):
        layout = w.get("layout") or {}
        size = layout.get("window_size") or layout.get("tile_size") or [0, 0]
        try:
            return float(size[0]) * float(size[1])
        except Exception:
            return 0.0

    matched.sort(key=area, reverse=True)
    return matched[0]


def get_focused_niri_window() -> Optional[dict]:
    """返回当前聚焦的 niri 窗口 dict（无则 None）。"""
    wins = list_niri_windows()
    for w in wins:
        if w.get("is_focused"):
            return w
    return None


def is_game_window_focused(process_name: str = "Yakuza6.exe") -> bool:
    """游戏窗口是否为当前聚焦窗口（niri 合成器状态，非 X 猜测）。"""
    focused = get_focused_niri_window()
    if not focused:
        return False
    return _match_game(focused, process_name)


def focus_window_niri(window_id: int) -> bool:
    code, out, err = _run(
        ["niri", "msg", "action", "focus-window", "--id", str(int(window_id))]
    )
    if code != 0:
        print(f">> niri focus-window 失败: {err or out}")
        return False
    return True


def window_center_local(win: dict) -> Optional[Tuple[int, int]]:
    """窗口客户区中心（相对窗口），供虚拟绝对鼠标在「已聚焦」后使用时的参考。"""
    layout = win.get("layout") or {}
    size = layout.get("window_size") or layout.get("tile_size")
    if not size:
        return None
    try:
        w, h = float(size[0]), float(size[1])
        return int(w / 2), int(h / 2)
    except Exception:
        return None


def focus_game_window(process_name: str = "Yakuza6.exe") -> Dict:
    """
    仅用合成器 API 聚焦，不移动真实鼠标、不注入真实键。
    """
    win = find_game_window_niri(process_name)
    if win is not None:
        wid = win.get("id")
        if wid is not None and focus_window_niri(int(wid)):
            return {
                "ok": True,
                "method": "niri-focus",
                "window": win,
                "local_center": window_center_local(win),
                "detail": (
                    f"id={wid} title={win.get('title')!r} "
                    f"app_id={win.get('app_id')!r}"
                ),
            }
        return {
            "ok": False,
            "method": "niri-focus",
            "window": win,
            "local_center": None,
            "detail": "找到游戏窗口但 focus 失败",
        }

    env_id = os.environ.get("DQN_NIRI_WINDOW_ID")
    if env_id and shutil.which("niri"):
        if focus_window_niri(int(env_id)):
            return {
                "ok": True,
                "method": "niri-env",
                "window": {"id": int(env_id)},
                "local_center": None,
                "detail": f"DQN_NIRI_WINDOW_ID={env_id}",
            }

    return {
        "ok": False,
        "method": "none",
        "window": None,
        "local_center": None,
        "detail": "未找到 Yakuza 游戏窗口（检查 niri msg windows / app_id）",
    }

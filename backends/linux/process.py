# -*- coding: utf-8 -*-
"""
Linux 游戏进程冻结（SIGSTOP / SIGCONT）。

用于可选的“帧步进”训练。对 Proton/Wine 主进程及其子进程一并操作。
注意: 冻结 GPU 渲染线程可能导致部分驱动异常，默认建议关闭，改用 FPS 节流。
"""
from __future__ import annotations

import os
import signal
import time
from typing import List, Optional, Set

import psutil


def find_process_ids(process_name: str) -> List[int]:
    target = process_name.lower()
    found: Set[int] = set()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            hit = name == target or target in name
            if not hit:
                for part in proc.info.get("cmdline") or []:
                    if target in os.path.basename(part).lower():
                        hit = True
                        break
            if hit:
                found.add(int(proc.info["pid"]))
                try:
                    for child in psutil.Process(proc.info["pid"]).children(recursive=True):
                        found.add(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(found)


class GameProcessFreezer:
    def __init__(self, process_name, target_fps=20.0, strict_passes=2):
        self.process_name = process_name
        self.frame_seconds = 1.0 / max(float(target_fps), 1.0)
        self.strict_passes = max(int(strict_passes), 1)
        self.pids: List[int] = []
        self.is_suspended = False

    def open(self):
        pids = find_process_ids(self.process_name)
        if not pids:
            raise RuntimeError(f"找不到游戏进程: {self.process_name}")
        self.pids = pids

    def _signal_all(self, sig):
        alive = []
        for pid in self.pids:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError as e:
                raise RuntimeError(
                    f"无法向 pid={pid} 发送信号（权限不足）: {e}"
                ) from e
        self.pids = alive or self.pids

    def suspend(self):
        self.open()
        if self.is_suspended:
            return
        for _ in range(self.strict_passes):
            self._signal_all(signal.SIGSTOP)
            time.sleep(0.001)
        self.is_suspended = True

    def resume(self):
        if not self.pids or not self.is_suspended:
            return
        self._signal_all(signal.SIGCONT)
        self.is_suspended = False

    def run_one_frame(self):
        self.resume()
        time.sleep(self.frame_seconds)
        self.suspend()

    def close(self, resume=True):
        if resume and self.is_suspended:
            try:
                self.resume()
            except Exception:
                pass
        self.pids = []
        self.is_suspended = False

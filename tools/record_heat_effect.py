# -*- coding: utf-8 -*-
"""
录制"极限热血模式"（按 Q）期间的血条帧，用于修血条检测。

现象：按 Q 之后血条会加特效、整体偏深蓝，而 `core/health_bar.fill_mask` 只认
饱和红（H∈0-8 或 170-180），于是整条血条匹配不到填充色 -> 该帧被判为不可信
（`no-fill-*`）直接丢弃。要修就得先量出特效期间填充像素的真实颜色。

用法（游戏需在前台、处于战斗中）:
    python tools/record_heat_effect.py --out samples_heat --seconds 12

脚本会：
  1. 聚焦游戏窗口
  2. 先录一段常态基线
  3. 按 Q，继续高速录制整个特效周期
  4. 每帧顺便打印两条血条轨道上"最亮那一列"的 BGR/HSV，方便直接看颜色怎么变
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends import load_platform
from core.health_bar import fill_mask
from core.observation import GameObservation


def bar_stats(frame, box):
    """返回轨道核心区里最亮像素的 BGR/HSV，以及当前 fill_mask 的命中率。"""
    if box is None:
        return None
    core = frame[box.y1:box.y2, box.x1:box.x2]
    if core.size == 0:
        return None
    v = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)[:, :, 2]
    idx = np.unravel_index(int(np.argmax(v)), v.shape)
    bgr = core[idx[0], idx[1]].astype(int)
    hsv = cv2.cvtColor(core[idx[0]:idx[0] + 1, idx[1]:idx[1] + 1], cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    return {
        "bgr": tuple(bgr.tolist()),
        "hsv": tuple(hsv.tolist()),
        "fill_frac": float(fill_mask(core).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples_heat")
    ap.add_argument("--seconds", type=float, default=12.0, help="按 Q 之后继续录多久")
    ap.add_argument("--baseline", type=float, default=3.0, help="按 Q 之前先录多久")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--capture", default=os.environ.get("DQN_CAPTURE_METHOD", "x11shm"))
    ap.add_argument("--no-press", action="store_true", help="只录不按 Q（你自己手动按）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundle = load_platform(capture_method=args.capture)
    game = GameObservation(bundle.capture)

    interval = 1.0 / max(args.fps, 1e-6)
    n = 0
    pressed_at = None

    try:
        if not args.no_press:
            print(">> 聚焦游戏窗口…")
            try:
                bundle.input.resume_game(process_name=game.process_name, press_escape=False)
            except Exception as exc:
                print(f">> 聚焦失败（继续）: {exc}")
            time.sleep(1.0)

        # 先让轨道锁定
        for _ in range(10):
            raw = game.get_raw_screen()
            if raw is not None:
                game.check_game_state(auto_locate=True)
            if all(t.box for t in game.trackers.values()):
                break
            time.sleep(0.2)
        print(f">> 轨道: {game.health_status()}")

        t_end = time.time() + args.baseline + args.seconds
        while time.time() < t_end:
            raw = game.get_raw_screen()
            if raw is None:
                time.sleep(interval)
                continue

            if pressed_at is None and time.time() >= t_end - args.seconds:
                if not args.no_press:
                    print(">> 按 Q（极限热血模式）")
                    bundle.input.extrem_heat_mode()
                pressed_at = time.time()

            phase = "base" if pressed_at is None else f"heat{time.time() - pressed_at:05.1f}"
            readings = game.read_health(raw, allow_locate=False)
            line = [f"[{n:03d}] {phase}"]
            for name in ("player", "boss"):
                st = bar_stats(raw, game.trackers[name].box)
                r = readings[name]
                val = "None" if r.value is None else f"{r.value:.3f}"
                if st:
                    line.append(
                        f"{name}={val}({r.reason}) BGR{st['bgr']} HSV{st['hsv']} fill={st['fill_frac']:.2f}"
                    )
                else:
                    line.append(f"{name}=未锁定")
            print("  ".join(line))

            cv2.imwrite(os.path.join(args.out, f"{phase}_{n:04d}.png"), raw)
            n += 1
            time.sleep(interval)
    finally:
        bundle.capture.release()
        bundle.input.shutdown()
        print(f">> 已保存 {n} 帧到 {args.out}/")


if __name__ == "__main__":
    main()

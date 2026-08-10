# -*- coding: utf-8 -*-
"""
采集游戏帧样本，用于血条检测调参/回归测试。

用法:
  python tools/record_frames.py --out samples --interval 0.5 --max 400

只保存与上一张"明显不同"的帧，避免菜单静止时刷屏。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--process", default=os.environ.get("DQN_PROCESS_NAME", "Yakuza6.exe"))
    ap.add_argument("--method", default=os.environ.get("DQN_CAPTURE_METHOD", "x11shm"))
    ap.add_argument("--diff", type=float, default=2.0,
                    help="平均像素差阈值，低于此值视为重复帧")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from backends.linux.capture import LinuxScreenGrabber
    g = LinuxScreenGrabber()
    grab = {
        "x11shm": g.grab_x11shm,
        "pipewire": g.grab_pipewire,
        "mss": g.grab_mss,
        "auto": g.grab_auto,
    }.get(args.method, g.grab_x11shm)

    prev_small = None
    saved = 0
    t0 = time.time()
    try:
        while saved < args.max:
            frame = grab(args.process)
            if frame is None:
                time.sleep(args.interval)
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev_small is not None:
                d = float(np.abs(small - prev_small).mean())
                if d < args.diff:
                    time.sleep(args.interval)
                    continue
            prev_small = small
            ts = time.time() - t0
            path = os.path.join(args.out, f"f{saved:04d}_t{ts:07.2f}.png")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"saved {path}", flush=True)
            time.sleep(args.interval)
    finally:
        g.release()
        print(f">> 共保存 {saved} 帧到 {args.out}/", flush=True)


if __name__ == "__main__":
    main()

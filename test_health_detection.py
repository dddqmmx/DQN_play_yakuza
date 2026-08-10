# -*- coding: utf-8 -*-
"""
对着当前游戏画面跑一次血条检测，输出读数和标注图。

用法:
  python test_health_detection.py
  python test_health_detection.py --frames 40     # 连续跑 40 帧看稳定性
"""
import argparse
import os
import time

import cv2

from backends import default_capture_method, load_platform
from config import GAME_CONFIG
from core.observation import GameObservation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--out", default="debug_images")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bundle = load_platform(capture_method=default_capture_method())
    game = GameObservation(bundle.capture)

    print(f">> 进程={GAME_CONFIG['process_name']} 捕获={default_capture_method()}")
    raw = None
    try:
        for i in range(args.frames):
            raw = game.get_raw_screen()
            if raw is None:
                print(f"[{i}] 抓不到画面，游戏在运行吗？")
                time.sleep(args.interval)
                continue
            game.read_health(raw)
            print(f"[{i}] {raw.shape[1]}x{raw.shape[0]}  {game.health_status()}")
            time.sleep(args.interval)
    finally:
        if raw is not None:
            cv2.imwrite(os.path.join(args.out, "full_screen.png"), raw)
            game.debug_health_regions(
                raw, show=False, save_path=os.path.join(args.out, "debug_locations.png")
            )
            for name, tr in game.trackers.items():
                if tr.box:
                    b = tr.box
                    cv2.imwrite(
                        os.path.join(args.out, f"{name}_track.png"),
                        raw[b.y1:b.y2, b.x1:b.x2],
                    )
            print(f">> 标注图已写入 {args.out}/")
        bundle.capture.release()


if __name__ == "__main__":
    main()

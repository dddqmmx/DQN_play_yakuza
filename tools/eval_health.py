# -*- coding: utf-8 -*-
"""
在录制的样本上回归测试血条检测。

用法:
  python tools/eval_health.py samples/            # 汇总
  python tools/eval_health.py samples/ --verbose  # 每帧
  python tools/eval_health.py samples/ --debug-out /tmp/dbg   # 输出标注图
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.health_bar as hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug-out", default=None)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.path, "*.png")))
    if not files:
        print("没有样本")
        return 1
    if args.debug_out:
        os.makedirs(args.debug_out, exist_ok=True)

    trackers = {
        "player": hb.BarTracker(hb.PLAYER_TARGET),
        "boss": hb.BarTracker(hb.BOSS_TARGET),
    }
    series = {k: [] for k in trackers}
    reasons = {k: {} for k in trackers}

    for idx, f in enumerate(files):
        if idx % args.every:
            continue
        frame = cv2.imread(f)
        if frame is None:
            continue
        readings = {}
        for name, tr in trackers.items():
            r = tr.read(frame)
            readings[name] = r
            series[name].append(r.value)
            key = r.reason if not r.ok else "ok"
            reasons[name][key] = reasons[name].get(key, 0) + 1
        if args.verbose:
            parts = []
            for name in trackers:
                r = readings[name]
                b = trackers[name].box
                bs = f"[{b.x1},{b.y1},{b.x2},{b.y2}]" if b else "[-]"
                v = "None " if not r.ok else f"{r.value:.3f}"
                parts.append(f"{name}={v} {bs} {r.reason}")
            print(f"{os.path.basename(f)[:20]:22s} " + " | ".join(parts))
        if args.debug_out:
            out = hb.draw_debug(frame, trackers, readings)
            cv2.imwrite(os.path.join(args.debug_out, os.path.basename(f)), out)

    print()
    print("=" * 68)
    for name in trackers:
        vals = series[name]
        good = [v for v in vals if v is not None]
        print(f"[{name}] 帧数={len(vals)} 有效={len(good)} "
              f"({100.0*len(good)/max(len(vals),1):.1f}%)")
        if trackers[name].box:
            b = trackers[name].box
            print(f"   锁定轨道: x={b.x1}..{b.x2} (w={b.w}) y={b.y1}..{b.y2} (h={b.h})")
        if good:
            print(f"   范围: {min(good):.3f} .. {max(good):.3f}")
            # 单调性：战斗中血量应只降不升（除回合切换）
            ups = []
            prev = None
            for v in vals:
                if v is None:
                    continue
                if prev is not None and v > prev + 0.02:
                    ups.append((prev, v))
                prev = v
            print(f"   回升次数(>0.02): {len(ups)}"
                  + (f"  例: {ups[:5]}" if ups else ""))
            # 相邻帧跳变
            jumps = []
            prev = None
            for v in vals:
                if v is None:
                    continue
                if prev is not None and abs(v - prev) > 0.15:
                    jumps.append((round(prev, 3), round(v, 3)))
                prev = v
            print(f"   大跳变(>0.15): {len(jumps)}"
                  + (f"  例: {jumps[:5]}" if jumps else ""))
        top = sorted(reasons[name].items(), key=lambda kv: -kv[1])[:6]
        print(f"   原因分布: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
CTM 决策延迟基准。

关键约束：game-node 按 target_fps 推进（默认 20fps = 50ms 一帧），一次决策必须
在这个预算内出结果，否则 AI 会拖慢游戏节奏。CTM 一次决策要跑 `iterations` 个
内部 tick，比旧 ProNet 贵；但计划制下 commit>1 时决策频率同比下降，实际每帧
成本是 `决策耗时 / 平均提交步数`。这个脚本把两个数都打出来。

    python tools/bench_ctm.py                 # 三档都测
    python tools/bench_ctm.py --profile small
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import CTM_PROFILES, DEVICE, GAME_CONFIG, MODEL_PROFILES


def bench_net(net, label, budget_ms, iters=50, warmup=10, plan_length=1):
    frame = GAME_CONFIG["frame_size"]
    stack = GAME_CONFIG["frame_stack_size"]
    x = torch.randint(0, 256, (1, stack, frame[1], frame[0]), dtype=torch.uint8, device=DEVICE)
    boss = torch.rand(1, 1, device=DEVICE)
    selfhp = torch.rand(1, 1, device=DEVICE)

    def once():
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=DEVICE.type if DEVICE.type != "cpu" else "cpu",
                dtype=torch.float16,
                enabled=(DEVICE.type != "cpu"),
            ):
                return net(x, boss, selfhp)

    for _ in range(warmup):
        once()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        once()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    n_params = sum(p.numel() for p in net.parameters())
    mean, p95 = times.mean(), np.percentile(times, 95)
    verdict = "OK  " if p95 <= budget_ms else "超预算"
    print(
        f"  [{verdict}] {label:<22} 均值 {mean:6.2f}ms  p95 {p95:6.2f}ms  "
        f"参数 {n_params / 1e6:5.2f}M"
    )
    if plan_length > 1:
        print(
            f"           计划长度 {plan_length}：若平均提交 k 步，每帧摊薄成本 = "
            f"{mean:.2f}/k ms（k=2 时 {mean / 2:.2f}ms，k={plan_length} 时 "
            f"{mean / plan_length:.2f}ms）"
        )
    return p95 <= budget_ms


def main():
    parser = argparse.ArgumentParser(description="CTM 决策延迟基准")
    parser.add_argument("--profile", choices=["small", "medium", "large", "all"], default="all")
    parser.add_argument("--fps", type=float, default=20.0, help="game-node 目标帧率，用来算预算")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--compare-pro", action="store_true", help="同时测旧 ProNet 作对照")
    args = parser.parse_args()

    budget_ms = 1000.0 / max(args.fps, 1e-6)
    profiles = ["small", "medium", "large"] if args.profile == "all" else [args.profile]
    n_actions = GAME_CONFIG["num_actions"]

    print("=" * 68)
    print(f"设备 {DEVICE} | 目标 {args.fps:.0f}fps -> 单次决策预算 {budget_ms:.1f}ms")
    print("=" * 68)

    from ctm_planner import CTMPlannerNet

    all_ok = True
    for prof in profiles:
        cfg = CTM_PROFILES[prof]
        net = CTMPlannerNet(n_actions, config=cfg).to(DEVICE)
        label = f"ctm/{prof} (T={cfg['iterations']}, D={cfg['d_model']})"
        all_ok &= bench_net(net, label, budget_ms, iters=args.iters,
                            plan_length=cfg["plan_length"])
        del net
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if args.compare_pro:
        from network_components import ProNet
        for prof in profiles:
            net = ProNet(n_actions, config=MODEL_PROFILES[prof]).to(DEVICE)
            bench_net(net, f"pro/{prof}", budget_ms, iters=args.iters)
            del net
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    print()
    if not all_ok:
        print(">> 有档位超预算：降低 CTM_CONFIG 的 iterations 或 token_grid，"
              "或把 game-node 的 --game-fps 调低。")
        print(">> 也可以先看摊薄成本——commit>1 时实际每帧开销要除以平均提交步数。")
    else:
        print(">> 全部在预算内。")


if __name__ == "__main__":
    main()

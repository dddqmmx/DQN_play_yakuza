import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Yakuza 6 AI训练程序")
    parser.add_argument(
        "--mode",
        choices=[
            "train", "debug", "test", "game-node",
            "ai-node", "decision-node", "calibrate",
        ],
        default="train",
        help="运行模式",
    )
    parser.add_argument("--checkpoint", type=str, help="指定检查点文件路径")
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    parser.add_argument("--host", default="0.0.0.0", help="本机监听地址，供 ai-node 使用")
    parser.add_argument("--port", type=int, default=None, help="本机监听端口，ai-node 默认 15001")
    parser.add_argument("--decision-host", default="127.0.0.1", help="game-node 连接的 AI 节点地址")
    parser.add_argument("--decision-port", type=int, default=15001, help="game-node 连接的 AI 节点端口")
    parser.add_argument(
        "--arch",
        choices=["ctm", "pro"],
        default="ctm",
        help="ai-node 使用的模型架构：ctm=计划型 Continuous Thought Machine（默认），"
             "pro=旧的 ProNet（Conv3D+FPN+GRU+QR-DQN），只出单步动作",
    )
    parser.add_argument(
        "--model-profile",
        choices=["small", "medium", "large"],
        default="large",
        help="ai-node 初始模型档位",
    )
    parser.add_argument("--game-fps", type=float, default=20.0, help="game-node 目标帧率")
    parser.add_argument("--freeze", action="store_true", help="game-node 启动时开启进程冻结")
    parser.add_argument("--no-freeze", action="store_true", help="兼容旧参数：关闭冻结控制")
    parser.add_argument(
        "--input-settle-ms",
        type=float,
        default=5.0,
        help="收到动作后、放行游戏前等待按键注入的毫秒数",
    )
    parser.add_argument(
        "--capture",
        default=None,
        help="捕获后端: windows=mss|bitblt|dxcam  linux=auto|pipewire|x11shm|mss",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 50)
    print("Yakuza 6 强化学习训练系统")
    print("=" * 50)
    print(f"运行模式: {args.mode}")

    try:
        if args.mode == "train":
            from training_manager import TrainingManager
            trainer = TrainingManager(arch=args.arch, model_profile=args.model_profile)
            print("\n正在启动训练模式...")
            print("请确保游戏已启动并处于可操作状态")
            trainer.train()

        elif args.mode == "debug":
            from training_manager import TrainingManager
            trainer = TrainingManager(arch=args.arch, model_profile=args.model_profile)
            print("\n正在启动调试模式...")
            print("用于调试血条检测和坐标校准")
            trainer.debug_mode()

        elif args.mode == "test":
            from training_manager import TrainingManager
            trainer = TrainingManager(arch=args.arch, model_profile=args.model_profile)
            print("\n正在启动测试模式...")
            print("加载已训练模型进行测试")
            trainer.agent.epsilon = 0.0
            trainer.train()

        elif args.mode == "game-node":
            from backends import default_capture_method, is_linux, is_wayland, load_platform
            from core.game_loop import GameClient

            method = args.capture or default_capture_method()
            bundle = load_platform(capture_method=method)
            print(f"平台: {bundle.name} | Wayland={is_wayland() if is_linux() else False} | 捕获={method}")
            if bundle.notes:
                print(bundle.notes)
            node = GameClient(
                platform=bundle,
                decision_host=args.decision_host,
                decision_port=args.decision_port,
                target_fps=args.game_fps,
                freeze_process=args.freeze and not args.no_freeze,
                input_settle_ms=args.input_settle_ms,
            )
            node.run()

        elif args.mode == "calibrate":
            from backends import default_capture_method, load_platform
            from core.observation import GameObservation
            import time

            method = args.capture or default_capture_method()
            bundle = load_platform(capture_method=method)
            game = GameObservation(bundle.capture)
            print(">> 血条自动标定模式：请进入战斗 HUD（可见双方血条）")
            print(">> 每 0.5s 尝试定位，成功后写入 locations.json 并退出；Ctrl+C 取消")
            try:
                while True:
                    raw = game.get_raw_screen()
                    if raw is None:
                        print("\r>> 等待画面…", end="", flush=True)
                        time.sleep(0.5)
                        continue
                    located = game.auto_locate_health_bars(raw, save=True, force=True)
                    if located:
                        out = "debug_auto_locate.png"
                        game.debug_health_regions(raw, show=False, save_path=out)
                        boss_hp, self_hp, _, _, _ = game.check_game_state(auto_locate=False)
                        print(
                            f"\n>> 标定完成 Boss={boss_hp} Self={self_hp} "
                            f"| 预览 {out} | locations.json 已更新"
                        )
                        break
                    print(
                        f"\r>> 未找到血条 frame={raw.shape[1]}x{raw.shape[0]}，重试中…",
                        end="",
                        flush=True,
                    )
                    time.sleep(0.5)
            finally:
                bundle.capture.release()

        elif args.mode in ("ai-node", "decision-node"):
            from decision_node import DecisionNode
            node = DecisionNode(
                host=args.host,
                port=args.port or 15001,
                model_profile=args.model_profile,
                checkpoint=args.checkpoint,
                arch=args.arch,
            )
            node.serve_forever()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")

    except Exception as e:
        print(f"\n\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n程序结束")


if __name__ == "__main__":
    main()

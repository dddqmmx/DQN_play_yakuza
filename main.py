import argparse


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Yakuza 6 AI训练程序')
    parser.add_argument('--mode', choices=['train', 'debug', 'test', 'game-node', 'ai-node', 'decision-node'],
                        default='train', help='运行模式')
    parser.add_argument('--checkpoint', type=str,
                        help='指定检查点文件路径')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='训练回合数')
    parser.add_argument('--host', default='0.0.0.0',
                        help='本机监听地址，供 ai-node 使用')
    parser.add_argument('--port', type=int, default=None,
                        help='本机监听端口，ai-node 默认 15001')
    parser.add_argument('--decision-host', default='127.0.0.1',
                        help='game-node 连接的 AI 节点地址')
    parser.add_argument('--decision-port', type=int, default=15001,
                        help='game-node 连接的 AI 节点端口')
    parser.add_argument('--model-profile', choices=['small', 'medium', 'large'], default='large',
                        help='ai-node 初始模型档位')
    parser.add_argument('--game-fps', type=float, default=20.0,
                        help='game-node 控制游戏进程运行/冻结的目标帧率')
    parser.add_argument('--freeze', action='store_true',
                        help='game-node 启动时开启进程冻结控制；默认关闭以避免图形异常')
    parser.add_argument('--no-freeze', action='store_true',
                        help='兼容旧参数：关闭冻结控制')
    parser.add_argument('--input-settle-ms', type=float, default=5.0,
                        help='game-node 收到动作后、放行游戏前等待按键注入的毫秒数')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 50)
    print("Yakuza 6 强化学习训练系统")
    print("=" * 50)
    print(f"运行模式: {args.mode}")

    try:
        if args.mode == 'train':
            import grabscreen_pro  # 必须在任何 torch/cuda 相关代码运行前导入，其内部会初始化 dxcam
            from training_manager import TrainingManager
            trainer = TrainingManager()
            print("\n正在启动训练模式...")
            print("请确保游戏已启动并处于可操作状态")
            # 此时 TrainingManager 已经注册了热键，可以响应暂停/继续
            trainer.train()

        elif args.mode == 'debug':
            import grabscreen_pro
            from training_manager import TrainingManager
            trainer = TrainingManager()
            print("\n正在启动调试模式...")
            print("用于调试血条检测和坐标校准")
            trainer.debug_mode()

        elif args.mode == 'test':
            import grabscreen_pro
            from training_manager import TrainingManager
            trainer = TrainingManager()
            print("\n正在启动测试模式...")
            print("加载已训练模型进行测试")
            # 设置为不探索模式
            trainer.agent.epsilon = 0.0
            trainer.train()

        elif args.mode == 'game-node':
            import grabscreen_pro
            from game_node import GameNode
            node = GameNode(
                decision_host=args.decision_host,
                decision_port=args.decision_port,
                target_fps=args.game_fps,
                freeze_process=args.freeze and not args.no_freeze,
                input_settle_ms=args.input_settle_ms,
            )
            node.run()

        elif args.mode in ('ai-node', 'decision-node'):
            from decision_node import DecisionNode
            node = DecisionNode(
                host=args.host,
                port=args.port or 15001,
                model_profile=args.model_profile,
                checkpoint=args.checkpoint,
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


if __name__ == '__main__':
    main()

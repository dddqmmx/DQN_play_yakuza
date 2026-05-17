# 分布式运行说明

这套入口把职责拆成两类节点：

- `ai-node`: 决策和训练在同一个进程内完成。它接收游戏状态，立即返回动作，同时把回传经验放进本地 replay buffer，由后台线程持续训练。
- `game-node`: 只负责截屏、血条检测、按键执行和游戏进程冻结/放行。

## 推荐两台设备

AI 设备：

```bash
python main.py --mode ai-node --host 0.0.0.0 --port 15001 --model-profile large
```

游戏设备：

```bash
python main.py --mode game-node --decision-host <AI设备IP> --decision-port 15001 --game-fps 20
```

`decision-node` 仍保留为 `ai-node` 的兼容别名，但推荐使用 `ai-node`。

## AI 服务端控制台

`ai-node` 启动后可以在控制台输入：

- `status`: 查看当前模型、保存点、训练步数、经验池大小。
- `save [path]`: 保存当前模型，不填路径则保存到当前保存点。
- `checkpoint <path>`: 切换保存点，并尝试加载该保存点。
- `model <small|medium|large> [path]`: 切换小/中/大模型，可选指定保存点。
- `exit`: 保存并退出。

## 冻结帧控制

`game-node` 现在默认不冻结游戏进程，避免冻结渲染线程导致图形错误。默认方案是“非冻结帧节流”：按 `--game-fps` 控制 AI 采样和动作执行节奏，让游戏自然运行。

游戏端快捷键：

- `0`: 启动/暂停 AI 控制。启动后才会开始发状态和执行动作。
- `F10`: 运行时切换冻结控制。出现图形异常时再按一次关闭。
- `F9`: 安全退出并释放所有长按键。

如果显式加 `--freeze`，`game-node` 会按以下顺序运行：

1. 冻结游戏进程。
2. 在冻结状态下截取当前状态并发送给 `ai-node`。
3. 收到动作后先注入按键。
4. 放行游戏进程 `1 / --game-fps` 秒。
5. 再次冻结游戏进程并读取下一状态。

这样网络通信和模型推理耗时发生在游戏冻结期间，不会把通信延迟转化为游戏内操作延迟。

如果暂时不想冻结进程，可加：

```bash
python main.py --mode game-node --decision-host <AI设备IP> --no-freeze
```

冻结 Windows 游戏进程通常需要以管理员权限启动终端。

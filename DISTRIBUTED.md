# 分布式运行说明

这套入口把职责拆成两类节点：

- `ai-node`: 决策和训练在同一个进程内完成。它接收游戏状态，立即返回动作，同时把回传经验放进本地 replay buffer，由后台线程持续训练。
- `game-node`: 只负责观测、血条检测、按键执行和游戏进程冻结/放行。

## 代码分层

- **`core/`**：协议编解码、`DecisionClient` / `CommandServer`、客户端训练循环、观测与动作逻辑（与 OS 无关）。
- **`backends/windows`**、**`backends/linux`**：捕获 / 输入 / 冻结 / 热键的具体实现。
- 根目录旧模块名保留为兼容 re-export。

消息类型（客户端 ↔ AI）：

| 客户端发出 | AI 回复 |
|------------|---------|
| `action_request` | `plan` |
| `transition` | `ack` |
| `reset_hidden` | `ack` |
| `save` | `ack` |

`plan` 回包的内容：

```json
{"type": "plan", "plan": [1, 1, 2, 9, 0, 3], "commit": 3,
 "confidence": 0.42, "decision_ms": 11.4, "train_steps": 20531}
```

- `plan`：模型规划的动作序列（"接下来打算做的 L 件事"）。
- `commit`：本次真正提交给游戏的步数，由模型对这条计划的**决断程度**决定 ——
  有把握就把连招打完，没把握就走一步看一步。客户端执行完这几步就重新请求，
  **下一条计划可以推翻上一条**；执行途中挨打（超过
  `GAME_CONFIG['plan_interrupt_damage']`）会立刻丢弃剩余步数重新规划。
- `confidence`：相对动作间隔 =（最优 Q − 次优 Q）/（最优 Q − 最差 Q），逐槽位取均值。
  用它而不是 CTM 的熵 certainty，是因为熵被 Q 的绝对尺度主导：实测 σ=2.0 的**纯随机**
  Q 能得到 0.42 的"certainty"，而 σ=0.3 下真正决断的 Q 只有 0.03 —— 那样模型会
  因为 Q 变大而凭噪声连招。相对间隔对尺度免疫：13 个动作毫无偏好时恒在 ≈0.147，
  最优动作高出 10σ 时 ≈0.72，所以固定阈值在整个训练期都成立。
  （熵 certainty 仍在网络内部用于挑 tick 和 loss 聚合 —— 同一次前向内尺度一致，
  那里的比较是有效的。）
- 客户端不需要知道对面是哪种模型：`--arch pro` 的旧 ProNet 只是回一条
  `plan=[a], commit=1` 的退化计划。

## 推荐两台设备

AI 设备：

```bash
# 默认 CTM 计划模型
python main.py --mode ai-node --host 0.0.0.0 --port 15001 --model-profile large
# 旧 ProNet 对照
python main.py --mode ai-node --host 0.0.0.0 --port 15001 --arch pro --model-profile large
```

游戏设备：

```bash
python main.py --mode game-node --decision-host <AI设备IP> --decision-port 15001 --game-fps 20
```

`decision-node` 仍保留为 `ai-node` 的兼容别名，但推荐使用 `ai-node`。

两种架构的保存点互不通用（`ctm_planner*.pth` / `dqn_optimized*.pth`），
加载错了会明确报错而不是静默半初始化。

## AI 服务端控制台

`ai-node` 启动后可以在控制台输入：

- `status`: 查看当前架构、模型档位、保存点、计划长度、训练步数、两个经验池大小。
- `save [path]`: 保存当前模型，不填路径则保存到当前保存点。
- `checkpoint <path>`: 切换保存点，并尝试加载该保存点。
- `model <small|medium|large> [path]`: 切换小/中/大模型档位（架构不变）。
- `exit`: 保存并退出。

## TensorBoard 指标

计划模型额外记录：

| 标量 | 含义 |
|------|------|
| `Loss/TD` | 槽位 0 的 QR-DQN TD 损失 |
| `Loss/Plan` | 槽位 1..L-1 的自举一致性损失 |
| `Plan/Drift` | 上一条计划挪掉已执行步数后与新计划的不一致率，即"改主意"的程度 |
| `Plan/CommitLen` | 平均提交步数（模型有多敢连招） |
| `Plan/Confidence` | 平均决断程度（相对动作间隔） |
| `Plan/CertainTick` | 被选中的内部 tick 序号（模型平均"想"多久） |

两个要盯的：

- `Plan/Drift` 不该趋近 0 —— 环境在变，该改主意就得改；但它长期居高不下说明
  计划头没学到东西。
- `Plan/Confidence` 若长期贴着 **0.147**（13 个动作毫无偏好时的理论基线）、
  `Plan/CommitLen` 恒为 1，说明模型还分不清动作好坏，此时它退化成单步 DQN。
  这不是阈值设错，是策略还没学出来。

## 冻结帧控制

`game-node` 现在默认不冻结游戏进程，避免冻结渲染线程导致图形错误。默认方案是“非冻结帧节流”：按 `--game-fps` 控制 AI 采样和动作执行节奏，让游戏自然运行。

游戏端快捷键：

- `0`: 启动/暂停 AI 控制。启动后才会开始发状态和执行动作。
- `F10`: 运行时切换冻结控制。出现图形异常时再按一次关闭。
- `F9`: 安全退出并释放所有长按键。

如果显式加 `--freeze`，`game-node` 会按以下顺序运行：

1. 冻结游戏进程。
2. 在冻结状态下截取当前状态并发送给 `ai-node`。
3. 收到计划后注入第一步按键。
4. 放行游戏进程 `1 / --game-fps` 秒。
5. 再次冻结、读状态；若计划还有未提交的步数就继续第 3 步，否则回到第 2 步重新规划。

这样网络通信和模型推理耗时发生在游戏冻结期间，不会把通信延迟转化为游戏内操作延迟。
计划制还顺带降低了通信频率：commit=k 时每 k 帧才有一次往返。

如果暂时不想冻结进程，可加：

```bash
python main.py --mode game-node --decision-host <AI设备IP> --no-freeze
```

冻结 Windows 游戏进程通常需要以管理员权限启动终端。

## Linux / Proton / Wayland

游戏节点可在 Linux 上运行（推荐 Proton + Wayland，如 niri）。详见 [LINUX.md](LINUX.md)。

要点：

- 观测默认 **PipeWire portal 流** 或 **XWayland 窗口 MIT-SHM**，不再依赖 Win32 截图。
- 输入为 **uinput 虚拟键盘**，不创建鼠标设备、不捕获指针。
- 冻结为可选 `SIGSTOP`；默认仍用 `--game-fps` 节流。

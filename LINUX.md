# Linux / Proton / Wayland 运行说明

## 架构

```
ctm_planner.py / ctm_agent.py / ctm_components.py   # CTM 计划模型（默认 --arch ctm）
network_components.py                              # 旧 ProNet（--arch pro）

core/                     # 平台无关
  protocol.py             # TCP 编解码
  client.py               # DecisionClient（发命令、收计划）
  command_server.py       # CommandServer（收命令）
  game_loop.py            # GameClient 训练循环（按计划提交 k 步）
  observation.py / actions.py / interfaces.py

backends/
  windows/                # SendInput + MSS/BitBlt/DXCAM + NtSuspend
  linux/                  # uinput + PipeWire/XShm + SIGSTOP
```

根目录 `game_node.py` / `directkeys.py` 等为兼容薄封装。

| 子系统 | Windows | Linux |
|--------|---------|-------|
| 观测 | MSS / BitBlt / DXCAM | **PipeWire** 或 **XWayland MIT-SHM** |
| 输入 | `SendInput` | **uinput**（无鼠标、不抓指针） |
| 冻结 | `NtSuspendProcess` | `SIGSTOP`（可选） |
| 热键 | `keyboard` | `evdev` / 终端 |

目标：**niri / Wayland** + **Proton** 跑 `Yakuza6.exe`，不捕获鼠标、可观测窗口。

## 推荐架构

AI 节点（可同机或另一台）：

```bash
python main.py --mode ai-node --host 0.0.0.0 --port 15001 --model-profile large
# 旧 ProNet 对照：加 --arch pro
```

先跑一次延迟基准，确认单次决策进得了 `--game-fps` 的预算：

```bash
python tools/bench_ctm.py --fps 20
```

游戏节点（本机 + Proton 游戏已启动）：

```bash
pip install -r requirements-game-linux.txt
python main.py --mode game-node --decision-host 127.0.0.1 --decision-port 15001 --game-fps 20
```

不要默认开 `--freeze`（GPU 驱动在 SIGSTOP 下可能花屏）。

## 观测后端（比截图更快）

默认 Wayland 下 `DQN_CAPTURE_METHOD=pipewire`：

1. **PipeWire + xdg-desktop-portal ScreenCast**（`OpenPipeWireRemote` fd + node）  
   首次弹窗请选择 **游戏窗口**（不要选装饰条）；restore token 写入 `.pw_restore_token`。
2. **XWayland MIT-SHM**（`auto` 时若有 X11 窗）  
3. **MSS** 最后回退  

**不使用** niri `screenshot-window` 截图。

```bash
export DQN_CAPTURE_METHOD=pipewire   # 推荐
# 勿再依赖 DQN_PIPEWIRE_NODE 单独连接（无 portal fd 会失败）
```

## 输入（不捕获鼠标）

`directkeys_linux` 通过 `/dev/uinput` 创建**仅键盘**虚拟设备：

- 无相对/绝对鼠标轴，不会抢鼠标
- 不调用 `XGrabPointer` / 指针 lock

权限：

```bash
# 临时
sudo chmod 666 /dev/uinput
# 或加入 input 组 + udev 规则（见 requirements-game-linux.txt 注释）
```

**注意**：uinput 事件进的是系统输入栈。在 niri 上若游戏窗口无焦点，键可能到不了 Proton。可选：

- 训练时给游戏窗口焦点，人用另一台键鼠/不碰键盘
- 或用 `gamescope` 嵌游戏，并保证 gamescope 接收键盘
- 或配置 niri 对游戏窗口 `focus` 策略

本实现刻意**不**做鼠标捕获/指针锁定。

## Proton 启动示例

```bash
# Steam: 兼容性 → Proton；或
export STEAM_COMPAT_DATA_PATH=~/.steam/steam/steamapps/compatdata/<APPID>
export STEAM_COMPAT_CLIENT_INSTALL_PATH=~/.steam/steam
~/.steam/steam/steamapps/common/Proton\ 10.0/proton run <path/to/Yakuza6.exe>
```

进程名默认 `Yakuza6.exe`（Wine 下通常仍可见）。覆盖：

```bash
export DQN_PROCESS_NAME=Yakuza6.exe
```

窗口分辨率尽量与战斗 HUD 一致。血条坐标支持**自动定位**：

```bash
# 独立标定（进战斗有血条时）
python main.py --mode calibrate --capture pipewire

# game-node 运行中按 F8，或等待无效血量时自动搜索
# 关闭自动定位: export DQN_AUTO_LOCATE=0
```

`locations.json` 会写入 `frame_size`；分辨率变化时自动按比例缩放。

## gamescope（可选）

```bash
gamescope -w 1360 -h 768 -f false -- proton run Yakuza6.exe
```

观测可选 PipeWire 选 gamescope 输出；输入仍走 uinput。

## 控制台指令（Linux 不用全局热键）

在运行 `game-node` / `train` 的**前台终端**输入命令后回车：

| 指令 | 别名 | 作用 |
|------|------|------|
| `0` | `start` `pause` `toggle` | 启动/暂停 AI |
| `resume` | `r` `focus` `esc` | 聚焦游戏窗口 + 虚拟点击 + ESC |
| `f8` | `calibrate` `locate` | 自动定位血条 |
| `f10` | `freeze` | 切换进程冻结 |
| `f9` | `quit` `exit` `stop` | 安全退出 |
| `f5` | `save` | 保存模型（train 模式） |
| `help` | `?` | 显示帮助 |

`resume` 全程**不占用真实鼠标/键盘**：

1. `niri msg action focus-window`（合成器 API，不移动实体指针）  
2. 独立 **uinput 虚拟鼠标** 设备点击（与物理鼠标分离）  
3. 独立 **uinput 虚拟键盘** 发送 ESC  

训练默认也只用虚拟键盘；不会 `SetCursorPos` / xdotool 搬你的真鼠标。

必须前台运行（不要 nohup 且关掉 stdin），否则无法输入指令。

## 依赖摘要

```bash
pacman -S python-evdev gstreamer gst-plugins-base gst-plugin-pipewire \
          python-gobject python-dbus xdg-desktop-portal xdg-desktop-portal-gnome
pip install -r requirements-game-linux.txt
```

niri 通常配合 `xdg-desktop-portal-gnome` 或 gtk portal 做 ScreenCast。

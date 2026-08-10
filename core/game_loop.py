# -*- coding: utf-8 -*-
"""
GameClient：平台无关的客户端训练/对局循环。

依赖:
  - DecisionClient     与 AI 通信（编解码/命令）
  - GameObservation    观测与奖励
  - ActionController   动作执行
  - ProcessControl     可选冻结
  - HotkeyBackend      热键
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from config import GAME_CONFIG, REWARD_CONFIG
from core.actions import ActionController
from core.client import DecisionClient
from core.interfaces import PlatformBundle, ProcessControlBackend
from core.observation import GameObservation


class GameClient:
    """原 GameNode 的核心逻辑，平台细节由 PlatformBundle 注入。"""

    def __init__(
        self,
        platform: PlatformBundle,
        decision_host: str,
        decision_port: int = 15001,
        target_fps: float = 20.0,
        freeze_process: bool = False,
        input_settle_ms: float = 5.0,
    ):
        self.platform = platform
        self.target_fps = float(target_fps)
        self.freeze_process = freeze_process
        self.input_settle_seconds = max(float(input_settle_ms), 0.0) / 1000.0

        self.decision = DecisionClient(decision_host, decision_port)
        self.game = GameObservation(platform.capture, GAME_CONFIG["process_name"])
        self.action_controller = ActionController(platform.input)
        # 计划执行途中挨打超过这个幅度就丢弃剩余计划、立刻重规划
        self.plan_interrupt_damage = float(GAME_CONFIG.get("plan_interrupt_damage", 0.01))
        self.freezer: Optional[ProcessControlBackend] = None
        if freeze_process:
            self.freezer = platform.process_factory(
                GAME_CONFIG["process_name"], target_fps
            )

        self.is_running = True
        self.is_paused = True
        # 全黑画面（死亡/结算）下按回车的间隔
        self.revive_enter_interval = 0.35
        self.state_lock = threading.Lock()
        self._setup_hotkeys()

    # ---- 热键 / 控制 ----
    def _setup_hotkeys(self):
        hk = self.platform.hotkeys
        hk.add_hotkey("0", self._toggle_pause)
        hk.add_hotkey("f9", self._stop)
        hk.add_hotkey("f10", self._toggle_freeze)
        hk.add_hotkey("f8", self._calibrate_locations)
        hk.add_hotkey("resume", self._resume_game)
        print("\n" + "=" * 42)
        print(f"Game Client [{self.platform.name}]")
        if self.platform.name == "linux":
            print(" 控制方式: 终端输入指令后回车（无全局热键）")
            print("   0 / start / pause   启动或暂停 AI")
            print("   resume / r / focus  聚焦游戏+虚拟键鼠ESC（不占真实鼠标）")
            print("   f8 / calibrate      自动定位血条")
            print("   f10 / freeze        切换进程冻结")
            print("   f9 / quit / exit    安全退出")
            print("   help                显示帮助")
        else:
            print(" [0]   启动/暂停 AI 控制")
            print(" [F8]  自动定位血条坐标")
            print(" [F10] 切换冻结控制")
            print(" [F9]  安全退出")
            print(" 控制台也可输入: resume")
        if self.platform.notes:
            for line in self.platform.notes.strip().splitlines():
                print(f" {line}")
        print("=" * 42 + "\n")

    def _toggle_pause(self):
        with self.state_lock:
            self.is_paused = not self.is_paused
            paused = self.is_paused
        if paused:
            self.action_controller.force_cancel_all()
            if self.freezer:
                self.freezer.close(resume=True)
            print("\n>> Game Client 已暂停，已释放所有长按键")
        else:
            self.game.reset_reward_stats()
            print("\n>> Game Client 已启动 AI 控制")

    def _stop(self):
        with self.state_lock:
            self.is_running = False
            self.is_paused = True
        self.action_controller.force_cancel_all()
        if self.freezer:
            self.freezer.close(resume=True)
        print("\n>> Game Client 正在安全退出")

    def _toggle_freeze(self):
        with self.state_lock:
            self.freeze_process = not self.freeze_process
            enable = self.freeze_process
        if enable:
            self.freezer = self.platform.process_factory(
                GAME_CONFIG["process_name"], self.target_fps
            )
            print("\n>> 冻结控制已开启。若出现图形异常，按 F10 关闭。")
        else:
            if self.freezer:
                self.freezer.close(resume=True)
            self.freezer = None
            print("\n>> 冻结控制已关闭，改用非冻结帧节流。")

    def _resume_game(self):
        """合成器聚焦 + uinput 虚拟键鼠 ESC/点击（不占用真实键鼠）。"""
        print("\n>> resume：聚焦游戏 + 虚拟输入 ESC（不碰真实鼠标）…")
        try:
            result = self.platform.input.resume_game(
                process_name=GAME_CONFIG.get("process_name"),
                press_escape=True,
            )
            print(
                f">> resume 完成 ok={result.get('ok')} method={result.get('method')} "
                f"virt_click={result.get('clicked')} esc={result.get('esc')} "
                f"real_mouse={result.get('real_mouse', False)}"
            )
        except Exception as e:
            print(f">> resume 失败: {e}")
            try:
                self.platform.input.press_esc()
            except Exception:
                pass

    def _calibrate_locations(self):
        print("\n>> 手动触发血条轨道定位…")
        raw = self.game.get_raw_screen()
        if raw is None:
            print(">> 无法截取画面，定位取消")
            return
        located = self.game.auto_locate_health_bars(raw, save=True, force=True)
        if located is None:
            print(">> 定位失败：请进入有血条的战斗 HUD 后再试")
            return
        out = "debug_auto_locate.png"
        boss_hp, self_hp, _, _, _ = self.game.check_game_state(auto_locate=False)
        self.game.debug_health_regions(raw, show=False, save_path=out)
        print(f">> 定位后读数: {self.game.health_status()} | 预览已写 {out}")

    # ---- 步进辅助 ----
    def _is_invalid_health_transition(self, prev_self, prev_boss, curr_self, curr_boss):
        tolerance = REWARD_CONFIG.get("health_regen_tolerance", 1e-6)
        return curr_boss > prev_boss + tolerance or curr_self > prev_self + tolerance

    def _read_settled(self, attempts: int = 3, delay: float = 0.05):
        """
        读血量，并给"连续 N 帧为 0 才算死"的确认逻辑一点时间收敛。

        不这么做的话：玩家血量归零的**第一帧**返回 `zero-unconfirmed`(None)，
        主循环会当成"读不到"直接跳到等待分支，于是 `done` 永远不触发 ——
        学习端收不到终局样本，回合也永远不结算（实测跑了 1350 步、
        玩家死了两次，回合结束次数仍是 0）。
        """
        raw = None
        for i in range(attempts):
            boss, selfhp, _, _, raw = self.game.check_game_state()
            if boss is not None and selfhp is not None:
                return boss, selfhp, raw
            if i + 1 < attempts:
                time.sleep(delay)
        return None, None, raw

    def _advance_game(self):
        if self.freezer:
            self.freezer.run_one_frame()
        else:
            time.sleep(1.0 / max(self.target_fps, 1.0))

    def _wait_for_valid_health(self):
        """
        等待可信血条。期间若画面几乎全黑（玩家死亡/结算/读盘），
        持续按回车推进 —— 否则血条永远读不到，循环会一直空转。
        """
        last_log = 0.0
        last_enter = 0.0
        black_since = None
        was_locked = all(t.box for t in self.game.trackers.values())
        while self.is_running:
            if self.freezer:
                self.freezer.resume()
            boss_hp, self_hp, _, _, raw = self.game.check_game_state(auto_locate=True)

            # 刚锁定成功就把轨道存下来，下次启动直接复用
            now_locked = all(t.box for t in self.game.trackers.values())
            if now_locked and not was_locked:
                self.game.locations_frame_size = (
                    (raw.shape[1], raw.shape[0]) if raw is not None else None
                )
                self.game.save_locations()
                print(f">> 血条轨道已锁定并保存 | {self.game.health_status()}")
            was_locked = now_locked

            if boss_hp is not None and self_hp is not None and self_hp > 0:
                if self.freezer:
                    self.freezer.suspend()
                return boss_hp, self_hp, raw

            now = time.time()

            # 全黑画面 → 死亡/结算，狂按回车复活
            if raw is not None and self.game.is_black_screen(raw):
                if black_since is None:
                    black_since = now
                    print(">> 画面几乎全黑（判定死亡/结算），开始按回车推进…")
                if now - last_enter >= self.revive_enter_interval:
                    self.action_controller.press_enter()
                    last_enter = now
            elif black_since is not None:
                print(f">> 画面已恢复（黑屏持续 {now - black_since:.1f}s）")
                black_since = None

            if now - last_log > 5.0:
                shape = None if raw is None else f"{raw.shape[1]}x{raw.shape[0]}"
                dark = "" if raw is None else f" dark={self.game.dark_fraction(raw):.2f}"
                print(
                    f">> 等待有效血条中… frame={shape}{dark} | "
                    f"{self.game.health_status()}"
                )
                last_log = now
            time.sleep(0.2)

    def _resurrect_player(self, timeout=30.0):
        print(">> 玩家死亡，开始复活流程")
        self.action_controller.force_cancel_all()
        if self.freezer:
            self.freezer.resume()

        start_time = time.time()
        while self.is_running and time.time() - start_time < timeout:
            self.action_controller.press_enter()
            time.sleep(self.revive_enter_interval)
            boss_hp, self_hp, _, _, raw = self.game.check_game_state()
            if self_hp is not None and self_hp > 0.1:
                print(">> 玩家已复活，重新进入冻结控制")
                time.sleep(0.8)
                if self.freezer:
                    self.freezer.suspend()
                return boss_hp, self_hp, raw

        if self.freezer:
            self.freezer.suspend()
        print(">> 复活超时，转入等待（仍会按回车推进黑屏）")
        return self._wait_for_valid_health()

    # ---- 主循环 ----
    def run(self):
        self.decision.connect()
        print(f">> 已连接 Decision: {self.decision.host}:{self.decision.port}")
        print(
            f">> 平台={self.platform.name} | FPS={self.target_fps:.1f} | "
            f"冻结={self.freeze_process}"
        )
        print(">> 当前默认暂停。按 [0] 开始/暂停 AI 控制。")
        import os
        if os.environ.get("DQN_AUTO_START", "").lower() in ("1", "true", "yes"):
            self.is_paused = False
            self.game.reset_reward_stats()
            print(">> DQN_AUTO_START：已自动启动 AI 控制")

        try:
            boss_hp = self_hp = raw = state = None
            boss_hp_start = 0.0
            episode_start = time.time()
            step_count = 0
            episode_reward = 0.0
            damage_out = damage_in = 0.0
            plans_made = steps_committed = interrupts = 0

            while self.is_running:
                if self.is_paused:
                    state = None
                    time.sleep(0.1)
                    continue

                if state is None:
                    if self.freezer:
                        self.freezer.open()
                        self.freezer.suspend()
                    boss_hp, self_hp, raw = self._wait_for_valid_health()
                    state = self.game.get_state_from_frame(raw)
                    self.game.reset_reward_stats()
                    boss_hp_start = boss_hp
                    episode_start = time.time()
                    step_count = 0
                    episode_reward = 0.0
                    damage_out = damage_in = 0.0
                    plans_made = steps_committed = interrupts = 0

                plan, commit, confidence, decision_ms, _ = self.decision.request_plan(
                    state, boss_hp, self_hp
                )
                plans_made += 1
                steps_committed += commit
                done = False
                next_self_hp = self_hp

                # ---- 按计划连打 commit 步；中途出任何意外都 break 回去重规划 ----
                for slot in range(commit):
                    action = plan[slot]
                    self.action_controller.take_action(action)
                    if self.input_settle_seconds > 0:
                        time.sleep(self.input_settle_seconds)
                    self._advance_game()

                    next_boss_hp, next_self_hp, next_raw = self._read_settled()
                    if next_boss_hp is None or next_self_hp is None:
                        print(f">> 暂时无法读取血条，等待恢复 | {self.game.health_status()}")
                        boss_hp, self_hp, raw = self._wait_for_valid_health()
                        state = self.game.get_state_from_frame(raw)
                        break

                    if self._is_invalid_health_transition(
                        self_hp, boss_hp, next_self_hp, next_boss_hp
                    ):
                        print(
                            ">> 检测到血量回升垃圾帧，已跳过 | "
                            f"boss {boss_hp:.4f}->{next_boss_hp:.4f}, "
                            f"self {self_hp:.4f}->{next_self_hp:.4f}"
                        )
                        time.sleep(1.0 / max(self.target_fps, 1.0))
                        break

                    next_state = self.game.get_state_from_frame(next_raw)
                    reward = self.game.calculate_reward(
                        self_hp, boss_hp, next_self_hp, next_boss_hp, action
                    )
                    if reward is None:
                        print(">> 奖励函数拒绝垃圾帧，已跳过")
                        break
                    done = next_self_hp <= 0 or next_boss_hp <= 0

                    self.decision.send_transition({
                        "state": state,
                        "boss_health": boss_hp,
                        "self_health": self_hp,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "next_boss_health": next_boss_hp,
                        "next_self_health": next_self_hp,
                        "done": done,
                        # 只作诊断用：AI 节点存样本时不区分动作来自计划的第几格
                        "plan_slot": slot,
                    })

                    took = max(0.0, self_hp - next_self_hp)
                    state = next_state
                    damage_out += max(0.0, boss_hp - next_boss_hp)
                    damage_in += took
                    boss_hp = next_boss_hp
                    self_hp = next_self_hp
                    step_count += 1
                    episode_reward += reward

                    if step_count % 50 == 0:
                        elapsed = max(time.time() - episode_start, 1e-6)
                        print(
                            f"Step {step_count} | R={episode_reward:7.2f} | "
                            f"打出={damage_out:.3f} 挨打={damage_in:.3f} | "
                            f"self={self_hp:.3f} boss={boss_hp:.3f} | "
                            f"{step_count / elapsed:.1f}步/s 决策{decision_ms:.0f}ms | "
                            f"计划{plan} 提交{commit} 决断度{confidence:.2f} "
                            f"招式{self.game.action_variety()}种 "
                            f"均提交{steps_committed / max(plans_made, 1):.1f} "
                            f"中断{interrupts}"
                        )

                    if done:
                        break

                    # 挨打了：原来的计划是在没挨打的前提下定的，作废重来
                    if took > self.plan_interrupt_damage and slot + 1 < commit:
                        interrupts += 1
                        break

                if done:
                    total_damage = max(0.0, boss_hp_start - next_boss_hp)
                    duration = max(time.time() - episode_start, 1e-6)
                    print(
                        f">>> 回合结束 | Reward: {episode_reward:.2f} | "
                        f"Damage: {total_damage:.3f} | DPS: {total_damage / duration:.4f} | "
                        f"均提交{steps_committed / max(plans_made, 1):.1f} 中断{interrupts}"
                    )
                    self.action_controller.force_cancel_all()
                    self.decision.reset_hidden()
                    if next_self_hp <= 0:
                        boss_hp, self_hp, raw = self._resurrect_player()
                    else:
                        if self.freezer:
                            self.freezer.resume()
                        time.sleep(1.0)
                        if self.freezer:
                            self.freezer.suspend()
                        boss_hp, self_hp, raw = self._wait_for_valid_health()
                    state = self.game.get_state_from_frame(raw)
                    self.game.reset_reward_stats()
                    boss_hp_start = boss_hp
                    episode_start = time.time()
                    step_count = 0
                    episode_reward = 0.0
                    damage_out = damage_in = 0.0
                    plans_made = steps_committed = interrupts = 0
        finally:
            self.action_controller.force_cancel_all()
            if self.freezer:
                self.freezer.close(resume=True)
            self.platform.input.shutdown()
            self.platform.capture.release()
            self.platform.hotkeys.stop()
            self.decision.close()


# 兼容旧名
GameNode = GameClient

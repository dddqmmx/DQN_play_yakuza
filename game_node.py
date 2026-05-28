import time
import threading

import keyboard

from actions import ActionController
from config import GAME_CONFIG, REWARD_CONFIG
from distributed_protocol import connect_with_retry, recv_message, send_message
from game_interface import GameInterface
from game_process_control import GameProcessFreezer


class GameNode:
    def __init__(self, decision_host, decision_port=15001, target_fps=20.0, freeze_process=False,
                 input_settle_ms=5.0):
        self.decision_host = decision_host
        self.decision_port = decision_port
        self.target_fps = float(target_fps)
        self.freeze_process = freeze_process
        self.input_settle_seconds = max(float(input_settle_ms), 0.0) / 1000.0
        self.game = GameInterface()
        self.action_controller = ActionController(max_combo_slots=GAME_CONFIG.get('max_combo_slots', 5))
        self.freezer = GameProcessFreezer(GAME_CONFIG["process_name"], target_fps) if freeze_process else None
        self.sock = None
        self.is_running = True
        self.is_paused = True
        self.state_lock = threading.Lock()

        self._setup_hotkeys()

    def _setup_hotkeys(self):
        keyboard.add_hotkey('0', self._toggle_pause)
        keyboard.add_hotkey('f9', self._stop)
        keyboard.add_hotkey('f10', self._toggle_freeze)
        print("\n" + "=" * 34)
        print("Game Node 快捷键")
        print(" [0]   启动/暂停 AI 控制")
        print(" [F10] 切换冻结控制")
        print(" [F9]  安全退出")
        print("=" * 34 + "\n")

    def _toggle_pause(self):
        with self.state_lock:
            self.is_paused = not self.is_paused
            paused = self.is_paused
        if paused:
            self.action_controller.force_cancel_all()
            if self.freezer:
                self.freezer.close(resume=True)
            print("\n>> Game Node 已暂停，已释放所有长按键")
        else:
            self.game.reset_reward_stats()
            print("\n>> Game Node 已启动 AI 控制")

    def _stop(self):
        with self.state_lock:
            self.is_running = False
            self.is_paused = True
        self.action_controller.force_cancel_all()
        if self.freezer:
            self.freezer.close(resume=True)
        print("\n>> Game Node 正在安全退出")

    def _toggle_freeze(self):
        with self.state_lock:
            self.freeze_process = not self.freeze_process
            enable = self.freeze_process
        if enable:
            self.freezer = GameProcessFreezer(GAME_CONFIG["process_name"], self.target_fps)
            print("\n>> 冻结控制已开启。若出现图形异常，按 F10 关闭。")
        else:
            if self.freezer:
                self.freezer.close(resume=True)
            self.freezer = None
            print("\n>> 冻结控制已关闭，改用非冻结帧节流。")

    def _connect(self):
        if self.sock is None:
            self.sock = connect_with_retry(self.decision_host, self.decision_port)

    def _send_transition(self, transition):
        send_message(self.sock, {"type": "transition", **transition})
        recv_message(self.sock)

    def _request_action(self, state, boss_hp, self_hp, episode_done=False):
        send_message(self.sock, {
            "type": "action_request",
            "state": state,
            "boss_health": boss_hp,
            "self_health": self_hp,
            "episode_done": episode_done,
        })
        reply = recv_message(self.sock)
        if reply.get("type") != "action":
            raise RuntimeError(f"Decision 返回异常: {reply}")
        return int(reply["action"]), float(reply.get("decision_ms", 0.0))

    def _is_invalid_health_transition(self, prev_self, prev_boss, curr_self, curr_boss):
        tolerance = REWARD_CONFIG.get('health_regen_tolerance', 1e-6)
        return curr_boss > prev_boss + tolerance or curr_self > prev_self + tolerance

    def _advance_game(self):
        if self.freezer:
            self.freezer.run_one_frame()
        else:
            time.sleep(1.0 / max(self.target_fps, 1.0))

    def _wait_for_valid_health(self):
        while self.is_running:
            if self.freezer:
                self.freezer.resume()
            boss_hp, self_hp, _, _, raw = self.game.check_game_state()
            if boss_hp is not None and self_hp is not None and self_hp > 0:
                if self.freezer:
                    self.freezer.suspend()
                return boss_hp, self_hp, raw
            time.sleep(0.2)

    def _resurrect_player(self, timeout=30.0):
        print(">> 玩家死亡，开始复活流程")
        self.action_controller.force_cancel_all()
        if self.freezer:
            self.freezer.resume()

        start_time = time.time()
        while self.is_running and time.time() - start_time < timeout:
            self.action_controller.press_enter()
            time.sleep(0.25)
            boss_hp, self_hp, _, _, raw = self.game.check_game_state()
            if self_hp is not None and self_hp > 0.1:
                print(">> 玩家已复活，重新进入冻结控制")
                time.sleep(0.8)
                if self.freezer:
                    self.freezer.suspend()
                return boss_hp, self_hp, raw

        if self.freezer:
            self.freezer.suspend()
        print(">> 复活超时，继续等待有效血条")
        return self._wait_for_valid_health()

    def run(self):
        self._connect()
        print(f">> Game 节点连接到 Decision: {self.decision_host}:{self.decision_port}")
        print(f">> 游戏端目标帧率: {self.target_fps:.1f} FPS，冻结控制: {self.freeze_process}")

        print(">> 当前默认暂停。按 [0] 开始/暂停 AI 控制。")

        try:
            boss_hp = self_hp = raw = state = None
            boss_hp_start = 0.0
            episode_start = time.time()
            step_count = 0
            episode_reward = 0.0

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

                action, decision_ms = self._request_action(state, boss_hp, self_hp)
                self.action_controller.take_action(action)
                if self.input_settle_seconds > 0:
                    time.sleep(self.input_settle_seconds)
                self._advance_game()

                next_boss_hp, next_self_hp, _, _, next_raw = self.game.check_game_state()
                if next_boss_hp is None or next_self_hp is None:
                    print(">> 暂时无法读取血条，等待恢复")
                    boss_hp, self_hp, raw = self._wait_for_valid_health()
                    state = self.game.get_state_from_frame(raw)
                    continue

                if self._is_invalid_health_transition(self_hp, boss_hp, next_self_hp, next_boss_hp):
                    print(
                        ">> 检测到血量回升垃圾帧，已跳过 | "
                        f"boss {boss_hp:.4f}->{next_boss_hp:.4f}, "
                        f"self {self_hp:.4f}->{next_self_hp:.4f}"
                    )
                    time.sleep(1.0 / max(self.target_fps, 1.0))
                    continue

                next_state = self.game.get_state_from_frame(next_raw)
                reward = self.game.calculate_reward(self_hp, boss_hp, next_self_hp, next_boss_hp, action)
                if reward is None:
                    print(">> 奖励函数拒绝垃圾帧，已跳过")
                    continue
                done = next_self_hp <= 0 or next_boss_hp <= 0

                self._send_transition({
                    "state": state,
                    "boss_health": boss_hp,
                    "self_health": self_hp,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "next_boss_health": next_boss_hp,
                    "next_self_health": next_self_hp,
                    "done": done,
                })

                state = next_state
                boss_hp = next_boss_hp
                self_hp = next_self_hp
                step_count += 1
                episode_reward += reward

                if step_count % 50 == 0:
                    print(
                        f"Step: {step_count} | Reward: {episode_reward:.2f} | "
                        f"Decision: {decision_ms:.1f}ms"
                    )

                if done:
                    total_damage = max(0.0, boss_hp_start - next_boss_hp)
                    duration = max(time.time() - episode_start, 1e-6)
                    print(
                        f">>> 回合结束 | Reward: {episode_reward:.2f} | "
                        f"Damage: {total_damage:.3f} | DPS: {total_damage / duration:.4f}"
                    )
                    self.action_controller.force_cancel_all()
                    send_message(self.sock, {"type": "reset_hidden"})
                    recv_message(self.sock)
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
        finally:
            self.action_controller.force_cancel_all()
            if self.freezer:
                self.freezer.close(resume=True)
            if self.sock:
                self.sock.close()

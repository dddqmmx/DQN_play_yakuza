# -*- coding: utf-8 -*-
"""观测与奖励：平台无关。捕获通过 CaptureBackend 注入。"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config import FILE_PATHS, GAME_CONFIG, HEALTH_CONFIG, REWARD_CONFIG
from core.health_bar import (
    BOSS_TARGET,
    PLAYER_TARGET,
    BarBox,
    BarReading,
    BarTracker,
    dark_fraction,
    draw_debug,
    is_black_screen,
)
from core.interfaces import CaptureBackend


class GameObservation:
    """帧预处理、血条检测、奖励计算。"""

    def __init__(self, capture: CaptureBackend, process_name: str = None):
        self.capture = capture
        self.process_name = process_name or GAME_CONFIG["process_name"]
        self.window_size = GAME_CONFIG["window_size"]
        self.frame_size = GAME_CONFIG["frame_size"]
        self.frame_stack_size = GAME_CONFIG["frame_stack_size"]
        self.frame_stack = deque(maxlen=self.frame_stack_size)

        # 血条跟踪器：轨道锁定 + 每帧读填充长度（见 core/health_bar.py）
        self.trackers = {
            "player": BarTracker(PLAYER_TARGET),
            "boss": BarTracker(BOSS_TARGET),
        }
        self.last_readings: dict = {}
        self.locations_frame_size = None
        self._auto_locate_interval = float(
            os.environ.get("DQN_AUTO_LOCATE_INTERVAL", "2.5")
        )
        self.auto_locate_enabled = os.environ.get(
            "DQN_AUTO_LOCATE", "1"
        ).lower() not in ("0", "false", "no")

        # 连续读到 0 才认定死亡/击杀，避免单帧误判触发回合结束
        self.zero_confirm_frames = int(
            HEALTH_CONFIG.get("zero_confirm_frames", 2)
        )
        self._zero_streak = {"player": 0, "boss": 0}

        self.load_locations()
        self.location_lock = threading.Lock()

        self.combo_streak = 0
        self.last_damage_time = 0
        self.last_action = 0
        self.passive_streak = 0
        self.steps_since_last_grab = 0
        self.attack_types_in_combo = set()
        self.passive_time = 0.0
        self.time_since_last_grab = 0.0
        self.last_step_time = time.time()
        window = int(REWARD_CONFIG.get("monotony_window", 20))
        self.recent_actions = deque(maxlen=window)
        self.recent_damage = deque(maxlen=window)
        self.same_action_time = 0.0

        # ---- 极限热血模式帧自动采样 ----
        # 按 Q 后血条会被蓝->紫动画盖掉，`core/health_bar` 目前只有在能同时看到
        # 空区时才敢报数（见 heat-saturated）。缺的样本是"特效期 + boss 已掉血"，
        # 手动摆拍很难碰上，所以让训练自己攒：读数原因里带 heat 的帧直接存盘。
        self.heat_dump_dir = os.environ.get("DQN_HEAT_DUMP_DIR", "samples_heat_auto")
        self.heat_dump_max = int(os.environ.get("DQN_HEAT_DUMP_MAX", "400"))
        self.heat_dump_interval = float(os.environ.get("DQN_HEAT_DUMP_INTERVAL", "0.2"))
        self.heat_dump_enabled = self.heat_dump_max > 0
        self._heat_dumped = 0
        self._heat_last_dump = 0.0

    # ------------------------------------------------------------ 坐标持久化
    @property
    def locations(self) -> dict:
        """兼容旧接口：返回当前锁定的轨道框。"""
        out = {}
        for name, tr in self.trackers.items():
            key = "player" if name == "player" else "boss"
            out[key] = tr.box.as_pairs() if tr.box else [[0, 0], [0, 0]]
        return out

    def load_locations(self, filename=FILE_PATHS["locations"]):
        """
        读取上次锁定的轨道。旧格式（`detect_health` 时代那种整条屏幕宽的框）
        一律丢弃 —— 那些坐标本身就是错的，重新自动定位更快。
        """
        default_size = tuple(GAME_CONFIG["window_size"])
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except Exception:
            self.locations_frame_size = default_size
            return

        fs = data.get("frame_size")
        size = (int(fs[0]), int(fs[1])) if fs and len(fs) == 2 else default_size
        self.locations_frame_size = size
        if data.get("format") != "track-v2":
            print(">> locations.json 是旧格式（填充框而非轨道），忽略并重新定位")
            return
        for name, tr in self.trackers.items():
            tr.load_dict(data.get(name), size)
        locked = [n for n, t in self.trackers.items() if t.box]
        if locked:
            print(f">> 已载入锁定血条轨道: {', '.join(locked)}")

    def save_locations(self, filename=FILE_PATHS["locations"]):
        payload = {"format": "track-v2"}
        if self.locations_frame_size:
            payload["frame_size"] = list(self.locations_frame_size)
        for name, tr in self.trackers.items():
            d = tr.to_dict()
            if d:
                payload[name] = d
        with open(filename, "w") as f:
            json.dump(payload, f, indent=4)

    def get_raw_screen(self):
        return self.capture.grab(self.process_name)

    def apply_locations(self, located: dict, save: bool = True):
        """兼容旧接口：手工指定轨道框。"""
        with self.location_lock:
            for key in ("player", "boss"):
                if key not in located:
                    continue
                box = BarBox.from_pairs(located[key])
                if box.valid():
                    self.trackers[key].box = box
                    self.trackers[key].frame_size = tuple(
                        located.get("frame_size", self.locations_frame_size)
                    )
        if save:
            self.save_locations()

    def auto_locate_health_bars(self, screen=None, save: bool = True, force: bool = False):
        """
        搜索并锁定血条轨道。成功返回 dict，失败返回 None。
        实际锁定逻辑在 BarTracker 内（需连续多帧一致）。
        """
        if screen is None:
            screen = self.get_raw_screen()
        if screen is None:
            return None

        if force:
            for tr in self.trackers.values():
                tr.reset()

        found = {}
        for name, tr in self.trackers.items():
            box = tr.box
            if box is None:
                # force 时连喂几帧同一画面以满足 confirm_frames
                for _ in range(tr.confirm_frames if force else 1):
                    box = tr.locate(screen)
                    if box is not None:
                        break
            if box is not None:
                found[name] = box.as_pairs()

        if not found:
            return None
        self.locations_frame_size = (screen.shape[1], screen.shape[0])
        if save:
            self.save_locations()
        found["frame_size"] = list(self.locations_frame_size)
        print(
            ">> 血条轨道已锁定 | "
            + " ".join(
                f"{n}={t.box.x1}..{t.box.x2}@y{t.box.y1}"
                for n, t in self.trackers.items() if t.box
            )
        )
        return found

    def get_health_regions(self, screen):
        """兼容旧接口：返回两条轨道的图像切片（现已不参与血量计算）。"""
        regions = []
        for name in ("player", "boss"):
            b = self.trackers[name].box
            if b is None:
                regions.append(None)
            else:
                regions.append(screen[b.y1:b.y2, b.x1:b.x2])
        return regions[0], regions[1]

    def preprocess_frame(self, image):
        """
        返回 **uint8** 灰度帧（0~255），不在这里归一化。

        状态会成千上万份地躺在 replay buffer 和 TCP 包里：float32 下一份
        4x160x160 就是 400KB，20000 容量的 buffer 光状态就要十几 GB。存 uint8
        直接省 4 倍，归一化挪到网络 forward 的入口做（标准 DQN 做法）。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)

    def get_state(self):
        screen = self.get_raw_screen()
        return self.get_state_from_frame(screen)

    def get_state_from_frame(self, screen):
        processed = self.preprocess_frame(screen)
        if len(self.frame_stack) == 0:
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(processed)
        else:
            self.frame_stack.append(processed)
        return np.stack(self.frame_stack, axis=0)

    def read_health(self, screen, allow_locate: bool = True) -> dict:
        """
        读两条血条。返回 {"player": BarReading, "boss": BarReading}。
        `value is None` 表示本帧不可信，调用方必须跳过。
        """
        out = {}
        for name, tr in self.trackers.items():
            r = tr.read(screen, allow_locate=allow_locate)
            # 单帧读到 0 不算数：连续 N 帧才认定死亡/击杀
            if r.ok and r.value <= 0.0:
                self._zero_streak[name] += 1
                if self._zero_streak[name] < self.zero_confirm_frames:
                    r = BarReading(None, r.confidence, "zero-unconfirmed")
            elif r.ok:
                self._zero_streak[name] = 0
            out[name] = r
        self.last_readings = out
        self._maybe_dump_heat_frame(screen, out)
        return out

    def _maybe_dump_heat_frame(self, screen, readings: dict):
        """
        读数原因里带 `heat` 就把原始帧存下来，攒热血特效的回归样本。

        限流 + 配额，免得把磁盘写满：`DQN_HEAT_DUMP_MAX=0` 可完全关掉。
        文件名带上两条血条的读数和原因，事后 `tools/eval_health.py` 一看就知道
        哪些帧是"特效期 + 已掉血"——那正是判断覆盖动画有没有盖住空区的关键样本。
        """
        if not self.heat_dump_enabled or screen is None:
            return
        if self._heat_dumped >= self.heat_dump_max:
            return
        if not any("heat" in (r.reason or "") for r in readings.values()):
            return
        now = time.time()
        if now - self._heat_last_dump < self.heat_dump_interval:
            return

        try:
            os.makedirs(self.heat_dump_dir, exist_ok=True)
            def tag(name):
                r = readings[name]
                v = "none" if r.value is None else f"{r.value:.3f}"
                return f"{name[0]}{v}-{(r.reason or '').replace('/', '_')}"
            fname = f"{self._heat_dumped:04d}_{tag('player')}_{tag('boss')}.png"
            cv2.imwrite(os.path.join(self.heat_dump_dir, fname), screen)
            self._heat_dumped += 1
            self._heat_last_dump = now
            if self._heat_dumped in (1, 50, 100, 200, 400):
                print(f">> 已采集热血特效样本 {self._heat_dumped} 帧 -> {self.heat_dump_dir}/")
        except Exception as exc:
            print(f">> 采样热血帧失败（已关闭该功能）: {exc}")
            self.heat_dump_enabled = False

    def detect_health(self, region, is_boss=True):
        """
        兼容旧接口。**已废弃**：旧实现取"最右侧红像素"，任何偏红背景都会读成满血。
        血量请走 `read_health` / `check_game_state`。
        """
        raise NotImplementedError(
            "detect_health 已移除：请使用 read_health()/check_game_state()，"
            "它们返回 BarReading（value 可能为 None，表示该帧不可信）"
        )


    def reset_reward_stats(self):
        self.combo_streak = 0
        self.last_damage_time = 0
        self.passive_time = 0.0
        self.time_since_last_grab = 0.0
        self.attack_types_in_combo = set()
        self.last_step_time = time.time()
        # "一成不变"统计
        window = int(REWARD_CONFIG.get("monotony_window", 20))
        self.recent_actions = deque(maxlen=window)
        self.recent_damage = deque(maxlen=window)
        self.same_action_time = 0.0

    def action_variety(self) -> int:
        """最近窗口内出现过多少种不同动作。给日志看策略有没有僵死。"""
        return len(set(getattr(self, "recent_actions", ()) or ()))

    def _monotony_penalty(self, action, damage_dealt, is_aggressive, dt) -> float:
        """
        惩罚"一成不变的招式组合"，两条独立的项：

        1. **同一招连按且打不出伤害** —— 命中伤害立刻清零。
        2. **最近窗口里招式种类过少，且这套打法没打出伤害** —— 对所有动作生效
           （不只是进攻），否则"一直站着不动"会变成规避第 1 条的漏洞。

        两项都用伤害门控。理由是一样的：在格斗游戏里反复按同一招本来就可能是最优解，
        无条件罚重复会把策略推离最优，跟"damage 是唯一主奖励"直接冲突。只有
        "按了半天没效果"才该罚。量级刻意压在伤害奖励之下（一次命中约值 2.5~5 分，
        这里满额时约 -0.2/步）。
        """
        penalty = 0.0

        if damage_dealt > 0:
            self.same_action_time = 0.0
        elif is_aggressive and action == self.last_action:
            self.same_action_time += dt
            threshold = REWARD_CONFIG.get("repeat_no_damage_seconds", 0.4)
            if self.same_action_time > threshold:
                per_sec = REWARD_CONFIG.get("repeat_no_damage_penalty_per_sec", -1.0)
                penalty += (self.same_action_time - threshold) * per_sec * dt
        else:
            self.same_action_time = 0.0

        self.recent_actions.append(action)
        self.recent_damage.append(damage_dealt)
        if len(self.recent_actions) == self.recent_actions.maxlen:
            variety = len(set(self.recent_actions))
            min_variety = int(REWARD_CONFIG.get("monotony_min_variety", 3))
            damage_floor = REWARD_CONFIG.get("monotony_damage_floor", 0.01)
            # 这套单调打法确实在掉血 -> 不管它；打不动才罚
            if variety < min_variety and sum(self.recent_damage) <= damage_floor:
                per_sec = REWARD_CONFIG.get("monotony_penalty_per_sec", -0.6)
                penalty += (min_variety - variety) * per_sec * dt

        return penalty

    def calculate_reward(self, prev_self, prev_boss, curr_self, curr_boss, action):
        now = time.time()
        dt = now - self.last_step_time
        if dt > 1.0:
            dt = 1.0
        self.last_step_time = now

        regen_tolerance = REWARD_CONFIG.get("health_regen_tolerance", 1e-6)
        if curr_boss > prev_boss + regen_tolerance or curr_self > prev_self + regen_tolerance:
            return None

        damage_dealt = max(0.0, prev_boss - curr_boss)
        damage_taken = max(0.0, prev_self - curr_self)

        time_penalty_per_sec = REWARD_CONFIG.get("time_penalty", -0.01) * 20.0
        main_reward = damage_dealt * REWARD_CONFIG["damage_dealt_multiplier"]
        special_bonus = 0.0
        penalty = time_penalty_per_sec * dt

        is_attack = action in [1, 2]
        is_grab = action == 9
        is_aggressive = is_attack or is_grab
        is_passive = action in [0, 4, 5, 6, 8]

        if is_attack:
            special_bonus += REWARD_CONFIG.get("attack_action_bonus", 0.02)

        if is_grab:
            self.time_since_last_grab = 0.0
        else:
            self.time_since_last_grab += dt
            grab_threshold_sec = REWARD_CONFIG.get("grab_frequency_threshold", 100) / 20.0
            if self.time_since_last_grab > grab_threshold_sec:
                multiplier = self.time_since_last_grab - grab_threshold_sec
                penalty_per_sec = REWARD_CONFIG.get("grab_absence_penalty_base", -0.01) * 20.0
                penalty += multiplier * penalty_per_sec * dt

        if damage_dealt > 0:
            self.passive_time = 0.0
            if is_grab:
                special_bonus += REWARD_CONFIG.get("grab_success_bonus", 0.02)
            if REWARD_CONFIG["enable_combo_reward"]:
                if now - self.last_damage_time <= REWARD_CONFIG["combo_timeout"]:
                    self.combo_streak += 1
                    if action in [1, 2, 9]:
                        if action not in self.attack_types_in_combo and len(self.attack_types_in_combo) > 0:
                            if action == 9:
                                special_bonus += REWARD_CONFIG.get("grab_combo_bonus", 1.0)
                            else:
                                special_bonus += REWARD_CONFIG.get("combo_variety_bonus", 0.2)
                        self.attack_types_in_combo.add(action)
                else:
                    self.combo_streak = 1
                    self.attack_types_in_combo = {action} if action in [1, 2, 9] else set()
                self.last_damage_time = now
                if self.combo_streak > 1:
                    special_bonus += self.combo_streak * REWARD_CONFIG["combo_base_reward"]
        elif is_aggressive:
            self.passive_time = 0.0
        elif not is_passive:
            self.passive_time = 0.0

        if damage_taken > 0:
            penalty -= damage_taken * REWARD_CONFIG["damage_taken_multiplier"]
            self.combo_streak = 0
            self.attack_types_in_combo = set()

        if is_passive:
            self.passive_time += dt
            threshold_sec = REWARD_CONFIG.get("cowardice_threshold", 10) / 20.0
            if self.passive_time > threshold_sec:
                multiplier = self.passive_time - threshold_sec
                penalty_per_sec = REWARD_CONFIG.get("passive_action_penalty_base", -0.01) * 20.0
                penalty += multiplier * penalty_per_sec * dt

        if curr_self < REWARD_CONFIG["low_health_threshold"]:
            # 按秒计：其余时间类惩罚都乘了 dt，这里若按"每步"算，
            # 15fps 下会变成 -30/秒，直接盖过伤害奖励，训练必然发散。
            penalty += REWARD_CONFIG["low_health_penalty"] * dt

        if curr_boss <= 0 and prev_boss > 0:
            special_bonus += REWARD_CONFIG["boss_defeat_reward"]

        # 一成不变的招式（必须在覆盖 self.last_action 之前算）
        penalty += self._monotony_penalty(action, damage_dealt, is_aggressive, dt)

        self.last_action = action
        if damage_dealt > 0:
            bonus_cap = min(
                REWARD_CONFIG.get("special_bonus_step_cap", 3.0),
                max(
                    REWARD_CONFIG.get("special_bonus_cap_without_damage", 0.2),
                    main_reward * REWARD_CONFIG.get("special_bonus_damage_ratio", 0.25),
                ),
            )
        else:
            bonus_cap = REWARD_CONFIG.get("special_bonus_cap_without_damage", 0.2)
        return main_reward + min(special_bonus, bonus_cap) + penalty

    def check_game_state(self, auto_locate: bool = None):
        """
        返回 (boss_hp, self_hp, self_region, boss_region, raw_screen)。

        血量为 `None` 表示**该帧读不到**（HUD 不在 / 过场 / 尚未锁定），
        调用方必须跳过，**不能**当成 0。
        """
        raw_screen = self.get_raw_screen()
        if raw_screen is None:
            return None, None, None, None, None

        if auto_locate is None:
            auto_locate = self.auto_locate_enabled

        readings = self.read_health(raw_screen, allow_locate=auto_locate)
        self_hp = readings["player"].value
        boss_hp = readings["boss"].value
        self_region, boss_region = self.get_health_regions(raw_screen)
        return boss_hp, self_hp, self_region, boss_region, raw_screen

    def is_black_screen(self, frame) -> bool:
        """画面是否几乎全黑：玩家死亡/结算/读盘，需要按回车推进。"""
        return is_black_screen(frame)

    def dark_fraction(self, frame) -> float:
        return dark_fraction(frame)

    def health_status(self) -> str:
        """给日志用的一行读数说明。"""
        parts = []
        for name in ("player", "boss"):
            r = self.last_readings.get(name)
            tr = self.trackers[name]
            box = f"x{tr.box.x1}..{tr.box.x2}" if tr.box else "未锁定"
            if r is None:
                parts.append(f"{name}=? [{box}]")
            elif r.ok:
                parts.append(f"{name}={r.value:.3f} [{box}]")
            else:
                parts.append(f"{name}=None({r.reason}) [{box}]")
        return " | ".join(parts)

    def debug_health_regions(self, raw_screen, show: bool = True, save_path: str = None):
        debug_img = draw_debug(raw_screen, self.trackers, self.last_readings)
        if save_path:
            cv2.imwrite(save_path, debug_img)
        if show:
            try:
                cv2.imshow("DEBUG", debug_img)
                cv2.waitKey(1)
            except Exception:
                pass
        return debug_img


# 兼容旧名
GameInterface = GameObservation

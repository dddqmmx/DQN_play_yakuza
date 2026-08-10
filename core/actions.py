# -*- coding: utf-8 -*-
"""
动作控制器：依赖 InputBackend，与平台解耦。

只有 13 个原子动作。这里**没有连招**——预设连招（原先由 ComboManager 从
combos.json 读出来、挂在动作 13~17 当宏用）已经整条删掉：

  - 那套宏在执行期间会用 `is_executing_combo` 把后续所有决策直接丢弃，
    模型在整段连招里对被打断/被抓/血量骤降完全无法反应；
  - 更根本的问题是连招不是策略的一部分，模型只是在"挑一个录好的宏"。

现在连招由模型自己规划：CTMPlannerNet 一次输出一条动作序列，GameClient 按
确定度提交前 k 步，挨打立刻中断重规划。所以这里只需要老老实实按键。
"""
from __future__ import annotations

import threading

from core.interfaces import InputBackend


class ActionController:
    def __init__(self, input_backend: InputBackend):
        self.keys = input_backend

        self.base_action_map = {
            0: self._no_action,
            1: self.keys.weak_attack,
            2: self.keys.strong_attack,
            3: self.keys.start_forward,
            4: self.keys.start_back,
            5: self.keys.start_left,
            6: self.keys.start_right,
            7: self.keys.dodge,
            8: self.keys.start_defense,
            9: self.keys.grab,
            10: self.keys.striking_pose_start,
            11: self.keys.striking_pose_cancel,
            12: self.keys.extrem_heat_mode,
        }
        self.stop_map = {
            3: self.keys.stop_forward,
            4: self.keys.stop_back,
            5: self.keys.stop_left,
            6: self.keys.stop_right,
            8: self.keys.stop_defense,
        }
        self.is_striking = False
        self.active_long_press = None

    def _no_action(self):
        pass

    def take_action(self, action_id):
        """执行一个原子动作。计划里的每一步都走这里。"""
        if action_id in self.base_action_map:
            self._handle_base_action(action_id)

    def _handle_base_action(self, action):
        # 换动作前先松掉上一个长按键（移动/防御），否则会一直按着
        if self.active_long_press is not None and self.active_long_press != action:
            if self.active_long_press in self.stop_map:
                self.stop_map[self.active_long_press]()
            self.active_long_press = None

        if action == 10:
            self.is_striking = True
        elif action == 11:
            self.is_striking = False

        if action in self.stop_map:
            # 长按类：按下就保持，直到下一个不同动作把它松开
            if self.active_long_press != action:
                self.active_long_press = action
                self.base_action_map[action]()
        elif action != 0:
            # 点按类（攻击/闪避/抓取…）：后端里带 sleep，扔线程里免得堵住决策循环
            thread = threading.Thread(target=self.base_action_map[action], daemon=True)
            thread.start()

    def press_enter(self):
        self.keys.press_enter()

    def get_action_count(self):
        return len(self.base_action_map)

    def is_currently_striking(self):
        return self.is_striking

    def force_cancel_all(self):
        if self.active_long_press in self.stop_map:
            self.stop_map[self.active_long_press]()
        self.active_long_press = None
        if self.is_striking:
            self.keys.striking_pose_cancel()
            self.is_striking = False

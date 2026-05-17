import threading
import time
import json
import os
import directkeys
import queue


class ComboManager:
    """连招管理器：发现、保存和加载连招，支持异步保存"""

    def __init__(self, combo_file="combos.json", max_combo_slots=5):
        self.combo_file = combo_file
        self.max_combo_slots = max_combo_slots
        self.combos = self.load_combos()
        self.current_sequence = []
        self.max_sequence_len = 5
        
        # 异步保存队列
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # 选出的最优连招槽位，用于模型调用
        self.combo_slots = []
        self._update_combo_slots()

    def load_combos(self):
        if os.path.exists(self.combo_file):
            try:
                with open(self.combo_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载连招文件失败: {e}")
                return {}
        return {}

    def _save_worker(self):
        """后台保存线程，防止 I/O 阻塞主循环"""
        while True:
            try:
                data = self.save_queue.get()
                if data is None: break
                
                with open(self.combo_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                # print(f">> 连招已异步保存至 {self.combo_file}")
                self.save_queue.task_done()
            except Exception as e:
                print(f"异步保存连招失败: {e}")
            time.sleep(1) # 限制写入频率

    def save_combos(self):
        """将保存任务放入队列"""
        # 只在有变化时放入队列，且队列中只保留最新的
        if self.save_queue.empty():
            self.save_queue.put(self.combos.copy())

    def record_action(self, action):
        """记录动作序列"""
        if action in [1, 2, 7, 9]: # 只记录攻击、闪避、抓取等瞬间动作
            self.current_sequence.append(action)
            if len(self.current_sequence) > self.max_sequence_len:
                self.current_sequence.pop(0)

    def discover_combo(self, reward):
        """如果奖励高，则将当前序列保存为连招"""
        if reward > 10.0 and len(self.current_sequence) >= 2:
            combo_key = ",".join(map(str, self.current_sequence))
            changed = False
            if combo_key not in self.combos:
                self.combos[combo_key] = {
                    "actions": list(self.current_sequence),
                    "success_count": 1,
                    "avg_reward": reward,
                    "last_seen": time.time()
                }
                print(f">> 发现新连招: {combo_key} (奖励: {reward:.1f})")
                changed = True
            else:
                # 更新统计
                c = self.combos[combo_key]
                c["success_count"] += 1
                c["avg_reward"] = (c["avg_reward"] * 0.9) + (reward * 0.1)
                c["last_seen"] = time.time()
                # 如果奖励显著提升，也标记为改变
                if reward > c["avg_reward"]: changed = True
            
            if changed:
                self.save_combos()
                self._update_combo_slots()

    def _update_combo_slots(self):
        """从已发现的连招中挑选最优的几个放入槽位供模型使用"""
        if not self.combos:
            self.combo_slots = []
            return
            
        # 按照 平均奖励 * 成功次数 排序，选出最具价值的连招
        sorted_combos = sorted(
            self.combos.values(), 
            key=lambda x: x['avg_reward'] * min(x['success_count'], 10), 
            reverse=True
        )
        self.combo_slots = [c['actions'] for c in sorted_combos[:self.max_combo_slots]]

    def get_combo_from_slot(self, slot_idx):
        """获取槽位中的连招"""
        if slot_idx < len(self.combo_slots):
            return self.combo_slots[slot_idx]
        return None


class ActionController:
    """游戏动作控制器 (支持长按与连招)"""

    def __init__(self, max_combo_slots=5):
        self.combo_manager = ComboManager(max_combo_slots=max_combo_slots)
        self.max_combo_slots = max_combo_slots
        
        # 基础动作映射
        self.base_action_map = {
            0: self._no_action,
            1: directkeys.weak_attack,
            2: directkeys.strong_attack,
            3: directkeys.start_forward,
            4: directkeys.start_back,
            5: directkeys.start_left,
            6: directkeys.start_right,
            7: directkeys.dodge,
            8: directkeys.start_defense,
            9: directkeys.grab,
            10: directkeys.striking_pose_start,
            11: directkeys.striking_pose_cancel,
            12: directkeys.extrem_heat_mode
        }

        # 停止函数映射 (用于长按释放)
        self.stop_map = {
            3: directkeys.stop_forward,
            4: directkeys.stop_back,
            5: directkeys.stop_left,
            6: directkeys.stop_right,
            8: directkeys.stop_defense
        }

        self.is_striking = False
        self.active_long_press = None # 当前正在长按的动作ID
        self.is_executing_combo = False

    def _no_action(self):
        pass

    def take_action(self, action_id):
        """执行动作，支持长按逻辑和连招槽位"""
        # 如果正在执行连招，忽略普通指令 (或者可以设计为打断)
        if self.is_executing_combo and action_id < len(self.base_action_map):
            return

        # 1. 处理基础动作
        if action_id < len(self.base_action_map):
            self._handle_base_action(action_id)
        # 2. 处理连招槽位 (例如 13, 14, 15, 16, 17)
        else:
            combo_idx = action_id - len(self.base_action_map)
            combo_actions = self.combo_manager.get_combo_from_slot(combo_idx)
            if combo_actions:
                self.execute_combo(combo_actions)

    def _handle_base_action(self, action):
        # 处理长按释放
        if self.active_long_press is not None and self.active_long_press != action:
            if self.active_long_press in self.stop_map:
                self.stop_map[self.active_long_press]()
            self.active_long_press = None

        # 记录动作用于连招发现
        self.combo_manager.record_action(action)

        # 执行新动作
        if action in self.base_action_map:
            # 管理蓄力状态
            if action == 10: self.is_striking = True
            elif action == 11: self.is_striking = False

            # 如果是移动或防御，设为当前长按动作
            if action in [3, 4, 5, 6, 8]:
                if self.active_long_press != action:
                    self.active_long_press = action
                    self.base_action_map[action]() # 开始按住
            else:
                # 瞬时动作使用线程执行
                if action != 0:
                    thread = threading.Thread(target=self.base_action_map[action])
                    thread.daemon = True
                    thread.start()

    def execute_combo(self, actions):
        """执行一个动作序列"""
        if self.is_executing_combo: return
        
        def _run():
            self.is_executing_combo = True
            try:
                for act in actions:
                    self._handle_base_action(act)
                    time.sleep(0.2)
            finally:
                self.is_executing_combo = False
                
        thread = threading.Thread(target=_run)
        thread.daemon = True
        thread.start()

    def press_enter(self):
        directkeys.press_enter()

    def get_action_count(self):
        return len(self.base_action_map) + self.max_combo_slots

    def is_currently_striking(self):
        return self.is_striking

    def force_cancel_all(self):
        """强制停止所有动作 (如回合结束时)"""
        if self.active_long_press in self.stop_map:
            self.stop_map[self.active_long_press]()
        self.active_long_press = None
        if self.is_striking:
            directkeys.striking_pose_cancel()
            self.is_striking = False
        self.is_executing_combo = False

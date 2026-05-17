import json
import threading
import time
from collections import deque
import cv2
import numpy as np
from grabscreen import grab_screen_by_process_name
from config import GAME_CONFIG, HEALTH_CONFIG, REWARD_CONFIG, FILE_PATHS


class GameInterface:
    """游戏交互接口 (优化版)"""

    def __init__(self, process_name=GAME_CONFIG['process_name']):
        self.process_name = process_name
        self.window_size = GAME_CONFIG['window_size']
        self.frame_size = GAME_CONFIG['frame_size']
        self.frame_stack_size = GAME_CONFIG['frame_stack_size']

        # 帧堆叠缓存 (使用 grayscale 减少 3x 内存占用和计算量)
        self.frame_stack = deque(maxlen=self.frame_stack_size)

        # 血条区域坐标
        self.locations = {
            'player': [[0, 0], [0, 0]],
            'boss': [[0, 0], [0, 0]]
        }
        self.load_locations()
        self.location_lock = threading.Lock()

        # 预先计算掩码所需的阈值
        self.boss_low = np.array([
            HEALTH_CONFIG['boss_rgb_thresholds']['red_range'][0],
            HEALTH_CONFIG['boss_rgb_thresholds']['green_range'][0],
            HEALTH_CONFIG['boss_rgb_thresholds']['blue_range'][0]
        ])
        self.boss_high = np.array([
            HEALTH_CONFIG['boss_rgb_thresholds']['red_range'][1],
            HEALTH_CONFIG['boss_rgb_thresholds']['green_range'][1],
            HEALTH_CONFIG['boss_rgb_thresholds']['blue_range'][1]
        ])
        
        self.player_low = np.array([
            HEALTH_CONFIG['player_rgb_thresholds']['red_range'][0],
            HEALTH_CONFIG['player_rgb_thresholds']['green_range'][0],
            HEALTH_CONFIG['player_rgb_thresholds']['blue_range'][0]
        ])
        self.player_high = np.array([
            HEALTH_CONFIG['player_rgb_thresholds']['red_range'][1],
            HEALTH_CONFIG['player_rgb_thresholds']['green_range'][1],
            HEALTH_CONFIG['player_rgb_thresholds']['blue_range'][1]
        ])

        # 奖励计算相关
        self.combo_streak = 0
        self.last_damage_time = 0
        self.last_action = 0
        self.passive_streak = 0 # 连续消极动作计数
        self.steps_since_last_grab = 0 # 距离上次抓取经过的步数
        self.attack_types_in_combo = set()

    def load_locations(self, filename=FILE_PATHS['locations']):
        try:
            with open(filename, 'r') as f:
                self.locations = json.load(f)
        except Exception:
            pass

    def save_locations(self, filename=FILE_PATHS['locations']):
        with open(filename, 'w') as f:
            json.dump(self.locations, f, indent=4)

    def get_raw_screen(self):
        return grab_screen_by_process_name(self.process_name)

    def get_health_regions(self, screen):
        with self.location_lock:
            (x1, y1), (x2, y2) = self.locations['player']
            self_region = screen[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
            (x3, y3), (x4, y4) = self.locations['boss']
            boss_region = screen[y3:y4, x3:x4] if y4 > y3 and x4 > x3 else None
        return self_region, boss_region

    def preprocess_frame(self, image):
        """预处理单帧图像: 缩放 + 灰度化"""
        # BGR -> Gray (更轻量)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        # Normalize
        return (resized / 255.0).astype(np.float32)

    def get_state(self):
        """获取状态: 4帧灰度堆叠 (Channel=4)"""
        screen = self.get_raw_screen()
        return self.get_state_from_frame(screen)

    def get_state_from_frame(self, screen):
        """优化：从已有的帧中获取状态，避免重复截屏"""
        processed = self.preprocess_frame(screen)

        if len(self.frame_stack) == 0:
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(processed)
        else:
            self.frame_stack.append(processed)

        # shape: (4, 224, 224)
        return np.stack(self.frame_stack, axis=0)

    def detect_health(self, region, is_boss=True):
        """优化后的血条检测"""
        if region is None or region.size == 0:
            return 0.0

        # BGR -> RGB
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # 使用 numpy 向量化操作代替 bitwise_and
        low = self.boss_low if is_boss else self.player_low
        high = self.boss_high if is_boss else self.player_high
        
        mask = cv2.inRange(rgb, low, high)
        
        # 统计掩码中白色像素的比例
        health_pixels_per_col = np.any(mask > 0, axis=0)
        if not np.any(health_pixels_per_col):
            return 0.0
            
        # 找到最右边的血条像素位置
        indices = np.where(health_pixels_per_col)[0]
        if len(indices) == 0:
            return 0.0
        
        # 血量 = 最后一个像素位置 / 总宽度
        health = (indices[-1] + 1) / region.shape[1]
        return float(np.clip(health, 0.0, 1.0))

    def reset_reward_stats(self):
        """每个回合开始时重置奖励相关的状态"""
        self.combo_streak = 0
        self.last_damage_time = 0
        self.passive_time = 0.0 # 修改为按时间(秒)计算的消极累计
        self.time_since_last_grab = 0.0 # 修改为按时间(秒)计算的未抓取累计
        self.attack_types_in_combo = set()
        self.last_step_time = time.time()

    def calculate_reward(self, prev_self, prev_boss, curr_self, curr_boss, action):
        now = time.time()
        dt = now - self.last_step_time
        # 防止暂停或卡顿时产生过大的时间增量
        if dt > 1.0: dt = 1.0 
        self.last_step_time = now

        damage_dealt = max(0.0, prev_boss - curr_boss)
        damage_taken = max(0.0, prev_self - curr_self)
        
        # 将原本每步扣除的时间惩罚改为每秒扣除，假设原来约 20FPS (0.05s/步)，这里按比例放缩
        time_penalty_per_sec = REWARD_CONFIG.get('time_penalty', -0.01) * 20.0
        main_reward = damage_dealt * REWARD_CONFIG['damage_dealt_multiplier']
        special_bonus = 0.0
        penalty = time_penalty_per_sec * dt

        # 1. 动作倾向分析
        is_attack = action in [1, 2, 9]
        is_passive = action in [0, 4, 5, 6, 8] # 无动作, 后退, 左右, 防御
        is_grab = (action == 9)

        # 2. 基础攻击动作奖励 (鼓励尝试进攻)
        if is_attack:
            special_bonus += REWARD_CONFIG.get('attack_action_bonus', 0.02)
            
        # 2.1 抓取动作引导
        if is_grab:
            self.time_since_last_grab = 0.0
            special_bonus += REWARD_CONFIG.get('grab_action_bonus', 0.1)
        else:
            self.time_since_last_grab += dt
            # 假设之前阈值为 100步 (~5秒)
            grab_threshold_sec = REWARD_CONFIG.get('grab_frequency_threshold', 100) / 20.0
            if self.time_since_last_grab > grab_threshold_sec:
                # 阶梯惩罚: 长期不使用抓取扣分，按超出的秒数累加
                multiplier = self.time_since_last_grab - grab_threshold_sec
                penalty_per_sec = REWARD_CONFIG.get('grab_absence_penalty_base', -0.01) * 20.0
                penalty += multiplier * penalty_per_sec * dt

        # 3. 伤害奖励与连击逻辑
        if damage_dealt > 0:
            self.passive_time = 0.0 # 造成伤害重置消极时间
            
            if REWARD_CONFIG['enable_combo_reward']:
                if now - self.last_damage_time <= REWARD_CONFIG['combo_timeout']:
                    self.combo_streak += 1
                    
                    # 连招多样性奖励: 如果在连击中切换了攻击类型
                    if action in [1, 2, 9]:
                        if action not in self.attack_types_in_combo and len(self.attack_types_in_combo) > 0:
                            if action == 9:
                                special_bonus += REWARD_CONFIG.get('grab_combo_bonus', 1.0)
                            else:
                                special_bonus += REWARD_CONFIG.get('combo_variety_bonus', 0.2)
                        self.attack_types_in_combo.add(action)
                else:
                    self.combo_streak = 1
                    self.attack_types_in_combo = {action} if action in [1, 2, 9] else set()
                
                self.last_damage_time = now
                if self.combo_streak > 1:
                    special_bonus += self.combo_streak * REWARD_CONFIG['combo_base_reward']
        elif is_attack:
            # 攻击但没打中，也重置消极计数
            self.passive_time = 0.0
        elif not is_passive:
            # 前进、闪避等积极动作重置消极计数
            self.passive_time = 0.0

        # 4. 受伤惩罚
        if damage_taken > 0:
            penalty -= damage_taken * REWARD_CONFIG['damage_taken_multiplier']
            # 被打断连招重置
            self.combo_streak = 0
            self.attack_types_in_combo = set()
            
        # 5. 消极/逃跑行为惩罚 (阶梯式)
        if is_passive:
            self.passive_time += dt
            # 假设之前阈值为 10步 (~0.5秒)
            threshold_sec = REWARD_CONFIG.get('cowardice_threshold', 10) / 20.0
            if self.passive_time > threshold_sec:
                # 阶梯惩罚: 按超出的秒数累加惩罚
                multiplier = self.passive_time - threshold_sec
                penalty_per_sec = REWARD_CONFIG.get('passive_action_penalty_base', -0.01) * 20.0
                penalty += multiplier * penalty_per_sec * dt

        # 6. 删除无条件防御奖励漏洞
        # (如果在防御/闪避时没受伤害，原本给予微量奖励，但这会导致疯狂刷闪避/防御的“骗保”行为)
        # 防御/闪避的真正收益是不扣血(避免 damage_taken 的巨大惩罚)

        # 7. 血量状态
        if curr_self < REWARD_CONFIG['low_health_threshold']:
            penalty += REWARD_CONFIG['low_health_penalty']

        # 8. 最终胜利
        if curr_boss <= 0 and prev_boss > 0:
            special_bonus += REWARD_CONFIG['boss_defeat_reward']

        self.last_action = action
        if damage_dealt > 0:
            bonus_cap = min(
                REWARD_CONFIG.get('special_bonus_step_cap', 3.0),
                max(
                    REWARD_CONFIG.get('special_bonus_cap_without_damage', 0.2),
                    main_reward * REWARD_CONFIG.get('special_bonus_damage_ratio', 0.25)
                )
            )
        else:
            bonus_cap = REWARD_CONFIG.get('special_bonus_cap_without_damage', 0.2)
        reward = main_reward + min(special_bonus, bonus_cap) + penalty
        return reward

    def check_game_state(self):
        raw_screen = self.get_raw_screen()
        if raw_screen is None:
            return None, None, None, None, None
        self_region, boss_region = self.get_health_regions(raw_screen)
        if self_region is None or boss_region is None:
            return None, None, None, None, raw_screen

        boss_hp = self.detect_health(boss_region, is_boss=True)
        self_hp = self.detect_health(self_region, is_boss=False)
        return boss_hp, self_hp, self_region, boss_region, raw_screen

    def debug_health_regions(self, raw_screen):
        debug_img = raw_screen.copy()
        cv2.rectangle(debug_img, tuple(self.locations['player'][0]), tuple(self.locations['player'][1]), (0, 255, 0), 2)
        cv2.rectangle(debug_img, tuple(self.locations['boss'][0]), tuple(self.locations['boss'][1]), (0, 0, 255), 2)
        cv2.imshow('DEBUG', debug_img)
        cv2.waitKey(1)

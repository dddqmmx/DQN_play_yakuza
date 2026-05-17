import time
import keyboard
import threading
import queue
import json
import os
import numpy as np
from config import TRAINING_CONFIG, REWARD_CONFIG, GAME_CONFIG
from actions import ActionController
from game_interface import GameInterface
from dqn_agent import DQNAgent


class DamageStats:
    """伤害统计与评估系统：防止消极怠工与骗保行为"""

    def __init__(self, filename="damage_stats.json", window_size=50):
        self.filename = filename
        self.window_size = window_size
        self.history = []
        self.max_avg_damage = 0.0
        self.consecutive_low_damage = 0
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])[-self.window_size:]
                    self.max_avg_damage = data.get('max_avg_damage', 0.0)
            except Exception as e:
                print(f">> 加载伤害统计失败: {e}")

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump({
                    'history': self.history,
                    'max_avg_damage': self.max_avg_damage
                }, f)
        except Exception as e:
            print(f">> 保存伤害统计失败: {e}")

    def update_and_evaluate(self, episode_damage, episode_duration):
        """更新统计并返回表现奖励/惩罚"""
        dps = episode_damage / episode_duration if episode_duration > 0 else 0.0
        self.history.append(dps)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        current_avg = np.mean(self.history)
        # 只有在样本足够时才更新最高平均值，防止初期波动剧烈
        if len(self.history) >= 10:
            self.max_avg_damage = max(self.max_avg_damage, current_avg)
        
        # 计算基准线 (防止骗保：基准线不低于历史最高平均的 75%)
        baseline = max(current_avg, self.max_avg_damage * 0.75)
        
        performance_reward = 0.0
        status_msg = ""

        # 1. 严重低于平均水平 (消极惩罚)
        if dps < baseline * 0.6:
            self.consecutive_low_damage += 1
            # 只有连续出现低伤害才重罚，防止单次意外（如开局被秒）误伤
            if self.consecutive_low_damage >= 2:
                performance_reward = -10.0 * min(self.consecutive_low_damage, 3)
                status_msg = f"检测到消极行为 (DPS: {dps:.4f} < 均值: {baseline:.4f})，触发阶梯惩罚: {performance_reward}"
        else:
            self.consecutive_low_damage = 0
        
            # 2. 远高于平均水平 (表现奖励)
            if dps > baseline * 1.25:
                performance_reward = (dps - baseline) * 20.0
                status_msg = f"表现卓越！(DPS: {dps:.4f} > 均值: {baseline:.4f})，获得额外奖励: {performance_reward:.2f}"
                
            # 3. 停滞不前惩罚 (长期维持在低位波动)
            elif len(self.history) >= 15:
                std = np.std(self.history[-10:])
                if std < 0.05 and current_avg < self.max_avg_damage * 0.8:
                    performance_reward = -5.0
                    status_msg = "检测到训练停滞 (DPS长期无波动且低于历史最佳)，给予刺激性惩罚: -5.0"

        self.save()
        return performance_reward, status_msg


class TrainingManager:
    """极速版训练管理器：支持异步训练与优化截屏"""

    def __init__(self):
        self.action_controller = ActionController(max_combo_slots=GAME_CONFIG.get('max_combo_slots', 5))
        self.game = GameInterface()
        self.agent = DQNAgent(self.action_controller.get_action_count())
        self.damage_stats = DamageStats()

        # 状态控制
        self.is_paused = True
        self.is_running = True
        self.episode_count = 0
        self.total_reward = 0
        
        # 异步训练控制
        self.train_thread = threading.Thread(target=self._async_train_worker, daemon=True)
        self.train_thread.start()

        # 性能追踪
        self.last_step_time = time.time()
        self.fps_list = []

        self._setup_hotkeys()

    def _setup_hotkeys(self):
        keyboard.add_hotkey('f5', self._save_checkpoint)
        keyboard.add_hotkey('0', self._toggle_pause)
        keyboard.add_hotkey('f9', self._stop_training)
        print("\n" + "="*30)
        print("控制快捷键:")
        print(" [0] - 开始/暂停训练")
        print(" [F5] - 手动保存模型")
        print(" [F9] - 安全退出程序")
        print("="*30 + "\n")

    def _async_train_worker(self):
        """后台训练线程"""
        print(">> 异步训练线程已启动")
        while self.is_running:
            try:
                # 即使没有新数据，也持续从经验池学习
                if not self.is_paused and self.agent.memory.is_ready(self.agent.batch_size):
                    self.agent.update_model()
                    time.sleep(0.001) # 防止死循环过快占用所有 CPU
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"训练线程错误: {e}")
                time.sleep(1)

    def _save_checkpoint(self):
        with self.agent.net_lock:
            self.agent.save_checkpoint()
        self.game.save_locations()
        print(">> 检查点已手动保存")

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        status = "▶️ 已继续" if not self.is_paused else "⏸️ 已暂停"
        print(f"\n{status}")
        if not self.is_paused:
            self.last_step_time = time.time()

    def _stop_training(self):
        self.is_running = False
        print("\n>> 正在停止训练并保存数据...")

    def fast_resurrection(self):
        """更快速、可靠的自动复活流程"""
        print(">> 检测到角色死亡/战斗结束，执行复活程序...")
        
        for _ in range(5):
            self.action_controller.press_enter()
            time.sleep(0.2)

        start_time = time.time()
        while self.is_running:
            if self.is_paused:
                time.sleep(0.5)
                continue
                
            # 优化：只截一次屏
            boss_hp, self_hp, _, _, _ = self.game.check_game_state()
            if self_hp is not None and self_hp > 0.1:
                print(">> 角色已就绪，战斗开始！")
                time.sleep(1.0)
                return True
            
            self.action_controller.press_enter()
            
            if time.time() - start_time > 30:
                print(">> 复活超时，请检查游戏窗口状态")
                return False
                
            time.sleep(0.5)
        return False

    def run_episode(self):
        """单个回合的高效循环 (异步训练版)"""
        episode_reward = 0
        step_count = 0
        
        # 初始状态获取
        boss_hp, self_hp, _, _, raw = self.game.check_game_state()
        if boss_hp is None or self_hp is None or self_hp <= 0:
            if not self.fast_resurrection():
                return 0, 0
            boss_hp, self_hp, _, _, raw = self.game.check_game_state()

        # 记录起始血量用于结算伤害表现
        boss_hp_start = boss_hp
        
        # 优化：利用 check_game_state 已经拿到的 raw screen
        state = self.game.get_state_from_frame(raw)
        self.game.reset_reward_stats() # 重置本回合的奖励统计数据
        self.episode_count += 1
        hidden = None
        
        print(f"\n--- 回合 #{self.episode_count} 开始 ---")
        episode_start_time = time.time()

        while self.is_running and not self.is_paused:
            step_start = time.time()
            
            # 1. 决策
            if self.agent.use_recurrent:
                action, hidden = self.agent.select_action(state, boss_hp, self_hp, hidden)
            else:
                action = self.agent.select_action(state, boss_hp, self_hp)
                
            # 2. 执行
            self.action_controller.take_action(action)

            # 3. 动态等待
            wait_time = 0.02 if action == 0 else 0.04
            time.sleep(wait_time)

            # 4. 获取新状态 (关键：只截一次屏)
            next_boss_hp, next_self_hp, _, _, next_raw = self.game.check_game_state()
            if next_boss_hp is None: break
                
            next_state = self.game.get_state_from_frame(next_raw)

            # 5. 计算奖励
            reward = self.game.calculate_reward(self_hp, boss_hp, next_self_hp, next_boss_hp, action)
            done = next_self_hp <= 0 or next_boss_hp <= 0
            
            # 5.1 表现评估 (仅在回合结束时计算一次)
            if done:
                episode_duration = time.time() - episode_start_time
                total_damage = max(0, boss_hp_start - next_boss_hp)
                perf_reward, msg = self.damage_stats.update_and_evaluate(total_damage, episode_duration)
                if msg: print(f">> {msg}")
                reward += perf_reward

            # 5.2 连招发现与统计
            self.action_controller.combo_manager.discover_combo(reward)
            
            # 6. 存储经验
            self.agent.store_transition(
                state, boss_hp, self_hp, action,
                reward, next_state, next_boss_hp, next_self_hp, done
            )

            # 7. 更新当前状态
            state = next_state
            boss_hp = next_boss_hp
            self_hp = next_self_hp
            episode_reward += reward
            step_count += 1

            # 性能统计
            step_duration = time.time() - step_start
            fps = 1.0 / step_duration if step_duration > 0 else 0
            
            if step_count % 50 == 0:
                print(f"Step: {step_count} | Reward: {episode_reward:.2f} | FPS: {fps:.1f} | Epsilon: {self.agent.epsilon:.4f}")

            if done:
                print(f">>> 回合结束 | 总奖励: {episode_reward:.2f} | 总步数: {step_count}")
                self.action_controller.force_cancel_all() # 确保释放所有长按键
                break

        return episode_reward, step_count

    def train(self):
        """主训练控制流程"""
        if self.agent.load_checkpoint():
            print(">> 成功加载已有模型权重")
        else:
            print(">> 未找到检查点，从头开始训练")

        print(">> 系统已就绪。当前处于暂停状态，请按 '0' 键开始训练。")

        try:
            while self.is_running:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ep_reward, steps = self.run_episode()
                
                if steps == 0:
                    # 如果没有成功运行步数（可能没进游戏），多等一会儿再试，防止死循环占用太高或意外退出
                    time.sleep(1.0)
                    continue

                if steps > 0:
                    self.total_reward += ep_reward
                    
                    self.agent.writer.add_scalar('Episode/Reward', ep_reward, self.episode_count)
                    self.agent.writer.add_scalar('Episode/Steps', steps, self.episode_count)
                    
                    if self.episode_count % 5 == 0:
                        with self.agent.net_lock:
                            self.agent.save_checkpoint()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n>> 检测到强制退出...")
        finally:
            self._save_checkpoint()
            self.agent.close()

    def debug_mode(self):
        """调试模式"""
        print(">> 进入调试模式，按 ESC 退出")
        while not keyboard.is_pressed('esc'):
            boss_hp, self_hp, _, _, raw = self.game.check_game_state()
            if raw is not None:
                self.game.debug_health_regions(raw)
                if boss_hp is not None:
                    print(f"\r[DEBUG] Boss HP: {boss_hp:.3f} | Player HP: {self_hp:.3f} | 窗口正常", end="")
            else:
                print("\r[DEBUG] 无法获取游戏窗口！", end="")
            time.sleep(0.05)
        print("\n>> 退出调试模式")

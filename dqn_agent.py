import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import threading
import queue
import time
import copy
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE, TRAINING_CONFIG, FILE_PATHS, NETWORK_CONFIG
from network_components import OptimizedNet, ProNet
from replay_buffer import PrioritizedReplayBuffer, Transition


class DQNAgent:
    """更聪明的DQN智能体：支持异步保存、动态动作空间扩展 (升级为 Pro 版: QR-DQN + N-step)"""

    def __init__(self, num_actions, network_config=None, checkpoint_file=None, log_dir=None):
        self.num_actions = num_actions
        self.network_config = (network_config or NETWORK_CONFIG).copy()
        self.checkpoint_file = checkpoint_file or FILE_PATHS['checkpoint']
        self.use_recurrent = self.network_config.get('use_recurrent', True)
        self.use_noisy = self.network_config.get('use_noisy', True)

        # 网络初始化 (升级为 ProNet)
        self.policy_net = ProNet(num_actions, input_channels=4, config=self.network_config).to(DEVICE)
        self.target_net = ProNet(num_actions, input_channels=4, config=self.network_config).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000)
        self.scaler = torch.amp.GradScaler(device=DEVICE.type if DEVICE.type != 'cpu' else 'cpu', enabled=(DEVICE.type != 'cpu'))

        # 经验回放
        self.memory = PrioritizedReplayBuffer(TRAINING_CONFIG['memory_capacity'])

        self.batch_size = TRAINING_CONFIG['batch_size']
        self.gamma = TRAINING_CONFIG['gamma']
        
        # N-step return
        self.n_step = TRAINING_CONFIG.get('n_step', 3)
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        # QR-DQN
        self.num_quantiles = self.network_config.get('num_quantiles', 51)
        # 固定分位数中点：[1/(2N), 3/(2N), ..., (2N-1)/(2N)]
        self.quantiles = torch.linspace(0.0, 1.0, self.num_quantiles + 1, device=DEVICE)[1:] - 0.5 / self.num_quantiles
        self.quantiles = self.quantiles.view(1, 1, -1) # [1, 1, num_quantiles]

        # Noisy Nets 不需要 epsilon-greedy
        self.epsilon = 0.0 if self.use_noisy else TRAINING_CONFIG['epsilon_start']
        self.epsilon_decay = TRAINING_CONFIG['epsilon_decay']
        self.epsilon_min = TRAINING_CONFIG['epsilon_min']
        
        self.tau = TRAINING_CONFIG.get('tau', 0.005)
        self.reward_scale = TRAINING_CONFIG.get('reward_scale', 0.1)
        self.steps = 0
        self.writer = SummaryWriter(log_dir or FILE_PATHS['tensorboard_logs'])
        
        # 线程锁
        self.net_lock = threading.Lock()
        
        # 异步保存队列
        self.save_queue = queue.Queue(maxsize=1)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    def _save_worker(self):
        """后台保存模型权重"""
        while True:
            try:
                task = self.save_queue.get()
                if task is None: break
                
                filename, checkpoint = task
                torch.save(checkpoint, filename)
                # print(f">> 模型已异步保存至 {filename}")
                self.save_queue.task_done()
            except Exception as e:
                print(f"异步保存模型失败: {e}")
            time.sleep(0.1)

    def select_action(self, state, boss_health, self_health, hidden=None):
        """选择动作"""
        if not self.use_noisy and random.random() < self.epsilon:
            if self.use_recurrent:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    boss_t = torch.as_tensor([boss_health], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    self_t = torch.as_tensor([self_health], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    with self.net_lock:
                        _, next_hidden = self.policy_net(state_t, boss_t, self_t, hidden)
                return random.randint(0, self.num_actions - 1), next_hidden
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            boss_t = torch.as_tensor([boss_health], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            self_t = torch.as_tensor([self_health], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast(device_type=DEVICE.type if DEVICE.type != 'cpu' else 'cpu', 
                                  dtype=torch.float16, 
                                  enabled=(DEVICE.type != 'cpu')):
                with self.net_lock:
                    if self.use_recurrent:
                        q_dist, next_hidden = self.policy_net(state_t, boss_t, self_t, hidden)
                        q_values = q_dist.mean(dim=2)
                        action = q_values.argmax().item()
                        return action, next_hidden
                    else:
                        q_dist = self.policy_net(state_t, boss_t, self_t)
                        q_values = q_dist.mean(dim=2)
                        return q_values.argmax().item()

    def store_transition(self, state, boss_health, self_health, action,
                         reward, next_state, next_boss_health, next_self_health, done):
        transition = Transition(
            state.copy() if hasattr(state, 'copy') else state, 
            boss_health, self_health, action,
            reward, 
            next_state.copy() if hasattr(next_state, 'copy') else next_state, 
            next_boss_health, next_self_health, done
        )
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) == self.n_step:
            reward_sum = 0
            for i, t in enumerate(self.n_step_buffer):
                reward_sum += t.reward * (self.gamma ** i)
                if t.done:
                    break
            
            n_step_trans = self.n_step_buffer[0]
            last_trans = t # 实际的终止状态或者第N步状态
            
            n_step_transition = Transition(
                n_step_trans.state, n_step_trans.boss_health, n_step_trans.self_health, n_step_trans.action,
                reward_sum, last_trans.next_state, last_trans.next_boss_health, last_trans.next_self_health, last_trans.done
            )
            self.memory.add(n_step_transition)

    def update_model(self):
        if not self.memory.is_ready(self.batch_size):
            return

        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=DEVICE)
        boss_hp = torch.as_tensor(batch.boss_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        self_hp = torch.as_tensor(batch.self_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        action = torch.as_tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=DEVICE) * self.reward_scale
        next_state = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=DEVICE)
        next_boss = torch.as_tensor(batch.next_boss_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_self = torch.as_tensor(batch.next_self_health, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        done = torch.as_tensor(batch.done, dtype=torch.float32, device=DEVICE)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)

        with self.net_lock:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

            with torch.amp.autocast(device_type=DEVICE.type if DEVICE.type != 'cpu' else 'cpu', 
                                  dtype=torch.float16, 
                                  enabled=(DEVICE.type != 'cpu')):
                if self.use_recurrent:
                    current_q_dist, _ = self.policy_net(state, boss_hp, self_hp, None)
                    action_expanded = action.unsqueeze(-1).expand(-1, 1, self.num_quantiles)
                    current_quantiles = current_q_dist.gather(1, action_expanded).squeeze(1)
                    
                    with torch.no_grad():
                        next_q_dist_target, _ = self.target_net(next_state, next_boss, next_self, None)
                        next_q_dist_policy, _ = self.policy_net(next_state, next_boss, next_self, None)
                        next_actions = next_q_dist_policy.mean(dim=2).argmax(1, keepdim=True)
                        next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, 1, self.num_quantiles)
                        next_quantiles = next_q_dist_target.gather(1, next_actions_expanded).squeeze(1)
                else:
                    current_q_dist = self.policy_net(state, boss_hp, self_hp)
                    action_expanded = action.unsqueeze(-1).expand(-1, 1, self.num_quantiles)
                    current_quantiles = current_q_dist.gather(1, action_expanded).squeeze(1)
                    
                    with torch.no_grad():
                        next_q_dist_target = self.target_net(next_state, next_boss, next_self)
                        next_q_dist_policy = self.policy_net(next_state, next_boss, next_self)
                        next_actions = next_q_dist_policy.mean(dim=2).argmax(1, keepdim=True)
                        next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, 1, self.num_quantiles)
                        next_quantiles = next_q_dist_target.gather(1, next_actions_expanded).squeeze(1)

                target_quantiles = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * (self.gamma ** self.n_step) * next_quantiles
                target_quantiles = target_quantiles.unsqueeze(1) # [batch, 1, num_quantiles]
                current_quantiles = current_quantiles.unsqueeze(2) # [batch, num_quantiles, 1]
                
                td_errors = target_quantiles - current_quantiles # [batch, num_quantiles, num_quantiles]
                huber_loss = F.smooth_l1_loss(current_quantiles.expand_as(td_errors), target_quantiles.expand_as(td_errors), reduction='none')
                
                weights_q = torch.abs(self.quantiles - (td_errors.detach() < 0).float())
                loss = (weights_q * huber_loss).sum(dim=2).mean(dim=1)
                
                mean_td_error = td_errors.mean(dim=(1,2)).abs().detach().cpu().numpy()
                self.memory.update_priorities(indices, mean_td_error)
                
                loss = (weights * loss).mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.steps % 10 == 0:
                self.scheduler.step()

            with torch.no_grad():
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

            if not self.use_noisy:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
            self.steps += 1
            if self.steps % 100 == 0:
                self.writer.add_scalar('Loss/TD_Loss', loss.item(), self.steps)

    def save_checkpoint(self, filename=None):
        filename = filename or self.checkpoint_file
        """将保存任务提交到异步队列 (带锁深拷贝，防止训练冲突)"""
        with self.net_lock:
            # 必须在锁内克隆 state_dict 到 CPU，否则异步保存会与训练线程冲突导致崩溃
            checkpoint = {
                'policy_state': {k: v.cpu().clone() for k, v in self.policy_net.state_dict().items()},
                'target_state': {k: v.cpu().clone() for k, v in self.target_net.state_dict().items()},
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'num_actions': self.num_actions,
                'network_config': self.network_config
            }
        
        # 如果队列已满，丢弃旧的，确保最后一次保存总是最新的
        if self.save_queue.full():
            try: 
                self.save_queue.get_nowait()
                self.save_queue.task_done()
            except: 
                pass
        self.save_queue.put((filename, checkpoint))

    def load_checkpoint(self, filename=None):
        filename = filename or self.checkpoint_file
        try:
            import os
            if not os.path.exists(filename):
                return False
                
            checkpoint = torch.load(filename, map_location=DEVICE, weights_only=False)
            
            # 动作空间兼容性处理
            saved_actions = checkpoint.get('num_actions', 13)
            if saved_actions != self.num_actions:
                print(f">> 警告: 检测到动作空间变化 ({saved_actions} -> {self.num_actions})，正在执行权重对齐...")
                self._expand_output_layer(checkpoint['policy_state'])
                self._expand_output_layer(checkpoint['target_state'])
                # 重新初始化优化器，因为参数形状变了
                self.optimizer = optim.AdamW(
                    self.policy_net.parameters(),
                    lr=TRAINING_CONFIG['learning_rate'],
                    weight_decay=TRAINING_CONFIG['weight_decay']
                )
            
            with self.net_lock:
                self.policy_net.load_state_dict(checkpoint['policy_state'], strict=False)
                self.target_net.load_state_dict(checkpoint['target_state'], strict=False)
                
                if saved_actions == self.num_actions:
                    try: self.optimizer.load_state_dict(checkpoint['optimizer'])
                    except: print(">> 优化器权重不匹配，已重置")
                    
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.steps = checkpoint.get('steps', 0)
            return True
        except Exception as e:
            print(f">> 加载检查点时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _expand_output_layer(self, state_dict):
        """动态扩展输出层权重以适应新动作空间"""
        for key in ['advantage_head.weight_mu', 'advantage_head.weight_sigma', 
                    'advantage_head.bias_mu', 'advantage_head.bias_sigma',
                    'advantage_head.weight', 'advantage_head.bias']: # 兼容 Noisy 和 Linear
            if key in state_dict:
                old_weight = state_dict[key]
                new_shape = list(old_weight.shape)
                new_shape[0] = self.num_actions
                
                new_weight = torch.randn(new_shape, device=old_weight.device) * 0.01
                # 拷贝旧权重
                slice_idx = min(old_weight.shape[0], self.num_actions)
                new_weight[:slice_idx] = old_weight[:slice_idx]
                state_dict[key] = new_weight

    def close(self):
        """安全关闭：确保模型保存完成"""
        print(">> 正在等待后台保存任务完成...")
        self.save_queue.put(None)
        self.save_thread.join()
        self.writer.close()
        print(">> 资源已释放")

import numpy as np
import threading
from collections import namedtuple
from config import REPLAY_CONFIG

Transition = namedtuple('Transition',
                        ('state', 'boss_health', 'self_health', 'action',
                         'reward', 'next_state', 'next_boss_health',
                         'next_self_health', 'done'))

class SumTree:
    """SumTree for prioritized sampling (Thread-safe)"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        self.lock = threading.Lock()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        with self.lock:
            return self.tree[0]

    def add(self, p, data):
        with self.lock:
            idx = self.write + self.capacity - 1
            self.data[self.write] = data
            self.update_locked(idx, p)

            self.write += 1
            if self.write >= self.capacity:
                self.write = 0
            if self.n_entries < self.capacity:
                self.n_entries += 1

    def update(self, idx, p):
        with self.lock:
            self.update_locked(idx, p)

    def update_locked(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        with self.lock:
            idx = self._retrieve(0, s)
            dataIdx = idx - self.capacity + 1
            return idx, self.tree[idx], self.data[dataIdx]

class PrioritizedReplayBuffer:
    """Optimized Prioritized Replay Buffer using SumTree (Thread-safe)"""
    def __init__(self, capacity, alpha=REPLAY_CONFIG.get('alpha', 0.6), beta=REPLAY_CONFIG.get('beta', 0.4)):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.01

    def add(self, transition):
        # np.max 在大切片上可能较慢，且需要锁保护
        with self.tree.lock:
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = 1.0
        self.tree.add(max_p, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        
        total = self.tree.total()
        if total == 0:
            return [], [], []
            
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / (total + 1e-8)
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-8)

        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors):
        errors += self.epsilon
        clipped_errors = np.minimum(errors, 1.0)
        ps = np.power(clipped_errors, self.alpha)
        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)

    def is_ready(self, batch_size):
        return self.tree.n_entries >= batch_size

    def __len__(self):
        return self.tree.n_entries


class UniformReplayBuffer:
    """
    均匀采样的环形 buffer，给计划一致性 loss 用。

    为什么不复用上面的 PER：槽位一致性是**回归**（把 Q_j(s) 拟合到
    Q_{j-1}^target(s')），没有 TD error 可言，优先级无从谈起；而且它必须吃
    **1 步** transition —— PER 里存的是 n-step 聚合过的，那里的 s' 已经是 t+n 了，
    对不上"下一帧的计划往前挪一格"这个关系。
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = [None] * self.capacity
        self.write = 0
        self.n_entries = 0
        self.lock = threading.Lock()

    def add(self, transition):
        with self.lock:
            self.data[self.write] = transition
            self.write = (self.write + 1) % self.capacity
            self.n_entries = min(self.n_entries + 1, self.capacity)

    def sample(self, batch_size):
        with self.lock:
            if self.n_entries == 0:
                return []
            idxs = np.random.randint(0, self.n_entries, size=batch_size)
            return [self.data[i] for i in idxs]

    def is_ready(self, batch_size):
        return self.n_entries >= batch_size

    def __len__(self):
        return self.n_entries

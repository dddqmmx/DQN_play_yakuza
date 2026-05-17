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

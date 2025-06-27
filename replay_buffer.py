import random
import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition, td_error):
        priority = float((abs(td_error) + 1e-5) ** self.alpha)

        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
    
    # Importance Sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        weights = torch.FloatTensor(weights).unsqueeze(1)  # For broadcasting during loss

    # Now unpack samples
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, weights, indices


    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = float((abs(td_error) + 1e-5) ** self.alpha)


    def __len__(self):
        return len(self.buffer)

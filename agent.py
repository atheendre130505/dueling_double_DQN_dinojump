import torch
import torch.nn as nn
import torch.nn.functional as F
from DuelingDqn import DuelingDQN 
import numpy as np
import pygame
from dino3 import DinoEnv

state_dim = 9
action_dim = 3

class Agent:
    def __init__(self,replay_buffer):
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 256
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.05
        self.target_update = 10
        self.replay_buffer = replay_buffer
        self.build_model()

    def build_model(self):
        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = F.smooth_l1_loss
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(action_dim)
        torch_state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_value = self.policy_net(torch_state)
        return torch.argmax(q_value).item()
        
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, actions, rewards, next_state, done, weights, indices = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action = self.policy_net(next_state).argmax(dim=1, keepdim=True)
            target = rewards + self.gamma * self.target_net(next_state).gather(1, next_action) * (1 - done)

        q_value = self.policy_net(state).gather(1, actions)
        td_error = target - q_value.detach()

    # Weighted loss
        loss = (F.mse_loss(q_value, target, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Update priorities
        self.replay_buffer.update_priorities(indices, td_error)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()

        
    
def test_agent(agent):
    env = DinoEnv()
    state = env.reset()
    done = False
    agent.epsilon = 0.0
    agent.policy_net.eval()

    while not done:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                exit()
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        env.draw()
        pygame.time.delay(30)
        state = next_state
    print("Test complete.")




    
    

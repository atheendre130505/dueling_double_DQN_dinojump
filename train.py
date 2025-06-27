# train.py
import os
import numpy as np
from dino3 import DinoEnv
from agent import Agent, test_agent  # ← include test_agent from agent.py
from replay_buffer import PrioritizedReplayBuffer

# Hyperparameters
NUM_EPISODES = 5000
MODEL_PATH = "dino_ddqn.pth"

# Initialize
env = DinoEnv()
buffer = PrioritizedReplayBuffer(capacity=10000)
agent = Agent(buffer)
episode_rewards = []

# Load model if exists
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    agent.load_model(MODEL_PATH)
else:
    print("Training from scratch.")

# Training loop
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        buffer.add((state, action, reward, next_state, done), td_error=1.0)
        agent.train()
        state = next_state
        total_reward += reward
        
    episode_rewards.append(total_reward)

    # Update target network
    if episode % agent.target_update == 0:
        agent.update_target_network()
    
    print(total_reward)
    # Every 100 episodes: average reward + save
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode} | Avg Reward: {avg_reward:.2f}")
        agent.save_model(MODEL_PATH)

    # Every 500 episodes: run visual test
    if episode % 200 == 0:
        print("Running test episode...")
        test_agent(agent)

# Final save
agent.save_model(MODEL_PATH)

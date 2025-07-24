# dueling_double_DQN_dinojump
project uses dueling double dqn + personalized replay buffer to help the agent learn the game

files contained: agent.py, dino3.py, DuelingDqn, dino_ddqn.path(contains weights for the nerual network), replay_buffer/prioritised replay buffer, and train.py.

![Screenshot from 2025-06-28 02-06-46](https://github.com/user-attachments/assets/99203f5a-c4a4-4354-a349-8b974e78c3d2)

Dueling Double DQN for Chrome Dino Game
This project implements an advanced deep reinforcement learning agent that learns to play the Chrome Dino Run game. It uses a Dueling Double Deep Q-Network (Dueling Double DQN) architecture combined with a Prioritized Experience Replay buffer to achieve efficient and robust learning.

Table of Contents
Project Overview

Key Concepts

Project Structure

Technology Stack

Setup and Installation

How to Run

Training the Agent

Running with Pre-trained Weights

Architecture Details

Contributing

License

Project Overview
The goal of this project is to train an AI agent to master the Chrome Dino game by making intelligent decisions (jumping or not jumping) to avoid obstacles. The agent learns from raw pixel input of the game screen and improves its gameplay over time through trial and error, guided by a sophisticated reinforcement learning algorithm.

This implementation showcases the power of modern RL techniques to solve problems in a visual domain.

Key Concepts
Dueling Double DQN: An enhancement over the standard DQN algorithm.

Double DQN: Mitigates the overestimation of Q-values, leading to more stable and reliable training.

Dueling Architecture: Separates the estimation of the state value function (V(s)) and the advantage function (A(s,a)). This allows the network to learn which states are valuable without having to learn the effect of each action at each state, resulting in better policy evaluation.

Prioritized Experience Replay: A technique that improves learning efficiency by replaying important transitions more frequently. Transitions are prioritized based on the magnitude of their temporal-difference (TD) error. This allows the agent to focus on experiences from which it can learn the most.

Project Structure
The repository contains the following key files:

agent.py: Defines the Agent class, which encapsulates the agent's behavior, including action selection and interaction with the learning network.

dino3.py: The game environment. This script handles the game logic, state updates, and rendering.

DuelingDqn.py: Contains the PyTorch implementation of the Dueling DQN neural network architecture.

train.py: The main script used to start and manage the training process.

replay_buffer.py: Implements the Prioritized Experience Replay buffer.

dino_ddqn.pth: A file containing the pre-trained weights for the neural network, allowing you to run the agent without training from scratch.

Technology Stack
Language: Python

Core Libraries:

PyTorch (for the neural network)

NumPy (for numerical operations)

OpenCV (for image processing)

Pygame (for running the game environment)

Setup and Installation
Clone the repository:

bash
git clone https://github.com/atheendre130505/dueling_double_DQN_dinojump.git
cd dueling_double_DQN_dinojump
Create a virtual environment (recommended):

bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

bash
pip install torch torchvision torchaudio numpy opencv-python pygame
How to Run
Training the Agent
To train the agent from the beginning, run the train.py script. This will start the game environment and the training loop. The agent's progress and model weights will be saved periodically.

bash
python train.py
Running with Pre-trained Weights
To see the trained agent in action immediately, you can use the provided dino_ddqn.pth weights file. Make sure the train.py script is configured to load the model in evaluation mode.

Open the train.py file.

Ensure that the model loading section is active and the model is set to evaluation mode (model.eval()).

Run the script:

bash
python train.py
The agent will start playing the game using the learned policy from the pre-trained weights.

Architecture Details
Game Environment (dino3.py): Provides the state of the game (as screen pixels) to the agent and receives an action in return. It then executes the action and returns the new state and a reward signal.

Agent (agent.py): Acts as the brain. It preprocesses the image data from the game, feeds it into the Dueling DQN, and selects an action based on the network's output (Q-values).

Dueling DQN (DuelingDqn.py): A deep neural network that takes the processed game screen as input and outputs the expected cumulative reward (Q-value) for each possible action (e.g., jump, do nothing).

Replay Buffer (replay_buffer.py): Stores the agent's experiences (state, action, reward, next state). During training, batches of these experiences are sampled to update the neural network's weights, with more "surprising" experiences being sampled more often.

Training Loop (train.py): Orchestrates the entire process. It initializes the agent and environment, runs episodes of the game, collects experiences, and periodically trains the network using data from the replay buffer.

Contributing
Contributions are welcome! If you have suggestions for improvements or want to add new features, please follow these steps:

Fork the repository.

Create a new feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

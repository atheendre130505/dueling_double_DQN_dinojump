import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.featurelayer = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)

        self.value_stream = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self,x):
        x=F.relu(self.featurelayer(x))
        x=F.relu(self.layer2(x))

        value = self.value_stream(x)
        advantage=self.advantage_stream(x)

        q_val = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_val
        
    


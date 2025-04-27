import torch
import torch.nn as nn
import torch.optim as optim

class MetaPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x
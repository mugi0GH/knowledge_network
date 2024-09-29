import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.fc1 = nn.Linear(torch.prod(torch.tensor(input_dim.shape)).item(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1) # .reshape(tensor_3d.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x
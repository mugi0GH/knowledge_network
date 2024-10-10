import torch
from torch import nn
from torchrl.modules.models.exploration import NoisyLinear

class model(nn.Module):
    def __init__(self,input_size,action_dim,noisy_net=False,role = '') -> None:
        super().__init__()
        self.L1 = nn.Linear(input_size.shape[0],512)
        if noisy_net:
            self.L2 = NoisyLinear(512,action_dim)
        else:
            self.L2 = nn.Linear(512,action_dim)
        # self.output = nn.Softmax(dim=-1)
        # self.output = torch.argmax
        if role == 'target':
            for module in self.modules(): 
                module.training = False
    def forward(self,x):
        x = torch.relu(self.L1(torch.Tensor(x)))
        x = torch.relu(self.L2(x))
        # x = self.output(x)
        return x
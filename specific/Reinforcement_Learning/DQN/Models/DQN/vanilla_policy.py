import torch
from torch import nn

class model(nn.Module):
    def __init__(self,input_size,output_size) -> None:
        super().__init__()
        self.L1 = nn.Linear(input_size.shape[0],512)
        self.L2 = nn.Linear(512,output_size)
        # self.output = nn.Softmax(dim=-1)
        # self.output = torch.argmax

    def forward(self,x):
        x = torch.relu(self.L1(torch.Tensor(x)))
        x = torch.relu(self.L2(x))
        # x = self.output(x)
        return x
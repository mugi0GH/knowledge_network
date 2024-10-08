import torch
from torch import nn
import numpy as np

class model(nn.Module):
    def __init__(self,input_size,output_size) -> None:
        super().__init__()
        self.lr = 0.001
        self.theta = 0

        self.backbone = nn.Sequential(
            nn.Linear(input_size.shape[0],512),
            nn.ReLU(),
            nn.Linear(512,output_size)
        )
        self.output = nn.Softmax()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
    def forward(self,x):
        x = self.backbone(x)
        x = self.output(x)
        return x
    
    def train(self,trajectories,returns):
        # 计算损失：log概率 * 回报的负值
        loss = []
        for log_prob, G in zip(trajectories, returns):
            loss.append(-log_prob * G)

        loss = torch.stack(loss).sum()

        # 进行优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.theta +=self.lr * 
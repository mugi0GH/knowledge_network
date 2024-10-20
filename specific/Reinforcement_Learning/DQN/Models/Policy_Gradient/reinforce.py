from Update_modules.initialization.weight_init import he_init_weights
import torch
from torch import nn
import numpy as np

class actor(nn.Module):
    def __init__(self,state_dim,action_dim,hypers) -> None:
        super().__init__()
        # self.theta = 0
        self.hypers = hypers
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,512),
            nn.ReLU(),
            nn.Linear(512,action_dim)
        )
        self.output = nn.Softmax(dim=-1)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hypers['LR'], weight_decay=1e-2)

        # 调用初始化函数
        he_init_weights(self)

    def forward(self,x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(self.hypers['DEVICE'], dtype=torch.float32)
        else:
            x = x.to(self.hypers['DEVICE'], dtype=torch.float32)
        # x = torch.tensor(x).to(self.hypers['DEVICE'])
        x = self.backbone(x)
        x = self.output(x)
        return x
    
    def train(self,log_probs,returns):
        # 计算损失：log概率 * 回报的负值
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G  # 目标函数的负梯度

        # 进行优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.theta +=self.lr * 
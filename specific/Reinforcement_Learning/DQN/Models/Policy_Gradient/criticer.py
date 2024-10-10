from torch import nn
import torch.optim as optim
import torch

class critic(nn.Module):
    def __init__(self,state_dim,action_dim,hypers) -> None:
        self.hypers = hypers
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,512),
            nn.ReLU())
        
        self.output = nn.Sequential(
            nn.Linear(512,action_dim)
        )
        self.critic_optimizer = optim.Adam(self.parameters(), lr=self.hypers["LR"])

    def forward(self,x):
        x = torch.tensor(x).to(self.hypers['DEVICE'])
        x = self.backbone(x)
        return self.output(x)
    
    def optimize(self,state,next_state,reward):
        # 前向传播，计算当前状态和下一个状态的值
        value = self.forward(state)  # V(s_t)
        next_value = self.forward(next_state)  # V(s_{t+1})
        """-----------------------公式------------------------------"""
        # 计算 TD 目标和 TD 误差
        td_target = reward + self.hypers["GAMMA"] * next_value  # TD 目标
        td_error = td_target - value  # TD 误差

        # 计算均方误差损失
        critic_loss = td_error.pow(2).mean()
        """------------------------------------------------------------"""
        # 反向传播并更新 Critic 网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return td_error
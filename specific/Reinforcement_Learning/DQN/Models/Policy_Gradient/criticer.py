from torch import nn
from Update_modules.initialization.weight_init import he_init_weights
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
            # nn.Linear(512,action_dim) # Q(s,a)
            nn.Linear(512,1)
        )
        self.critic_optimizer = optim.Adam(self.parameters(), lr=self.hypers["LR"])

        # 调用初始化函数
        he_init_weights(self)

    def forward(self, x):
        # 确保输入是 tensor 并且在正确的设备上
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.hypers['DEVICE'], dtype=torch.float32)
        else:
            x = x.to(self.hypers['DEVICE'])
        x = self.backbone(x)
        return self.output(x)
    
    def optimize(self, states, next_states, rewards, dones):
        # 前向传播，计算当前状态和下一个状态的值
        value = self.forward(states)  # V(s_t)
        next_value = self.forward(next_states)  # V(s_{t+1})

        # 检查是否是终止状态
        td_target = rewards + self.hypers["GAMMA"] *  next_value.detach() * (1 - dones)  # TD 目标 Gt = r_t + γV(s_t+1) (考虑终止状态)
        td_error = td_target - value  # TD 误差 Gt - Vt

        # 计算均方误差损失
        critic_loss = td_error.pow(2).mean()

        # 反向传播并更新 Critic 网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        self.critic_optimizer.step()

        return td_error

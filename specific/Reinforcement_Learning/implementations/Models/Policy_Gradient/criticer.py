from torch import nn
from Update_modules.initialization.weight_init import he_init_weights
import torch.optim as optim
import torch
import copy

class critic(nn.Module):
    def __init__(self,state_dim,hypers:dict) -> None:
        self.hypers = hypers
        if not hypers.get('critic_LR'):
            self.hypers['critic_LR'] = 1e-3
        
        if not hypers.get('v_loss_iter'):
            self.hypers['v_loss_iter'] = 10

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            # nn.Tanh(),
            )
        
        self.output = nn.Sequential(
            # nn.Linear(512,action_dim) # Q(s,a)
            nn.Linear(128,1),
            # nn.Tanh()
        )
        # self.critic_optimizer = torch.optim.AdamW(self.parameters(), lr= self.hypers['critic_LR'], weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers['LR'])

        # 调用初始化函数
        he_init_weights(self)

        # Huber_loss
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, x):
        # 确保输入是 tensor 并且在正确的设备上
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.hypers['DEVICE'], dtype=torch.float32)
        else:
            x = x.to(self.hypers['DEVICE'])
        x = self.backbone(x)
        return self.output(x)
    
    def back_prop(self,td_error):
        # 计算均方误差损失
        # critic_loss = 0.5 * td_error.pow(2).mean()
        critic_loss = td_error.pow(2)
        # critic_loss = self.huber_loss(td_target, value)
        # print(f"Critic Loss: {critic_loss.item()}")
        # 反向传播并更新 Critic 网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # 防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        self.critic_optimizer.step()

    def optimize(self, states, next_states, rewards, dones):
        
        # 确保输入是张量，并在第0维添加维度以形成批量
        if states.ndim == 1:
            states = torch.tensor(states).to(self.hypers["DEVICE"])  # 将 NumPy 数组转换为 PyTorch 张量
            next_states = torch.tensor(next_states).to(self.hypers["DEVICE"])  # 将 NumPy 数组转换为 PyTorch 张量
            rewards = torch.tensor(rewards).to(self.hypers["DEVICE"])  # 将 NumPy 数组转换为 PyTorch 张量
            dones = torch.tensor(dones).to(self.hypers["DEVICE"])  # 将 NumPy 数组转换为 PyTorch 张量
        
        if self.hypers.get('n_step_TD') and self.hypers['n_step_TD']:
            '''N-step TD'''
            # 前向传播，计算当前状态和下一个状态的值
            value = self.forward(states)  # V(s_t)
            n_steps = self.hypers['TD_steps']
            gamma = self.hypers['GAMMA']

            # 初始化 Gt，与 rewards 大小相同
            Gt = torch.zeros_like(rewards, dtype=torch.float32)

            # 计算 n-step 回报
            for t in range(len(rewards)):
                G = 0
                discount = 1
                # 最大累积步数不能超过序列长度
                for k in range(n_steps):
                    if t + k < len(rewards):
                        idx = t + k
                        G += discount * rewards[idx]
                        discount *= gamma
                        if dones[idx]:
                            break  # 如果遇到终止状态，停止累积
                    else:
                        break  # 超出序列长度，停止累积
                else:
                    # 如果未遇到终止状态，添加第 t+n 步的状态价值估计
                    if t + n_steps < len(value):
                        G += discount * value[t + n_steps]
                Gt[t] = G

            # 计算 TD 误差
            td_error = Gt - value
            self.back_prop(td_error)
        else:
            '''TD'''
            for _ in range(self.hypers['v_loss_iter']):
                # 前向传播，计算当前状态和下一个状态的值
                value = self.forward(states)  # V(s_t)
                next_value = self.forward(next_states)  # V(s_{t+1})
                
                # 检查是否是终止状态
                td_target = rewards + (self.hypers["GAMMA"] *  next_value.detach() * (1 - dones))  # TD 目标 Gt = r_t + V(s_t+1) (考虑终止状态)
                
                td_error = td_target - value  # TD 误差 Gt - Vt
                self.back_prop(td_error)

        return td_error

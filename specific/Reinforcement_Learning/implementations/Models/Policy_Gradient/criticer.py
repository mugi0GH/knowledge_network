from torch import nn
from Update_modules.initialization.weight_init import he_init_weights
import torch.optim as optim
import torch
import copy

class critic(nn.Module):
    def __init__(self,state_dim,action_dim,hypers:dict) -> None:
        self.hypers = hypers
        self.action_dim = action_dim
        if not hypers.get('critic_LR'):
            self.hypers['critic_LR'] = 1e-3
        
        if not hypers.get('v_loss_iter'):
            self.hypers['v_loss_iter'] = 10

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,self.action_dim),
            )
        
        self.output = nn.Sequential(
            # nn.Linear(512,action_dim) # Q(s,a)
            # nn.Linear(128,1),
            nn.Linear(self.action_dim,1) # V(s)
            # nn.Tanh()
        )
        # self.critic_optimizer = torch.optim.AdamW(self.parameters(), lr= self.hypers['critic_LR'], weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers['critic_LR'])

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
        critic_loss = critic_loss.mean()
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
        
        if self.hypers.get('GAE') and self.hypers['GAE']:
            '''GAE (Generalized Advantage Estimation)
            δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
            A_t = δ_t + (γλ)·(1-done_t)·A_{t+1}   （从后往前递推）
            λ=1 退化为 Monte Carlo，λ=0 退化为单步 TD
            '''
            gamma = self.hypers['GAMMA']
            lam   = self.hypers['TD_lambda']

            # 更新前用当前 critic 计算 advantage（解耦，避免过拟合残差）
            with torch.no_grad():
                values      = self.forward(states)
                next_values = self.forward(next_states)
                deltas = rewards + gamma * next_values * (1 - dones) - values  # [n,1]

                advantages = torch.zeros_like(deltas)
                gae = torch.zeros_like(deltas[-1])
                for t in reversed(range(len(deltas))):
                    gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
                    advantages[t] = gae

            # 更新 critic v_loss_iter 次
            for _ in range(self.hypers['v_loss_iter']):
                value      = self.forward(states)
                next_value = self.forward(next_states)
                td_target  = rewards + gamma * next_value.detach() * (1 - dones)
                self.back_prop(td_target - value)

            return advantages  # 作为 td_error 传给 actor

        elif self.hypers.get('n_step_TD') and self.hypers['n_step_TD']:
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
            # 先用当前 critic 算一次 td_error 作为 actor 的 advantage（更新前的估计）
            with torch.no_grad():
                adv_value = self.forward(states)
                adv_next  = self.forward(next_states)
                adv_target = rewards + (self.hypers["GAMMA"] * adv_next * (1 - dones))
                td_error = adv_target - adv_value  # 返回给 actor 用

            # 再更新 critic v_loss_iter 次
            for _ in range(self.hypers['v_loss_iter']):
                value      = self.forward(states)
                next_value = self.forward(next_states)
                td_target  = rewards + (self.hypers["GAMMA"] * next_value.detach() * (1 - dones))
                self.back_prop(td_target - value)

        return td_error

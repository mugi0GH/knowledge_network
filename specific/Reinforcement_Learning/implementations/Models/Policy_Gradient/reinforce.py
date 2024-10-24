from Update_modules.initialization.weight_init import he_init_weights
from Update_modules.reward_oper import discounted_reward
import torch
from torch import nn
import numpy as np

class actor(nn.Module):
    def __init__(self,state_dim,action_dim,hypers) -> None:
        super().__init__()


        self.hypers = hypers
        self.backbone = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(128,action_dim)
        )
        self.output = nn.Softmax(dim=-1)

        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hypers['LR'], weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers['LR'])

        # 调用初始化函数
        he_init_weights(self)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            x = torch.tensor(state, dtype=torch.float32)
        x = x.to(self.hypers['DEVICE'], dtype=torch.float32)  # 确保 x 在正确的设备上并且类型正确
        x = self.backbone(x)
        x = self.output(x)
        return x
    
    def act(self,state):
        action_probs = self.forward(state)
        # 使用概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        # 记录每个动作的log概率
        log_prob = action_dist.log_prob(action)

        return action,log_prob
    
    # 损失函数实现
    def reinforce_loss(self,log_probs, rewards):

        discounted_rewards = discounted_reward(rewards).to(self.hypers["DEVICE"])

        # 计算损失：log概率 * 回报的负值
        '''隐式版本：训练函数'''
        # 计算 REINFORCE 损失
        loss = -torch.sum(log_probs * discounted_rewards)  # 累加损失
        return loss
        '''显式版本：训练函数'''
        # loss = []
        # for log_prob, reward in zip(log_probs, discounted_rewards):
        #     loss.append(-log_prob * reward)
        # return torch.stack(loss).sum()
        
    # 新的 REINFORCE 损失函数，加入了基线
    def reinforce_loss_with_baseline(self,log_probs, rewards, values):
        if type(log_probs) is list:
            # 将 log_probs 和 rewards 向量化，避免 for 循环
            log_probs = torch.stack(log_probs)  # 将 log_probs 转换为张量
            # rewards = torch.stack(rewards)  # 将 rewards 转换为张量
            discounted_rewards = discounted_reward(rewards).to(self.hypers["DEVICE"])
            
            # 计算加入基线后的损失
            loss = []
            for log_prob, reward, value in zip(log_probs, discounted_rewards, values):
                advantage = reward - value  # Advantage = R - V
                loss.append(-log_prob * advantage)
            # return torch.stack(loss).sum()
            return torch.stack(loss).mean()
        else:
            # 单步更新
            td_error = values 
            loss = -(log_probs * td_error.detach())
            return loss
    
    def train(self, log_probs, rewards,values=None):

        if values is not None and self.hypers['if_baseline']:
            loss = self.reinforce_loss_with_baseline(log_probs, rewards, values)
        else:
            loss = self.reinforce_loss(log_probs, rewards)

        # 进行优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

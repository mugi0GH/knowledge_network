import torch

class integrated_model():
    def __init__(self,actor,critic=None,hypers=None) -> None:
        self.actor = actor.to(hypers["DEVICE"])
        self.critic = critic.to(hypers["DEVICE"])
        self.hypers = hypers
    
    def forward(self,state):
        # state = torch.tensor(state).to(self.hyper["DEVICE"])
        '''actor'''
        state = torch.tensor(state).float().unsqueeze(0).to(self.hypers["DEVICE"])  # 将状态转化为张量
        probs = self.actor(state)  # 获取动作概率分布
        
        action_dist = torch.distributions.Categorical(probs)  # 创建分布
        action = action_dist.sample()  # 采样动作

        # 记录每个动作的log概率
        log_prob = action_dist.log_prob(action)
        self.actor.log_prob = log_prob

        return action.item()  # 返回动作

    def optimize(self,state,next_state,reward):
        td_error = self.critic.optimize(state,next_state,reward)

        # 更新 Actor 网络 (使用优势函数)
        log_prob = self.actor.log_prob  # 获取当前动作的 log 概率
        actor_loss = (-log_prob * td_error.detach()).mean()  # 使用优势函数 A(s_t, a_t) 更新 Actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        
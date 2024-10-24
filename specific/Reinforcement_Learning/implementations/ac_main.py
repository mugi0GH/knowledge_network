import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from Visualization.line_chart import  plot_performance

# 定义策略网络（Actor）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率
        )
    
    def forward(self, x):
        return self.actor(x)

# 定义价值网络（Critic）
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出状态价值
        )
    
    def forward(self, x):
        return self.critic(x)

def main():
    # env = gym.make('CartPole-v1',render_mode='human')
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    num_episodes = 1000
    gamma = 0.99
    ep_rewards = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        state = torch.FloatTensor(state)
        ep_reward = 0

        while True:
            # env.render()  # 如需渲染环境，取消注释
            probs = actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            if terminated or truncated:
                done = True
            else:
                done = False
            next_state = torch.FloatTensor(next_state)

            ep_reward += reward
            # 计算价值
            value = critic(state)
            next_value = critic(next_state)

            # 计算 TD 误差（Advantage）
            td_error = reward + gamma * next_value * (1 - int(done)) - value

            # 更新 Critic（最小化 TD 误差的平方）
            critic_loss = td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # 更新 Actor（基于策略梯度）
            actor_loss = -dist.log_prob(action) * td_error.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state

            if done:
                # print(f"Episode {episode+1}, Reward: {ep_reward}")
                ep_rewards.append(ep_reward)
                plot_performance(ep_rewards)
                break

    env.close()

if __name__ == '__main__':
    main()

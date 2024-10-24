import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),  # 可以尝试替换为 nn.Tanh()
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 输出为概率分布
        )

    def forward(self, x):
        return self.network(x)

# 定义价值网络（用于基线）
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出状态价值
        )

    def forward(self, x):
        return self.network(x)

# 折扣回报计算
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

    num_episodes = 1000
    gamma = 0.99

    for episode in range(num_episodes):
        state = env.reset()[0]
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []

        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            value = value_net(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # next_state, reward, done, _ = env.step(action.item())
            next_state, reward, terminated, truncated, info = env.step(action.item())
            if terminated or truncated:
                done = True
            else:
                done = False

            log_prob = dist.log_prob(action)
            ep_reward += reward

            # 存储数据
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state

            if done:
                # 计算优势 A(s,a) = G_t - V(s)
                returns = compute_returns(rewards, gamma)
                returns = torch.FloatTensor(returns).unsqueeze(1)
                values = torch.cat(values)
                advantages = returns - values.detach()

                # 更新策略网络
                policy_loss = - (torch.cat(log_probs) * advantages).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # 更新价值网络
                value_loss = nn.MSELoss()(values, returns)
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                print(f"Episode {episode+1}, Reward: {ep_reward}")
                break

    env.close()

if __name__ == "__main__":
    main()

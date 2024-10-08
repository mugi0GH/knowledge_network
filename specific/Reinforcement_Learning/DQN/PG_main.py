import gymnasium as gym
from Visualization.line_chart import plot_durations,plot_rewards
from Models.Policy_Gradient.reinforce import model
from itertools import count
import torch
import numpy as np

EPISODES = 1000
DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")
GAMMA = 0.99
rewards_record = []

def main():
    env = gym.make("LunarLander-v3",render_mode='human')
    state, info = env.reset()

    reinforce = model(input_size = state,output_size = env.action_space.n).to(DEVICE)

    for ep in range(EPISODES):
        state, info = env.reset()
        reward_steps = []
        trajectories = []
        returns = []
        G = 0
        rewards_ep=0

        for step in count():
            # 获取动作概率分布
            action_probs = reinforce(torch.tensor(state).to(DEVICE))
            
            # 使用概率分布采样动作
            # action_dist = torch.distributions.Categorical(action_probs)
            # action = action_dist.sample()
            action = torch.multinomial(action_probs, 1).item()

            # 记录每个动作的log概率
            # log_prob = action_dist.log_prob(action)
            log_prob = torch.log(action_probs[action])
            
            # 收集log概率
            trajectories.append(log_prob)

            observation, reward, terminated, truncated, info = env.step(int(action))

            # 收集奖励
            reward_steps.append(reward)      

            # episode总奖励
            rewards_ep+=reward      

            if terminated or truncated:
                # episode_durations.append(step + 1)
                # plot_durations(episode_durations)
                rewards_record.append(rewards_ep)
                plot_rewards(rewards_record)
                break

        # 在每个episode结束后，计算损失并进行优化
        # 累计回报R_t（对于每个t的折扣未来奖励之和）
        for r in reversed(reward_steps):
            G = r + GAMMA * G
            returns.insert(0, G)

        # 标准化或归一化奖励，可以帮助稳定训练 (optional)
        returns = torch.tensor(returns).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        reinforce.train(trajectories,returns)

    env.close()

if __name__ == '__main__':
    main()
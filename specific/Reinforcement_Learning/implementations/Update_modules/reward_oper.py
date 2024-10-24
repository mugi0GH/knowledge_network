import torch
def discounted_reward(rewards):
    # 计算每个动作对应的总回报
    discounted_rewards = []
    G = 0
    gamma = 0.99  # 折扣因子

    # 累计回报R_t（对于每个t的折扣未来奖励之和）
    for r in rewards[::-1]:
        G = r + gamma * G
        discounted_rewards.insert(0, G)
    
    # 将回报标准化（可选，提升稳定性）
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    return discounted_rewards
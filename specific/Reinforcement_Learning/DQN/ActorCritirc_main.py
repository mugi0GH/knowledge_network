import torch
import gymnasium as gym
from Models.Policy_Gradient.reinforce import actor
from Models.Policy_Gradient.criticer import critic
from Models.Policy_Gradient.integration import integrated_model
from itertools import count
from Visualization.line_chart import plot_rewards


DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")
hypers=dict(
{
    "EPISODEs": 1000,
    "GAMMA":0.99,
    "actor_LR":1e-4,
    "critic_LR":1e-3,
    "LR":1e-3, # 如果 actor critic的网络结构类似，或懒得各自设不同的学习速率
    "BATCH":64, # 5000
    "DEVICE":DEVICE,
    # TRPO参数
    "TRPO":False,
    "delta": 0.02 # KL 散度阈值
})
rewards_record = []

def main():
    # env = gym.make("LunarLander-v3")
    env = gym.make("CartPole-v1",render_mode="rgb_array")
    state, info = env.reset()
    hypers['state_shape'] = state.shape[0]
    state_dim = len(state)
    action_dim = env.action_space.n  # 动作的维度
    
    ac = actor(state_dim,action_dim,hypers=hypers)
    cr = critic(state_dim,action_dim,hypers=hypers)
    ac_cr = integrated_model(actor=ac,critic=cr,hypers=hypers)

    for ep in range(hypers['EPISODEs']):
        state, info = env.reset()
        next_state = None
        rewards_ep = 0
        done = False
        for step in count():
            action = ac_cr.act(state)
            # action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                done = True
            ac_cr.optimize(state, action, next_state, reward, done)
            
            state = next_state 

            rewards_ep+=reward
            
            # 处理终止条件
            if done:
                rewards_record.append(rewards_ep)
                plot_rewards(rewards_record)
                break

if __name__ == '__main__':
    main()
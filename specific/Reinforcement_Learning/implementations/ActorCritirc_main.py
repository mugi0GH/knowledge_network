import torch
import gymnasium as gym
from Models.Policy_Gradient.reinforce import actor
from Models.Policy_Gradient.criticer import critic
from Models.Policy_Gradient.integration import integrated_model
from itertools import count
from Visualization.line_chart import plot_rewards,plot_performance
import copy

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")
hypers=dict(
{
    "EPISODEs": 5000,
    "GAMMA":0.99,
    "actor_LR":1e-4,
    "critic_LR":1e-3,
    "LR":1e-4, # 如果 actor critic的网络结构类似，或懒得各自设不同的学习速率
    "BATCH":4000, # 5000
    "DEVICE":DEVICE,
    "n_step_TD": False,
    "v_loss_iter":80,
    "TD_lambda": 0.9, # TD(λ)的平滑因子 λ
    "TD_steps": 5, # 多步td
    # Vanilla Policy Gradient
    "VPG":True,
    # TRPO参数
    "TRPO":False,
    "max_iter": 50,
    "delta": 0.01, # KL 散度阈值
    "beta": 0.4   # 步长缩减系数，减少保守性
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
        done = 0
        for step in count():
            action = ac_cr.act(state)
            # action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                done = 1
            else:
                done = 0
            ac_cr.optimize(state, action, next_state, reward, done)

            state = copy.deepcopy(next_state)
                 
            # episode总奖励
            rewards_ep+=reward     

            # 处理终止条件
            if done:
                rewards_record.append(rewards_ep)
                # plot_rewards(rewards_record)
                plot_performance(rewards_record)
                break

if __name__ == '__main__':
    main()
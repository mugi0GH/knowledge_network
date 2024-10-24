import gymnasium as gym
from Visualization.line_chart import plot_durations,plot_rewards
from Models.Policy_Gradient.reinforce import actor
from Models.Policy_Gradient.criticer import critic
from itertools import count
import torch
import numpy as np
'''
reinforce通常也被视作VPG
原始版本为蒙特卡罗方法：回合结束后，计算R(τ)；
此时损失函数计算方式为：loss = -torch.sum(log_probs * returns)

加入优势函数后，则用：时序差分法
因为，优势函数有 值估计模型参与
# 计算 Actor 损失：基于 TD 误差的 Advantage = TD Error
loss = -(log_probs * td_error.detach()).mean()
'''
EPISODES = 1000
DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")
hypers=dict(
{
    "EPISODEs": 1000,
    "GAMMA":0.99,
    "LR":1e-3,
    "DEVICE":DEVICE,
    "epsidoe_update": True, # 基于回合更新
    "step_update": False,
    "v_loss_iter": 5, # 值估计模型 基于单步的迭代次数
    "if_baseline":True # 是否带基线
})
rewards_record = []

def main():
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v3")
    # env = gym.make("LunarLander-v3",render_mode='human')
    state, info = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    reinforce = actor(state_dim=state_dim,action_dim = action_dim,hypers=hypers).to(DEVICE)
    if hypers['if_baseline']:
        value_estimator = critic(state_dim=state_dim,hypers=hypers).to(DEVICE)

    for ep in range(EPISODES):
        state, info = env.reset()
        reward_steps = []
        log_probs = [] # 概率分布的底数的列表
        values = []
        rewards_ep=0
        done = 0
        for step in count():
            # 获取动作 及其 概率分布的底数
            action, log_prob = reinforce.act(state)

            next_state, reward, terminated, truncated, info = env.step(int(action))

            if terminated or truncated:
                done = 1
            else:
                done = 0

            '''数据收集'''
            if hypers['epsidoe_update']:
                value = value_estimator.forward(state)
                # 收集单步奖励
                reward_steps.append(reward)      
                # 计算值函数 V(s)
                log_probs.append(log_prob)
                values.append(value)
            elif hypers["step_update"]:
                td_error = value_estimator.optimize(state,next_state,reward,done)
                reinforce.train(log_prob,reward,td_error)
            state = next_state
            # episode总奖励
            rewards_ep+=reward      

            if done:
                # episode_durations.append(step + 1)
                # plot_durations(episode_durations)
                rewards_record.append(rewards_ep)
                plot_rewards(rewards_record,show_result=False)

                if hypers['if_baseline'] and hypers['epsidoe_update']:
                    reinforce.train(log_probs,reward_steps,values)
                if not hypers['if_baseline'] and hypers['epsidoe_update']:
                    reinforce.train(log_probs,reward_steps)
                break

    env.close()

if __name__ == '__main__':
    main()
import torch
import gymnasium as gym
from Models.Policy_Gradient.reinforce import actor
from Models.Policy_Gradient.criticer import critic
from Models.Policy_Gradient.integration import integrated_model
from itertools import count
from Visualization.line_chart import plot_rewards,plot_performance
import copy

# 1. 切换算法

# "VPG": False,
# "TRPO": True,

# 2. 调整 TRPO 相关参数
# "BATCH": 1000,    # TRPO 需要更多数据估计 Fisher 矩阵，从 500 → 1000
# "max_iter": 10,   # CG 通常 10 步内收敛，50 是浪费；line search 同用此值
# "delta": 0.01,    # KL 约束，标准值，可以先保持
# "beta": 0.5,      # 回溯系数从 0.4 → 0.5，标准值，步子缩减不用那么激进

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
    "LR":3e-4,
    "BATCH":1000,        # TRPO 建议 ≥500 步（CartPole ~25 局）；episode 模式下此值无效
    "DEVICE":DEVICE,
    # advantage 估计方式（三选一）
    "GAE": True,        # Generalized Advantage Estimation（推荐）
    "n_step_TD": False,
    "v_loss_iter":30,
    "TD_lambda": 0.95,  # GAE 的 λ，0.95 是 PPO 默认值；0=单步TD，1=MC
    "TD_steps": 5,
    # 更新频率：True=每回合更新，False=攒够 BATCH 步再更新
    "update_per_episode": False,
    # 算法选择（二选一）
    "VPG":True,
    "TRPO":False,
    # TRPO 参数
    "max_iter": 10,
    "delta": 0.01,
    "beta": 0.5
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
                print(f"Episode {ep+1}, Reward: {rewards_ep}")
                # plot_rewards(rewards_record)
                plot_performance(rewards_record)
                break

if __name__ == '__main__':
    main()
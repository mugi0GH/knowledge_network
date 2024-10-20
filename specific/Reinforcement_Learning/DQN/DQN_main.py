import gymnasium as gym
import highway_env
# from Models.DQN.vanilla_policy import model
from Models.DQN.Dueling_policy import model
from Models.DQN.integration import integrated_model
import torch
from itertools import count
from Visualization.line_chart import plot_durations,plot_rewards

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")
EPISODES = 1000
hypers=dict(
{
    "CAPACITY":1024,
    "BATCH_SIZE":128,
    "GAMMA":0.99,
    "EPS_START":0.9,
    "EPS_END":0.05,
    "EPS_DECAY":1000,
    "TAU":0.005,
    "LR":1e-4,
    "Noisy":False,
    "PER":False,
    "if_DDQN":True,
    "DEVICE":DEVICE
})
episode_durations = []
total_rewards = []

def main():
    # env = gym.make('highway-v0', render_mode='human')
    env = gym.make("CartPole-v1",render_mode="rgb_array")
    # env = gym.make("LunarLander-v2",render_mode='human')
    state, info = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    policy_net = model(state_dim = state_dim,action_dim = action_dim,noisy_net = hypers['Noisy'],role = 'policy').to(DEVICE)
    target_net = model(state_dim = state_dim,action_dim = action_dim,noisy_net = hypers['Noisy'],role = 'target').to(DEVICE)

    dqn = integrated_model(policy_net,target_net,hypers)

    for ep in range(EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dqn.state = state
        rewards_ep = 0
        done = False
        for step in count():
            # env.action_space.sample()
            action = dqn.select_action()
            if action is None:
                action = torch.tensor([[env.action_space.sample()]], device=hypers["DEVICE"], dtype=torch.long)

            observation, reward, terminated, truncated, info = env.step(int(action))
            rewards_ep+=reward
            if terminated or truncated:
                done = True
            dqn.train(observation,action,reward,terminated,done)
            if done:
                # episode_durations.append(step + 1)
                # plot_durations(episode_durations)
                total_rewards.append(rewards_ep)
                plot_rewards(total_rewards)
                break

    env.close()

if __name__ == '__main__':
    main()
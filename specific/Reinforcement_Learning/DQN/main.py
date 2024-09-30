import gymnasium as gym
from Models.DQN.vanilla_policy import model
from Models.DQN.integration import integrated_model
import torch
from itertools import count
from Visualization.line_chart import plot_durations

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")
EPISODES = 1000
hypers={
        "CAPACITY":1024,
        "BATCH_SIZE":128,
        "GAMMA":0.99,
        "EPS_START":0.9,
        "EPS_END":0.05,
        "EPS_DECAY":1000,
        "TAU":0.005,
        "LR":1e-4,
        "DEVICE":DEVICE
    }
episode_durations = []

def main():
    # env = gym.make("CartPole-v1")# ,render_mode='human'
    env = gym.make("CartPole-v1",render_mode='human')
    state, info = env.reset()

    policy_net = model(state,env.action_space.n).to(DEVICE)
    target_net = model(state,env.action_space.n).to(DEVICE)
    dqn = integrated_model(policy_net,target_net,hypers)

    for ep in range(EPISODES):
        state, info = env.reset(seed=42)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dqn.state = state

        for step in count():
            # action = int(policy_net(state)) # env.action_space.sample()
            action = dqn.select_action()
            if action is None:
                action = torch.tensor([[env.action_space.sample()]], device=hypers["DEVICE"], dtype=torch.long)

            observation, reward, terminated, truncated, info = env.step(int(action))

            dqn.train(observation,action,reward,terminated)
            if terminated or truncated:
                episode_durations.append(step + 1)
                plot_durations(episode_durations)
                break

    env.close()

if __name__ == '__main__':
    main()
import torch
import math
import random
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from algorithms.agents.DQN.Tricks.Replay_memory import ReplayMemory
from algorithms.agents.DQN.policies.MultiLayerPerceptron import model as DQN
import numpy as np
from algorithms.agents.DQN.components import optimize_model,eps_decay

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
class model():
    def __init__(self,action_space,state) -> None:
        self.BATCH_SIZE = 2 # 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.episode_durations = []
        self.steps_done = 0
        
        LR = 1e-4

        # if GPU is to be used
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
            )
            
        self.action_space = action_space
    # Get number of actions from gym action space
    # n_actions = env.action_space.n
        n_actions = action_space.n
    # Get the number of state observations
    # state, info = env.reset()
        # n_observations = n_observations

        self.policy_net = DQN(state, n_actions).to(self.device)
        self.target_net = DQN(state, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(150000)
    def optimization(self):
        return optimize_model(self)
    
    def select_action(self,state):
        return eps_decay(self,state)

    def plot_durations(self):
        return plot_durations(self)

    def step(self,state):
        # state = torch.tensor(np.reshape(state, (-1, 1)), dtype=torch.float).to(self.device)
        action = self.select_action(state)
        self.optimization()
        return action

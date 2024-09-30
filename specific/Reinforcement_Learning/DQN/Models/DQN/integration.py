import torch
from Update_modules.off_policy.memory.ExperienceReplay import ReplayMemory
from torch import nn
import random
import math
import matplotlib.pyplot as plt

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class integrated_model:
    def __init__(self,policy_net,target_net,hypers,DQN=False) -> None:
        if hypers is None:
            self.hypers=dict(
            {
                "CAPACITY":1024,
                "BATCH_SIZE":128,
                "GAMMA":0.99,
                "EPS_START":0.9,
                "EPS_END":0.05,
                "EPS_DECAY":1000,
                "TAU":0.005,
                "LR":1e-4,
            })

        self.DQN = DQN
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(policy_net.state_dict())
        self.hypers = hypers
        self.steps_done = 0
        self.optimizer = torch.optim.AdamW(policy_net.parameters(),lr=float(self.hypers["LR"]),amsgrad = True)
        self.memory_pool = ReplayMemory(hypers["CAPACITY"])
        self.episode_durations = []

    def train(self,observation,action,reward,terminated): 
        reward = torch.tensor([reward],device=self.hypers["DEVICE"])

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation,device=self.hypers["DEVICE"]).unsqueeze(0)

        self.memory_pool.push(self.state,action,next_state,reward)

        self.state = next_state

        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.hypers["TAU"] + target_net_state_dict[key]*(1-self.hypers["TAU"])
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.memory_pool) < self.hypers["BATCH_SIZE"]:
            return
        transitions = self.memory_pool.sample(self.hypers["BATCH_SIZE"])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.hypers["DEVICE"], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.hypers["BATCH_SIZE"], device=self.hypers["DEVICE"])

        if self.DQN:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        else:
            with torch.no_grad():
                # 用在线网络选择下一个动作
                next_actions = self.policy_net(non_final_next_states).max(1).indices
                # 用目标网络评估该动作的Q值
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hypers["GAMMA"]) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self):
        # global steps_done
        sample = random.random()
        eps_threshold = self.hypers["EPS_END"] + (self.hypers["EPS_START"] - self.hypers["EPS_END"]) * \
            math.exp(-1. * self.steps_done / self.hypers["EPS_DECAY"])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(self.policy_net(self.state).max(1).indices.view(1))
                return self.policy_net(self.state).max(1).indices.view(1, 1)
            # return torch.tensor([[env.action_space.sample()]], device=self.hypers["DEVICE"], dtype=torch.long)
        return None
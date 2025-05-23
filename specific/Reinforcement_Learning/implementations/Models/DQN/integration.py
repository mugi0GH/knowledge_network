import torch
from Update_modules.memory.ExperienceReplay import ReplayMemory
from Update_modules.memory.PrioritizedExperienceReplay import SumTree
from torch import nn
import random
import math
import matplotlib.pyplot as plt

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

class integrated_model:
    def __init__(self,policy_net,target_net,hypers) -> None:
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
                "Noisy":False,
                "PER":False,
                "if_DDQN":True
            })

        # self.if_DDQN = self.hypers["if_DDQN"] # 网络结构
        # self.PER = self.hypers["PER"]
        self.policy_net = policy_net # 在线网络 或 策略网络
        self.target_net = target_net # 目标网络 或 评价网络

        self.target_net.load_state_dict(policy_net.state_dict()) # 权重复制 保证在初始化后权重一样
        self.hypers = hypers # 超参
        self.steps_done = 0 # 每个轮回里的步数
        # 权重更新函数
        self.optimizer = torch.optim.AdamW(policy_net.parameters(),lr=float(self.hypers["LR"]),amsgrad = True)

        #是否使用PER
        if self.hypers["PER"]:
            self.memory_pool = SumTree(hypers["CAPACITY"])
            self.epsilon = 1e-5
        else:
            self.memory_pool = ReplayMemory(hypers["CAPACITY"])
        
        # 每轮回的训练结果
        self.episode_durations = []

    def train(self,observation,action,reward,terminated,done): 
        reward = torch.tensor([reward],device=self.hypers["DEVICE"])

        if terminated:
            if self.hypers['PER']:
                next_state = torch.zeros_like(torch.tensor(observation),device=self.hypers["DEVICE"]).unsqueeze(0)
            else:
                next_state = None
        else:
            next_state = torch.tensor(observation,device=self.hypers["DEVICE"]).unsqueeze(0)

        # 记录动作，状态，下个状态与奖赏的关系到经验池内
        if self.hypers["PER"]:
            self.memory_pool.push(priority=self.memory_pool.total_priority()+1,data=Transition(self.state,action,next_state,reward))
        else:
            self.memory_pool.push(self.state,action,next_state,reward,done)
            
        self.state = next_state

        # 优化权重
        self.optimize_model()

        # 软更新目标网络权重，硬更新为直接复制没有tau
        # θ′ ← τ θ + (1 −τ )θ′
        # 获取权重
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        # 更新权重
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.hypers["TAU"] + target_net_state_dict[key]*(1-self.hypers["TAU"])
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.memory_pool) < self.hypers["BATCH_SIZE"]:
            return
        
        if self.hypers["PER"]:
            segment = self.memory_pool.total_priority() / self.hypers["BATCH_SIZE"]  # 将总优先级划分为 batch_size 段
            batch = [] # data batch
            leaf_index_batch = [] # 子叶节点
            tree_batch = [] # 被抽到的索引
            for i in range(self.hypers["BATCH_SIZE"]):
                sample = random.uniform(segment * i, segment * (i + 1))  # 从每个段中均匀采样
                SumTree_batch = self.memory_pool.get_leaf(sample)
                if type(SumTree_batch[2]) is int:
                    print("Bug dectated",SumTree_batch[2])
                    exit()
                batch.append(SumTree_batch[2])  # 从 Sum Tree 中采样
                tree_batch.append(SumTree_batch[1])
                leaf_index_batch.append(SumTree_batch[0])
                # print(sample,SumTree_batch[1],tree_batch[0])
            batch = Transition(*zip(*batch))
        else:
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

        if self.hypers["if_DDQN"]:
            with torch.no_grad():
                # 用在线网络选择下一个动作
                next_actions = self.policy_net(non_final_next_states).max(1).indices
                # 用目标网络评估该动作的Q值
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hypers["GAMMA"]) + reward_batch

        # 计算 Huber损失，时序差分
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.hypers["PER"]:
            # 基于误差量决定优先级
            td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1))
            # priority = abs(loss) + self.epsilon
            for i in range(self.hypers["BATCH_SIZE"]):
                # 假设 sampled_indices 是 Sum Tree 中采样的经验索引
                # 这里更新第 i 个样本的优先级，通常加上一个 epsilon 防止优先级为零
                new_priority = td_errors[i].item() + self.epsilon
                self.memory_pool.update(leaf_index_batch[i], new_priority)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self):
        self.steps_done += 1
        if self.hypers["Noisy"]:
             with torch.no_grad():
                return self.policy_net(self.state).max(1).indices.view(1, 1)

        # global steps_done
        sample = random.random()
        eps_threshold = self.hypers["EPS_END"] + (self.hypers["EPS_START"] - self.hypers["EPS_END"]) * \
            math.exp(-1. * self.steps_done / self.hypers["EPS_DECAY"])
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(self.policy_net(self.state).max(1).indices.view(1))
                return self.policy_net(self.state).max(1).indices.view(1, 1)
            # return torch.tensor([[env.action_space.sample()]], device=self.hypers["DEVICE"], dtype=torch.long)
        return None
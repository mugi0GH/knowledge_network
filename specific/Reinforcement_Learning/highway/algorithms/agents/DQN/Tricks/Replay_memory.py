from collections import namedtuple, deque
import random
import torch
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'info'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        # return zip(*random.sample(self.memory, batch_size))
        states, actions, next_states, rewards = zip(*random.sample(self.memory, batch_size))
        return (torch.FloatTensor(states).to('cuda'),
                torch.LongTensor(actions).unsqueeze(0).to('cuda'),
                torch.FloatTensor(next_states).to('cuda'),
                torch.FloatTensor(rewards).to('cuda')
                # torch.FloatTensor(dones).to('cuda')
                )
    def __len__(self):
        return len(self.memory)

class important_sampling:
    def __init__(self) -> None:
        pass
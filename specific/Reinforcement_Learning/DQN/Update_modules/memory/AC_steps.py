import torch

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else
"mps" if torch.backends.mps.is_available() else
"cpu")


class batch():
    def __init__(self,capacity,state_shape) -> None:
        self.capacity = capacity
        self.state_shape = state_shape

        self.states = torch.empty(capacity,state_shape).to(DEVICE)
        self.actions = torch.empty(capacity,1).to(DEVICE)
        self.next_states = torch.empty(capacity,state_shape).to(DEVICE)
        self.rewards = torch.empty(capacity,1).to(DEVICE)
        self.dones = torch.empty(capacity,1).to(DEVICE)
        self.td_errors = torch.empty(capacity,1).to(DEVICE)
        self.log_probs = torch.empty(capacity,1).to(DEVICE)

        self.ele_num = 0

    def push(self,state, action, next_state, reward, done,td_error):
        # 将数据放入相应的张量位置
        self.states[self.ele_num] = torch.tensor(state).to(DEVICE)
        self.actions[self.ele_num] = torch.tensor(action).to(DEVICE)
        self.next_states[self.ele_num] = torch.tensor(next_state).to(DEVICE)
        self.rewards[self.ele_num] = torch.tensor(reward).to(DEVICE)
        self.dones[self.ele_num] = torch.tensor(done).to(DEVICE)
        self.td_errors[self.ele_num] = torch.tensor(done).to(DEVICE)

        self.ele_num+=1

    def reset(self):
        self.states = torch.empty(self.capacity,self.state_shape).to(DEVICE)
        self.actions = torch.empty(self.capacity,1).to(DEVICE)
        self.next_states = torch.empty(self.capacity,self.state_shape).to(DEVICE)
        self.rewards = torch.empty(self.capacity,1).to(DEVICE)
        self.dones = torch.empty(self.capacity,1).to(DEVICE)

        self.ele_num = 0
    def __len__(self):
        return self.ele_num
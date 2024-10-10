import torch
from torch import nn
# from Structure_modules.noisy_linear import NoisyLinear
from torchrl.modules.models.exploration import NoisyLinear

class model(nn.Module):
    def __init__(self,state_dim,action_dim,noisy_net=False,role = '') -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
            )
        if noisy_net:
            self.state_value = nn.Sequential(
                NoisyLinear(512,256),
                # nn.ReLU(),
                NoisyLinear(256,1), # 输出单一状态价值
                ) 
            self.advantages = nn.Sequential(
                NoisyLinear(512,256),
                # nn.ReLU(),
                NoisyLinear(256,output_size), # 输出与动作数目相同的优势值
                )
        else:
            self.state_value = nn.Sequential(
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,1), # 输出单一状态价值
                ) 
            self.advantages = nn.Sequential(
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,action_dim) # 输出与动作数目相同的优势值
                )
            
        if role == 'target':
            for module in self.modules(): 
                module.training = False
                # if isinstance(module, NoisyLinear):
                #     print(module.training)
            
    def forward(self,x):
        # 通过特征提取层
        x = self.backbone(x)
        # 计算状态价值和优势值
        state_value = self.state_value(x)  # 输出形状为 [batch_size, 1]
        advantages = self.advantages(x) # 输出形状为 [batch_size, num_actions]

        # 组合状态价值和优势值
        q_values = state_value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
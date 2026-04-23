from torch import nn
import torch
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model 必须为偶数"
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # 根据pos和i创建一个常量PE矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for two_i in range(0, d_model, 2):
                pe[pos, two_i] = math.sin(pos / (10000 ** (two_i / d_model)))
                pe[pos, two_i + 1] = math.cos(pos / (10000 ** (two_i / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # pe 已是 buffer，会随模块 .to(device) 自动迁移，无需手动 .cuda()
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

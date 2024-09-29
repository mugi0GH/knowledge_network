# import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,heads,d_model,dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # 线性层
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model,d_model)

    def attention(q,k,v,d_k,mask=None,dropout = None):
        
        # 注意力公式
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)

        # 掩盖那些为了补全长而增加的单元，使其通过Softmax计算后为0
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0. -1e9)

        scores = F.softmax(scores,dim = -1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores,v)
        return output
    
    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # 向量存值
        k = self.k_linear(k).view(bs,-1,self.h,self.d_k)
        q = self.k_linear(q).view(bs,-1,self.h,self.d_k)
        v = self.k_linear(v).view(bs,-1,self.h,self.d_k)

        # 矩阵转置, K,Q,V
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # 计算attention
        scores = attention(q,k,v,self.d_k,mask,self.dropout)

        # 连接多个头并输入最后的线性层
        concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)

        output = self.out(concat)

        return output
from torch import nn
from N0_Utils import Embedder, get_clones
from N2_MultiHeadAttention import MultiHeadAttention
from N3_FeedForward import FeedForward
from N4_Resi_Norm import Norm
from N1_PositionalEncoder import PositionalEncoder

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads,d_model, dropout = dropout)
        self.ff = FeedForward(d_model,dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # Post‑LayerNorm：原始设计，在每个子层之后进行残差连接和层归一化。
    # def forward(self,x,mask):
    #     attn_output = self.attn(x,x,x,mask)
    #     attn_output = self.dropout_1(attn_output)
    #     x=x+attn_output
    #     x=self.norm_1(x)
    #     ff_output = self.ff(x)
    #     ff_output = self.dropout_2(ff_output)
    #     x = x + ff_output
    #     x = self.norm_2(x)
    #     return x
    
    # Pre‑LayerNorm：在每个子层之前进行层归一化，然后再进行残差连接。这种设计有助于稳定训练，尤其是在深层网络中。
    '''
    训练稳定性高：残差连接直接传递未缩放的输入，避免了 Post‑LN 中早期梯度消失/爆炸的问题。
    可以去掉 warm‑up：Post‑LN 通常需要学习率 warm‑up 来稳定初期训练，Pre‑LN 往往不需要。
    更深的模型：Pre‑LN 在几百层（如 GPT‑3、LLaMA）上表现良好，而 Post‑LN 很难训练极深网络。
    '''
    def forward(self, x, mask):
        # 注意：先 norm，再进 attention
        norm_x = self.norm_1(x)
        attn_output = self.attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout_1(attn_output)

        # FFN 部分同理
        norm_x = self.norm_2(x)
        ff_output = self.ff(norm_x)
        x = x + self.dropout_2(ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size,d_model)
        self.pe = PositionalEncoder(d_model,dropout = dropout)
        self.layers = get_clones(EncoderLayer(d_model,heads,dropout),N)
        self.norm = Norm(d_model)

    def forward(self,src,mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        return self.norm(x)
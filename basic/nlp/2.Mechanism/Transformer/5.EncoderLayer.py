class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.l):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads,d_model, dropout = dropout)
        self.ff = FeedForward(d_model,dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output = self.attn(x,x,x,mask)
        attn_output = self.dropout_1(attn_output)
        x=x+attn_output
        x=self.norm_1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm_2(x)
        return x

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout):
        super.__init__()
        self.N = N
        self.embed = Embedder(vocab_size,d_model)
        self.pe = PositionalEncodewr(d_model,dropout = dropout)
        self.layers = get_clones(EncoderLayer(d_model,heads,dropout),N)
        self.norm = Norm(d_model)

    def forward(self,src,mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        return self.norm(x)
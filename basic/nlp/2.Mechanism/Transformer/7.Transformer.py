class Transformer(nn.Module):
    def __init__(self,src_vocab,trg_vocab,d_model,N,heads,dropout):
        super.__init__()
        self.encoder = Encoder(src_vocab,d_model,N,heads,dropout)
        self.decoder = Decoder(trg_vocab,d_model,N,heads,dropout)
        self.out = nn.Linear(d_model,trg_vocab)

    def forward(self,src,trg,src_mask,trg_mask):
        e_outpouts = self.encoder(src,src_mask)
        d_output = self.decoder(trg,e_outpouts,src_mask,trg_mask)
        d_output = self.out(d_output)
        output = self.out(d_output)
        return output
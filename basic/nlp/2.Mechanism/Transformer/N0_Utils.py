import copy
from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

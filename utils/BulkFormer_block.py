import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from performer_pytorch import Performer

class BulkFormer_block(nn.Module):
    def __init__(self, dim, gene_length, bin_head=4, full_head=4, bins=10, p_repeat=1):
        super().__init__()
        self.dim = dim
        self.gene_length = gene_length
        self.bins = bins
        self.p_repeat = p_repeat
        self.bin_head = bin_head
        self.full_head = full_head

        # 图卷积层
        self.g = GCNConv(dim, dim, cached=True, add_self_loops=False)

        # 全局 performer
        self.f = nn.Sequential(*[
            Performer(dim=self.dim, heads=self.full_head, depth=1,
                      dim_head=self.dim // self.full_head,
                      attn_dropout=0.05, ff_dropout=0.1)
            for _ in range(self.p_repeat)
        ])

        self.layernorm = nn.LayerNorm(self.dim)

    def forward(self, x, graph):
 
        # === 图卷积 ===
        x = self.layernorm(x)
        x = x + self.g(x, graph)
        # === performer ===
        x = self.f(x)

        return x




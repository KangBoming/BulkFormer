import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from performer_pytorch import Performer


class GBFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        gene_length: int,
        bin_head: int = 4,
        full_head: int = 4,
        bins: int = 10,
        p_repeat: int = 1,
    ):
        super().__init__()

        self.dim = dim
        self.gene_length = gene_length
        self.bins = bins
        self.p_repeat = p_repeat
        self.bin_head = bin_head
        self.full_head = full_head

        self.g = GCNConv(dim, dim, cached=True, add_self_loops=False)
        # self.g = GATv2Conv(dim, dim, add_self_loops=False)
        self.which_b = nn.Sequential(
            nn.Linear(self.dim, 1),
        )
        self.b = nn.ModuleList([
            Performer(
                dim=self.dim,
                heads=self.bin_head,
                depth=1,
                dim_head=self.dim // self.bin_head,
                attn_dropout=0.2,
                ff_dropout=0.2,
            )
            for _ in range(self.bins)
        ])
        self.f = nn.Sequential(*[
            Performer(
                dim=self.dim,
                heads=self.full_head,
                depth=1,
                dim_head=self.dim // self.full_head,
                attn_dropout=0.2,
                ff_dropout=0.2,
            )
            for _ in range(self.p_repeat)
        ])
        self.layernorm = nn.LayerNorm(self.dim)
        # self.layer_pos_emb = FixedPositionalEmbedding(self.dim//self.bin_head, self.gene_length)
    
    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        b, g, e = x.shape

        x = self.layernorm(x)
        x = x + self.g(x, graph)

        if self.bins > 0:
            # sort
            which_b = self.which_b(x).squeeze(-1)  # [B, G]
            order = torch.sort(which_b, dim=1, descending=True)[1]
            order = order.unsqueeze(-1).repeat(1, 1, e)
            n = (g - 1) // self.bins + 1
            
            # forward
            x = x.gather(1, order)
            # layer_pos_emb = self.layer_pos_emb(x)
            xs = torch.split(x, n, dim=1)
            # ps = torch.split(layer_pos_emb, n, dim=1)
            xs = [
                layer(x)
                for x, layer in zip(xs, self.b)
            ]
            xs = torch.cat(xs, dim=1)
            
            # reverse
            x = torch.empty_like(xs)
            x = x.scatter_(1, order, xs)
        
        x = self.f(x)
        
        return x


import torch
import torch.nn as nn
from bulkformer.models.encoder import GBFormer
from bulkformer.models.rope import PositionalExprEmbedding


class BulkFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        graph: torch.Tensor,
        gene_emb: torch.Tensor,
        gene_length: int,
        bin_head: int = 4,
        full_head: int = 4,
        bins: int = 10,
        gb_repeat: int = 3,
        p_repeat: int = 1,
    ):
        super().__init__()
        
        self.dim = dim
        self.gene_length = gene_length
        
        self.bins = bins
        self.bin_head = bin_head
        self.full_head = full_head

        self.gb_repeat = gb_repeat
        self.p_repeat = p_repeat

        self.graph = graph

        
        self.gene_emb = nn.Parameter(gene_emb)

        self.gene_emb_proj = nn.Sequential(
            nn.Linear(self.gene_emb.shape[1], 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
        )
        
        self.expr_emb = PositionalExprEmbedding(self.dim) 

        self.x_proj = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
        )

        self.gb_formers = nn.ModuleList([
            GBFormer(self.dim, self.gene_length, self.bin_head, self.full_head, self.bins, self.p_repeat)
            for _ in range(self.gb_repeat)
        ])

        self.layernorm = nn.LayerNorm(self.dim)
        
        self.ae_enc = nn.Sequential(
            nn.Linear(self.gene_length, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
        )
        
        self.head = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, repr_layers: list[int] | None = None) -> torch.Tensor | tuple[torch.Tensor, dict]:
        b, g = x.shape
        
        x = self.expr_emb(x) + self.gene_emb_proj(self.gene_emb) + self.ae_enc(x).unsqueeze(1)
        x = self.x_proj(x)

        hidden = {}
        for idx, layer in enumerate(self.gb_formers):
            x = layer(x, self.graph)
            if repr_layers and idx in repr_layers:
                hidden[idx] = x
                
        x = self.layernorm(x)
        if repr_layers and idx in repr_layers:
            hidden[idx] = x
        
        x = self.head(x).squeeze(-1)

        if repr_layers:
            return x, hidden
        else:
            return x


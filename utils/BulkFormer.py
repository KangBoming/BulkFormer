import torch
import torch.nn as nn
from utils.BulkFormer_block import BulkFormer_block
from utils.Rope import PositionalExprEmbedding


class BulkFormer(nn.Module):
    """BulkFormer model for gene-level representation learning and expression prediction."""

    def __init__(self, 
                 dim, graph, gene_emb, gene_length,
                 bin_head=4, full_head=4, bins=10,
                 gb_repeat=3, p_repeat=1):
        super().__init__()
        self.dim = dim
        self.gene_length = gene_length
        self.graph = graph

        # Legacy trainable gene embedding path (kept for reference).
        # self.gene_emb = nn.Parameter(gene_emb)

        # Gene embedding initialized from one-hot index embedding.
        self.gene_emb_onehot_layer = nn.Embedding(gene_length, dim)
        nn.init.xavier_uniform_(self.gene_emb_onehot_layer.weight)


        self.gene_emb_proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        self.expr_emb = PositionalExprEmbedding(dim)
        self.x_proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        # Main encoder blocks.
        self.gb_formers = nn.ModuleList([
            BulkFormer_block(dim, gene_length, bin_head, full_head, bins, p_repeat)
            for _ in range(gb_repeat)
        ])

        self.layernorm = nn.LayerNorm(dim)

        # Sample-level global expression context.
        self.global_expr_proj = nn.Sequential(
            nn.Linear(gene_length, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        # Per-gene prediction head.
        self.head = nn.Sequential(
            nn.LayerNorm(dim + 3),
            nn.Linear(dim + 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.ReLU()
        )

    def forward(self, x, mask_prob=None, output_expr = False):
        """Forward pass.

        Args:
            x: Input expression tensor of shape [batch, genes].
            mask_prob: Optional scalar mask ratio used as an auxiliary feature.
            output_expr: If True, return corrected gene expression predictions;
                otherwise return enriched gene embeddings.
        """
        b, g = x.shape
        x_input = x.clone()

        gene_emb_onehot = self.gene_emb_onehot_layer.weight
        # Build token embedding from expression + gene identity + sample context.
        gene_emb_proj = self.gene_emb_proj(gene_emb_onehot)

        x = self.expr_emb(x) + gene_emb_proj + self.global_expr_proj(x).unsqueeze(1).expand(-1, g, -1)

        x = self.x_proj(x)

        # Encoder trunk.
        for layer in self.gb_formers:
            x = layer(x, self.graph)
       
        # Gene token outputs.
        gene_emb = self.layernorm(x)

        # Sample-level statistical features.
        # Mask ratio feature.
        mask_scalar = torch.full((b, g, 1), mask_prob or 0.0, device=x.device)
        # Mask indicators.
        mask_token_val = -10.0
        mask = (x_input == mask_token_val).float()       # 1 indicates masked genes
        valid_mask = 1 - mask                            # 1 indicates observed genes

        # Mean expression over observed (non-masked) genes.
        expr_mean = (x_input * valid_mask).sum(dim=1, keepdim=True) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        expr_mean = expr_mean.unsqueeze(-1).expand(-1, g, -1)

        # Non-zero expression ratio.
        nonzero_ratio = (x_input != 0).float().sum(dim=1, keepdim=True) / g
        nonzero_ratio = nonzero_ratio.unsqueeze(-1).expand(-1, g, -1)

        # Concatenate auxiliary features: [batch, genes, dim + 3].
        gene_emb_output = torch.cat([gene_emb, mask_scalar, expr_mean, nonzero_ratio],dim=-1)  

        # Per-gene prediction.
        pred = self.head(gene_emb_output).squeeze(-1)

        # Mean-correction to align masked-position predictions with observed scale.
        # 1) Mean of predictions on observed genes.
        pred_valid_mean = (pred * valid_mask).sum(dim=1, keepdim=True) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        # 2) Mean of observed expressions on observed genes.
        observed_mean = (x_input * valid_mask).sum(dim=1, keepdim=True) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        # 3) Apply correction only to masked positions.
        pred_corrected = pred.clone()
        pred_corrected = pred_corrected - mask * (pred_valid_mean - observed_mean)

        if output_expr:
            return pred_corrected
        else:
            return gene_emb_output




from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor):
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))


class PandoraEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim, dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor):
        batch, length, _ = x.shape
        residual = x
        qkv = self.qkv(self.attn_norm(x)).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        key_mask = attention_mask[:, None, None, :].bool()
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.dim)
        x = residual + self.dropout(self.out_proj(attended))
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class BulkFormerAtlas(nn.Module):
    """Pandora-inspired encoder trained with symmetric InfoNCE; see docs/models/atlas.md."""

    def __init__(
        self,
        num_genes: int,
        dim: int = 2048,
        layers: int = 16,
        heads: int = 16,
        ffn_dim: int = 5376,
        projection_dim: int = 512,
        projection_hidden_dim: int | None = None,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        self.cls_token_id = self.num_genes
        self.gradient_checkpointing = gradient_checkpointing
        projection_hidden_dim = projection_hidden_dim or dim

        self.gene_embedding = nn.Embedding(self.num_genes + 1, dim)
        self.expression_projection = nn.Linear(1, dim)
        self.input_norm = RMSNorm(dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [PandoraEncoderBlock(dim, heads, ffn_dim, dropout) for _ in range(layers)]
        )
        self.output_norm = RMSNorm(dim)

        self.masked_expression_head = SwiGLU(dim, ffn_dim, 1, dropout)
        self.constructive_sample_head = SwiGLU(dim, projection_hidden_dim, projection_dim, dropout)
        self.constructive_gene_head = SwiGLU(dim, projection_hidden_dim, projection_dim, dropout)
        self.contrastive_head = SwiGLU(dim, projection_hidden_dim, projection_dim, dropout)

        self.constructive_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.constructive_bias = nn.Parameter(torch.tensor(0.0))
        self.contrastive_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.contrastive_bias = nn.Parameter(torch.tensor(-10.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.expression_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.expression_projection.bias)

    def encode(self, gene_ids: Tensor, expression: Tensor, attention_mask: Tensor):
        x = self.gene_embedding(gene_ids.long()) + self.expression_projection(expression.float().unsqueeze(-1))
        x = self.input_dropout(self.input_norm(x))
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        return self.output_norm(x)

    def forward_view(self, view: dict[str, Tensor]):
        hidden = self.encode(view["gene_ids"], view["input_expr"], view["attention_mask"])
        masked_prediction = self.masked_expression_head(hidden).squeeze(-1)

        sample_embedding = hidden[:, 0]
        sample_factor = F.normalize(self.constructive_sample_head(sample_embedding), dim=-1)
        raw_gene_embedding = self.gene_embedding(view["gene_ids"])
        gene_factor = F.normalize(self.constructive_gene_head(raw_gene_embedding), dim=-1)
        constructive_scale = self.constructive_log_scale.exp().clamp(max=100.0)
        constructive_prediction = constructive_scale * (sample_factor.unsqueeze(1) * gene_factor).sum(-1)
        constructive_prediction = constructive_prediction + self.constructive_bias

        contrastive_embedding = F.normalize(self.contrastive_head(sample_embedding), dim=-1)
        return {
            "hidden": hidden,
            "sample_embedding": sample_embedding,
            "masked_prediction": masked_prediction,
            "constructive_prediction": constructive_prediction,
            "contrastive_embedding": contrastive_embedding,
        }

    def forward(self, view1: dict[str, Tensor], view2: dict[str, Tensor]):
        output1 = self.forward_view(view1)
        output2 = self.forward_view(view2)
        # Returning these parameters keeps them inside DDP's traversed graph.
        output1["contrastive_log_scale"] = self.contrastive_log_scale
        output1["contrastive_bias"] = self.contrastive_bias
        return output1, output2

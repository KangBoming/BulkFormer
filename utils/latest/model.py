from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from performer_pytorch import Performer
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from .graph import WeightedGeneGraph


@dataclass(frozen=True)
class BulkFormerV1PlusConfig:
    num_genes: int = 19_973
    dim: int = 768
    depth: int = 12
    heads: int = 12
    dim_head: int = 64
    ff_mult: int = 4
    nb_features: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.05
    graph_hidden_dim: int = 192
    graph_dropout: float = 0.1
    graph_scale_init: float = 0.01
    expression_base: float = 100.0
    expression_scale_init: float = 1.0
    mask_value: float = -10.0
    mask_ratio_scaling_enabled: bool = True
    mask_ratio_reference: float = 0.10
    mask_ratio_max_scale: float = 2.0
    mask_ratio_min_visible_fraction: float = 0.05
    fix_performer_projection_matrices: bool = True


class FixedSinusoidalExpressionEncoder(nn.Module):
    """V1 scalar encoding with balanced RMS and an all-zero mask representation."""

    def __init__(self, dim: int, base: float = 100.0, mask_value: float = -10.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("expression encoder dimension must be a positive even number")
        if base <= 1.0:
            raise ValueError("expression encoding base must be greater than one")
        self.dim = int(dim)
        self.mask_value = float(mask_value)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, expression: Tensor) -> Tensor:
        if expression.ndim != 2:
            raise ValueError("expression must have shape [batch, genes]")
        masked = expression.eq(self.mask_value)
        safe_expression = expression.masked_fill(masked, 0.0).float()
        phase = safe_expression.unsqueeze(-1) * self.inv_freq
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1)
        # Each observed sine/cosine pair has squared norm one. sqrt(2)
        # therefore gives every observed token exact RMS one across channels.
        encoded = encoded * math.sqrt(2.0)
        return encoded.masked_fill(masked.unsqueeze(-1), 0.0)


class NonAffineRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(value, (self.dim,), weight=None, eps=self.eps)


class GatedWeightedGCNBlock(nn.Module):
    """One-hop, pre-norm, bottleneck GCN with a near-zero residual gate."""

    def __init__(self, config: BulkFormerV1PlusConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.dim)
        self.input_projection = nn.Linear(
            config.dim, config.graph_hidden_dim, bias=False
        )
        self.convolution = GCNConv(
            config.graph_hidden_dim,
            config.graph_hidden_dim,
            cached=True,
            add_self_loops=False,
            normalize=True,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.graph_hidden_dim, config.dim, bias=False
        )
        self.dropout = nn.Dropout(config.graph_dropout)
        self.residual_scale = nn.Parameter(
            torch.full((config.dim,), config.graph_scale_init)
        )

    def forward(
        self,
        hidden: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        update = self.input_projection(self.norm(hidden))
        update = self.convolution(update, edge_index, edge_weight)
        update = self.output_projection(F.gelu(update))
        return hidden + self.residual_scale * self.dropout(update)


class BulkFormerV1Plus(nn.Module):
    """V1Plus full-vocabulary masked-expression model.

    The sequence always contains exactly ``num_genes`` gene tokens. There is no
    sample token, state embedding, global dense expression projection, or sample
    pooling path. Source-unavailable and dynamically masked values share the
    all-zero expression encoding; only the external loss mask distinguishes them.
    """

    def __init__(
        self,
        config: BulkFormerV1PlusConfig,
        graph: WeightedGeneGraph,
    ):
        super().__init__()
        _validate_config(config, graph)
        self.config = config

        self.register_buffer("edge_index", graph.edge_index, persistent=False)
        self.register_buffer("edge_weight", graph.edge_weight, persistent=False)

        self.gene_embedding = nn.Embedding(config.num_genes, config.dim)
        nn.init.normal_(
            self.gene_embedding.weight, mean=0.0, std=1.0 / math.sqrt(config.dim)
        )
        self.gene_norm = NonAffineRMSNorm(config.dim)
        self.expression_encoder = FixedSinusoidalExpressionEncoder(
            config.dim,
            base=config.expression_base,
            mask_value=config.mask_value,
        )
        self.expression_scale = nn.Parameter(
            torch.tensor(config.expression_scale_init)
        )
        self.input_norm = nn.LayerNorm(config.dim)

        self.graph_block = GatedWeightedGCNBlock(config)
        self.pre_performer_norm = nn.LayerNorm(config.dim)

        self.performer = Performer(
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            dim_head=config.dim_head,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=None,
            reversible=False,
            generalized_attention=False,
            no_projection=False,
            causal=False,
            local_attn_heads=0,
            attn_dropout=config.attention_dropout,
            ff_dropout=config.dropout,
        )
        if config.fix_performer_projection_matrices:
            self.performer.fix_projection_matrices_()

        self.output_norm = nn.LayerNorm(config.dim)
        self.expression_head = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, 1),
        )

    def forward(
        self,
        expression: Tensor,
        *,
        return_gene_embeddings: bool = False,
    ) -> dict[str, Tensor]:
        if expression.ndim != 2:
            raise ValueError("expression must have shape [batch, genes]")
        batch_size, num_genes = expression.shape
        if num_genes != self.config.num_genes:
            raise ValueError(
                f"expected {self.config.num_genes} genes, received {num_genes}"
            )
        gene_tokens = self.gene_norm(self.gene_embedding.weight).unsqueeze(0)
        gene_tokens = gene_tokens.expand(batch_size, -1, -1)
        expression_tokens = self.expression_encoder(expression)
        visible_fraction, mask_ratio_scale = self.expression_mask_ratio_scale(
            expression
        )
        expression_tokens = expression_tokens * mask_ratio_scale[:, None, None]
        hidden = self.input_norm(
            gene_tokens + self.expression_scale * expression_tokens
        )

        hidden = self.graph_block(hidden, self.edge_index, self.edge_weight)
        hidden = self.pre_performer_norm(hidden)
        hidden = self.performer(hidden)
        gene_embeddings = self.output_norm(hidden)
        prediction = self.expression_head(gene_embeddings).squeeze(-1)

        output = {
            "expression_prediction": prediction,
            "input_visible_fraction": visible_fraction,
            "mask_ratio_scale": mask_ratio_scale,
        }
        if return_gene_embeddings:
            output["gene_embeddings"] = gene_embeddings
        return output

    def expression_mask_ratio_scale(
        self,
        expression: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return per-sample visible fractions and ESM-style expression scales."""
        visible_fraction = expression.ne(self.config.mask_value).float().mean(dim=1)
        if not self.config.mask_ratio_scaling_enabled:
            return visible_fraction, torch.ones_like(visible_fraction)
        reference_visible_fraction = 1.0 - self.config.mask_ratio_reference
        scale = reference_visible_fraction / visible_fraction.clamp_min(
            self.config.mask_ratio_min_visible_fraction
        )
        return visible_fraction, scale.clamp(max=self.config.mask_ratio_max_scale)


def _validate_config(
    config: BulkFormerV1PlusConfig,
    graph: WeightedGeneGraph,
) -> None:
    if config.num_genes <= 0 or graph.num_nodes != config.num_genes:
        raise ValueError("model vocabulary and graph node count must be identical")
    if config.dim <= 0 or config.dim % 2:
        raise ValueError("model dimension must be a positive even number")
    if config.heads * config.dim_head != config.dim:
        raise ValueError("heads * dim_head must equal model dimension")
    if config.depth <= 0 or config.graph_hidden_dim <= 0:
        raise ValueError("model depth and graph hidden dimension must be positive")
    if not 0.0 <= config.mask_ratio_reference < 1.0:
        raise ValueError("mask_ratio_reference must be in [0, 1)")
    if not 0.0 < config.mask_ratio_min_visible_fraction <= 1.0:
        raise ValueError("mask_ratio_min_visible_fraction must be in (0, 1]")
    if config.mask_ratio_max_scale < 1.0:
        raise ValueError("mask_ratio_max_scale must be at least one")


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}

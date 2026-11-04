from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ._backbone import FullTranscriptomeBackbone, LatestConfig
from ._graph import WeightedGeneGraph
from ._missingness import MissingnessCalibration, MissingnessCalibrationConfig, input_missing_fraction

FIXED_FUSION = {
    "operation": "sum", "trainable_coefficients": False,
    "fixed_coefficients": {"gene_identity": 1.0, "expression": 1.0, "global_context": 1.0},
}


@dataclass(frozen=True)
class GlobalContextConfig:
    hidden_dim: int = 2560
    activation: str = "relu"
    input_policy: str = "masked_expression_with_sentinel"
    normalization: str = "non_affine_layer_norm"
    normalization_eps: float = 1e-8
    injection: str = "broadcast_add_before_input_norm"

    def __post_init__(self):
        if self.hidden_dim <= 0 or not math.isfinite(self.normalization_eps) or self.normalization_eps <= 0:
            raise ValueError("invalid global context configuration")
        if self.activation != "relu" or self.normalization != "non_affine_layer_norm":
            raise ValueError("unsupported global context activation or normalization")
        if self.input_policy != "masked_expression_with_sentinel":
            raise ValueError("global context must read only masked inputs")
        if self.injection != "broadcast_add_before_input_norm":
            raise ValueError("unsupported global context injection")


class SparseGCNBlock(nn.Module):
    """Execute the same weighted GCN using a cached normalized CSR matrix."""

    def __init__(self, block: nn.Module, graph: WeightedGeneGraph):
        super().__init__()
        for name in ("norm", "input_projection", "convolution", "output_projection", "dropout"):
            setattr(self, name, getattr(block, name))
        self.residual_scale = block.residual_scale
        edges, weights = gcn_norm(
            graph.edge_index, graph.edge_weight, graph.num_nodes,
            improved=False, add_self_loops=False, flow="source_to_target", dtype=torch.float32,
        )
        # A[target, source] multiplies source features into their target nodes.
        adjacency = torch.sparse_coo_tensor(
            edges.flip(0), weights.float(), (graph.num_nodes, graph.num_nodes),
        ).coalesce().to_sparse_csr()
        self.register_buffer("normalized_adj_t", adjacency, persistent=False)

    def forward(self, hidden: Tensor, _edge_index: Tensor, _edge_weight: Tensor) -> Tensor:
        update = self.convolution.lin(self.input_projection(self.norm(hidden)))
        batch, nodes, width = update.shape
        # FP32 edge weights also promote messages to FP32 in the original scatter path.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            flat = update.transpose(0, 1).reshape(nodes, batch * width).float()
            update = torch.sparse.mm(self.normalized_adj_t, flat)
            update = update.reshape(nodes, batch, width).transpose(0, 1)
        update = self.output_projection(F.gelu(update))
        return hidden + self.residual_scale * self.dropout(update)


class BulkFormerLatest(FullTranscriptomeBackbone):
    """Three normalized inputs, fixed equal-weight fusion, and no GenePT branch."""

    def __init__(self, config: LatestConfig, graph: WeightedGeneGraph,
                 global_context: GlobalContextConfig, *, graph_aggregation: str = "scatter",
                 missingness_calibration: MissingnessCalibrationConfig | None = None):
        if config.mask_ratio_scaling_enabled or config.expression_scale_init != 1.0:
            raise ValueError("Latest requires fixed input fusion without mask-ratio scaling")
        if graph_aggregation not in {"scatter", "sparse_csr"}:
            raise ValueError("unsupported GCN aggregation backend")
        super().__init__(config, graph)
        del self.expression_scale
        self.global_context_config = global_context
        self.global_expr_proj = nn.Sequential(
            nn.Linear(config.num_genes, global_context.hidden_dim), nn.ReLU(),
            nn.Linear(global_context.hidden_dim, config.dim),
        )
        eps = global_context.normalization_eps
        self.gene_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=eps)
        self.expression_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=eps)
        self.global_context_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=eps)
        self.graph_aggregation = graph_aggregation
        if graph_aggregation == "sparse_csr":
            self.graph_block = SparseGCNBlock(self.graph_block, graph)
        self.missingness_calibration = (
            MissingnessCalibration(missingness_calibration) if missingness_calibration is not None else None
        )
        self._last_calibration = None

    def branch_diagnostics(self) -> dict[str, float | bool | str]:
        diagnostics = {
            "gene_identity_scale": 1.0, "expression_scale": 1.0, "global_context_scale": 1.0,
            "input_coefficients_trainable": False, "genept_enabled": False,
            "graph_aggregation": self.graph_aggregation,
            "missingness_calibration_enabled": self.missingness_calibration is not None,
        }
        if self._last_calibration is not None:
            missing, gain, offset = self._last_calibration
            diagnostics.update({
                "calibration_last_batch_missing_fraction_mean": missing.mean().item(),
                "calibration_last_batch_gain_mean": gain.mean().item(),
                "calibration_last_batch_offset_mean": offset.mean().item(),
            })
        return diagnostics

    def forward(self, expression: Tensor, *, return_gene_embeddings: bool = False) -> dict[str, Tensor]:
        if expression.ndim != 2 or expression.shape[1] != self.config.num_genes:
            raise ValueError(f"expression must have shape [batch, {self.config.num_genes}]")
        identity = self.gene_norm(self.gene_embedding.weight)
        visible_fraction, mask_ratio_scale = self.expression_mask_ratio_scale(expression)
        expression_tokens = self.expression_norm(self.expression_encoder(expression))
        context = self.global_context_norm(self.global_expr_proj(expression))
        hidden = self.input_norm(identity.unsqueeze(0) + expression_tokens + context.unsqueeze(1))
        hidden = self.graph_block(hidden, self.edge_index, self.edge_weight)
        hidden = self.pre_performer_norm(hidden)
        hidden = self.performer(hidden)
        gene_embeddings = self.output_norm(hidden)
        prediction = self.expression_head(gene_embeddings).squeeze(-1)
        missing = input_missing_fraction(expression, self.config.mask_value)
        output = {
            "expression_prediction": prediction,
            "input_visible_fraction": visible_fraction, "mask_ratio_scale": mask_ratio_scale,
            "input_missing_fraction": missing,
        }
        if self.missingness_calibration is not None:
            calibrated, gain, offset = self.missingness_calibration(prediction, missing)
            output.update({
                "expression_prediction": calibrated,
                "uncalibrated_expression_prediction": prediction,
                "missingness_gain": gain, "missingness_offset": offset,
            })
            self._last_calibration = (missing.detach(), gain.detach(), offset.detach())
        if return_gene_embeddings:
            output["gene_embeddings"] = gene_embeddings
        return output

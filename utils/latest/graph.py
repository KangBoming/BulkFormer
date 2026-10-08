from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class WeightedGeneGraph:
    """Standard PyG source-to-target representation of a fixed weighted graph."""

    num_nodes: int
    edge_index: Tensor
    edge_weight: Tensor


def load_weighted_gene_graph(
    graph_dir: str | Path,
    expected_num_nodes: int,
    *,
    validate_symmetry: bool = True,
    validate_unique_edges: bool = True,
) -> WeightedGeneGraph:
    """Load the official V1Plus graph without transposing its edge orientation."""
    graph_dir = Path(graph_dir)
    edge_index = torch.load(
        graph_dir / "edge_index.pt", map_location="cpu", weights_only=True
    ).long()
    edge_weight = torch.load(
        graph_dir / "edge_weight.pt", map_location="cpu", weights_only=True
    ).float()

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, edges], got {edge_index.shape}")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight must contain one value per directed edge")
    if edge_index.numel() == 0:
        raise ValueError("gene graph must contain at least one edge")
    if edge_index.min().item() < 0 or edge_index.max().item() >= expected_num_nodes:
        raise IndexError("gene graph contains a node outside the configured vocabulary")
    if not torch.isfinite(edge_weight).all().item():
        raise ValueError("gene graph contains non-finite weights")
    if (edge_weight <= 0).any().item():
        raise ValueError("V1Plus GCN requires strictly positive edge weights")
    if (edge_index[0] == edge_index[1]).any().item():
        raise ValueError("V1Plus graph must not contain self-loops")

    keys = edge_index[0] * expected_num_nodes + edge_index[1]
    if validate_unique_edges and torch.unique(keys).numel() != keys.numel():
        raise ValueError("gene graph contains duplicate directed edges")
    if validate_symmetry:
        order = torch.argsort(keys)
        reverse_keys = edge_index[1] * expected_num_nodes + edge_index[0]
        reverse_order = torch.argsort(reverse_keys)
        if not torch.equal(keys[order], reverse_keys[reverse_order]):
            raise ValueError("gene graph is not structurally symmetric")
        if not torch.allclose(
            edge_weight[order], edge_weight[reverse_order], rtol=0.0, atol=1e-7
        ):
            raise ValueError("reverse graph edges do not have identical weights")

    return WeightedGeneGraph(
        num_nodes=expected_num_nodes,
        edge_index=edge_index.contiguous(),
        edge_weight=edge_weight.contiguous(),
    )

"""Inference API for the pinned bulkformer-latest release."""

from .model import BulkFormerV1Plus as BulkFormerLatest
from .model import BulkFormerV1PlusConfig as BulkFormerLatestConfig
from .inference import (
    extract_feature, load_latest_model, main_gene_selection,
    normalize_data, prepare_latest_bundle,
)

__all__ = [
    "BulkFormerLatest", "BulkFormerLatestConfig", "load_latest_model",
    "normalize_data", "main_gene_selection", "extract_feature", "prepare_latest_bundle",
]

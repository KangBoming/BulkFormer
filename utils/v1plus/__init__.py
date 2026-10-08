"""Compatibility imports for the former V1 Plus release name.

New workflows should use ``utils.latest``. These names resolve to the same
model implementation and retain the former bundle destination directory.
"""

from ..latest.model import BulkFormerV1Plus, BulkFormerV1PlusConfig
from ..latest.inference import (
    extract_feature, load_latest_model as load_v1plus_model,
    main_gene_selection, normalize_data, prepare_latest_bundle,
)
from pathlib import Path


def prepare_v1plus_bundle(download_path, output_dir="model", download_manifest=None):
    manifest = download_manifest or (
        Path(__file__).resolve().parents[2] / "model" / "v1plus_downloads.json"
    )
    return prepare_latest_bundle(download_path, output_dir, manifest)


__all__ = [
    "BulkFormerV1Plus", "BulkFormerV1PlusConfig", "load_v1plus_model",
    "normalize_data", "main_gene_selection", "extract_feature", "prepare_v1plus_bundle",
]

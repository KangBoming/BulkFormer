"""Load the official release and extract features with the complete V1 Plus model."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import load_weighted_gene_graph
from .model import BulkFormerV1Plus, BulkFormerV1PlusConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_v1plus_bundle(download_path, output_dir='model', download_manifest=None):
    """Verify the complete checkpoint and its five accompanying resource files.

    ``download_path`` is a directory containing the six Google Drive files.
    If it is already ``output_dir/bulkformer_v1plus``, verify it in place.
    Otherwise copy the verified files into that directory atomically. An
    existing, different model directory is never replaced.
    """
    manifest_path = Path(download_manifest) if download_manifest else (
        Path(__file__).resolve().parents[2] / 'model' / 'v1plus_downloads.json'
    )
    metadata = json.loads(manifest_path.read_text())
    if metadata.get('format_version') != 2:
        raise ValueError('Unsupported download manifest format')
    required = {'model.pt', 'model_config.json', 'gene_vocab.csv',
                'edge_index.pt', 'edge_weight.pt', 'manifest.json'}
    if set(metadata.get('files', {})) != required:
        raise ValueError('Download manifest must describe the six official release files')
    source = Path(download_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Download the six release files into a directory: {source}')
    output_dir = Path(output_dir).resolve()
    destination = output_dir / 'bulkformer_v1plus'
    if destination.exists() and source != destination:
        raise FileExistsError(f'{destination} already exists; load it with load_v1plus_model')
    for name in sorted(required):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing release file: {path}')
        expected = metadata['files'][name]
        if path.stat().st_size != expected['bytes'] or _sha256(path) != expected['sha256']:
            raise ValueError(f'Release file failed SHA256 verification: {name}')
    if source == destination:
        return destination
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.v1plus-', dir=output_dir) as temporary:
        staging = Path(temporary) / 'bulkformer_v1plus'
        staging.mkdir()
        for name in sorted(required):
            shutil.copyfile(source / name, staging / name)
        staging.rename(destination)
    return destination


def load_v1plus_model(model_dir, device=None):
    """Return a frozen model, its ordered gene vocabulary, and release metadata.

    ``model_dir`` is the directory containing the six official release files.
    The checkpoint contains tensors and plain metadata, so it is
    loaded using ``weights_only=True``. Every required artifact is authenticated
    against the bundle manifest before use.
    """
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest.get('architecture') != 'BulkFormerV1Plus-v1':
        raise ValueError('Expected the official BulkFormer V1 Plus inference bundle')
    for name in ['model.pt', 'model_config.json', 'gene_vocab.csv', 'edge_index.pt', 'edge_weight.pt']:
        path = model_dir / name
        expected = manifest['files'][name]
        if path.stat().st_size != expected['bytes'] or _sha256(path) != expected['sha256']:
            raise ValueError(f'Release file differs from manifest: {name}')
    configuration = json.loads((model_dir / 'model_config.json').read_text())
    config = BulkFormerV1PlusConfig(**configuration['model'])
    vocab = pd.read_csv(model_dir / 'gene_vocab.csv')
    if (len(vocab) != config.num_genes or not np.array_equal(vocab['vocab_index'], np.arange(config.num_genes))
            or vocab['canonical_ensg'].nunique() != config.num_genes):
        raise ValueError('The release vocabulary must contain the complete ordered gene axis')
    lengths = vocab['gene_length_bp'].to_numpy(dtype=np.float64)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError('The release contains invalid gene lengths')
    graph = load_weighted_gene_graph(model_dir, config.num_genes,
                                    validate_symmetry=False, validate_unique_edges=False)
    payload = torch.load(model_dir / 'model.pt', map_location='cpu', mmap=True, weights_only=True)
    if (payload.get('architecture') != 'BulkFormerV1Plus-v1'
            or payload.get('model_config') != configuration['model']
            or payload.get('checkpoint_metadata') != manifest['checkpoint_metadata']):
        raise ValueError('The checkpoint and bundle configuration disagree')
    model = BulkFormerV1Plus(config, graph)
    model.load_state_dict(payload['model'], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != manifest['trainable_parameters']:
        raise ValueError('Unexpected model parameter count')
    target = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.requires_grad_(False).eval().to(target)
    return model, vocab, manifest


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.ndim != 2 or frame.empty:
        raise ValueError('Provide a nonempty DataFrame with samples as rows and Ensembl IDs as columns')
    result = frame.copy()
    result.columns = [str(gene).split('.', 1)[0] for gene in result.columns]
    if result.columns.has_duplicates:
        raise ValueError('Duplicate Ensembl IDs after removing version suffixes; provide one column per gene')
    return result


def normalize_data(X_df: pd.DataFrame, gene_vocab: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to natural ln(TPM+1) using the release gene lengths.

    Ensembl version suffixes are removed. The TPM denominator uses vocabulary-
    matched genes, as in the registered V1 Plus evaluation preprocessing.
    Missing entries can be NaN; measured zeros remain zero. No guessed gene
    lengths are substituted for genes outside the vocabulary.
    """
    frame = _canonical_columns(X_df)
    columns = [gene for gene in gene_vocab['canonical_ensg'] if gene in frame.columns]
    if not columns:
        raise ValueError('No input Ensembl IDs match the released gene vocabulary')
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('Raw counts must be non-negative and finite, with NaN only for missing entries')
    missing = np.isnan(values)
    values[missing] = 0
    lengths = gene_vocab.set_index('canonical_ensg').loc[columns, 'gene_length_bp'].to_numpy(dtype=np.float32)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError('All matched genes require positive release gene lengths')
    values /= lengths[None, :] / np.float32(1000)
    denominator = values.sum(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError('Each sample needs a positive count sum over vocabulary-matched genes')
    values /= denominator
    values *= np.float32(1e6)
    np.log1p(values, out=values)
    values[missing] = np.nan
    return pd.DataFrame(values, index=frame.index, columns=columns)


def main_gene_selection(X_df: pd.DataFrame, gene_list):
    """Align natural ln(TPM+1) values to the complete released gene axis.

    Return ``(aligned_dataframe, globally_missing_genes, gene_metadata)``.
    Source-unavailable genes and NaN entries become -10. True expression zeros
    remain observed zero values. Extra input genes are excluded.
    """
    frame = _canonical_columns(X_df)
    genes = list(gene_list)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError('gene_list must contain unique ordered Ensembl IDs')
    if not set(genes).intersection(frame.columns):
        raise ValueError('No input Ensembl IDs match the released gene vocabulary')
    missing_genes = [gene for gene in genes if gene not in frame.columns]
    aligned = frame.reindex(columns=genes).astype(np.float32).fillna(-10.0)
    values = aligned.to_numpy()
    observed = values != -10.0
    if not np.isfinite(values).all() or (values[observed] < 0).any() or (values[observed] > 13.81552).any():
        raise ValueError('Expected natural ln(TPM+1) in [0, ln(1000001)], or -10 for missing genes')
    if not observed.any(axis=1).all():
        raise ValueError('Each sample needs at least one observed gene')
    metadata = pd.DataFrame({'mask': [int(gene in missing_genes) for gene in genes]}, index=genes)
    return aligned, missing_genes, metadata


def extract_feature(model, expr_array, output_feature_type='sample_level',
                    aggregate_type='mean', batch_size=1, interested_gene_idx=None,
                    amp_dtype='bf16'):
    """Return float32 NumPy sample, contextual-gene, or expression features.

    Inputs must contain the complete gene axis in release order. Sample vectors
    use the mean over observed genes, matching the released phenotype benchmark.
    ``interested_gene_idx`` selects contextual output genes or pooling genes; the
    encoder still receives the complete gene axis. No random masking is added.
    Missing genes are excluded from sample pooling for each individual sample.
    """
    if torch.is_tensor(expr_array):
        expr_array = expr_array.detach().cpu().numpy()
    values = np.asarray(expr_array, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != model.config.num_genes:
        raise ValueError(f'Expected a nonempty [samples, {model.config.num_genes}] input')
    observed = values != model.config.mask_value
    if not np.isfinite(values).all() or (values[observed] < 0).any() or (values[observed] > 13.81552).any():
        raise ValueError('Input must be finite natural ln(TPM+1), or -10 for missing genes')
    if not observed.any(axis=1).all():
        raise ValueError('Each sample needs at least one observed gene')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if output_feature_type not in ['sample_level', 'gene_level', 'expr_level']:
        raise ValueError('output_feature_type must be sample_level, gene_level, or expr_level')
    if aggregate_type != 'mean':
        raise ValueError('The released phenotype protocol uses aggregate_type=mean')
    if amp_dtype not in ['bf16', 'fp16', 'fp32']:
        raise ValueError('amp_dtype must be bf16, fp16, or fp32')
    device = next(model.parameters()).device
    indices = None
    if interested_gene_idx is not None:
        selected = np.asarray(interested_gene_idx)
        if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu'
                or (selected < 0).any() or (selected >= model.config.num_genes).any()
                or len(np.unique(selected)) != len(selected)):
            raise ValueError('interested_gene_idx must contain unique valid integer positions')
        indices = torch.tensor(selected.astype(np.int64), device=device)
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': None}[amp_dtype]
    if device.type == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16; set amp_dtype=fp16 or fp32')
    use_amp = device.type == 'cuda' and dtype is not None
    model.eval()
    chunks = []
    with torch.inference_mode():
        for first in range(0, len(values), batch_size):
            expression = torch.tensor(values[first:first + batch_size], device=device)
            with torch.amp.autocast(device.type, dtype=dtype, enabled=use_amp):
                output = model(expression, return_gene_embeddings=output_feature_type != 'expr_level')
                if output_feature_type == 'expr_level':
                    result = output['expression_prediction']
                else:
                    hidden = output['gene_embeddings']
                    visible = expression.ne(model.config.mask_value)
                    if indices is not None:
                        hidden = hidden.index_select(1, indices)
                        visible = visible.index_select(1, indices)
                    if output_feature_type == 'gene_level':
                        result = hidden
                    else:
                        count = visible.sum(dim=1, keepdim=True)
                        if (count == 0).any():
                            raise ValueError('A sample has no observed genes in the selected pooling subset')
                        # Match the benchmark's mean reduction, including its
                        # BF16 rounding, while supporting per-sample missingness.
                        result = torch.stack([
                            hidden[row, visible[row]].mean(dim=0)
                            for row in range(len(hidden))
                        ])
            chunks.append(result.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)

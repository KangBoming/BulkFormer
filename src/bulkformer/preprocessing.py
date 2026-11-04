"""Explicit expression scales and separate normalization contracts for each model."""
import warnings

import numpy as np
import pandas as pd


def _normalize(frame, vocabulary, input_scale, log_base):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Input must be a nonempty samples-by-Ensembl-genes DataFrame')
    if not frame.index.is_unique:
        raise ValueError('Sample IDs must be unique')
    genes = pd.Index(frame.columns.astype(str)).str.replace(r'\.\d+$', '', regex=True)
    if not genes.is_unique:
        raise ValueError('Gene IDs must be unique after removing Ensembl version suffixes')
    if input_scale not in {'counts', 'tpm', 'log1p_tpm', 'log2_tpm'}:
        raise ValueError('Choose counts, tpm, log1p_tpm or log2_tpm explicitly; array values are unsupported')
    target = pd.Index(vocabulary.ensembl_gene)
    source_positions = genes.get_indexer(target)
    matched = np.flatnonzero(source_positions >= 0)
    if not len(matched):
        raise ValueError('No Ensembl gene IDs match the released vocabulary')
    values = frame.iloc[:, source_positions[matched]].to_numpy(dtype=np.float64, copy=True)
    observed = ~np.isnan(values)
    if np.isinf(values).any() or (values[observed] < 0).any():
        raise ValueError('Observed expression must be finite and nonnegative; represent unavailable values with NaN')
    if not observed.any(axis=1).all():
        raise ValueError('Every sample must have at least one observed matched gene')
    values[~observed] = 0
    if input_scale == 'counts':
        lengths = vocabulary.gene_length_bp.to_numpy(dtype=np.float64)[matched]
        if not np.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError('Released gene lengths must be positive and finite')
        values /= lengths[None, :] / 1000.0
        totals = values.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError('Cannot normalize a sample with zero matched library size')
        values *= 1e6 / totals[:, None]
        values = np.log1p(values)
    elif input_scale == 'tpm':
        values = np.log1p(values)
    elif input_scale == 'log2_tpm' and log_base != 2:
        values *= np.log(2.0)
    values = values.astype(np.float32)
    if log_base == 2 and input_scale != 'log2_tpm':
        # Match the frozen PRIMO pipeline: natural-log float32, then float32 scaling.
        values *= np.float32(1.4426950408889634)
    if not np.isfinite(values).all():
        raise ValueError('Normalization produced nonfinite values')
    values[~observed] = -10.0
    aligned = np.full((len(frame), len(target)), -10.0, dtype=np.float32)
    aligned[:, matched] = values.astype(np.float32)
    return aligned


def _coverage(expression, index):
    seen = expression >= 0
    return pd.DataFrame({'observed_genes': seen.sum(axis=1),
                         'missing_fraction': 1.0 - seen.mean(axis=1)}, index=index)


def prepare_atlas_input(expression, vocabulary, normalization_vocabulary, *, input_scale):
    """Normalize on the 41,623-gene axis before selecting coding genes or HVGs.

    Counts are normalized over matched genes. Incomplete source coverage gives
    TPM-like values, not an exact reconstruction of the training denominator.
    Already normalized TPM is used as supplied, without a second library rescaling.
    """
    normal = _normalize(expression, normalization_vocabulary, input_scale, log_base=2)
    columns = vocabulary.matrix_column.to_numpy(dtype=np.int64)
    if not np.array_equal(normalization_vocabulary.ensembl_gene.to_numpy()[columns], vocabulary.ensembl_gene.to_numpy()):
        raise ValueError('Coding vocabulary does not match its normalization axis')
    aligned = normal[:, columns]
    return aligned, _coverage(aligned, expression.index)


def prepare_latest_input(expression, vocabulary, *, input_scale):
    """Return all 19,973 gene positions; observed zeros remain distinct from missing genes."""
    aligned = _normalize(expression, vocabulary, input_scale, log_base=np.e)
    coverage = _coverage(aligned, expression.index)
    if (coverage.missing_fraction > .2).any():
        warnings.warn('Some samples exceed 20% missingness, outside the additional masking range used in training. Inspect coverage and evaluate on your task.', stacklevel=2)
    return aligned, coverage


def select_variable_genes(expression, n_genes=4096):
    """Select one fixed panel for a cohort, with stable tie breaking and gene order.

    For prospective supervised evaluation, fit this panel on training samples
    and reuse it on held-out samples. PRIMO used unlabeled whole-cohort selection.
    """
    values = np.asarray(expression)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('HVG selection needs at least two finite expression profiles')
    if not isinstance(n_genes, int) or isinstance(n_genes, bool) or n_genes <= 0:
        raise ValueError('n_genes must be a positive integer')
    available = np.flatnonzero((values >= 0).all(axis=0))
    variance = np.var(values[:, available], axis=0, dtype=np.float64)
    available, variance = available[variance > 0], variance[variance > 0]
    if len(available) < n_genes:
        raise ValueError(f'Need {n_genes} observed variable genes; found {len(available)}. Do not pad or duplicate genes.')
    order = np.lexsort((available, -variance))
    return np.sort(available[order[:n_genes]])

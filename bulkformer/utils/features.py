"""
Feature extraction and data preprocessing utilities for BulkFormer.

These functions are designed to be used in both CLI and programmatic contexts
(e.g., notebooks, scripts, pipelines).
"""

from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def normalize_data(X_df: pd.DataFrame, gene_length_dict: dict) -> pd.DataFrame:
    """
    Normalize RNA-seq count data to log-transformed TPM values.
    
    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix where rows represent samples and columns represent genes.
    gene_length_dict : dict
        A dictionary mapping gene identifiers to gene lengths (in base pairs).
    
    Returns
    -------
    log_tpm_df : pandas.DataFrame
        A DataFrame containing log-transformed TPM values.
    """
    gene_names = X_df.columns
    gene_lengths_kb = np.array([gene_length_dict.get(gene, 1000) / 1000 for gene in gene_names])
    counts_matrix = X_df.values
    rate = counts_matrix / gene_lengths_kb
    sum_per_sample = rate.sum(axis=1)
    sum_per_sample[sum_per_sample == 0] = 1e-6
    sum_per_sample = sum_per_sample.reshape(-1, 1)
    tpm = rate / sum_per_sample * 1e6
    log_tpm = np.log1p(tpm)
    log_tpm_df = pd.DataFrame(log_tpm, index=X_df.index, columns=X_df.columns)
    return log_tpm_df


def align_genes(X_df: pd.DataFrame, gene_list: list) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """
    Align expression matrix to a predefined gene list.
    
    Parameters
    ----------
    X_df : pandas.DataFrame
        Gene expression matrix with rows as samples and columns as genes.
    gene_list : list
        Predefined list of gene identifiers to align to.
    
    Returns
    -------
    X_df : pandas.DataFrame
        Aligned expression matrix.
    to_fill_columns : list
        Genes that were added with placeholder values.
    var : pandas.DataFrame
        DataFrame with mask indicating imputed genes.
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    
    padding_df = pd.DataFrame(
        np.full((X_df.shape[0], len(to_fill_columns)), -10),
        columns=to_fill_columns,
        index=X_df.index
    )
    
    X_df = pd.DataFrame(
        np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
        index=X_df.index,
        columns=list(X_df.columns) + list(padding_df.columns)
    )
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var


def extract_features(
    model,
    expr_array: np.ndarray,
    high_var_gene_idx: list,
    feature_type: str,
    aggregate_type: str,
    device: str,
    batch_size: int,
    return_expr_value: bool = False,
    esm2_emb: Optional[torch.Tensor] = None,
    valid_gene_idx: Optional[list] = None,
) -> torch.Tensor | np.ndarray:
    """
    Extract features from expression data using BulkFormer model.
    
    Parameters
    ----------
    model : BulkFormer
        The BulkFormer model instance.
    expr_array : np.ndarray
        Expression matrix as numpy array (samples × genes).
    high_var_gene_idx : list
        Indices of highly variable genes for aggregation.
    feature_type : str
        Type of features to extract: 'transcriptome_level' or 'gene_level'.
    aggregate_type : str
        Aggregation method for transcriptome_level: 'max', 'mean', 'median', or 'all'.
    device : str
        Device to use: 'cuda' or 'cpu'.
    batch_size : int
        Batch size for inference.
    return_expr_value : bool, optional
        If True, return predicted expression values instead of embeddings.
    esm2_emb : torch.Tensor, optional
        ESM2 protein embeddings for gene-level features.
    valid_gene_idx : list, optional
        Indices of valid (non-imputed) genes for gene-level features.
    
    Returns
    -------
    result : torch.Tensor or np.ndarray
        Extracted features or predicted expression values.
    """
    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_emb_list = []
    all_expr_value_list = []
    
    with torch.no_grad():
        if feature_type == 'transcriptome_level':
            for (X,) in tqdm(myloader, total=len(myloader), desc="Extracting features"):
                X = X.to(device)
                output, emb = model(X, [2])
                all_expr_value_list.append(output.detach().cpu().numpy())
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:, high_var_gene_idx, :]
                
                if aggregate_type == 'max':
                    final_emb = np.max(emb_valid, axis=1)
                elif aggregate_type == 'mean':
                    final_emb = np.mean(emb_valid, axis=1)
                elif aggregate_type == 'median':
                    final_emb = np.median(emb_valid, axis=1)
                elif aggregate_type == 'all':
                    max_emb = np.max(emb_valid, axis=1)
                    mean_emb = np.mean(emb_valid, axis=1)
                    median_emb = np.median(emb_valid, axis=1)
                    final_emb = max_emb + mean_emb + median_emb
                
                all_emb_list.append(final_emb)
            result_emb = np.vstack(all_emb_list)
            result_emb = torch.tensor(result_emb, device='cpu', dtype=torch.float32)
        
        elif feature_type == 'gene_level':
            for (X,) in tqdm(myloader, total=len(myloader), desc="Extracting features"):
                X = X.to(device)
                output, emb = model(X, [2])
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:, valid_gene_idx, :]
                all_emb_list.append(emb_valid)
                all_expr_value_list.append(output.detach().cpu().numpy())
            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb, device='cpu', dtype=torch.float32)
            esm2_emb_selected = esm2_emb[valid_gene_idx]
            esm2_emb_expanded = esm2_emb_selected.unsqueeze(0).expand(all_emb_tensor.shape[0], -1, -1)
            esm2_emb_expanded = esm2_emb_expanded.to('cpu')
            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)
    
    if return_expr_value:
        return np.vstack(all_expr_value_list)
    else:
        return result_emb


#!/usr/bin/env python3
"""
BulkFormer CLI - Feature extraction from expression data
"""

from pathlib import Path
from typing import Optional
import typer
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.typing import SparseTensor
from tqdm import tqdm
from collections import OrderedDict


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


def main(
    input_file: Path = typer.Argument(..., help="Path to expression data CSV file (samples as rows, genes as columns)"),
    output_file: Path = typer.Argument(..., help="Path to save extracted features"),
    feature_type: str = typer.Option("transcriptome_level", "--type", "-t", help="Feature type: transcriptome_level, gene_level, or expression_imputation"),
    aggregate_type: str = typer.Option("max", "--aggregate", "-a", help="Aggregation method: max, mean, median, or all (for transcriptome_level)"),
    is_count_data: bool = typer.Option(False, "--counts", "-c", help="Input is raw count data (will be normalized to log-TPM)"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size for inference"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to use: cuda or cpu"),
    model_dir: Path = typer.Option("model", "--model-dir", help="Directory containing model checkpoint"),
    data_dir: Path = typer.Option("data", "--data-dir", help="Directory containing data files"),
) -> None:
    """
    Extract features from bulk RNA-seq expression data using BulkFormer.
    
    Examples:
    
        # Extract transcriptome-level embeddings (default)
        bulkformer extract input.csv output.pt
        
        # Extract from raw count data (will normalize to log-TPM)
        bulkformer extract input.csv output.pt --counts
        
        # Extract gene-level embeddings
        bulkformer extract input.csv output.pt --type gene_level
        
        # Expression imputation
        bulkformer extract input.csv output.csv --type expression_imputation
    """
    from bulkformer import BulkFormer, model_params
    
    typer.echo(f"🧬 BulkFormer Feature Extraction")
    typer.echo(f"{'='*50}")
    
    # Validate inputs
    if not input_file.exists():
        typer.echo(f"❌ Error: Input file not found: {input_file}", err=True)
        raise typer.Exit(1)
    
    if feature_type not in ['transcriptome_level', 'gene_level', 'expression_imputation']:
        typer.echo(f"❌ Error: Invalid feature type: {feature_type}", err=True)
        typer.echo("Valid options: transcriptome_level, gene_level, expression_imputation", err=True)
        raise typer.Exit(1)
    
    # Check if model and data files exist
    model_path = model_dir / "Bulkformer_ckpt_epoch_29.pt"
    graph_path = data_dir / "G_gtex.pt"
    weights_path = data_dir / "G_gtex_weight.pt"
    gene_emb_path = data_dir / "esm2_feature_concat.pt"
    gene_info_path = data_dir / "bulkformer_gene_info.csv"
    high_var_path = data_dir / "high_var_gene_list.pt"
    
    missing_files = []
    for path in [model_path, graph_path, weights_path, gene_emb_path, gene_info_path, high_var_path]:
        if not path.exists():
            missing_files.append(str(path))
    
    if missing_files:
        typer.echo("❌ Error: Missing required files:", err=True)
        for f in missing_files:
            typer.echo(f"  - {f}", err=True)
        typer.echo("\nRun: bulkformer download all", err=True)
        raise typer.Exit(1)
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        typer.echo("⚠️  Warning: CUDA not available, using CPU instead")
        device = "cpu"
    
    typer.echo(f"\n📁 Input file: {input_file}")
    typer.echo(f"📁 Output file: {output_file}")
    typer.echo(f"🔧 Feature type: {feature_type}")
    typer.echo(f"🔧 Device: {device}")
    typer.echo(f"🔧 Batch size: {batch_size}")
    
    # Load expression data
    typer.echo(f"\n📊 Loading expression data...")
    expr_df = pd.read_csv(input_file, index_col=0)
    typer.echo(f"   Shape: {expr_df.shape[0]} samples × {expr_df.shape[1]} genes")
    
    # Normalize if raw counts
    if is_count_data:
        typer.echo(f"🔄 Normalizing count data to log-TPM...")
        gene_length_path = data_dir / "gene_length_df.csv"
        if not gene_length_path.exists():
            typer.echo(f"❌ Error: gene_length_df.csv not found in {data_dir}", err=True)
            raise typer.Exit(1)
        gene_length_df = pd.read_csv(gene_length_path)
        gene_length_dict = gene_length_df.set_index('ensg_id')['length'].to_dict()
        expr_df = normalize_data(expr_df, gene_length_dict)
    
    # Load gene list and align
    typer.echo(f"🧬 Aligning genes to BulkFormer gene space...")
    gene_info = pd.read_csv(gene_info_path)
    gene_list = gene_info['ensg_id'].to_list()
    expr_df, to_fill_columns, var = align_genes(expr_df, gene_list)
    typer.echo(f"   {len(to_fill_columns)} genes imputed with placeholder values")
    
    # Get valid gene indices
    var.reset_index(inplace=True)
    valid_gene_idx = list(var[var['mask'] == 0].index)
    
    # Load model components
    typer.echo(f"\n🤖 Loading BulkFormer model...")
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    weights = torch.load(weights_path, map_location='cpu', weights_only=False)
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)
    gene_emb = torch.load(gene_emb_path, map_location='cpu', weights_only=False)
    high_var_gene_idx = torch.load(high_var_path, weights_only=False)
    
    model_params['graph'] = graph
    model_params['gene_emb'] = gene_emb
    model = BulkFormer(**model_params).to(device)
    
    # Load checkpoint
    typer.echo(f"📦 Loading pretrained checkpoint...")
    ckpt = torch.load(model_path, weights_only=False)
    new_state_dict = OrderedDict()
    for key, value in ckpt.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    
    # Extract features
    typer.echo(f"\n🚀 Extracting features...")
    
    if feature_type == 'expression_imputation':
        result = extract_features(
            model=model,
            expr_array=expr_df.values,
            high_var_gene_idx=high_var_gene_idx,
            feature_type='transcriptome_level',
            aggregate_type=aggregate_type,
            device=device,
            batch_size=batch_size,
            return_expr_value=True,
            esm2_emb=gene_emb,
            valid_gene_idx=valid_gene_idx,
        )
        # Save as CSV with gene names
        result_df = pd.DataFrame(result, index=expr_df.index, columns=expr_df.columns)
        result_df.to_csv(output_file)
        typer.echo(f"   Shape: {result.shape}")
    else:
        result = extract_features(
            model=model,
            expr_array=expr_df.values,
            high_var_gene_idx=high_var_gene_idx,
            feature_type=feature_type,
            aggregate_type=aggregate_type,
            device=device,
            batch_size=batch_size,
            return_expr_value=False,
            esm2_emb=gene_emb,
            valid_gene_idx=valid_gene_idx,
        )
        # Save as PyTorch tensor
        torch.save(result, output_file)
        typer.echo(f"   Shape: {result.shape}")
    
    typer.echo(f"\n✅ Features saved to: {output_file}")
    typer.echo(f"{'='*50}")


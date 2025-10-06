#!/usr/bin/env python3
"""
BulkFormer CLI - Feature extraction from expression data
"""

from pathlib import Path
from collections import OrderedDict
import typer
import torch
import pandas as pd
from torch_geometric.typing import SparseTensor
from bulkformer.utils import normalize_data, align_genes, extract_features


def main(
    input_file: Path = typer.Argument(..., help="Path to expression data CSV file (samples as rows, genes as columns)"),
    output_file: Path = typer.Argument(..., help="Path to save extracted features"),
    feature_type: str = typer.Option("transcriptome_level", "--type", "-t", help="Feature type: transcriptome_level, gene_level, or expression_imputation"),
    aggregate_type: str = typer.Option("max", "--aggregate", "-a", help="Aggregation method: max, mean, median, or all (for transcriptome_level)"),
    is_count_data: bool = typer.Option(False, "--counts", "-c", help="Input is raw count data (will be normalized to log-TPM)"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size for inference"),
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


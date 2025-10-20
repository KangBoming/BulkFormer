#!/usr/bin/env python3
"""
BulkFormer Data and Model Downloader

Downloads pretrained models and data files from Zenodo.
"""

from pathlib import Path
from typing import Optional, List
import requests
from tqdm import tqdm
import typer
from typing_extensions import Annotated


app = typer.Typer(
    name="download",
    help="Download BulkFormer pretrained models and data files from Zenodo",
    add_completion=False,
)


# Zenodo record information
ZENODO_RECORD = "15559368"
ZENODO_DOI = f"10.5281/zenodo.{ZENODO_RECORD}"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# File mappings: filename -> destination directory
DATA_FILES = {
    "G_gtex.pt": "data",
    "G_gtex_weight.pt": "data",
    "esm2_feature_concat.pt": "data",
    "demo.csv": "data",
    "high_var_gene_list.pt": "data",
}

MODEL_FILES = {
    "Bulkformer_ckpt_epoch_29.pt": "model",
}

ALL_FILES = {**DATA_FILES, **MODEL_FILES}


def get_zenodo_files() -> dict:
    """Fetch file information from Zenodo API."""
    try:
        response = requests.get(ZENODO_API_URL)
        response.raise_for_status()
        data = response.json()
        
        files_info = {}
        for file_entry in data.get("files", []):
            filename = file_entry["key"]
            files_info[filename] = {
                "url": file_entry["links"]["self"],
                "size": file_entry["size"],
                "checksum": file_entry["checksum"],
            }
        return files_info
    except Exception as e:
        typer.echo(f"Error fetching Zenodo metadata: {e}", err=True)
        raise typer.Exit(code=1)


def download_file(url: str, dest_path: Path, file_size: int) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        file_size: Size of file in bytes
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists and has correct size
    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        if existing_size == file_size:
            typer.echo(f"✓ {dest_path.name} already exists (correct size), skipping")
            return
        else:
            typer.echo(f"⚠ {dest_path.name} exists but size mismatch, re-downloading")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, "wb") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    typer.echo(f"✓ Downloaded {dest_path.name}")


@app.command()
def data(
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory (default: current directory)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-download even if files exist"),
    ] = False,
) -> None:
    """Download data files only."""
    base_dir = output_dir or Path.cwd()
    typer.echo(f"Downloading data files from Zenodo ({ZENODO_DOI})...")
    
    zenodo_files = get_zenodo_files()
    
    for filename, dest_subdir in DATA_FILES.items():
        if filename not in zenodo_files:
            typer.echo(f"⚠ Warning: {filename} not found in Zenodo record", err=True)
            continue
        
        file_info = zenodo_files[filename]
        dest_path = base_dir / dest_subdir / filename
        
        if dest_path.exists() and not force:
            if dest_path.stat().st_size == file_info["size"]:
                typer.echo(f"✓ {filename} already exists, skipping")
                continue
        
        typer.echo(f"Downloading {filename} ({file_info['size'] / 1024 / 1024:.1f} MB)...")
        download_file(file_info["url"], dest_path, file_info["size"])
    
    typer.echo("✓ Data download complete!")


@app.command()
def model(
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory (default: current directory)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-download even if files exist"),
    ] = False,
) -> None:
    """Download pretrained model only."""
    base_dir = output_dir or Path.cwd()
    typer.echo(f"Downloading pretrained model from Zenodo ({ZENODO_DOI})...")
    
    zenodo_files = get_zenodo_files()
    
    for filename, dest_subdir in MODEL_FILES.items():
        if filename not in zenodo_files:
            typer.echo(f"⚠ Warning: {filename} not found in Zenodo record", err=True)
            continue
        
        file_info = zenodo_files[filename]
        dest_path = base_dir / dest_subdir / filename
        
        if dest_path.exists() and not force:
            if dest_path.stat().st_size == file_info["size"]:
                typer.echo(f"✓ {filename} already exists, skipping")
                continue
        
        typer.echo(f"Downloading {filename} ({file_info['size'] / 1024 / 1024:.1f} MB)...")
        download_file(file_info["url"], dest_path, file_info["size"])
    
    typer.echo("✓ Model download complete!")


@app.command()
def all(
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory (default: current directory)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-download even if files exist"),
    ] = False,
) -> None:
    """Download all files (data + model)."""
    base_dir = output_dir or Path.cwd()
    typer.echo(f"Downloading all files from Zenodo ({ZENODO_DOI})...")
    typer.echo("")
    
    zenodo_files = get_zenodo_files()
    
    # Calculate total size
    total_size = sum(
        zenodo_files[f]["size"] 
        for f in ALL_FILES.keys() 
        if f in zenodo_files
    )
    typer.echo(f"Total download size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    typer.echo("")
    
    # Download all files
    for filename, dest_subdir in ALL_FILES.items():
        if filename not in zenodo_files:
            typer.echo(f"⚠ Warning: {filename} not found in Zenodo record", err=True)
            continue
        
        file_info = zenodo_files[filename]
        dest_path = base_dir / dest_subdir / filename
        
        if dest_path.exists() and not force:
            if dest_path.stat().st_size == file_info["size"]:
                typer.echo(f"✓ {filename} already exists, skipping")
                continue
        
        typer.echo(f"Downloading {filename} ({file_info['size'] / 1024 / 1024:.1f} MB)...")
        download_file(file_info["url"], dest_path, file_info["size"])
    
    typer.echo("")
    typer.echo("✓ All downloads complete!")


@app.command()
def list() -> None:
    """List available files on Zenodo."""
    typer.echo(f"Available files on Zenodo ({ZENODO_DOI}):")
    typer.echo("")
    
    zenodo_files = get_zenodo_files()
    
    typer.echo("Data files (data/):")
    for filename in DATA_FILES.keys():
        if filename in zenodo_files:
            size_mb = zenodo_files[filename]["size"] / 1024 / 1024
            typer.echo(f"  - {filename} ({size_mb:.1f} MB)")
        else:
            typer.echo(f"  - {filename} (not found)")
    
    typer.echo("")
    typer.echo("Model files (model/):")
    for filename in MODEL_FILES.keys():
        if filename in zenodo_files:
            size_mb = zenodo_files[filename]["size"] / 1024 / 1024
            typer.echo(f"  - {filename} ({size_mb:.1f} MB)")
        else:
            typer.echo(f"  - {filename} (not found)")


@app.command()
def info() -> None:
    """Show Zenodo record information."""
    typer.echo(f"Zenodo Record: {ZENODO_RECORD}")
    typer.echo(f"DOI: {ZENODO_DOI}")
    typer.echo(f"URL: https://doi.org/{ZENODO_DOI}")
    typer.echo("")
    
    try:
        response = requests.get(ZENODO_API_URL)
        response.raise_for_status()
        data = response.json()
        
        metadata = data.get("metadata", {})
        typer.echo(f"Title: {metadata.get('title', 'N/A')}")
        typer.echo(f"Publication Date: {metadata.get('publication_date', 'N/A')}")
        
        creators = metadata.get("creators", [])
        if creators:
            typer.echo("Creators:")
            for creator in creators:
                typer.echo(f"  - {creator.get('name', 'N/A')}")
        
        typer.echo("")
        typer.echo(f"Total files: {len(data.get('files', []))}")
        total_size = sum(f["size"] for f in data.get("files", []))
        typer.echo(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
        
    except Exception as e:
        typer.echo(f"Error fetching info: {e}", err=True)


if __name__ == "__main__":
    app()


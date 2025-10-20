#!/usr/bin/env python3
"""
BulkFormer CLI - Main entry point
"""

import typer
from bulkformer.cli.download import app as download_app
from bulkformer.cli.extract import main as extract_main
from bulkformer.cli.verify import main as verify_main

app = typer.Typer(
    name="bulkformer",
    help="BulkFormer: A large-scale foundation model for human bulk transcriptomes",
    add_completion=False,
)

# Add download commands as a subcommand group
app.add_typer(download_app, name="download", help="Download model and data files")

# Add extract as a direct command
app.command(name="extract", help="Extract features from bulk RNA-seq expression data")(extract_main)

# Add verify as a direct command
@app.command()
def verify() -> None:
    """Verify installation and check dependencies."""
    verify_main()


if __name__ == "__main__":
    app()


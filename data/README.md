# Example data and gene resources

## bulkformer-latest

[latest_demo_counts.csv](latest_demo_counts.csv) contains **four synthetic samples** with samples as rows and Ensembl IDs as columns. NumPy seed 42 was used to generate the counts; measured zeros are retained and eight genes are omitted to demonstrate missing-gene alignment. This example contains no biological samples.

The model's ordered 19,973-gene vocabulary, gene lengths, weighted graph, and configuration are all supplied in its [Google Drive model folder](https://drive.google.com/drive/folders/1AfekiOBNs6GzPBbA9BDlpuoolZIWbGQ2). See the [download instructions](../model/README.md) and [latest notebook](../notebooks/bulkformer-latest_feature_extraction.ipynb).

## Original model-family resources

The original models use the preprocessed resources in the [Zenodo record](https://doi.org/10.5281/zenodo.15744294), including:

- `PreBULK.h5ad.zip`: Original pretraining data.
- `bulkformer_gene_info.csv`: Original model-family vocabulary.
- `G_tcga.pt` and `G_tcga_weight.pt`: Original weighted graph.
- `esm2_feature_concat.pt`: Original workflow's protein features.
- `demo_count_adata.h5ad` and `interested_gene_list.pt`: Example data and selected genes.
- Downstream task resources.

The original notebook also reads demonstration count/normalized CSVs; provide those files or replace the input paths with your own data. The vocabulary and gene-length CSVs checked into this directory belong to the original workflow. Use resources matching the model family and release described in its [model guide](../docs/models/original-models.md).

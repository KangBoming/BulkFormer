# Jupyter notebooks

| Model family | Notebook | Environment |
|---|---|---|
| **bulkformer-latest** | [Feature extraction](bulkformer-latest_feature_extraction.ipynb) | [configs/bulkformer-latest.yaml](../configs/bulkformer-latest.yaml) |
| Original 37M–147M | [Original feature extraction](../bulkformer_extract_feature.ipynb) | [bulkformer.yaml](../bulkformer.yaml) |

The latest notebook prepares raw counts or normalized expression, aligns the complete vocabulary, loads verified weights, extracts sample/gene/expression features, and saves outputs. Its included data are synthetic. It automatically locates the repository root when opened from this directory.

The original notebook retains its repository-root working directory and resources. See the [usage guide](../docs/usage.md) for current input and GPU conventions.

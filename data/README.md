## 📦 Preprocessed Data & Resources

The following **preprocessed resource files** for running BulkFormer demos and scripts can be downloaded from **Zenodo**:

🔗 **Latest Zenodo record (always use the newest version):**   https://doi.org/10.5281/zenodo.15744294

**Available files:**
- `PreBULK.h5ad.zip(Pretraining data for BulkFormer)`
- `bulkformer_gene_info.csv`
- `G_tcga.pt`  
- `G_tcga_weight.pt`  
- `esm2_feature_concat.pt`  
- `demo_count_adata.h5ad`  
- `interested_gene_list.pt` (optional example pooling gene set)
- `Downtask-related data`


> ⚠️ **Important:** All of the above files have been **updated** in the latest release.

> Please **re-download them from the current Zenodo record**, as any previously downloaded

> versions may be outdated and **incompatible with the latest BulkFormer codebase**.

## Example pooling gene set

`interested_gene_list.pt` contains an **illustrative set of 2,000 genes** for sample-level embedding aggregation. It is optional, and users can choose their own pooling genes to suit a downstream task. It is not a required model input or a fixed gene set for every dataset.

The list matching MD5 `39c404b895ea80a76f7031888c908be5` contains **zero-based positional indices** into the row order of the released `bulkformer_gene_info.csv`, rather than Ensembl gene ID strings. Keep this gene order unchanged when aligning expression data or mapping a custom gene set.

The paper did not use this example list for embedding aggregation, including in the gene-wise essentiality experiments. Whenever sample-level representations were constructed by pooling gene embeddings, the paper used **mean pooling across all genes**. In the feature extraction notebook, use `interested_gene_idx=None` with `aggregate_type='mean'` to follow that pooling protocol. A pooling subset only selects model outputs for aggregation; it does not reduce the full input gene vocabulary.

### How the example list was generated

The historical notebooks record this procedure:

1. Start with an ARCHS4-derived gene expression matrix.
2. Normalize counts to TPM using gene lengths and the sample-specific sum of length-normalized counts.
3. Apply the natural logarithm, `ln(TPM + 1)`.
4. Calculate each gene's expression standard deviation across samples using `torch.std(..., dim=0)`.
5. Rank genes by decreasing standard deviation, retain the top 2,000 genes, and save their positional indices.

Neither DepMap expression data nor DepMap gene dependency labels were used to select this list. The same variability-based criterion can be applied to a user's own cohort, as in the custom HVG example below.

The following code shows the selection step once the source matrix is prepared. `X` must be a samples-by-genes PyTorch tensor of `ln(TPM + 1)` values, with columns in the released vocabulary order:

```python
import pandas as pd
import torch

gene_std_df = pd.DataFrame({
    "gene_std": X.std(dim=0).cpu().numpy()
})
gene_std_df.sort_values("gene_std", ascending=False, inplace=True)
indices = gene_std_df.index[:2000].tolist()
torch.save(indices, "interested_gene_list.pt")
```

Reproducing the exact historical list requires the same source expression matrix and gene order. Applying this selection rule to another cohort can produce a different gene set.

### Choosing a custom Top 2,000 HVG set

Users can aggregate embeddings over genes relevant to their own research question. For example, calculate highly variable genes (HVGs) across the bulk transcriptomic cohort being analyzed, then mean-pool over the top 2,000 HVGs present in the BulkFormer vocabulary. In our analyses, this has often improved downstream performance. The appropriate preprocessing, variability metric, and number of genes depend on the task. For predictive benchmarks, estimate gene variability within each training split.

The example below uses across-sample expression standard deviation as the HVG selection metric. It assumes that `log_tpm_df` is a samples-by-genes pandas DataFrame of `ln(TPM + 1)` values, with unique Ensembl gene IDs as columns. Use the observed expression matrix before adding placeholder values for missing vocabulary genes. The code selects up to 2,000 vocabulary genes with positive standard deviation; fewer genes are retained if fewer meet these criteria.

```python
import pandas as pd

gene_info = pd.read_csv("data/bulkformer_gene_info.csv")
gene_to_idx = {
    gene_id: idx
    for idx, gene_id in enumerate(gene_info["ensg_id"])
}

cohort_gene_std = (
    log_tpm_df.loc[:, log_tpm_df.columns.isin(gene_to_idx)]
    .std(axis=0)
    .dropna()
)
ranked_hvgs = (
    cohort_gene_std[cohort_gene_std > 0]
    .sort_values(ascending=False)
    .head(2000)
)
custom_gene_idx = [
    gene_to_idx[gene_id] for gene_id in ranked_hvgs.index
]
if not custom_gene_idx:
    raise ValueError("No variable genes match the BulkFormer vocabulary.")
```

In the notebook's sample-level extraction call, set `interested_gene_idx=custom_gene_idx` and `aggregate_type='mean'`. Continue passing the full vocabulary-aligned expression matrix as `expr_array`.

To inspect the Ensembl IDs represented by the released example list:

```python
import pandas as pd
import torch

indices = torch.load(
    "data/interested_gene_list.pt", map_location="cpu", weights_only=True
)
gene_info = pd.read_csv("data/bulkformer_gene_info.csv")
gene_ids = gene_info.iloc[indices]["ensg_id"].tolist()
```

These pooling choices apply to the original 37M–147M model family. We recommend **BulkFormer-147M**, the largest model in this family, when computational resources permit. See [issue #12](https://github.com/KangBoming/BulkFormer/issues/12#issuecomment-6384729462) for the related clarification.

















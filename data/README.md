## 📦 Preprocessed Data & Resources

The following **preprocessed resource files** required for running BulkFormer demos and scripts can be downloaded from **Zenodo**:

🔗 **Latest Zenodo record (always use the newest version):**   https://doi.org/10.5281/zenodo.15744294

**Available files:**
- `PreBULK.h5ad.zip(Pretraining data for BulkFormer)`
- `bulkformer_gene_info.csv`
- `G_tcga.pt`  
- `G_tcga_weight.pt`  
- `esm2_feature_concat.pt`  
- `demo_count_adata.h5ad`  
- `interested_gene_list.pt`
- `Downtask-related data`


> ⚠️ **Important:** All of the above files have been **updated** in the latest release.

> Please **re-download them from the current Zenodo record**, as any previously downloaded

> versions may be outdated and **incompatible with the latest BulkFormer codebase**.


















## V1 Plus demo resources

`v1plus_demo_counts.csv` contains **four synthetic samples**, with samples as rows and Ensembl IDs as columns. Counts were generated with NumPy seed 42; measured zeros are retained, and eight genes are omitted to demonstrate missing-gene alignment. It contains no biological samples and is only an inference example.

The V1 Plus gene vocabulary, gene lengths, and weighted graph are included in its [Google Drive model bundle](../model/V1_PLUS.md#download-and-run). Use those resources with [bulkformer_v1plus_extract_feature.ipynb](../bulkformer_v1plus_extract_feature.ipynb).

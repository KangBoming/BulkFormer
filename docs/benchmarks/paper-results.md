# Original publication benchmark results

This table preserves the original repository's reported benchmark values for the original BulkFormer model family and comparison methods. It belongs to the original publication workflow. The new [latest vs 147M phenotype comparison](../../benchmarks/phenotype/README.md) uses different dataset preparation, pooling, classifier settings, and folds; scores from the two tables should not be combined into a model-to-model improvement calculation.

Publication: [BulkFormer: A large-scale foundation model for bulk transcriptomes](https://doi.org/10.1016/j.cels.2026.101657), *Cell Systems* (2026).

| Task (Metric)                                   | BulkFormer | Geneformer | GeneCompass | scGPT | scFoundation | scLong | State | Cell2Sentence |
|-------------------------------------------------|-----------:|-----------:|------------:|------:|-------------:|-------:|------:|--------------:|
| Bulk RNA-seq Data Imputation (PCC ↑)            | **0.954**  | NA* | NA*         | NA*   | 0.142        | 0.041  | NA*   | NA*           |
| DiSignAtlas classification (weighted F1 ↑)      | **0.949**  | 0.749| 0.882       | 0.885 | 0.874        | 0.810  | 0.909 | 0.890         |
| CVD classification (weighted F1 ↑)              | **0.978**  | 0.884 | 0.937       | 0.953 | 0.955        | 0.892  | 0.944 | 0.943         |
| TCGA classification (weighted F1 ↑)             | **0.907**  | 0.528      | 0.774       | 0.813 | 0.791        | 0.514  | 0.851 | 0.770         |
| GTEx classification (weighted F1 ↑)             | **0.969**  | 0.848      | 0.919       | 0.936 | 0.939        | 0.678  | 0.926 | 0.940         |
| Drug Response Prediction (mean PCC ↑)           | **0.373**  | 0.246      | 0.300       | 0.356 | 0.303        | 0.238  | 0.336 | 0.328         |
| Gene Essentiality Prediction (mean PCC ↑)       | **0.186**  | 0.104      | 0.123       | 0.143 | 0.155        | 0.111  | 0.137 | 0.118         |

**Note**

1. The best-performing values for each task are highlighted in bold.
2. NA*: Geneformer, GeneCompass, scGPT, State and Cell2Sentence were not directly pretrained to model gene expression values, and therefore cannot perform transcriptome imputation tasks.




The LINCS compound-perturbation benchmark was removed in the 2025-12-11 repository update because its measurement platform differs from bulk RNA-seq. The drug-response task above remains part of the reported results. See the [changelog](../../CHANGELOG.md).

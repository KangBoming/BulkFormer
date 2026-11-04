# BulkFormer-Atlas

**242.37M parameters · 22,636 coding genes · 1,024-dimensional CLS representation**

[Weights and resources](https://drive.google.com/drive/folders/1OTLnUnnorCqL3nNVK0hfk0Om006WCsrJ) · [GPU tutorial](../../notebooks/bulkformer-atlas_sample_embeddings.ipynb) · [Model configuration](../../configs/atlas_model.json)

## Architecture

| Component | Configuration |
|---|---|
| Complete model / encoder parameters | 242,372,356 / 228,738,048 |
| Transformer width / layers / heads | 1,024 / 16 / 16 |
| Attention | Full bidirectional self-attention |
| Feed-forward block | SwiGLU, hidden width 2,816 |
| Normalization / network dropout | RMSNorm / 0 |
| Gene input | Learned identity + linear expression projection |
| Sample readout | Native CLS, before the 512-dimensional projection head |
| Input panel | Up to 4,096 observed coding genes, fixed within a cohort |
| Expression scale | log2(TPM + 1) |

The encoder has no positional embeddings, GCN, GenePT or global-transcriptome MLP. The complete checkpoint retains its expression and projection heads for strict loading; the released sample workflow uses the native CLS representation.

The encoder design is Pandora-inspired. This is an independently produced BulkFormer checkpoint, not the official Pandora weight file. [Architecture comparison](../architecture.md)

## Usage and evaluation

Use Atlas for frozen sample features in cohort classification, clinical score regression and other patient-level predictors. Normalize expression with the supplied vocabulary before selecting coding genes, and retain the selected panel for subsequent samples. [Input guide](../input-guide.md#atlas)

Immutable checkpoint ID: `atlas-infonce-epoch10-update320000`. The official PRIMO submission name is `BulkFormer-variant`; its evaluation uses native CLS embeddings. [Results and protocol](../../benchmarks/primo/README.md)

The PRIMO result supports this pipeline's aggregate performance, rather than universal superiority on every task. Source-scale exceptions and public-task selection are documented in the benchmark card; overlap between anonymized PRIMO cohorts and pretraining has not been fully audited.

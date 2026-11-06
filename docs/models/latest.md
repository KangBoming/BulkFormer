# BulkFormer-Latest

**154.55M parameters · 19,973 coding genes · 768-dimensional contextual gene representations**

[Weights and resources](https://drive.google.com/drive/folders/1wKXEgKk_oPkmaCITi4ASs-05FHl6qkvZ) · [GPU tutorial](../../notebooks/bulkformer-latest_feature_extraction.ipynb) · [Model configuration](../../configs/latest_model.json)

## Architecture

| Component | Configuration |
|---|---|
| Parameters | 154,551,299 |
| Performer width / layers / heads | 768 / 12 / 12 |
| Random attention features / feed-forward width | 256 per head / 3,072 |
| Input branches | Gene identity, sinusoidal expression encoding and global context |
| Input fusion | Separately normalized branches; fixed coefficients 1:1:1 |
| Global expression MLP | 19,973 → 2,560 → 768 |
| Weighted GCN hidden width | 256 |
| Feed-forward / graph / attention dropout | 0.1 / 0.1 / 0.05 |
| Missingness calibration | 64-hidden-unit network; bounded expression gain and offset |
| Expression scale | ln(TPM + 1) |

All gene positions are encoded together. A residual weighted GCN precedes the Performer, and the expression head adjusts its outputs using the actual missing fraction. No GenePT input or CLS token is used. Fixed input coefficients specify how branches are fused; they do not freeze the other network parameters. [Architecture comparison](../architecture.md)

## Usage and evaluation

Provide the complete available coding transcriptome for contextual gene extraction and expression recovery. To summarize HVGs or DEGs, select their contextual outputs after encoding and then pool them. [Input guide](../input-guide.md#latest)

Current checkpoint: **final epoch 20 / update 148,300**, `latest-20261106-epoch20-update148300`. The historical PRIMO entry `BulkFormer-V3Plus` was evaluated with epoch 19 / update 140,885 and mean pooling over available genes. Those scores do not evaluate this final checkpoint. [Results and protocol](../../benchmarks/primo/README.md)

The gene-level recommendation follows the architecture and output interface; a matched Atlas-versus-Latest gene-level benchmark remains to be established. Some downstream phenotype cohorts overlap with source collections, and overlap with anonymized PRIMO cohorts has not been fully audited.

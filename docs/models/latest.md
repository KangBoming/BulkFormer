# BulkFormer-Latest

**154,551,299 parameters; 19,973 genes; 768-d contextual gene representations.** Immutable checkpoint: `latest-20261104-epoch19-update140885`. [Weights and resources](https://drive.google.com/drive/folders/1wKXEgKk_oPkmaCITi4ASs-05FHl6qkvZ). This release pins epoch 19, selected by the mean ID/OOD reconstruction MSE at 10% masking.

The complete input is encoded using three normalized branches with fixed coefficients 1:1:1, a 256-wide weighted GCN and a 12-layer, 12-head Performer. The global MLP has dimensions 19,973 → 2,560 → 768. Performer uses 256 random features per head and feed-forward width 3,072. Feed-forward and graph dropout are 0.1; attention dropout is 0.05. A 64-hidden-unit calibration network predicts sample-wide expression gain and offset from actual input missingness. There is no GenePT input or CLS token.

The corpus contains 719,174 profiles after source processing: 668,208 ARCHS4, 3,899 DiSignAtlas, 27,279 GDC and 19,788 GTEx. The split contains 711,826 training, 3,746 ID validation and 3,602 OOD validation samples. Source collections overlap with some downstream phenotype cohorts, so those evaluations are not an independent external-generalization claim.

Training: one GPU, microbatch 4, accumulation 24, effective batch 96, BF16, AdamW betas (0.9, 0.95), weight decay 0.01, clipping 1.0. The first 1% of updates warm up to 1e-4; the rate then stays constant. Twenty epochs were planned. The released checkpoint is epoch 19 / update 140,885; the complete twentieth-epoch result is not part of this release. Dynamic masks are sampled uniformly between 10% and 20% per sample. Input feature coefficients are fixed; all ordinary model parameters and calibration weights remain trainable.

Validation covers seven mask rates (5%, 10%, 15%, 20%, 30%, 50%, 70%) after each epoch. Selection uses 10% masking, where epoch 19 yields **ID MSE 0.1477805855**, **OOD MSE 0.2003244350**, and **mean 0.1740525102**. Recovery checkpoints are saved every 1,000 updates and completed epochs are retained. [Training JSON](../../configs/latest_training.json) · [Full validation table](../../benchmarks/latest_pretraining.csv).

Recommended use: full-axis contextual gene embeddings, selected gene/gene-set features and expression recovery. Sample vectors can also be obtained by pooling contextual outputs. This recommendation follows the model's architecture and reconstruction objective; a direct Atlas-versus-Latest gene-level benchmark has not yet established superiority.

The historical PRIMO entry `BulkFormer-V3Plus` uses this same epoch-19 weight file and mean pooling over available genes. Its name on the external board has not been changed. The source training architecture identifier is retained only in checkpoint provenance for reproducibility. [Input guide](../input-guide.md) · [Architecture](../architecture.md).

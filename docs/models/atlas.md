# BulkFormer-Atlas

**242,372,356 parameters; 22,636 coding genes; 1,024-d CLS.** Immutable checkpoint: `atlas-infonce-epoch10-update320000`. [Weights and resources](https://drive.google.com/drive/folders/1OTLnUnnorCqL3nNVK0hfk0Om006WCsrJ). Historical PRIMO submission name: `BulkFormer-variant`.

The model uses 16 Transformer blocks, 16 heads, width 1,024 and SwiGLU hidden width 2,816, with network dropout 0. The encoder accounts for 228,738,048 parameters; the complete file also retains the pretraining heads and two unused legacy contrastive scalars. The contrastive projection is 512-d; downstream PRIMO features are the pre-projection CLS.

Training used 1,028,046 ARCHS4 profiles and 1,548 validation profiles, with no GSE overlap between those splits. Each sample produces two independently sampled 4,096-gene views. Gaussian gene/sample noise has scale 0.1; token dropout and expression masking among retained genes each have probability 0.2. Masked reconstruction targets the clean original values. The expected masked fraction of the initial panel is 16%.

AdamW: betas (0.9, 0.95), weight decay 0.05, clipping 1.0, BF16, four GPUs with eight samples per GPU and no gradient accumulation. Learning rate: 3% warmup to 1e-4, then cosine decay to 1e-5 over the planned 40-epoch horizon. The run completed ten epochs and entered epoch 11; this file is the best checkpoint at update 320,000, with validation total loss **1.27032864**. Validation ran every 2,000 updates and was capped at 1,024 of the 1,548 validation samples. Recovery checkpoints were saved every 5,000 updates.

The framework is Pandora-inspired; symmetric InfoNCE replaces the initial local sigmoid loss. This checkpoint was trained from scratch and is not the official Pandora authors' weight file. [Method details](../architecture.md) · [Training JSON](../../configs/atlas_training.json).

Recommended use: frozen sample features for cohort classification, clinical score regression and other patient-level predictors. The PRIMO result supports the aggregate performance of this particular pipeline, not universal superiority on every sample task. Input panels must be fixed within the intended cohort/evaluation protocol. Avoid treating targeted panels or microarray values as equivalent to complete count data.

PRIMO source-scale exceptions and public-task selection are recorded in [the benchmark card](../../benchmarks/primo/README.md). Overlap between the anonymized PRIMO cohorts and pretraining has not been fully audited.

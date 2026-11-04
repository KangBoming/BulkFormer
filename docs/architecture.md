# Atlas and Latest

## BulkFormer-Atlas

Atlas is a sample-oriented encoder. Each gene token combines a learned identity embedding with a linear projection of log2(TPM + 1), followed by RMSNorm. A CLS token and the selected gene tokens pass through 16 bidirectional Transformer blocks with width 1,024, 16 attention heads and SwiGLU feed-forward layers of width 2,816.

The sample feature is the native 1,024-dimensional CLS output, before the 512-dimensional projection head. The model has no positional embeddings, GCN, GenePT or global-transcriptome MLP inputs. The public input workflow uses one fixed panel of up to 4,096 coding genes per cohort.

Atlas's encoder design is inspired by Pandora. The released weights are independently produced BulkFormer weights; PRIMO results are not a controlled architecture ablation or a comparison to the official Pandora checkpoint. [Model card](models/atlas.md) · [Architecture JSON](../configs/atlas_model.json)

## BulkFormer-Latest

Latest encodes all 19,973 gene positions. Gene identity, sinusoidal expression encoding and a global expression MLP (19,973 → 2,560 → 768) are separately normalized and combined with fixed coefficients of 1:1:1. LayerNorm precedes a residual weighted GCN with hidden width 256, followed by a 12-layer Performer with width 768 and 12 heads.

The model returns contextual gene embeddings and expression predictions. A small network reads the actual input missingness and predicts a bounded gain and offset for the expression head. Measured zeros remain observed. There is no GenePT input or CLS token. [Model card](models/latest.md) · [Architecture JSON](../configs/latest_model.json)

## Choosing an output

Use Atlas CLS features as the starting point for the patient/sample tasks supported by PRIMO. Use Latest for full-transcriptome contextual gene extraction and expression recovery, or pool selected contextual outputs for gene-set summaries. These are workflow recommendations; a matched gene-level benchmark is still needed before claiming superiority between the two models.

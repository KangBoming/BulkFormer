# Official PRIMO bulk results

Snapshot: **2026-11-04 12:33:58 +08:00**, 19 leaderboard entries, all 44 released bulk datasets and 6,390 samples covered. [Live leaderboard](https://primomics-primo-eval.hf.space/?tab=leaderboard) · [Official evaluation code](https://github.com/PRIMOmics/primo-eval).

| Public name | Official submission alias | Rank | Elo | Clinical Pearson | Endotype AUROC | Perturbation centered Spearman | Prognosis AUROC | Treatment AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Atlas | BulkFormer-variant | 8/19 | 972 | .481 | .783 | .327 | .732 | .618 |
| Atlas + HVG | BulkFormer-variant + HVG | **1/19** | **1027** | .484 | .795 | .336 | .725 | .633 |
| Latest | BulkFormer-V3Plus | 14/19 | 838 | .469 | .749 | .279 | .708 | .577 |
| Latest + HVG | BulkFormer-V3Plus + HVG | 6/19 | 985 | .483 | .790 | .329 | .716 | .629 |

The renaming in this repository does not create a new submission or alter the external board. The raw rendered-board snapshot and aggregate CSV preserve the official names. The first-place claim applies to **Atlas + HVG**, not Atlas alone. Rankings change as entries change, and Elo differences are not confidence intervals.

The board also retains an older entry literally named `BulkFormer-Latest`. It belongs to the previous release, not the epoch-19 Latest checkpoint documented here; use the aliases in the table to identify the correct rows.

Atlas uses epoch 10 / update 320,000, native pre-projection CLS (1,024 dimensions), BF16 and batch 2. Each dataset uses one fixed set of 4,096 positive-variance mapped coding genes selected without labels across that whole dataset, shared across its samples, tasks and folds. Normalization precedes selection and retains the original 41,623-gene denominator and gene lengths. This whole-cohort selection is transductive; prospective uses should fit selection on training data.

Latest uses epoch 19 / update 140,885, the same checkpoint released here. It encodes the complete available transcriptome and averages contextual gene outputs over observed mapped genes, including measured zeros, yielding 768 dimensions. Inference used BF16, batch 4, no random masking, and no PRIMO fine-tuning. No HVG restriction was applied to its input or pooling.

The organizer independently constructs `+ HVG` by concatenating the representation with its **1,200-gene expression baseline**. Those features are not Atlas's 4,096 input genes and are not contextual gene embeddings. The evaluation uses the official frozen probe and split protocol, not the random-forest phenotype benchmark or original paper settings.

Input auditing retained relevant limitations: d037 Ion AmpliSeq uses per-million scaling without a gene-length divisor; d035/d036 retain count-like proxies because source normalization is incompletely specified; mouse-derived d010 uses mapped human orthologs with human-length limitations. Missing source genes prevent exact reconstruction of the full training normalization denominator. These score records should not be presented as validation on uniformly complete human raw RNA-seq counts.

Model/checkpoint/readout choices were informed by the 16 public bulk tasks. Private labels were not accessed by the local extraction pipeline. Pretraining overlap with anonymized PRIMO cohorts has not been completely audited. The original BulkFormer family paper is a reference for the family, not a report of these two checkpoints.

| Checkpoint | Source training-file SHA256 |
|---|---|
| Atlas epoch 10 | `206bd2d0f8aaf693a7e0a0c7596ff543a3715514cd722dac4cb518b219549982` |
| Latest epoch 19 | `22327988bc1d25adda9a633f4b65053ce8173ebb448096e81e50dc09c0f29a38` |

Atlas submission ID: `20261104T042607.174314Z-d716276bdf19455091413a714f81c1ba`. The submitted feature archive SHA256 is `52aef7a4dd6f3efa9222d52624df3c857c0addf8ad243ff4f0651a78e9ce2d37`. Training files include optimizer state; public inference-file hashes differ because the release excludes it, while model tensors are verified identical.

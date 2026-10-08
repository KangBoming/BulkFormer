# BulkFormer V1 Plus

## Released checkpoint

**BulkFormer V1 Plus Stage1 epoch 17** is a pretrained human bulk-transcriptome model with **101,323,778 parameters (101.3M)**. It produces 768-dimensional sample representations, contextual gene representations, and expression predictions over a fixed **19,973-gene** vocabulary.

| Property | Value |
|---|---|
| Checkpoint | Stage1 epoch 17, optimizer update 126,055 |
| Hidden dimension | 768 |
| Performer layers / heads | 12 / 12 |
| Attention head dimension | 64 |
| Graph bottleneck dimension | 192 |
| Input expression | Natural `ln(TPM + 1)` |
| Missing gene value | `-10`; measured zero remains `0` |
| Checkpoint selection metric | Mean ID/OOD masked-expression MSE at 10% masking: 0.1891963980 |

## Architecture

Each gene token combines a trainable gene identity vector with a fixed sinusoidal encoding of expression. The gene identity branch uses RMS normalization; observed expression encodings have unit RMS. A learned scalar and a per-sample visible-fraction correction scale the expression branch before input LayerNorm.

```text
19,973 genes in the released vocabulary order
  -> gene identity embedding + scaled sinusoidal expression encoding
  -> input LayerNorm
  -> one weighted GCN residual block (768 -> 192 -> 768)
  -> pre-Performer LayerNorm + 12 Performer layers + output LayerNorm
  -> contextual gene representations (19,973 x 768)
  -> expression head (768 -> 768 -> 1)
```

The GCN uses the bundled positive, symmetric TCGA project-balanced coexpression graph. Its residual scale is trainable per channel. Performer projection matrices are fixed and included in the checkpoint. Sample features are obtained by averaging the contextual vectors of observed genes, including genes with measured zero expression.

This release has its own architecture, vocabulary, and graph. It does not use GenePT, ESM2, a full-transcriptome dense projection, or the three appended sample summary features of the original BulkFormer models. Use the V1 Plus loader and bundled resources together.

## Pretraining

The V1 Plus corpus contains 719,174 profiles from ARCHS4, GDC, GTEx, and DiSignAtlas: 711,826 training samples, 3,746 ID validation samples, and 3,602 OOD validation samples. The released checkpoint completed 17 Stage1 epochs.

Stage1 masks 5–15% of available expression positions independently per sample; measured zeros can be selected, and unavailable genes are excluded from the loss. The objective is MSE on masked, available positions. Training used BF16, AdamW, a global batch size of 96, and a linear warmup followed by cosine decay. This download contains the selected Stage1 weights.

## Frozen-feature phenotype evaluation

These historical results were measured on the exact released checkpoint using mean pooling over available genes and a fixed cuML random forest: 500 trees, maximum depth 16, `max_features="sqrt"`, `min_samples_leaf=2`, and seed 42. The reported score is **pooled out-of-fold weighted F1**, across 10 folds. DiSignAtlas and TCGA use registered stratified folds; GTEx uses donor-grouped stratified folds. Encoder extraction used BF16 with batch size 32. No classifier hyperparameter search was performed.

| Task | Samples | Classes | Weighted F1 |
|---|---:|---:|---:|
| DiSignAtlas disease/state classification | 10,688 | 19 | 0.9293 |
| TCGA cancer type classification | 9,780 | 33 | 0.9173 |
| GTEx tissue classification | 19,616 | 54 | 0.9114 |

The protocol and full precision scores are recorded in [v1plus_phenotype_metrics.json](v1plus_phenotype_metrics.json). These results use a different evaluation protocol from the original paper's main table. The notebook demonstrates feature extraction on synthetic counts; it does not rerun the phenotype benchmark. Source collections can overlap pretraining sources, so these scores do not establish performance on independent external cohorts.

## Download and run

1. Create the environment from the repository root:

   ```bash
   conda env create -f bulkformer_v1plus.yaml
   conda activate bulkformer-v1plus
   ```

2. Download the complete [model.pt](https://drive.google.com/file/d/1ly4CNE6XHqHKSP1a5c0UGUeaSVC0Was1/view?usp=drivesdk) and the five resource files from the [V1 Plus Google Drive folder](https://drive.google.com/drive/folders/1AfekiOBNs6GzPBbA9BDlpuoolZIWbGQ2) into `model/bulkformer_v1plus/`. The folder also provides the environment YAML and this model introduction. If Drive wraps a folder download in a ZIP, extract it first so the six files are directly inside the model directory.

   | File | Purpose |
   |---|---|
   | `model.pt` | Complete encoder and expression-head checkpoint |
   | `model_config.json` | Architecture and input configuration |
   | `gene_vocab.csv` | Ordered 19,973-gene vocabulary, symbols and gene lengths |
   | `edge_index.pt` | Weighted graph edge indices |
   | `edge_weight.pt` | Graph edge weights |
   | `manifest.json` | Release metadata and resource checksums |

3. Open [bulkformer_v1plus_extract_feature.ipynb](../bulkformer_v1plus_extract_feature.ipynb) from the repository root and run its cells in order. The notebook verifies the complete checkpoint and all resource files before loading them.

   ```python
   from utils.v1plus import prepare_v1plus_bundle, load_v1plus_model
   model_dir = prepare_v1plus_bundle("model/bulkformer_v1plus", "model")
   model, gene_vocab, manifest = load_v1plus_model(model_dir, device="cuda")
   ```

`model.pt` is one complete file, **406,165,934 bytes**, with SHA256:

```text
0fb65ec123386ef753651cfcc43e2a64728c719307fe565a5ffe16f5d8cf3cfa
```

The download manifest [v1plus_downloads.json](v1plus_downloads.json) records the size and SHA256 of all six files. `prepare_v1plus_bundle` verifies a download directory in place, or copies its verified contents to `model/bulkformer_v1plus/` when a different download directory is supplied. The checkpoint includes model tensors and plain metadata and is loaded with `weights_only=True`; optimizer states and training samples are excluded. All 427 saved model tensors were verified identical to the retained training checkpoint.

## Input and output conventions

- **Raw counts:** use `normalize_data(counts, gene_vocab)`, with samples as rows and Ensembl gene IDs as columns. Normalization uses the released gene lengths and vocabulary-matched genes in the TPM denominator, matching the evaluation pipeline.
- **Already normalized data:** provide natural `ln(TPM+1)` directly. Convert `log2(TPM+1)` by multiplying by `ln(2)`. Do not renormalize logarithmic values as counts.
- **Gene alignment:** `main_gene_selection` removes Ensembl version suffixes, orders the complete axis, and fills unavailable genes or NaN entries with `-10`. Duplicate IDs after version removal require upstream resolution. Gene symbols are available for annotation but are not the input matching keys.
- **Zeros:** measured zeros are observed values and participate in sample pooling. For a source that lacks a gene, omit its column or use NaN; filling it with zero changes the model input.
- **Sample features:** shape `[samples, 768]`, using an observed-gene mean for each sample.
- **Contextual gene features:** shape `[samples, selected_genes, 768]`. Selected output genes do not reduce the full gene axis processed by the encoder. Contextual vectors depend on the input sample; they differ from the static gene identity table.
- **Expression predictions:** shape `[samples, 19973]`, on the learned `ln(TPM+1)` target scale. The head is unconstrained, so negative predictions are possible. It predicts all positions; to impute selected observed genes, set those inputs to `-10` before inference and retain their true values separately.

Start with one GPU, BF16 and `batch_size=1`. This complete-gene-axis model still processes 19,973 tokens when only a few output genes are requested. Increase batch size according to available memory. The helper supports CPU FP32 and CUDA BF16/FP16/FP32.

## Code verification

Run the input-mapping, pooling, and download-integrity regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The architecture implementation in `utils/v1plus/model.py` and `utils/v1plus/graph.py` matches the implementation used for this checkpoint. All nine notebook code cells were executed on one NVIDIA A800 GPU in BF16, using four synthetic samples and the full 19,973-gene input axis. With deterministic CUDA algorithms enabled, public-package sample features, selected gene features, and expression predictions matched the original training model exactly. All 427 GPU state tensors were identical. See [v1plus_release_validation.json](v1plus_release_validation.json) for the verification record. The registered phenotype encoder uses an algebraically equivalent expression-scaling order; its maximum absolute sample-feature difference was 0.000371 on the tested input.

# Using bulkformer-latest

## Environment and GPU selection

From the repository root:

```bash
conda env create -f configs/bulkformer-latest.yaml
conda activate bulkformer-latest
jupyter lab notebooks/bulkformer-latest_feature_extraction.ipynb
```

The environment pins PyTorch 2.5.1, PyTorch Geometric 2.6.1, Performer 1.1.4, NumPy 2.0.2, and pandas 2.2.3. Choose a physical GPU before importing torch in the notebook, or set `CUDA_VISIBLE_DEVICES` when launching Jupyter. Restart the kernel after changing GPU selection if CUDA was already initialized.

The relocated notebook finds the repository root from either the root or `notebooks/` working directory. The Python API examples assume the repository root. Download the six model files using the [model instructions](../model/README.md).

## Input conventions

Provide a pandas DataFrame with samples as rows and Ensembl gene IDs as columns. The model uses a fixed 19,973-gene vocabulary.

| Input | Preparation |
|---|---|
| Raw counts | `normalize_data(counts, gene_vocab)` uses bundled gene lengths and vocabulary-matched genes in the TPM denominator |
| TPM | Apply `numpy.log1p` to non-negative TPM values |
| Natural `ln(TPM+1)` | Use directly |
| `log2(TPM+1)` | Multiply by `numpy.log(2.0)` |

`main_gene_selection` strips Ensembl version suffixes, orders the complete gene axis, and fills absent genes and NaN entries with **-10**. Measured expression zero stays **0** and is included in sample pooling. Duplicate IDs after stripping version suffixes require upstream resolution. Gene symbols are annotations; matching uses Ensembl IDs. Do not feed logarithmic values into count normalization.

## Load and extract

```python
from utils.latest import prepare_latest_bundle, load_latest_model, extract_feature

model_dir = prepare_latest_bundle("model/bulkformer-latest")
model, gene_vocab, manifest = load_latest_model(model_dir, device="cuda")
sample_features = extract_feature(model, aligned_expression, batch_size=1)
gene_features = extract_feature(
    model, aligned_expression, output_feature_type="gene_level",
    interested_gene_idx=[0, 1, 2], batch_size=1,
)
expression_predictions = extract_feature(
    model, aligned_expression, output_feature_type="expr_level", batch_size=1,
)
```

`aligned_expression` is a NumPy array of natural `ln(TPM+1)` values in release vocabulary order. The loader freezes the encoder, verifies its resources, and loads the full encoder and expression head with `weights_only=True`.

| Output | Shape | Details |
|---|---|---|
| `sample_level` | samples × 768 | Mean of observed contextual gene vectors, including measured zeros |
| `gene_level` | samples × selected genes × 768 | Sample-dependent representations; selection reduces returned output only |
| `expr_level` | samples × 19,973 | Predictions on the learned natural `ln(TPM+1)` scale |

No additional random masking is applied during extraction. For imputation, set selected inputs to `-10` and retain their true values separately. The expression head is unconstrained and can return negative values. Static gene identity weights are a different representation from contextual gene outputs.

## Memory and precision

Start with **one GPU and batch size 1**. The encoder always processes the full gene axis. Returning a small output subset does not reduce the input token count. Select BF16 on supported GPUs or FP16 otherwise; CPU inference uses FP32. The notebook selects a supported precision automatically. The API accepts `amp_dtype="bf16"`, `"fp16"`, or `"fp32"`.

Use the [notebook](../notebooks/bulkformer-latest_feature_extraction.ipynb) to save sample features, selected gene features with gene IDs, and expression predictions with sample IDs. Results go to `outputs/latest_demo/` by default.

## Original models and earlier imports

The original 37M–147M models use their [original workflow](models/original-models.md). Their graphs, vocabulary, and configuration are model-family specific.

`utils.v1plus` remains a compatibility import for the earlier release name. The maintained API is `utils.latest`; the current model directory is `model/bulkformer-latest/`. Internal architecture identifiers in the checkpoint continue to record the trained V1 Plus implementation.

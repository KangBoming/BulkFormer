# bulkformer-latest vs bulkformer-147M

## Checkpoints

The latest model is the **V1 Plus Stage1 epoch 17 / update 126,055** checkpoint, published as `bulkformer-latest`. Immutable ID: `v1plus-stage1-epoch17-update126055`.

| Property | bulkformer-147M | bulkformer-latest |
|---|---|---|
| Nominal parameters | 147M | 101.3M (101,323,778) |
| Vocabulary | 20,010 genes | 19,973 genes |
| Hidden dimension | 640 | 768 |
| Returned sample feature dimension | 643: 640 contextual + 3 summary channels | 768 contextual channels |
| Extraction precision | FP16 | BF16 |
| Extraction batch size | 2 | 32 |
| Source evaluation date | 2026-09-15 | 2026-10-02 |

## Primary result

Scores are **pooled out-of-fold weighted F1**. Each test sample receives a prediction from a classifier that excluded its fold. The gain is `(latest − 147M) × 100` percentage points, calculated from unrounded pooled scores.

| Task | Samples | Classes | bulkformer-147M | bulkformer-latest | Gain |
|---|---:|---:|---:|---:|---:|
| DiSignAtlas disease/state | 10,688 | 19 | 0.9148 | **0.9293** | +1.45 pp |
| TCGA cancer type | 9,780 | 33 | 0.8813 | **0.9173** | +3.60 pp |
| GTEx tissue | 19,616 | 54 | 0.8607 | **0.9114** | +5.07 pp |

The retained fold results also show higher weighted F1 for the latest checkpoint in all 10 folds of each task. Fold results and pooled results are different summaries: averaging fold F1 values does not produce pooled out-of-fold F1.

## Shared evaluation setup

- Frozen encoders; no encoder fine-tuning.
- Identical dataset sample order, labels, registered fold assignments, and classifier hyperparameters.
- Natural `ln(TPM+1)` input; each model uses its own vocabulary and matching resources.
- Mean pooling over all source-available contextual genes, including measured zeros. No extra random inference masking or ESM2/GenePT feature fusion.
- 10 stratified folds for DiSignAtlas and TCGA; 10 donor-grouped stratified folds for GTEx.
- cuML `RandomForestClassifier`: 500 trees, depth 16, `max_features="sqrt"`, `min_samples_leaf=2`, `n_bins=128`, `n_streams=1`, bootstrap enabled, `max_samples=1.0`, Gini splits, seed 42.
- No classifier hyperparameter search. Historical 147M out-of-fold predictions and scores were reused after protocol and sample/fold identity checks.

Dataset preparation: DiSignAtlas uses the count-only 19-class case/state cohort; TCGA uses the 33-class expected-count cancer-type cohort; GTEx uses the 54-tissue GTEx v11 cohort with donors kept within one fold.

## Records

| File | Contents |
|---|---|
| [metrics.csv](metrics.csv) | Full precision pooled and fold summary scores for both models |
| [comparison.csv](comparison.csv) | Pooled weighted F1 gains in percentage points |
| [per_fold.csv](per_fold.csv) | 60 aggregate model/fold rows; no sample identifiers |
| [protocol.json](protocol.json) | Classifier, splits, metadata fingerprints, source checkpoint hashes, feature conventions |
| [latest_metrics.json](latest_metrics.json) | Previously retained metrics for the exact latest checkpoint |

These are retained completed evaluation results, exported from the registered benchmark. The [feature extraction notebook](../../notebooks/bulkformer-latest_feature_extraction.ipynb) demonstrates inference on synthetic counts; it does not rerun the three classification tasks. This directory does not include the registered cohort inputs or sample-level fold assignments needed for a complete rerun.

## Interpretation

The comparison evaluates each released representation under matched downstream classifier settings. Native vocabularies, parameter counts, feature dimensions, precision, and pretraining histories differ, so it does not isolate a single architecture change. Source collections can overlap pretraining sources; these results do not establish performance on independent external cohorts.

The [original paper table](../../docs/benchmarks/paper-results.md) uses a separate protocol. Its scores should not be substituted for the registered 147M baseline in this comparison.

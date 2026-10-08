# Changelog

## 2026-10-09 — bulkformer-latest

- Publish the V1 Plus Stage1 epoch 17 / update 126,055 checkpoint as **bulkformer-latest**: 101.3M parameters, 19,973 human genes, 768-dimensional representations.
- Provide one complete Google Drive weight file with configuration, vocabulary, graph, environment, and checksums in the model folder.
- Add a GPU-validated Jupyter workflow for sample embeddings, contextual gene embeddings, and expression predictions using synthetic example counts.
- Publish retained frozen-feature phenotype comparisons with bulkformer-147M on identical samples, fold assignments, and classifier settings.
- Organize current inference, notebooks, environments, documentation, assets, download metadata, and benchmark records into dedicated directories.
- Consolidate the homepage around model choice, quick start, and current performance. Preserve original-model entry points and earlier V1 Plus compatibility imports.

Checkpoint ID: `v1plus-stage1-epoch17-update126055`. The public rename preserves the model tensors and trained architecture.

## 2025-12-11 — Original model-family update

- Update the original model code, data resources, and weights.
- Remove the LINCS compound-perturbation benchmark due to its measurement-modality mismatch with bulk RNA-seq.
- Add optional ESM2 fusion to gene-level feature extraction and improve large-cohort inference memory usage.
- Include three sample summary features in the original model's returned gene representations.

The original reported scores are retained in [docs/benchmarks/paper-results.md](docs/benchmarks/paper-results.md).

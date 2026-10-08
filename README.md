# BulkFormer

### Foundation models for human bulk transcriptomes

BulkFormer combines gene coexpression graphs and Performer attention to represent complete bulk RNA-seq profiles. It provides sample embeddings, contextual gene embeddings, and expression predictions for downstream analysis.

**[Quick start](#quick-start)** · **[Models & downloads](#models-and-downloads)** · **[Performance](#performance)** · **[Documentation](docs/README.md)** · **[Publication](#publication)**

## bulkformer-latest

**The current release is `bulkformer-latest`**: a **101.3M** parameter encoder over **19,973 human genes**, with **768** hidden channels and **12** Performer layers. This is the V1 Plus Stage1 epoch 17 checkpoint, released under its new public name on **2026-10-09**. Its immutable checkpoint ID is `v1plus-stage1-epoch17-update126055`.

The model combines normalized trainable gene identity embeddings and sinusoidal expression encodings, followed by a weighted GCN and Performer encoder. Missing genes and measured zeros are handled separately; sample embeddings average the observed gene representations.

| Output | Shape | Example use |
|---|---|---|
| Sample embeddings | samples × 768 | Disease, cancer type, and tissue classification |
| Contextual gene embeddings | samples × selected genes × 768 | Gene representations in each sample context |
| Expression predictions | samples × 19,973 | Expression reconstruction and imputation |

**[Download complete weights](https://drive.google.com/file/d/1ly4CNE6XHqHKSP1a5c0UGUeaSVC0Was1/view?usp=drivesdk)** · **[Download configuration & resources](https://drive.google.com/drive/folders/1AfekiOBNs6GzPBbA9BDlpuoolZIWbGQ2)** · **[Model card](docs/models/bulkformer-latest.md)** · **[Jupyter notebook](notebooks/bulkformer-latest_feature_extraction.ipynb)**

See the [changelog](CHANGELOG.md) for this release and earlier updates.

## Quick start

### 1. Install

```bash
git clone https://github.com/KangBoming/BulkFormer.git
cd BulkFormer
conda env create -f configs/bulkformer-latest.yaml
conda activate bulkformer-latest
```

### 2. Download the model

Download the complete `model.pt` and its five accompanying resource files from the [Google Drive model folder](https://drive.google.com/drive/folders/1AfekiOBNs6GzPBbA9BDlpuoolZIWbGQ2) into:

```text
model/bulkformer-latest/
├── model.pt
├── model_config.json
├── gene_vocab.csv
├── edge_index.pt
├── edge_weight.pt
└── manifest.json
```

The weights are **one complete file (406 MB)**. The loader checks every downloaded file against the published [SHA256 manifest](model/downloads/bulkformer-latest.json). All model-specific configuration, vocabulary, and graph resources are in the same Drive folder.

### 3. Extract features on a GPU

```bash
jupyter lab notebooks/bulkformer-latest_feature_extraction.ipynb
```

Select your GPU in the first code cell. The notebook includes four synthetic count profiles and demonstrates sample, gene, and expression outputs. Start with batch size 1; the model processes the complete 19,973-gene axis for every sample.

For a Python workflow, run this from the repository root:

```python
import pandas as pd
from utils.latest import (
    prepare_latest_bundle, load_latest_model,
    normalize_data, main_gene_selection, extract_feature,
)

model_dir = prepare_latest_bundle("model/bulkformer-latest")
model, vocab, metadata = load_latest_model(model_dir, device="cuda")
counts = pd.read_csv("data/latest_demo_counts.csv", index_col=0)
expression = normalize_data(counts, vocab)
aligned, _, _ = main_gene_selection(expression, vocab["canonical_ensg"])
sample_features = extract_feature(model, aligned.to_numpy(), batch_size=1)
```

Inputs are samples × Ensembl gene IDs. Raw counts are converted to natural `ln(TPM+1)` with the released gene lengths. Already normalized inputs, missing values, and precision options are explained in the [usage guide](docs/usage.md).

## Models and downloads

| Model | Parameters | Hidden dim | Layers | Heads | Weights | Workflow |
|---|---:|---:|---:|---:|---|---|
| **bulkformer-latest** | **101.3M** | **768** | **12** | **12** | [Google Drive](https://drive.google.com/file/d/1ly4CNE6XHqHKSP1a5c0UGUeaSVC0Was1/view?usp=drivesdk) | [Latest notebook](notebooks/bulkformer-latest_feature_extraction.ipynb) |
| bulkformer-147M | 147M | 640 | 12 | 8 | [Google Drive](https://drive.google.com/file/d/1UtqN_vCh3669Fs-GU5CTE7F7UnuQCAzN/view?usp=drive_link) | [Original models](docs/models/original-models.md) |
| bulkformer-127M | 127M | 640 | 8 | 8 | [Google Drive](https://drive.google.com/file/d/1-5AdgIpkm8dOm9tuwOXcqS1sUyGN6vHg/view?usp=drive_link) | [Original models](docs/models/original-models.md) |
| bulkformer-93M | 93M | 512 | 6 | 8 | [Google Drive](https://drive.google.com/file/d/1s_3XoMaHiBfxi5C8D3bgwwzNrihafaIk/view?usp=drive_link) | [Original models](docs/models/original-models.md) |
| bulkformer-50M | 50M | 256 | 2 | 8 | [Google Drive](https://drive.google.com/file/d/12ZYGYrZIQJyodaVicrJpnY_8_JG-hamK/view?usp=drive_link) | [Original models](docs/models/original-models.md) |
| bulkformer-37M | 37M | 128 | 1 | 8 | [Google Drive](https://drive.google.com/file/d/1qY2qaXfKfDot9EMcOF9gr8T0jghrH7km/view?usp=drive_link) | [Original models](docs/models/original-models.md) |

Each model family requires its matching vocabulary, graph, and loader. The original models use the original [notebook](bulkformer_extract_feature.ipynb) and [environment](bulkformer.yaml); `bulkformer-latest` uses the dedicated workflow above. Full resource instructions are in the [model directory](model/README.md).

## Performance

### bulkformer-latest vs bulkformer-147M

Frozen sample embeddings were evaluated with the **same samples, registered 10-fold assignments, and fixed cuML random forest**. Scores are **pooled out-of-fold weighted F1**; gains are absolute percentage points.

| Task | Samples | Classes | bulkformer-147M | bulkformer-latest | Gain |
|---|---:|---:|---:|---:|---:|
| DiSignAtlas disease/state classification | 10,688 | 19 | 0.9148 | **0.9293** | **+1.45 pp** |
| TCGA cancer type classification | 9,780 | 33 | 0.8813 | **0.9173** | **+3.60 pp** |
| GTEx tissue classification | 19,616 | 54 | 0.8607 | **0.9114** | **+5.07 pp** |

The current checkpoint improves all three scores with approximately **31% fewer parameters** than the nominal 147M model. These results compare the released representations: the native vocabularies, feature dimensions (768 vs 643), and extraction precision (BF16 vs FP16) differ. GTEx folds are grouped by donor. Source collections can overlap pretraining sources.

See the [complete protocol and results](benchmarks/phenotype/README.md) for full precision scores, fold results, checkpoint hashes, and classifier settings. This comparison uses a separate protocol from the original publication; the [original paper benchmark table](docs/benchmarks/paper-results.md) remains available with its original scores.

## Repository guide

```text
BulkFormer/
├── configs/             Environment for bulkformer-latest
├── notebooks/           Latest feature extraction notebook
├── docs/                Model cards, usage, paper results, figures
├── benchmarks/          Aggregate phenotype results and protocol
├── model/               Download manifests, validation, local weights
├── data/                Synthetic example and original gene resources
├── utils/latest/        Latest architecture, graph, and inference API
├── utils/v1plus/        Compatibility imports for the former release name
└── tests/               Input, pooling, and download integrity checks
```

The original-model notebook, environment, `model/config.py`, and `utils/BulkFormer*.py` remain at their established paths. Start at the [documentation index](docs/README.md) or [notebook index](notebooks/README.md).

## Publication

Kang, B., Fan, R., Yi, M., Cui, C., & Cui, Q. (2026). **BulkFormer: A large-scale foundation model for bulk transcriptomes**. *Cell Systems*, 101657. [DOI: 10.1016/j.cels.2026.101657](https://doi.org/10.1016/j.cels.2026.101657).

```bibtex
@article{kang2026bulkformer,
  title = {BulkFormer: A large-scale foundation model for bulk transcriptomes},
  author = {Kang, Boming and Fan, Rui and Yi, Meizheng and Cui, Chunmei and Cui, Qinghua},
  journal = {Cell Systems},
  year = {2026},
  pages = {101657},
  doi = {10.1016/j.cels.2026.101657}
}
```

Please also identify the model name and checkpoint ID when reporting results for `bulkformer-latest`.

## License and contact

Code is distributed under the [MIT License](LICENSE).

- Boming Kang: [kangbm@bjmu.edu.cn](mailto:kangbm@bjmu.edu.cn)
- Qinghua Cui: [cuiqinghua@bjmu.edu.cn](mailto:cuiqinghua@bjmu.edu.cn)

# Model downloads and resources

## bulkformer-latest

**[Complete model.pt](https://drive.google.com/file/d/1ly4CNE6XHqHKSP1a5c0UGUeaSVC0Was1/view?usp=drivesdk)** · **[Google Drive model folder](https://drive.google.com/drive/folders/1AfekiOBNs6GzPBbA9BDlpuoolZIWbGQ2)** · **[Model card](../docs/models/bulkformer-latest.md)**

Current checkpoint: `v1plus-stage1-epoch17-update126055`, 101,323,778 parameters.

Download the following six files from the same folder and save them directly in `model/bulkformer-latest/`:

| File | Purpose |
|---|---|
| `model.pt` | Complete encoder and expression-head weights; 406,165,934 bytes |
| `model_config.json` | Architecture and input configuration |
| `gene_vocab.csv` | Ordered 19,973-gene vocabulary, symbols, and gene lengths |
| `edge_index.pt` | Weighted graph edge indices |
| `edge_weight.pt` | Graph weights |
| `manifest.json` | Checkpoint identity, release metadata, and resource checksums |

The Drive folder also contains the environment YAML, model introduction, download manifest, and checkpoint SHA256. If Drive wraps a folder download in a ZIP, extract it so the six files are directly inside the model directory. The checkpoint itself is one complete file.

```python
from utils.latest import prepare_latest_bundle, load_latest_model
model_dir = prepare_latest_bundle("model/bulkformer-latest")
model, gene_vocab, metadata = load_latest_model(model_dir, device="cuda")
```

The [download manifest](downloads/bulkformer-latest.json) provides each file's URL, size, and SHA256. Verification uses the manifest supplied by the repository. You can also pass a different download directory to `prepare_latest_bundle`; it copies verified files into `model/bulkformer-latest/` without overwriting an existing different destination.

Weight SHA256:

```text
0fb65ec123386ef753651cfcc43e2a64728c719307fe565a5ffe16f5d8cf3cfa
```

## Original models

| Model | Parameters | Hidden dim | Layers | Heads | Weights |
|---|---:|---:|---:|---:|---|
| bulkformer-147M | 147M | 640 | 12 | 8 | [Google Drive](https://drive.google.com/file/d/1UtqN_vCh3669Fs-GU5CTE7F7UnuQCAzN/view?usp=drive_link) |
| bulkformer-127M | 127M | 640 | 8 | 8 | [Google Drive](https://drive.google.com/file/d/1-5AdgIpkm8dOm9tuwOXcqS1sUyGN6vHg/view?usp=drive_link) |
| bulkformer-93M | 93M | 512 | 6 | 8 | [Google Drive](https://drive.google.com/file/d/1s_3XoMaHiBfxi5C8D3bgwwzNrihafaIk/view?usp=drive_link) |
| bulkformer-50M | 50M | 256 | 2 | 8 | [Google Drive](https://drive.google.com/file/d/12ZYGYrZIQJyodaVicrJpnY_8_JG-hamK/view?usp=drive_link) |
| bulkformer-37M | 37M | 128 | 1 | 8 | [Google Drive](https://drive.google.com/file/d/1qY2qaXfKfDot9EMcOF9gr8T0jghrH7km/view?usp=drive_link) |

Use the [original-model workflow](../docs/models/original-models.md), original `model/config.py`, and resources described in [data/README.md](../data/README.md).

## Directory organization

- `downloads/`: Public download manifests with model identity, URLs, and checksums.
- `validation/`: Published inference and GPU validation records.
- `bulkformer-latest/`: Locally downloaded model bundle, excluded from Git.
- `config.py`: Configuration selector for the original model family.

`V1_PLUS.md` and `v1plus_downloads.json` are retained for the earlier release name. The current public workflow uses `bulkformer-latest`.

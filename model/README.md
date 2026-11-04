# Model downloads

| Model | Complete release folder | Weight file |
|---|---|---|
| BulkFormer-Atlas | [Google Drive](https://drive.google.com/drive/folders/1OTLnUnnorCqL3nNVK0hfk0Om006WCsrJ) | `model.pt`, approximately 970 MB |
| BulkFormer-Latest | [Google Drive](https://drive.google.com/drive/folders/1wKXEgKk_oPkmaCITi4ASs-05FHl6qkvZ) | `model.pt`, approximately 619 MB |

Download the folder contents into `model/atlas/` or `model/latest/`, or use `bulkformer.download_model`. The weights are one complete FP32 file per model, not split archives. Both folders include `model_config.json`, `gene_vocab.csv`, `manifest.json`, `training_config.json`, `environment.yaml` and `MODEL_CARD.md`. Atlas additionally needs `normalization_vocab.csv`; Latest needs `edge_index.pt` and `edge_weight.pt`. Extract a Drive-generated ZIP before loading.

The loader verifies required resource checksums, matches configuration to checkpoint metadata, uses `weights_only=True`, and loads the model strictly. Weight files contain model tensors and plain metadata, without optimizer state, RNG state or training samples. Read access is public by link; editing remains restricted to the owner.

The download registry records immutable checkpoint IDs. `latest` is the API model key for the pinned release, not an automatic download of a live training file. Original model downloads remain in the [family homepage](../README.md#original-model-family).

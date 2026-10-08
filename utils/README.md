# Model implementation

| Path | Responsibility |
|---|---|
| `latest/model.py` | Exact trained V1 Plus architecture used by bulkformer-latest |
| `latest/graph.py` | Loading and validation of the released weighted gene graph |
| `latest/inference.py` | Download verification, model loading, input normalization, alignment, feature extraction |
| `latest/__init__.py` | Public `BulkFormerLatest` aliases and inference API |
| `v1plus/` | Compatibility imports for the former release name |
| `BulkFormer.py`, `BulkFormer_block.py`, `Rope.py` | Original 37M–147M architecture |

New usage imports from `utils.latest`. `BulkFormerLatest` and `BulkFormerLatestConfig` alias the original trained classes; checkpoint keys and the internal `BulkFormerV1Plus-v1` identifier preserve compatibility. The architecture and graph computation are unchanged by the public rename.

```python
from utils.latest import load_latest_model, normalize_data, main_gene_selection, extract_feature
```

The earlier `utils.v1plus` imports use this same implementation. Its `prepare_v1plus_bundle` retains the former `model/bulkformer_v1plus/` destination using the compatibility manifest.

See [docs/usage.md](../docs/usage.md) for examples and [model/README.md](../model/README.md) for model-specific resources.

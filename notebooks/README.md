# GPU tutorials

- [Atlas sample embeddings](bulkformer-atlas_sample_embeddings.ipynb): counts → fixed cohort gene panel → 1,024-dimensional CLS.
- [Latest gene features](bulkformer-latest_feature_extraction.ipynb): complete coding transcriptome → selected contextual gene vectors and expression recovery.

Install the package from the repository root with `pip install -e . --no-deps` after creating the documented environment. Download the relevant model folder before running the notebook. Both notebooks retain real GPU execution outputs on eight synthetic samples. Their tables demonstrate API behavior and are not clinical or biological performance claims. The original publication notebook is retained at the root of the main BulkFormer repository.

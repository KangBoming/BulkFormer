# Demo expression profiles

`demo_counts.csv` contains **eight synthetic profiles**, with samples in rows and human Ensembl gene IDs in columns. The columns cover the union of Atlas's normalization vocabulary and Latest's coding-gene vocabulary, so both tutorials start with complete source coverage.

The counts were generated with NumPy's `default_rng(42)`: one mean per gene sampled from a Gamma distribution (shape 2, scale 100), followed by negative-binomial draws with dispersion 5. No patient samples, phenotype labels or biological group effects are included.

These profiles demonstrate normalization, array alignment, feature extraction and saving outputs. Their embeddings or reconstruction values are not biological benchmark results. Replace this file with your own samples-by-Ensembl count matrix and set `input_scale` to the actual source scale.

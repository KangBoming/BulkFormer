# Preparing bulk transcriptomes

Input tables have samples in rows and unique human Ensembl gene IDs in columns. Version suffixes are removed; duplicates after that operation are rejected. Resolve ambiguous symbols and duplicate measurements before calling the API. Keep observed zero counts as zero, and use NaN for unavailable entries. The preparation functions align to the model vocabulary and represent unavailable positions internally as -10.

Declare the scale: `counts`, `tpm`, `log1p_tpm`, or `log2_tpm`. These options do not mean that microarray intensities, FPKM or unspecified normalized expression are interchangeable with counts. Recover the original source scale before conversion. Already normalized TPM is not re-normalized by the API.

## Atlas

For count data, use the released gene lengths and **all matched genes in the 41,623-gene normalization vocabulary before selecting coding genes or HVGs**. That denominator includes lncRNAs, although only the 22,636 coding identities are encoded. The released lengths retain historical fallback values; the normalization vocabulary marks them. Incomplete source coverage yields TPM-like values rather than the exact training denominator. The model receives log2(TPM+1).

Select a fixed panel of 4,096 observed, positive-variance coding genes for the cohort; use the same panel across samples. The default selector preserves model-vocabulary order and never fills missing positions with duplicates. Save the selected Ensembl IDs and indices. For prospective prediction, fit gene selection only on the training cohort. The reported PRIMO protocol instead used each full unlabeled released cohort; this is transductive preprocessing and must be disclosed.

The normalization vocabulary and input panel are different things. Do not normalize only the selected 4,096 genes, and do not replace CLS with the contrastive projection or an unvalidated pooling scheme when reproducing the reported results.

## Latest

Provide as complete a coding transcriptome as possible, aligned to all **19,973** positions. For raw counts, TPM is calculated over matched source-available vocabulary genes with the released lengths. The model receives natural ln(TPM+1). Missing source genes remain missing; measured zeros count toward coverage and sample pooling.

The expression calibrator reads the total missing fraction, including both unavailable source genes and values intentionally hidden for recovery. Sufficient gene coverage remains important, and performance at high missingness should be assessed on the intended data.

To represent a sample with HVGs or DEGs, run the **complete gene axis** first, then select contextual outputs and pool observed selected genes. `gene_indices` selects returned outputs and never truncates the encoder input. Infer gene selection without using held-out labels. For expression recovery, replace the values to hide with -10 after preparing input, then read `expression_prediction` at those indices. Predictions are on the natural-log expression scale and can be slightly negative; retain raw values for evaluation.

## Precision and reproducibility

Start with BF16 on a supported GPU and batch size 2 for Atlas or 1 for Latest. Use `precision="fp32"` when supported full precision is needed. The validation record compares exported weights and GPU outputs to the original implementations. CUDA sparse reductions in Latest can vary between executions and BF16 can amplify that variation; the record reports actual numerical errors rather than promising bitwise-identical outputs. Larger cohorts should be processed in batches, and full per-gene tensors should be requested only when needed.

In the release check, all exported model tensors matched the source checkpoints exactly. Atlas CLS outputs also matched the original GPU implementation and a retained PRIMO feature cache exactly. Final epoch-20 Latest's FP32 output comparison, with TF32 disabled, had relative L2 differences below 2e-6; BF16 differences were larger but below 1% for gene, sample and expression outputs. See the [numerical validation record](../model/validation/gpu.json) for the measured values. These checks validate the export and implementation, not biological performance.

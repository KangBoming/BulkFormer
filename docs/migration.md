# Release identities

The 2026-11-04 model family promotes **BulkFormer-Atlas** (the former PRIMO name `BulkFormer-variant`) for sample features and publishes a new **BulkFormer-Latest** checkpoint for full-transcriptome gene features.

The previous `bulkformer-latest` repository release referred to V1 Plus Stage1 epoch 17 / update 126,055, a 101.3M model. It is no longer an active model in the current tutorials or download selector. That repository revision is preserved under the [v1plus-20261009](https://github.com/KangBoming/bulkformer-latest/tree/v1plus-20261009) tag for existing analyses. Its weight file has not been silently replaced at its old Google Drive URL. Do not mix those resources with the new 154.55M Latest model.

The current Latest checkpoint is `latest-20261104-epoch19-update140885`; the original training identifier is recorded in its manifest for traceability. Use the new package API and new resource folder together. The old `utils.latest` API is not an alias for the new architecture. The original 37M–147M family remains available separately for publication reproducibility.

On PRIMO, the current Latest weights correspond to the historical submission name `BulkFormer-V3Plus`. The older leaderboard entry literally named `BulkFormer-Latest` refers to the previous release and must not be attributed to the new checkpoint.

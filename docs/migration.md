# Release identities

The 2026-11-04 model family promotes **BulkFormer-Atlas** (the former PRIMO name `BulkFormer-variant`) for sample features and publishes a new **BulkFormer-Latest** checkpoint for full-transcriptome gene features.

The previous `bulkformer-latest` repository release referred to V1 Plus Stage1 epoch 17 / update 126,055, a 101.3M model. It is no longer an active model in the current tutorials or download selector. That repository revision is preserved under the [v1plus-20261009](https://github.com/KangBoming/bulkformer-latest/tree/v1plus-20261009) tag for existing analyses. Its weight file has not been silently replaced at its old Google Drive URL. Do not mix those resources with the new 154.55M Latest model.

The current Latest checkpoint is `latest-20261106-epoch20-update148300`; the original training identifier is recorded in its manifest for traceability. Use the new package API and new resource folder together. The old `utils.latest` API is not an alias for the new architecture. The original 37M–147M family remains available separately for publication reproducibility.

On PRIMO, the historical submission `BulkFormer-V3Plus` uses epoch 19. The current final epoch-20 weights have not been evaluated on PRIMO. The older leaderboard entry literally named `BulkFormer-Latest` refers to the previous release and must not be attributed to the new checkpoint.

## Updating the epoch-19 Latest weights

Update this repository (`git pull`) and its installed package, then run `download_model("latest", "model/latest")`. The downloader checks the new pinned hashes and replaces older local resources. The architecture and extraction API are unchanged. The [epoch-19 archive](https://drive.google.com/drive/folders/1OYexRemCBEEcY5ClYgl51FtvgauI5V17) is retained for reproducing the historical PRIMO submission.

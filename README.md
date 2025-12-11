

## Abstract
Large language models have emerged as powerful foundation models that are transforming transcriptome analysis. However, existing RNA-seq foundation models are all pretrained on sparse scRNA-seq data, which typically detects only ~3000 genes per cell. This thus creates a critical gap in models (especially clinical models) specifically designed for bulk transcriptomes, a fundamentally different modality capable of profiling ~16,000 genes per sample. Here we propose BulkFormer, a large-scale foundation model for bulk transcriptome analysis. With ~150 million parameters covering about 20,000 protein-coding genes, BulkFormer is pretrained on over 500,000 human bulk RNA-seq profiles. BulkFormer incorporates a hybrid encoder architecture, combining a graph neural network to capture explicit gene-gene interactions and a performer module to model global expression dependencies. As a result, despite incurring substantially lower training costs than single-cell foundation models, BulkFormer consistently outperforms them in five downstream tasks: transcriptome imputation, disease annotation, prognosis modeling, drug response prediction, and gene essentiality prediction. Moreover, through a series of experiments, we demonstrate that the modality of the pretraining data exerts a critical influence on foundation model performance. Collectively, these results demonstrate BulkFormer’s power as a versatile and robust framework for bulk transcriptome modeling and biomedical discovery.


## Overview
![Overview](bulkformer_overview.png)


## Results
| Task (Metric)                       | BulkFormer | Geneformer | GeneCompass | scGPT | scFoundation | scLong |
|------------------------------------|------------|------------|-------------|-------|---------------|--------|
| Expression Imputation (PCC ↑)      | **0.954**  | NA*        | NA*         | NA*   | 0.142         | 0.041  |
| Disease Annotation (Weighted F1 ↑) | **0.939**  | 0.749      | 0.882       | 0.885 | 0.874         | 0.810  |
| Tissue Annotation (Weighted F1 ↑) | **0.963**  | 0.848      | 0.919       | 0.936 | 0.939         | 0.678  |
| Cancer Subtype Annotation (Weighted F1 ↑) | **0.833**  | 0.473      | 0.761       | 0.830 | 0.791         | 0.347  |
| Prognosis Modeling (AUROC ↑)       | **0.747**  | 0.647      | 0.709       | 0.723 | 0.726         | 0.584  |
| Drug Response Prediction (PCC ↑)   | **0.910**  | 0.873      | 0.872       | 0.877 | 0.880         | 0.843  |
| Compound Perturbation (PCC ↑)      | **0.493**  | 0.473      | 0.476       | 0.481 | 0.464         | 0.471  |
| Gene Essentiality Prediction (PCC ↑)| **0.931** | 0.897      | 0.881       | 0.907 | 0.852         | 0.889  |

Note：

The best-performing values for each task are highlighted in bold.

NA*: Geneformer, GeneCompass, and scGPT were not directly pretrained to model gene expression values, and therefore cannot perform transcriptome imputation tasks. 


## Main requirements
* python=3.12.7
* pytorch=2.5.1
* scikit-learn=1.5.2
* pandas=2.2.3
* numpy=2.0.2
* performer-pytorch=1.1.4

## Quick start
**Step1: clone the repo**
```
mkdir ./BulkFormer
cd BulkFormer
git clone https://github.com/KangBoming/BulkFormer.git
```
**Step2: create and activate the environment**
```
cd BulkFormer
conda env create -f bulkformer.yaml
conda activate bulkformer
```
**Step3: download pretrained model and data**
```
cd BulkFormer/model
Please follow the README.md file to download pretrained BulkFormer model.

cd BulkFormer/data
Please follow the README.md file to download related data.
```

**Step4: model infernece**
```
cd BulkFormer
Please follow bulkformer_extract_feature.ipynb
```

## Publication
A large-scale foundation model for human bulk transcriptomes

## License
This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/KangBoming/DeepAVC/blob/main/LICENSE) file for details

## Contact
Please feel free to contact us for any further questions

Boming Kang <kangbm@bjmu.edu.cn>

Qinghua Cui <cuiqinghua@bjmu.edu.cn>



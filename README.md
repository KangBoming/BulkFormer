# BulkFormer
BulkFormer: A large-scale foundation model for human bulk transcriptomes
## Abstract
Large language models (LLMs) have become foundation models leading to breakthroughs in the fields of transcriptome modeling. However, existing RNA-seq foundation models are predominantly pretrained on sparse scRNA-seq data, with few designed specifically for bulk RNA-seq. Here we proposed BulkFormer, a large-scale foundation model for bulk transcriptomes, with 150 million parameters covering about 20,000 protein-coding genes, pretrained on over 500,000 human bulk transcriptomic profiles. BulkFormer features a hybrid encoder architecture combining a graph neural network to capture explicit gene-gene interactions and a performer module to model global expression dependencies. As a result, BulkFormer consistently outperforms existing baseline models in diverse downstream tasks, including transcriptome imputation, disease annotation, prognosis modeling, drug response prediction, compound perturbation simulation, and gene essentiality scoring. Notably, BulkFormer enhances the discovery of clinically relevant biomarkers and reveals latent disease mechanisms by imputing biologically meaningful gene expression. All these results show the power of BulkFormer for bulk transcriptome modeling and analysis. 

## Overview


## Publication
BulkFormer: A large-scale foundation model for human bulk transcriptomes

## Main requirements
* python=3.7.13
* pytorch=1.10.0
* cudatoolkit=11.3.1
* scikit-learn=1.0.2
* pandas=1.3.5
* numpy=1.21.5
* fair-esm=2.0.0

## Quick start






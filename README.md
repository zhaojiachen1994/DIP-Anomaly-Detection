# DIP: Density-increasing Neighbor Path for Unsupervised Anomaly Detection
This repository contains the code for paper "Searching Density-increasing Path to Density Peaks for Unsupervised Anomaly Detection".

## Quick start
Run the `main.py` in the `Evaulate` dictionary.

We offer a demo to evaluate single DIP on **musk** dataset and evaluate ensemble DIP on **arrhythmia** dataset.

## Main idea of DIP
<div align=left><img src ="https://github.com/zhaojiachen1994/DIP-Anomaly-Detection/blob/main/figures/DIPidea.jpg" width="200" height="120"/></div>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/1d-gaussian-fit.png" width="300" height="150"/></div>
## Experimental results

DIP achieved better average results than a lot of existing traditional and deep methods. The evaluation metrics are AUC-ROC,
AUC-PR, and F1 score. This repository only show the F1 score performance, while the other results can be found in the original paper.
The dataset information and other datasets used in the paper can be download from [here](http://odds.cs.stonybrook.edu/).


### F1 score performance

|    ROC   | arrhythmia | ionosphere | lympho | mnist |  musk |  pima | satellite | satimage-2 | thyroid | vowels |  wbc  | Average |
|:--------:|:----------:|:----------:|:------:|:-----:|:-----:|:-----:|:---------:|:----------:|:-------:|:------:|:-----:|:-------:|
|   ABOD   |    0.762   |    0.922   |  0.946 | 0.733 | 0.167 | 0.678 |   0.554   |    0.793   |  0.909  |  0.965 | 0.908 |  0.757  |
|   CBLOF  |    0.786   |    0.890   |  0.979 | 0.802 | 1.000 | 0.646 |   0.719   |    0.999   |  0.931  |  0.915 | 0.939 |  0.873  |
|    FB    |    0.782   |    0.868   |  0.980 | 0.698 | 0.357 | 0.635 |   0.552   |    0.457   |  0.950  |  0.946 | 0.945 |  0.742  |
|   HBOS   |    0.815   |    0.653   |  0.998 | 0.613 | 1.000 | 0.686 |   0.750   |    0.983   |  0.959  |  0.686 | 0.953 |  0.827  |
|    ISF   |    0.808   |    0.846   |  0.995 | 0.807 | 0.997 | 0.664 |   0.730   |    0.992   |  0.979  |  0.752 | 0.919 |  0.862  |
|   LODA   |    0.757   |    0.801   |  0.667 | 0.724 | 0.893 | 0.623 |   0.611   |    0.986   |  0.828  |  0.713 | 0.926 |  0.775  |
|   LSCP   |    0.797   |    0.887   |  0.986 | 0.751 | 0.346 | 0.642 |   0.567   |    0.671   |  0.947  |  0.944 |  0.94 |  0.770  |
|    PCA   |    0.778   |    0.793   |  0.986 | 0.834 | 1.000 | 0.625 |   0.628   |    0.977   |  0.956  |  0.617 | 0.906 |  0.827  |
|    SOD   |    0.731   |    0.889   |  0.919 | 0.589 | 0.654 | 0.589 |   0.644   |    0.840   |  0.921  |  0.887 | 0.919 |  0.780  |
|    DAE   |    0.801   |    0.947   |  0.886 | 0.794 | 0.758 | 0.665 |   0.638   |    0.799   |  0.943  |  0.555 | 0.875 |  0.787  |
|   DAGMM  |    0.279   |    0.369   |  0.52  | 0.332 | 0.314 | 0.503 |   0.305   |    0.862   |  0.536  |  0.57  | 0.761 |  0.486  |
|   DSVDD  |    0.684   |    0.730   |  0.796 | 0.688 | 0.767 | 0.481 |   0.670   |    0.733   |  0.693  |  0.500 | 0.911 |  0.695  |
|  DIP-euc |    0.811   |    0.928   |  0.913 | 0.858 | 1.000 | 0.728 |   0.652   |    0.999   |  0.946  |  0.958 | 0.943 |  0.885  |
| EDIP-cos |    0.816   |    0.941   |  0.927 | 0.945 | 1.000 | 0.616 |   0.800   |    0.991   |  0.962  |  0.909 | 0.583 |  0.863  |
| EDIP-euc |    0.812   |    0.916   |  0.959 | 0.861 | 1.000 |  0.73 |   0.779   |    0.999   |  0.942  |  0.933 | 0.943 |  0.900  |
| EDIP-man |    0.829   |    0.875   |  0.979 | 0.824 | 1.000 | 0.738 |   0.789   |    0.998   |  0.958  |  0.921 | 0.948 |  0.900  |
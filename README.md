# DIP: Density-increasing Neighbor Path for Unsupervised Anomaly Detection
This repository contains the code for paper "Searching Density-increasing Path to Density Peaks for Unsupervised Anomaly Detection"
submitted to IEEE Transactions on Cybernetics.

## Quick start
Run the `main.py` in the `Evaulate` dictionary.

We offer a demo to evaluate single DIP on **musk** dataset and evaluate ensemble DIP on **arrhythmia** dataset.

## Experimental results

DIP achieved better average results than a lot of existing traditional and deep methods. The evaluation metrics are AUC-ROC,
AUC-PR, and F1 score. This repository only show the F1 score performance, while the other results can be found in the original paper.
The dataset information and other datasets used in the paper can be download from [here](http://odds.cs.stonybrook.edu/).


### F1 score performance
| F1 score | arrhythmia | ionosphere | lympho | mnist |  musk |  pima | satellite | satimage-2 | thyroid | vowels |  wbc  | Average |
|:--------:|:----------:|:----------:|:------:|:-----:|:-----:|:-----:|:---------:|:----------:|:-------:|:------:|:-----:|:-------:|
|   ABOD   |    0.400   |    0.842   |  0.565 | 0.307 | 0.024 | 0.517 |   0.374   |    0.172   |  0.050  |  0.558 | 0.354 |  0.378  |
|   CBLOF  |    0.476   |    0.755   |  0.768 | 0.328 |  1.00 | 0.471 |   0.553   |    0.951   |  0.154  |  0.253 | 0.548 |  0.568  |
|    FB    |    0.454   |    0.716   |  0.718 | 0.316 | 0.019 | 0.466 |   0.388   |    0.105   |  0.260  |  0.313 | 0.582 |  0.394  |
|   HBOS   |    0.487   |    0.446   |  0.947 | 0.146 | 1.000 | 0.533 |   0.571   |    0.745   |  0.479  |  0.133 | 0.590 |  0.552  |
|    ISF   |    0.480   |    0.641   |  0.813 | 0.323 | 0.854 | 0.494 |   0.586   |    0.859   |  0.578  |  0.194 | 0.524 |  0.576  |
|   LODA   |    0.438   |    0.623   |  0.108 | 0.261 | 0.244 |  0.44 |   0.492   |    0.854   |  0.187  |  0.184 | 0.516 |  0.395  |
|   LSCP   |    0.468   |    0.738   |  0.718 | 0.334 | 0.019 | 0.468 |   0.405   |    0.065   |  0.279  |  0.316 | 0.572 |  0.398  |
|    PCA   |    0.451   |    0.591   |  0.718 | 0.374 | 0.949 | 0.485 |   0.504   |    0.779   |  0.378  |  0.106 | 0.523 |  0.532  |
|    SOD   |    0.379   |    0.785   |  0.507 | 0.232 | 0.094 | 0.445 |   0.458   |    0.293   |  0.180  |  0.368 | 0.466 |  0.382  |
|    DAE   |    0.485   |    0.841   |  0.565 | 0.374 | 0.129 | 0.492 |   0.454   |    0.337   |  0.198  |  0.094 | 0.333 |  0.391  |
|   DAGMM  |    0.084   |    0.287   |  0.095 | 0.047 |  0.01 | 0.357 |   0.080   |    0.428   |  0.086  |  0.010 | 0.656 |  0.194  |
|   DSVDD  |    0.327   |    0.595   |  0.463 | 0.232 | 0.387 | 0.333 |   0.493   |    0.125   |  0.154  |  0.037 | 0.571 |  0.337  |
|  P2P-cos |    0.483   |    0.766   |  0.622 | 0.614 | 1.000 | 0.404 |   0.684   |    0.405   |  0.498  |  0.386 | 0.023 |  0.535  |
|  P2P-euc |    0.484   |    0.677   |  0.752 | 0.363 | 1.000 | 0.526 |   0.470   |    0.938   |  0.269  |  0.436 | 0.531 |  0.586  |
|  P2P-man |    0.497   |    0.610   |  0.768 | 0.284 | 1.000 | 0.529 |   0.555   |    0.957   |  0.363  |  0.388 | 0.582 |  0.593  |


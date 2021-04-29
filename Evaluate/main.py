# encdoing: utf-8
"""
@File:    main
@Author:  Jiachen Zhao
@Time:    2021/4/28 21:23
@Description: main function to run DIP model on real-world data sets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import compute_metrics
from tabulate import tabulate
from datasets.dataset import DATASET
from models.DIP import DIPModel


def run1(model, ds, seed):
    """
    Evaluate the DIPModel on a dataset, and output the performance metrics
    :param model: A DIPModel object
    :param ds: A DATASET object
    :param seed: random seed
    :return:
    """
    random_state = np.random.RandomState(seed)
    X_train, X_test, y_train, y_test = \
        train_test_split(ds.X, ds.y, test_size=ds.test_size, random_state=random_state)
    y_test_prob = model.fit_predict(X_train, X_test)
    # Compute the perfect y_test_pred, assuming the anomaly rate is known
    n_outlier = int(np.sum(y_test))
    y_test_pred = np.zeros(X_test.shape[0])
    y_test_pred[y_test_prob.argsort()[-n_outlier:][::-1]] = 1
    metrics = compute_metrics(y_true=y_test, y_pred=y_test_pred, y_prob=y_test_prob, verbose=True)
    return metrics


if __name__ == "__main__":
    # Evaluate Single DIP on the musk dataset
    print("Evaluate the single DIP on the musk dataset")
    ds = DATASET(name="musk", scalertype='MinMaxScaler', test_size=0.4)
    model = DIPModel(ModelName='DIP', n_neigh_list=[100], pathType='nearest',
                     distance="manhattan")
    run1(model, ds, seed=1)
    print('------'*11)
    # Evaluate Ensemble DIP on the arrhythmia dataset
    print("Evaluate Ensemble DIP on the arrhythmia dataset")
    ds = DATASET(name="arrhythmia", scalertype='MinMaxScaler', test_size=0.4)
    num_train = int(ds.numSample * 0.4)
    tt = np.linspace(0.2, 0.6, 9)
    n_neigh_lists = (num_train * tt).astype(int)    # Compute the n_neigh_lists
    model = DIPModel(ModelName='DIP', n_neigh_list=[100], pathType='nearest',
                     distance="manhattan")
    run1(model, ds, seed=1)


# encdoing: utf-8
"""
@File:    utils
@Author:  Jiachen Zhao
@Time:    2021/1/5 10:44
@Description: Helper functions
"""
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support as prf
import pandas as pd
import pickle5 as pickle
import collections


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def compute_metrics(y_true=None, y_pred=None, y_prob=None, decimal=4, verbose=False):
    """
    Compute the metrics for anomaly detection results
    :param verbose:
    :param decimal:
    :param y_true:
    :param y_pred:
    :param y_prob:
    :return:
    """
    metrics_dict = {}
    y_true = list(y_true.flatten())
    if y_pred is not None:
        y_pred = list(y_pred.flatten())
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        metrics_dict['accuracy'] = round(accuracy, decimal)
        metrics_dict['precision'] = round(precision, decimal)
        metrics_dict['recall'] = round(recall, decimal)
        metrics_dict['f_score'] = round(f_score, decimal)
    if y_prob is not None:
        y_prob = list(y_prob.flatten())
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        metrics_dict['auc_roc'] = round(auc_roc, decimal)
        metrics_dict['auc_pr'] = round(auc_pr, decimal)

    if verbose:
        print(pd.DataFrame(metrics_dict, index=['Results']))

    return metrics_dict


def ints2str(ints):
    '''
    Convert a list of integers into a string
    For example: [1, 2, 13] will turn into "1 2 13"
    '''
    string_ints = [str(int) for int in ints]
    str_of_ints = " ".join(string_ints)
    return str_of_ints


def sortDist(dict):
    """
    Sort a dict by its keys
    :param dict:
    :return:
    newdict: orederedDict object
    key_list: list, sorted keys
    value_list, list, values corresponding to sorted keys
    """
    newdict = collections.OrderedDict(sorted(dict.items()))
    key_list = list(newdict.keys())
    value_list = list(newdict.values())
    return newdict, key_list, value_list


if __name__ == "__main__":
    print('-----')

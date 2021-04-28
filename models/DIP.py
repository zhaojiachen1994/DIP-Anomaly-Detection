# encdoing: utf-8
"""
@File:    DIP
@Author:  Jiachen Zhao
@Time:    2021/4/28
@Description: Density-increasing Path model as described in ...
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class DIPModel:

    def __init__(self, ModelName='DIP', n_neigh_list=[5], pathType='nearest',
                 distance="euclidean"):
        """
        :param ModelName: str, default='DIP'. The name of the model.
        :param n_neigh_list: list, default=[5]
                             if len(n_neigh_list)==1, the neighborhood size to build knn graph
                             if len(n_neigh_list)>1,  a list of neighborhood sizes to use for multi-scale ensemble
        :param pathType: {'nearest', 'highest', 'average'}, default='nearest'.
                         The approach to define the density-increasing path
                         - 'nearest' will use the nearest neighbor that has higher density, original DIP
                         - 'highest' will use the neighbor that has highest density, variant 1 of DIP
                         - 'average' will consider all neighbors that has higher density
                            and average the step difficulty, variant 2 of DIP
        :param distance: {"euclidean", "manhattan", "cosine", "chebyshev"}, default="euclidean"
                         The distance metric to use for knn graph.
        """
        self.name = ModelName
        self.n_neigh_list = sorted(n_neigh_list)
        self.max_k = n_neigh_list[-1]
        self.pathType = pathType
        self.distance = distance
        self.neigh = NearestNeighbors(n_neighbors=self.max_k, metric=distance)

    def _fitBase(self, neigh_ind, neigh_dist):
        """
        Compute the path difficulty based on a knn graph G=[neigh_ind, neigh_dist]
        :param neigh_ind: np.array with shape [n_sample, n_neigh],
                          the index matrix of the knn graph learned by sklearn.NearestNeighbors
        :param neigh_dist: np.array with shape [n_sample, n_neigh],
                           the distance matrix of the knn graph learned by sklearn.NearestNeighbors
        :return:
                lrd: the local reachability density for each sample
                dist2peak: the path difficulty (anomaly score) for each sample
        """
        n_sample = neigh_ind.shape[0]
        dist_k = neigh_dist[neigh_ind, - 1]
        reach_dist_array = np.maximum(neigh_dist, dist_k)  # compute the local reachability distance
        lrd = 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)  # compute the local reachability density
        lrd_mat = lrd[neigh_ind]
        bigNeighborMusk = lrd_mat > lrd_mat[:, 0, np.newaxis] + 1e-10
        if self.pathType == 'nearest':
            bigNeighborInd = np.argmax(bigNeighborMusk, axis=1)
        elif self.pathType == 'average':
            bigNeighborInd = [np.argwhere(b_i).flatten() for b_i in bigNeighborMusk]
        elif self.pathType == 'highest':
            bigNeighborInd = np.argmax(lrd_mat, axis=1)
        peakflag = np.all(~bigNeighborMusk, axis=1)

        dist2peak = np.zeros(n_sample, dtype=float)  # Initialize the path difficulty as 0 for all data points
        sortedlrd_ind = (-lrd).argsort()

        for i in sortedlrd_ind:
            if peakflag[i]:
                dist2peak_i = 0
            else:
                bigNeighborInd_i = bigNeighborInd[i]
                end_i = neigh_ind[i, bigNeighborInd_i]  # next point in DIP
                length_i = neigh_dist[i, bigNeighborInd_i]  # distance of the current step
                height_i = lrd[end_i].flatten() / (
                            lrd[i] + 1e-10) - 1  # relative density difference of the current step
                step_i = np.maximum(length_i * height_i, 0)  # current step difficulty
                dist2peak_i = np.mean(step_i + dist2peak[end_i])  # path difficulty of the current point
            dist2peak[i] = dist2peak_i
        return lrd, dist2peak

    def fit(self, X):
        """
        Ensemble DIP with multiple neighbor sizes
        :param X: np.array with shape [n_sample, n_features]
        :return:
                neigh_dist: np.array with shape [n_sample, max_k],
                            the distance matrix of the knn graph with maximum k in n_neigh_list
                lrd_dict: dictionary, {k : corresponding lrd array learned by _fitBase}
                dist2peak_dict, dictionary, {k: corresponding path difficulty learned by _fitBase}
        """
        assert X.shape[0] >= self.max_k, 'neighborhood size is too large!'
        self.neigh.fit(X)
        neigh_dist, neigh_ind = self.neigh.kneighbors(X)

        lrd_dict = {}
        dist2peak_dict = {}
        for k in self.n_neigh_list:
            neigh_ind_k = neigh_ind[:, :k]
            neigh_dist_k = neigh_dist[:, :k]
            lrd_k, dist2peak_k = self._fitBase(neigh_ind_k, neigh_dist_k)
            lrd_dict[k] = lrd_k
            dist2peak_dict[k] = dist2peak_k

        return neigh_dist, lrd_dict, dist2peak_dict

    def _predictBase(self, neigh_ind_test, neigh_dist_test, neigh_dist_train, lrd_train, dist2peak_train):
        num_test = neigh_ind_test.shape[0]

        dist_k_test = neigh_dist_train[neigh_ind_test, -1]
        reach_dist_array_test = np.maximum(neigh_dist_test, dist_k_test)
        lrd_test = 1. / (np.mean(reach_dist_array_test, axis=1) + 1e-10)  # compute the lrd for test data
        lrd_mat_test = lrd_train[neigh_ind_test]

        bigNeighborMusk = lrd_mat_test > lrd_test[:, np.newaxis]
        peakflag = np.all(~bigNeighborMusk, axis=1)
        if self.pathType == 'nearest':
            bigNeighborInd = np.argmax(bigNeighborMusk, axis=1)
        elif self.pathType == 'average':
            bigNeighborInd = [np.argwhere(b_i).flatten() for b_i in bigNeighborMusk]
        elif self.pathType == 'highest':
            bigNeighborInd = np.argmax(lrd_mat_test, axis=1)

        dist2peak_test = np.zeros(num_test, float)

        for i in range(num_test):
            if peakflag[i]:
                dist2peak_test[i] = 0

            else:
                bigNeighborInd_i = bigNeighborInd[i]
                end_test_i = neigh_ind_test[i, bigNeighborInd_i]
                length_test_i = neigh_dist_test[i, bigNeighborInd_i]
                height_test_i = lrd_train[end_test_i] / (lrd_test[i]+1e-10)-1
                step_test_i = np.maximum(length_test_i * height_test_i, 0)
                dist2peak_test[i] = np.mean(dist2peak_train[end_test_i] + step_test_i)
        return dist2peak_test

    def predict(self, X_test, neigh_dist_train, lrd_dict_train, dist2peak_dict_train):
        """
        :param X_test: np.array, test data
        :param neigh_dist_train: output of fit()
        :param lrd_dict_train: output of fit()
        :param dist2peak_dict_train: output of fit()
        :return: dist2peak_test
        """
        # Find k neighbors for test data in the training set
        neigh_dist_test, neigh_ind_test = self.neigh.kneighbors(X_test, n_neighbors=self.max_k,
                                                                return_distance=True)
        # dist2peak_test_dict = {}
        dist2peak_test = np.zeros(X_test.shape[0], float)
        for k in self.n_neigh_list:
            neigh_dist_test_k = neigh_dist_test[:, :k]
            neigh_ind_test_k = neigh_ind_test[:, :k]
            neigh_dist_train_k = neigh_dist_train[:, :k]
            lrd_train_k = lrd_dict_train[k]
            dist2peak_train_k = dist2peak_dict_train[k]
            dist2peak_test_k = self._predictBase(neigh_ind_test_k, neigh_dist_test_k, neigh_dist_train_k,
                                                 lrd_train_k, dist2peak_train_k)
            dist2peak_test += dist2peak_test_k
            # dist2peak_test_dict[k] = dist2peak_test_k
        dist2peak_test /= len(self.n_neigh_list)
        return dist2peak_test

    def fit_predict(self, X_train, X_test):
        neigh_dist_train, lrd_dict_train, dist2peak_dict_train = self.fit(X_train)
        y_test_prob = self.predict(X_test, neigh_dist_train, lrd_dict_train, dist2peak_dict_train)
        return y_test_prob


if __name__ == "__main__":
    X = np.array(
        [[-1, -1], [-1, 0], [-1.1, 1], [0.1, 1], [0, -0.9], [1.1, 1.2], [1, 0.1], [1, -1.1], [3, 0], [1.5, 1.5]])
    X_test = np.array([[0, 0], [3, 2]])

    model = DIPModel(ModelName='DIP', n_neigh_list=[5], pathType='nearest',
                 distance="euclidean")
    dist2peak_test = model.fit_predict(X, X_test)
    print(dist2peak_test)

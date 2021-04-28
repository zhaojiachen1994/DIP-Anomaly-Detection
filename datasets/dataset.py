# encoding: utf-8
'''
@author: Jiachen Zhao
@time: 2020/2/1 20:29
@desc: The DATASET class to process the dataset
Datasets are downloaded from https://pyod.readthedocs.io/en/latest/benchmark.html
'''

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
import warnings
import os

warnings.filterwarnings("ignore")



class DATASET:
    def __init__(self, name, scalertype='StandarScaler', semirate=0.5, test_size=0.2):
        self.filepath = os.path.join(os.path.dirname(os.getcwd()), 'datasets')
        self.name = name
        self.X = None
        self.y = None
        self.dim = 0
        self.numSample = 0
        self.Download()
        self.rate = np.sum(self.y) / float(self.numSample)
        self.semirate = semirate
        self.test_size = test_size

        if scalertype == 'StandarScaler':
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
        elif scalertype == 'MinMaxScaler':
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)

    def Download(self):
        if self.name == 'arrhythmia':
            data = sio.loadmat(self.filepath + "/arrhythmia/arrhythmia.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'musk':
            data = sio.loadmat(self.filepath + "/musk/musk.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape


def check_datasets():
    names = sorted(["arrhythmia", "musk"])
    infdf = pd.DataFrame({'Name': [],
                          'num_samples': [],
                          'dim': [],
                          'rate': []
                          })
    for name in names:
        print(name)
        Dataset = DATASET(name=name, scalertype=None)
        infdf = infdf.append({'Name': Dataset.name,
                              'num_samples': int(Dataset.numSample),
                              'dim': int(Dataset.dim),
                              'rate': Dataset.rate
                              }, ignore_index=True)

    infTable = tabulate(infdf, headers='keys', tablefmt='psql', showindex="never")
    print(infTable)


if __name__ == "__main__":
    check_datasets()

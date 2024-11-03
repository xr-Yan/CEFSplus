#作者 : shangshilong
# coding:utf-8

import os
import numpy as np
import pandas as pd


class Relief:
    def __init__(self, data, labels, k=5):
        self.data = data
        self.labels = labels
        self.k = k
        self.weights = None

    def fit(self):
        n_samples, n_features = self.data.shape
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.sum(np.abs(self.data[i, :] + self.data[j, :]))
                distances[j, i] = distances[i, j]
        self.weights = np.zeros(n_features)
        for i in range(n_samples):
            nearest_pos = np.argsort(distances[i, :])[:self.k + 1]
            nearest_neg = np.argsort(-distances[i, :])[:self.k + 1]
            nearest_pos = nearest_pos[nearest_pos != i]
            nearest_neg = nearest_neg[nearest_neg != i]



data = 'all_t_pure_pro'
file_name = data + '.csv'


ge = pd.read_csv(file_name,low_memory=False)
print(ge)

#exit(666)
xy = ge.iloc[:,1:].values #抛去第一列
print(xy)
x = xy[:,:-1]#二维阵
y = xy[:,-1]#向量
print(x)
print(y)

relief_coef = pd.Series(relief.weights)
relief_coef.to_csv('relief_coef.csv')

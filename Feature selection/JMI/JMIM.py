#作者 : shangshilong
# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

def jmim(X, y, num_features):
    # 初始化已选择的特征列表和候选特征列表
    selected_features = []
    candidate_features = list(range(X.shape[1]))

    # 计算每个特征与标签之间的互信息
    mi = mutual_info_classif(X, y)

    # 选择第一个特征
    max_mi_idx = np.argmax(mi)
    selected_features.append(max_mi_idx)
    candidate_features.remove(max_mi_idx)

    # 选择剩余的特征
    for i in range(num_features - 1):
        # 计算已选择特征与候选特征之间的条件互信息
        cmi = np.zeros((len(candidate_features), len(selected_features)))
        for j, candidate in enumerate(candidate_features):
            for k, selected in enumerate(selected_features):
                cmi[j, k] = mutual_info_classif(X[:, [candidate]], X[:, selected])[0]

        # 计算每个候选特征与标签之间的 JMIM
        jmim_scores = mi[candidate_features]
        for j, candidate in enumerate(candidate_features):
            for k, selected in enumerate(selected_features):
                jmim_scores[j] -= np.max(cmi[j, :] - cmi[j, k])

        # 选择 JMIM 得分最高的特征
        max_jmim_idx = np.argmax(jmim_scores)
        selected_features.append(candidate_features[max_jmim_idx])
        candidate_features.remove(candidate_features[max_jmim_idx])

    return selected_features


df = pd.read_csv('all_t_pure_pro.csv',low_memory=False)
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

k = 20
# 使用JMI算法进行特征选择
X_new = jmim(X, y, k)
print(X_new)

df = pd.Series(X_new)
df.to_csv('X_new.csv')

"""

19, 21, 20, 22, 24, 1706, 1805, 925, 1963, 1215, 718, 268, 1164, 1081, 953, 301, 797, 979, 904, 2041, 1002, 1286, 750, 1554, 892, 1233, 1307, 1939, 1984, 1407, 796, 801, 241, 1439, 108, 609, 1922, 1282, 665, 1511, 1010, 414, 810, 1457, 1278, 1088, 1782, 1909, 1337, 1346, 1615, 2083, 627, 1623, 1132, 1844, 1484, 1285, 1359, 1709, 1678, 1180, 1185, 1856, 1438, 251, 1054, 1340, 217, 1523, 906, 776, 248, 1082, 789, 1734, 1110, 1783, 1989, 2025, 872, 2043, 1513, 2018, 2054, 1542, 713, 1618, 14, 131, 834, 1461, 965, 814, 1463, 939, 1784, 860, 1743, 138
"""
#作者 : shangshilong
# coding:utf-8

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn import metrics

def JMI(X, y, k):
    # 初始化选择的特征和条件互信息
    F = []
    MI = []

    # 计算每个特征与目标变量之间的互信息
    for i in range(X.shape[1]):
        MI.append(metrics.mutual_info_score(y, X[:,i]))

    # 选择互信息最高的特征
    F.append(np.argmax(MI))
    remaining_features = list(range(X.shape[1]))
    remaining_features.remove(F[0])

    # 选择 k-1 个特征
    for _ in range(k-1):
        # 初始化特征的条件互信息
        CMI = np.zeros(len(remaining_features))

        # 计算每个未选择的特征与已选择特征之间的条件互信息
        for i, f in enumerate(remaining_features):
            for j in range(len(F)):
                CMI[i] += metrics.mutual_info_score(X[:,f], X[:,F[j]])
            CMI[i] /= float(len(F))

        # 选择条件互信息最高的未选择特征
        best_idx = np.argmax(CMI)
        F.append(remaining_features[best_idx])
        remaining_features.remove(F[-1])

    return F



df = pd.read_csv('all_t_pure_pro.csv',low_memory=False)
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

k = 20
# 使用JMI算法进行特征选择
X_new = JMI(X, y, k)
print(X_new)

jmi_fs = pd.Series(X_new)
jmi_fs.to_csv('jmi_fs.csv')


""
1903, 111, 112, 129, 290, 113, 281, 407, 294, 415, 286, 298, 293, 408, 244, 296, 245, 246, 406, 372, 410, 409, 414, 285, 287, 291, 404, 405, 125, 292, 284, 365, 416, 366, 607, 361, 829, 334, 335, 339, 354, 940, 344, 413, 329, 343, 358, 351, 340, 403, 356, 337, 331, 773, 336, 341, 353, 364, 350, 357, 360, 599, 359, 855, 550, 555, 553, 554, 552, 288, 402, 551, 330, 597, 549, 295, 297, 280, 299, 279, 282, 300, 301, 1707, 316, 1675, 602, 309, 401, 124, 412, 123, 371, 380, 368, 322, 619, 547, 398, 543
""

"""
6551, 4067, 8265, 3866, 5758, 5020, 1032, 2476, 8187, 6316, 6598, 2307, 8384, 9277, 9910, 1768, 6600, 5618, 471, 3343, 3456, 5610, 328, 3994, 3199, 7648, 9290, 2770, 7154, 1878, 324, 9218, 3820, 7244, 2100, 6953, 4444, 8657, 5072, 2091, 4460, 546, 9499, 6578, 9490, 1185, 5459, 4742, 5153, 4787
"""
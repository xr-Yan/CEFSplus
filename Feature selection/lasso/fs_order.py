#作者 : shangshilong
#时间 : 2023/5/17 15:03
# coding:utf-8

import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata as rank

df = pd.read_csv('fs_array.csv')
print(df)

ddff = df.iloc[:,1:].values

l = []
for i in range(ddff.shape[1]):
    print(i)
    v = ddff[:,i]
    k = 0
    for item in v[::-1]:
        if item == 0:
            k = k + 1
        else:
            break


    l.append(k)

print(l)
fs_order_list = pd.Series(l)
fs_order_list.to_csv('fs_order_list_要升序.csv')
#print(rank(l))
"""
产生的表格中：第一列代表特征，第二列代表这个特征最后0的个数，0越多，说明这个特征越不重要，越早被压缩，所以要用升序

"""
#作者 : shangshilong
#时间 : 2023/7/27 18:35
# coding:utf-8

import numpy as np
import pandas as pd
import time
import copent
import copy
from scipy.stats import rankdata as rank
from numpy.random import normal as rnorm
from joblib import Parallel,delayed

"=================1.导入数据=================================="
data = pd.read_csv('all_t_pure_pro.csv',low_memory=False)
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values.reshape([-1,1])
Xy = data.iloc[:,1:].values
print(Xy)
print(Xy.shape)
"=================1.导入数据=================================="

"=================2.定义所需函数==============================="
#加随机噪音
def ccs(i,j,xy):
    m = xy.shape[0]
    z = xy[:, i] + max(abs(xy[:, i])) * 0.000001 * rnorm(0, 1, m)
    k = xy[:, j] + max(abs(xy[:, j])) * 0.000001 * rnorm(0, 1, m)
    #print(z,k)
    ccc = np.array([z, k])
    #print(ccc)
    t = copent.copent(ccc.T)
    #print(i,j,t)
    return t

#求平均值
def c20(i,j,xy):
    l = []
    for k in range(2):
        copula = ccs(i,j,xy)
        l.append(copula)

    return np.average(l)

#计算熵
def ccp(xy):
    l = xy.shape[1]
    m = xy.shape[0]
    xy_hat = np.zeros([m, l])

    for i in range(l):
        xy_hat[:, i] = xy[:, i] + max(abs(xy[:, i])) * 0.000001 * rnorm(0, 1, m)

    t = copent.copent(xy_hat)
    # print(i,j,t)
    return t

#求平均熵
def ccp20(xy):
    l = []
    for k in range(2):
        copula = ccp(xy)
        l.append(copula)

    return np.average(l)

#my_list = [2, 4, 1, 3, 4, 4, 5]
def n_argmax(list):
    max_value = max(list)

    max_indices = [index for index, item in enumerate(list) if item == max_value]

    return max_indices

#x = xy[:,:20]  #"###前20个先尝试"
#y = xy[:,-1]
def cc(i,xy):
    t = copent.copent(xy[:,[i,-1]])
    #print(i)
    return t

def comput_2(x,s_s,i): #x,ls,ln = x，已选特征的名字列表，单个未选特征的名字，返回l是未选特征的第二部分列表

    huhuhu_c = copent.copent(x[:, [i, s_s]])
    #print(huhuhu_c)
    return huhuhu_c


def compu_3(xy,ls,i):
    l = copy.deepcopy(ls)
    l.append(i)
    #print(l)
    c1 = ccp20(xy[:, l])
    l.append(-1)
    #print(l)
    c2 = ccp20(xy[:,l])

    return c1,c2 ##为负值，取大是正确的c2-c1

#主程序
def CEFS_plus(xy,k,n_jobs = 12): #xy--x&y的阵，最后一列是y（标签），k--要选的特征个数

    x = xy[:, :-1]  # "###前20个先尝试"
    y = xy[:, -1]

    length = x.shape[1]
    huhuhu = Parallel(n_jobs=n_jobs)(delayed(c20)(i,-1, xy) for i in range(length))
    #print(huhuhu)
    part1 = pd.Series(huhuhu)
    part1.to_csv('part1.csv')

    nphu = np.array(huhuhu)
    s_s = nphu.argmax()
    ls = [s_s]

    ln = [i for i in range(length)]
    del ln[ls[0]]
    nphu = np.delete(nphu,s_s)

    ks = len(ls)
    print('===================================================')
    print('全部特征个数:',xy.shape[1] - 1)
    print('要选特征个数:',k)
    print('已选特征个数:')
    print(ks)

    l_c1c2 = Parallel(n_jobs=n_jobs)(delayed(compu_part3)(xy,ls,i) for i in ln)
    c1c2 = np.array(l_c1c2)

    rank_c1 = rank(c1c2[:,0])
    rank_c2 = rank(c1c2[:,1])
    j_CEFS_plus = rank_c2 - rank_c1
    jm_index = n_argmax(j_CEFS_plus)


    ls.append(s_s)
    ks = ks + 1
    del ln[ns]
    nphu = np.delete(nphu,ns)

    while ks < k:
        print(ks)
        l_c1c2 = Parallel(n_jobs=n_jobs)(delayed(compu_part3)(xy, ls, i) for i in ln)

        c1c2 = np.array(l_c1c2)

        rank_c1 = rank(c1c2[:, 0])
        rank_c2 = rank(c1c2[:, 1])
        j_CEFS_plus = rank_c2 - rank_c1
        jm_index = n_argmax(j_CEFS_plus)

        ls.append(s_s)
        ks = ks + 1
        del ln[ns]
        nphu = np.delete(nphu, ns)

    print(k)
    print(ls)
    ls_select = pd.Series(ls)
    ls_select.to_csv('ls_select_5.csv')


    return ls

t = time.time()
CEFS_plus(Xy,2)
print('耗时:',(time.time() - t),'s')


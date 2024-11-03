#作者 : shangshilong
# coding:utf-8


import os
from msilib import knownbits

import numpy as np
import pandas as pd
import time
import copent
import torch
from numpy.f2py.capi_maps import lcb2_map
from scipy.stats import rankdata as rank




from joblib import Parallel,delayed

import copy




data = pd.read_csv('a_pro.csv',low_memory=False)

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values.reshape([-1,1])
Xy = data.iloc[:,1:].values
print(Xy)

"""l = [1,2,3,4,5]
l_a = np.array([1,2,3,4,5])
print(l_a)
print(np.argmax(l_a))
exit(666)"""






#my_list = [2, 4, 1, 3, 4, 4, 5]
def n_argmax(list):
    max_value = max(list)

    max_indices = [index for index, item in enumerate(list) if item == max_value]

    return max_indices

#print(n_argmax(my_list))
#print(len(n_argmax(my_list)))
#exit(666)









#x = xy[:,:20]  #"###前20个先尝试"
#y = xy[:,-1]

def cc(i,xy):
    t = copent.copent(xy[:,[i,-1]])
    #print(i)
    return t

###########################################################################

#huhuhu = Parallel(n_jobs=12)(delayed(cc)(i,xy) for i in range(20))
#print(huhuhu)
#sorted_huhu = sorted(huhuhu)
#print('sorted_huhu:\n',sorted_huhu)

#sorted_huhu_df = pd.Series(huhuhu)
#sorted_huhu_df.to_csv('GE_huhu_df.csv')




##huhuhu--list   to ndarray

'6 54 90 32  2 10 49 38 65  4 28'

###########################################################################
#GE_cop_list = Parallel(n_jobs=12)(delayed(mutual_info_regression)
# (x_ls,x[:, i],discrete_features=False,n_neighbors=3) for i in range(22544))
###########################################################################


def comput_2(x,s_s,i): #x,ls,ln = x，已选特征的名字列表，单个未选特征的名字，返回l是未选特征的第二部分列表

    huhuhu_c = copent.(x[:, [i, s_s]])
    print(huhuhu_c)
    return huhuhu_c

"""l = comput_2(x,ls,ln)
j_mrmr = huhuhu[ln] - l[ln]"""
#ls_0 = [nphu.argmax()]

#ln_0 = [i for i in range(20)]
#del ln_0[ls_0[0]]

#print(ls_0)
#print(ln_0)

#exit(0)

"""l = Parallel(n_jobs=12)(delayed(comput_part_2)(x,[0,1,2],i) for i in [3,4,5])
print('l:',l)
print('l * 2:',l * 2)"""

def compu_3(xy,ls,i):
    l = copy.deepcopy(ls)
    l.append(i)
    #print(l)
    c1 = copent.copent(xy[:, l])
    l.append(-1)
    #print(l)
    c2 = copent.copent(xy[:,l])

    return (c2 - c1) ##为负值，取大是正确的c2-c1




    ks = len(ls)
    print('===================================================')
    print('全部特征个数:',xy.shape[1] - 1)
    print('要选特征个数:',k)
    print('已选特征个数:')
    print(ks)
    """while ks < k:
        print(ks)"""
    #l = Parallel(n_jobs=n_jobs)(delayed(comput_part_2)(x,s_s,i) for i in ln)
        #print('*****',hu[ln])
        #print('*****', l)
    #print('前:',ls)
    l_3 = Parallel(n_jobs=n_jobs)(delayed(compu_part3)(xy,ls,i) for i in ln)
    #print('后:',ls)
    #rank_nphu = rank(nphu)
    #rank_l = rank(l)
    #j_mrmr = rank_nphu - rank_l
    #j_mrmr = nphu - l + l_3
        #print('haha:',j_mrmr)
    jm_index = n_argmax(l_3)



    if len(jm_index) == 1:
        s_s = ln[jm_index[0]]
        ns = jm_index[0]
    else:
        #t = ln[jm_index]
        j = np.argmax(nphu[jm_index])
        s_s = ln[jm_index[j]]
        ns = jm_index[j]

    ls.append(s_s)
    #print(ls)
    ks = ks + 1
    del ln[ns]
    #del l[ns]
    #print(s_s)
    #print(nphu)
    #print(len(nphu))
    nphu = np.delete(nphu,ns)
    #print(nphu)
    #print(len(nphu))
    #exit(666)

    #print('###:',nphu,ls,ln)
    #l_sum = np.array(l)


    while ks < k:
        print(ks)
        #print(l)

        #print(len(l1))
        #exit(0)
        #l = l1 + Parallel(n_jobs=12)(delayed(comput_part_2)(x, [ln[jm_index]], i) for i in ln)
        #print(len(Parallel(n_jobs=12)(delayed(comput_part_2)(x, [ln[jm_index]], i) for i in ln)))
        #print(np.array(l1)+np.array(Parallel(n_jobs=12)(delayed(comput_part_2)(x, [ln[jm_index]], i) for i in ln)))
        #print(len(np.array(l1)+np.array(Parallel(n_jobs=12)(delayed(comput_part_2)(x, [ln[jm_index]], i) for i in ln))))
        #l_sum = l_sum + Parallel(n_jobs=n_jobs)(delayed(comput_part_2)(x, s_s, i) for i in ln)
        #l_ave = l_sum / ks

        l_3 = Parallel(n_jobs=n_jobs)(delayed(compu_part3)(xy, ls, i) for i in ln)

        #rank_nphu = rank(nphu)
        #rank_l_ave = rank(l_ave)
        #j_mrmr = rank_nphu - rank_l_ave
        # j_mrmr = nphu - l + l_3
        # print('haha:',j_mrmr)
        jm_index = n_argmax(l_3)

        if len(jm_index) == 1:
            s_s = ln[jm_index[0]]
            ns = jm_index[0]
        else:
            # t = ln[jm_index]
            j = np.argmax(nphu[jm_index])
            s_s = lcb2_map[jm_index[j]]
            ns = knownbits[j]



        """j_mrmr = nphu - l_ave
        #j_mrmr = nphu - l_ave + l_3
        # print('haha:',j_mrmr)
        jm_index = j_mrmr.argmax()"""
        #s_s = ln[jm_index]
        #print(s_s)
        ls.append(s_s)
        #print(ls)
        ks = ks + 1
        del ln[ns]
        nphu = np.delete(nphu, ns)
        #l_sum = np.delete(l_sum, ns)

    #part2 = pd.Series(l_sum)
    #part2.to_csv('part2_l_sum.csv')
    print(k)
    print(ls)
    ls_select = pd.Series(ls)
    ls_select.to_csv('ls_select_ge_wu.csv')

    ln_select = pd.Series(ln,dtype=int)
    ln_select.to_csv('ln_select_ge.csv')

    return ls


"""print(huhuhu[[0,1,2]])
print(huhuhu[[0,1,2]] - l)"""
"[28, 2, 78, 4, 14, 64, 11, 10, 83, 65, 18, 32, 90, 73, 74, 6, 44, 7, 54, 38]"
"[28, 2, 14, 47, 74, 46, 81, 23, 37, 94, 96, 64, 17, 26, 89, 33, 59, 43, 78, 69]"
"[28, 2, 47, 78, 64, 74, 81, 23, 37, 92, 43, 14, 59, 46, 94, 35, 93, 25, 17, 77]"
t = time.time()
mrmr(Xy,100)
print('耗时:',(time.time() - t),'s')

"[71, 49, 18, 0, 93, 30, 68, 90, 65, 45]--10个"
"耗时: 69.3043565750122 s"
"耗时: 49.13134241104126 s"
"耗时: 49.809661626815796 s"
"1651.0810146331787"
"耗时: 497.2619721889496 s--要选特征个数: 4"
"[4, 3, 26, 13, 2, 17, 6, 23, 8, 31, 22, 25, 33, 30, 9, 11, 28, 5, 18, 21, 15, 32, 24, 7, 19, 12, 27, 20, 10, 14, 16, 29, 0, 1]"
"[4, 7, 28, 13, 8, 23, 2, 6, 26, 5, 9, 15, 22, 25, 3, 31, 27, 30, 19, 10, 20, 29, 12, 11, 33, 17, 0, 14, 21, 18, 32, 24, 16, 1]"
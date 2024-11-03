#作者 : shangshilong
#时间 : 2023/2/19 15:50
# coding:utf-8

import os
import pandas as pd
import numpy as np
import asgl
from numba import jit
from joblib import Parallel,delayed
import time

ge = pd.read_csv('mrna_z1.csv')
print(ge)
xy = ge.iloc[:,1:1905].values.T #(1904, 24369),最后一列是标签
print(xy)
x = xy[:,:-1]#二维阵
y = xy[:,-1]#向量
print(x,y)


Lambdas = np.logspace(-2, -1, 5)
print(Lambdas)

print(np.where(Lambdas==0.01))
#exit(0)


"""tvt_lasso = asgl.TVT(model='lm', penalization='lasso', lambda1=i, parallel=False,
                    error_type='MSE', random_state=42, train_size=12184, validate_size=6092) #24368 *0.5 = 12184"""


tvt_lasso = asgl.TVT(model='lm', penalization='lasso', lambda1=Lambdas, parallel=False,
                     error_type='MSE', random_state=42, train_size=12184, validate_size=6092)
lasso_result = tvt_lasso.train_validate_test(x=x, y=y)

lasso_prediction_error = lasso_result['test_error']
lasso_betas = lasso_result['optimal_betas'][1:]  # Remove intercept

st = time.time()

print('1:',time.time() - st)

print(lasso_prediction_error)
print(lasso_betas)
print('total feature:',len(lasso_betas))
print('selected feature:',np.sum(lasso_betas!=0))
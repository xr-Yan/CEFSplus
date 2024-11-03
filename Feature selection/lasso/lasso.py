#作者 : shangshilong
#时间 : 2023/2/18 11:27
# coding:utf-8

import os
import pandas as pd
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV


ge = pd.read_csv('breast亚型_ok_pure_归一化.csv',low_memory=False)
print(ge)

#exit(666)
xy = ge.iloc[:,1:].values #抛去第一列
print(xy)



x = xy[:,:-1]#二维阵
y = xy[:,-1]#向量
print(x)
print(y)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=42,train_size=0.8)


def optimal_lambda_value(X_train,y_train):
    Lambdas = np.logspace(-5, 1, 10)  # 10的-5到10的2次方
    # 构造空列表，用于存储模型的偏回归系数
    lasso_cofficients = []
    for Lambda in Lambdas:
        lasso = Lasso(alpha=Lambda,max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_cofficients.append(lasso.coef_)
    # 绘制Lambda与回归系数的关系
    plt.plot(Lambdas, lasso_cofficients)
    # 对x轴作对数变换
    plt.xscale('log')
    # 设置折线图x轴和y轴标签
    plt.xlabel('Lambda')
    plt.ylabel('Cofficients')
    # 显示图形
    plt.show()
    # LASSO回归模型的交叉验证
    lasso_cv = LassoCV(alphas=Lambdas,cv=5, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    # 输出最佳的lambda值
    lasso_best_alpha = lasso_cv.alpha_
    print(lasso_best_alpha)
    return lasso_best_alpha




#t = optimal_lambda_value(x,y)

#exit(666)


lasso = Lasso(alpha=0.00425,max_iter=3000).fit(x,y)

print('**********************************')
#print("Lasso alpha={:.9}".format(t))
print ("Lasso_xy set score:{:.2f}".format(lasso.score(x,y)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))
print(lasso.coef_)

lasso_coef_list = pd.Series(lasso.coef_)
lasso_coef_list.to_csv('lasso_coef_list.csv')


exit(666)



lasso = Lasso(alpha=0.023299518105153717).fit(X_train,y_train)

print('**********************************')
print("Lasso alpha=0.023299518105153717")
print ("training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))
CNA_lasso = pd.Series(lasso.coef_)
CNA_lasso.to_csv('CNA_lasso.csv')


"""lasso = Lasso(alpha=0.029150530628251757).fit(X_train,y_train)

print('**********************************')
print("Lasso alpha=0.029150530628251757")
print ("training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))


lasso = Lasso(alpha=0.03274549162877728).fit(X_train,y_train)

print('**********************************')
print("Lasso alpha=0.03274549162877728")
print ("training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))
GE_lasso = pd.Series(lasso.coef_)
GE_lasso.to_csv('GE_lasso.csv')"""


Lambdas = np.logspace(-2, -1, 50)
print(Lambdas)


"""参考https://blog.csdn.net/sinat_41858359/article/details/124765520?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168514665916800211576006%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168514665916800211576006&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-124765520-null-null.142^v88^control_2,239^v2^insert_chatgpt&utm_term=lasso%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9python%E4%BB%A3%E7%A0%81&spm=1018.2226.3001.4187"""


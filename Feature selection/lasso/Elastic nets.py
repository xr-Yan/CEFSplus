#作者 : shangshilong
#时间 : 2023/5/16 11:13
# coding:utf-8


import pandas as pd
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import ElasticNet,ElasticNetCV


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



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=42,train_size=0.8)


def optimal_lambda_value(X_train,y_train):
    l1_ratios = np.logspace(-1, 0, 5)# 10的-5到10的2次方,根据结果，细分到10的-2到10的-1次方之间
    alphas_l = np.logspace(-2, 2, 5)
    """# 构造空列表，用于存储模型的偏回归系数
    EN_cofficients = []
    fs_array = []
    fs_total = X_train.shape[1]
    k = fs_total

    j = 0
    for l1_ratio in l1_ratios:
        j = j + 1
        EN = ElasticNet(alpha=2,l1_ratio=l1_ratio,max_iter=10000)
        EN.fit(X_train, y_train)
        EN_cofficients.append(EN.coef_)
        fs_n = np.sum(EN.coef_ != 0)
        if fs_n == k:
            fs_array.append(EN.coef_)
            if k == 0:
                print('All Found!!!')
                break
            k = k - 1
        elif fs_n < k:
            print("ERROR_Not_Found:{}".format(k))
            print(fs_n)
            exit(888)
        print("Number of features used:{}".format(fs_n))

    df = pd.DataFrame(fs_array)
    df.to_csv('fs_array.csv')

    # 绘制Lambda与回归系数的关系
    plt.plot(l1_ratios[:j], EN_cofficients)
    # 对x轴作对数变换
    plt.xscale('log')
    # 设置折线图x轴和y轴标签
    plt.xlabel('l1_ratios')
    plt.ylabel('Cofficients')
    # plt.legend()
    # 显示图形
    plt.show()"""

    en_cv = ElasticNetCV(alphas=alphas_l,l1_ratio=[0.1,0.3,0.5,0.7,0.9],max_iter=2000,cv=5)
    en_cv.fit(X_train, y_train)
    # 输出最佳的lambda值
    en_best_alpha = en_cv.alpha_
    en_best_l1_ratio = en_cv.l1_ratio_
    print(en_best_alpha,en_best_l1_ratio)
    return en_best_alpha,en_best_l1_ratio




#optimal_lambda_value(x,y)

#exit(666)

"0.1 0.1--best"

EN = ElasticNet(alpha=0.1,l1_ratio=0.537,max_iter=3000).fit(x,y)

print('**********************************')
print("EN : alpha=0.00029763514416313193,l1_ratio=0.5748030158125351")
#print ("training set score:{:.2f}".format(EN.score(X_train,y_train)))
print ("set score:{:.2f}".format(EN.score(x,y)))
print ("Number of features used:{}".format(np.sum(EN.coef_!=0)))


en_coef_list = pd.Series(EN.coef_)
en_coef_list.to_csv('en_coef_list.csv')

exit(666)

EN = ElasticNet(alpha=0.023299518105153717).fit(X_train,y_train)

print('**********************************')
print("EN alpha=0.023299518105153717")
print ("training set score:{:.2f}".format(EN.score(X_train,y_train)))
print ("test set score:{:.2f}".format(EN.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(EN.coef_!=0)))
CNA_lasso = pd.Series(EN.coef_)
CNA_lasso.to_csv(data + '_EN.csv')


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

'0.00029763514416313193 0.5748030158125351'


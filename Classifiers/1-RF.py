#作者 : shangshilong
#时间 : 2023/5/7 17:49
# coding:utf-8

import os

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
import warnings
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from joblib import Parallel,delayed


warnings.filterwarnings("ignore")

df = pd.read_csv('breast亚型_ok_pure_归一化.csv',low_memory=False)
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
#print(X)
#print(y.ravel())






"---------------------------------------------------------------------------"
def accuracy(out, yb):
    #preds = (out>0.5)

    return (out == yb).mean()


def to_list(a):

    l = []
    for i in range(a.shape[0]):
        l.append(a[i,0])

    return l







def tt_svm(X,y):

    l_acc1 = []
    l_acc2 = []

    for k in range(10):

        #print("第{:1d}次10折交叉验证:***********************************************************\n".format(k+1))


        i = 0
        train_hat = np.array([]).reshape(-1,1)
        test_hat = np.array([]).reshape(-1,1)
        train_y = np.array([]).reshape(-1,1)
        test_y = np.array([]).reshape(-1,1)


        cv = KFold(n_splits = 10,shuffle = True)

        for train_index, test_index in cv.split(X):

            i = i + 1
            #print("第{:1d}折交叉验证:***********************************************************\n".format(i))
            #print(train_index)
            #print(test_index)
            #print(len(test_index))
            #print(max(train_index))
            #print(max(test_index))
            #X_train, X_test, y_train, y_test = X.iloc[train_index,:].values, X.iloc[test_index,:].values, y.iloc[train_index,:].values, y.iloc[test_index,:].values
            #X_train = X.iloc[train_index,:].values #没用
            X_test = X.iloc[test_index,:].values
            #y_train = y.values[train_index].reshape(-1, 1) #没用
            y_test = y.values[test_index].reshape(-1, 1)

            X_train = X.iloc[train_index,:].values
            y_train = y.values[train_index].reshape(-1, 1)
            #print(X_test,y_train,y_test)

            #这里不在划分成训练集和验证集

            #print(X_train.shape,y_train,X_val,y_val)



            # 创建一个RF分类器并进行预测
            clf = RandomForestClassifier()  # 创建RF训练模型
            clf.fit(X_train, y_train)  # 对训练集数据进行训练
            clf_y_predict1 = clf.predict(X_train)
            #print(clf_y_predict1)
            clf_y_predict1 = clf_y_predict1.reshape(-1, 1)
            #acc1 = accuracy(clf_y_predict1, y_train)
            #print(acc1)
            train_hat = np.vstack((train_hat,clf_y_predict1))
            train_y = np.vstack((train_y, y_train))



            clf_y_predict2 = clf.predict(X_test)
            clf_y_predict2 = clf_y_predict2.reshape(-1, 1)
            test_hat = np.vstack((test_hat,clf_y_predict2))
            test_y = np.vstack((test_y, y_test))
            #acc2 = accuracy(clf_y_predict2, y_test)


            #scores1 = clf.score(X_train, y_train)# 通过测试数据，得到测试标签
            #scores2 = clf.score(X_test, y_test)  # 测试结果打分

            #print(acc1,acc2,scores1,scores2)









            #print("第{:1d}折训练完成:***********************************************************\n".format(i))

        acc1 = accuracy(train_hat,train_y)
        acc2 = accuracy(test_hat,test_y)
        l_acc1.append(acc1)
        l_acc2.append(acc2)

    total_acc1 = np.mean(l_acc1)
    total_acc2 = np.mean(l_acc2)

    template = ("训练准确率:{:.8f},测试准确率:{:.8f}\n")
    print(template.format(total_acc1,total_acc2))
    return total_acc1,total_acc2


fs_df = pd.read_excel('mRMR.xlsx')
#print(fs_list.values)

fs_list = to_list(fs_df.values)
#print(ff)

def ssssvm(k,f):
    acc_a = []  # acc_a -- train_acc
    acc_b = []  # acc_b -- test_acc

    for i in range(f):
        t = i + 1
        print("选择{:1d}个特征:***********************************************************\n".format(t))

        fs = fs_list[:t]
        X_fs = X.iloc[:, fs]
        a, b = tt_svm(X_fs, y)
        acc_a.append(a)
        acc_b.append(b)

    m = max(acc_b)

    if m > 0.6:
        train_acc_Series = pd.Series(acc_a)
        train_acc_Series.to_csv('train_acc_sonar_SVM.csv')

        test_acc_Series = pd.Series(acc_b)
        test_acc_Series.to_csv('test_acc_sonar_SVM.csv')

        # 准确率
        plt.plot(range(1, len(acc_a) + 1), acc_a, label='train_acc')
        plt.plot(range(1, len(acc_b) + 1), acc_b, label='test_acc')
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Accuracy')
        plt.title('The Curve of Accuracy Changing with the Number of Selected Features')
        plt.legend()
        s = "my_plot_RF_" + str(k) + '_' + str(m) + '.png'
        plt.savefig(s, dpi=300)

    return k, m, acc_a, acc_b

    """if m > 0.81:

        train_acc_Series = pd.Series(acc_a)
        train_acc_Series.to_csv('train_acc_sonar_SVM.csv')

        test_acc_Series = pd.Series(acc_b)
        test_acc_Series.to_csv('test_acc_sonar_SVM.csv')

        # 准确率
        plt.plot(range(1,len(acc_a)+1), acc_a, label='train_acc')
        plt.plot(range(1,len(acc_b)+1), acc_b, label='test_acc')
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Accuracy')
        plt.title('The Curve of Accuracy Changing with the Number of Selected Features')
        plt.legend()
        s = "my_plot_" + str(k) + '_' + str(m) + '.png'
        plt.savef(ig(s,dpi=300)

        os.system"shutdown -s -t 60")"""

    # exit(85)


n_jobs = 1
u = Parallel(n_jobs=n_jobs)(delayed(ssssvm)(k,f=200) for k in range(1))
print(u)
uuu = pd.DataFrame(u)
uuu.to_csv('cz2_11_RF_df.csv')

#os.system("shutdown -s -t 60")











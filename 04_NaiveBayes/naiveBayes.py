#coding=utf-8
#Author:wangzy
#Date:2020-04-17
#Email:wangzycloud@163.com

import time
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

def loadDATA():
    iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    input_vecs = iris[:, [0, 1, 2, 3]].astype('float')
    labels = iris[:, [4]]
    for i in range(labels.shape[0]):
        if labels[i][0] == 'Iris-setosa':
            labels[i][0] = 1
        elif labels[i][0] == 'Iris-versicolor':
            labels[i][0] = 2
        elif labels[i][0] == 'Iris-virginica':
            labels[i][0] = 3
    return input_vecs, labels
def randomDATA(x_domain, y_domain, rate=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test
def zipDATA(x_train, y_train):
    return list(zip(x_train, y_train))
def featureTransfer(dataset):
    # 该函数实现输入空间特征向特征空间转换，这里只是进行了上取整，减少小数部分产生的特征。
    # 输入空间为数值型特征，特征数目太多，转换到特征空间共4*8=32个特征，共需求3*4*8=96个参数。
    # 特征空间有四个维度，每个维度可能的取值区间为[1,8]
    return np.ceil(dataset).astype(int)

class NaiveBayes(object):
    def __init__(self, dim_features, labels):
        # dim_features是一个二维数组，存放每个维度可能的取值。
        self.dim_features = dim_features
        self.labels = labels
        self.param_y = None
        self.param_yx = None

    def init_param(self):
        param_y = np.zeros((self.labels.shape[0],1))
        param_yx = []
        for i in range(len(self.dim_features)):
            temp = np.zeros((self.dim_features[i].shape[0],self.labels.shape[0]))
            param_yx.append(temp)
        return param_y,param_yx
    def count_y(self,y,trainDATA):
        count_y = 0
        for i in range(len(trainDATA)):
            if trainDATA[i][1][0] == y:
                count_y += 1
        return count_y
    def count_yx(self,dim,x,y,trainDATA):
        count_yx = 0
        for i in range(len(trainDATA)):
            if trainDATA[i][0][dim] == x and trainDATA[i][1][0] == y:
                count_yx +=1
        return count_yx

    def train(self,trainDATA):
        self.param_y,self.param_yx = self.init_param()

        count_yk = []
        for i in range(self.param_y.shape[0]):
            count_y = self.count_y(self.labels[i],trainDATA)
            Pyi = (count_y+1)/(len(trainDATA)+self.labels[0])
            self.param_y[i] = Pyi
            count_yk.append(count_y)

        for i in range(len(self.param_yx)):
            for j in range(self.dim_features[i].shape[0]):
                for k in range(self.labels.shape[0]):
                    count_yx = self.count_yx(i,self.dim_features[i][j],self.labels[k],trainDATA)
                    Pyxi = (count_yx + 1)/(count_yk[k]+len(count_yk))
                    self.param_yx[i][j][k] = Pyxi
    def predict(self, x):
        Pyi_x = []
        for i in range(x.shape[0]):
            index = np.where(self.dim_features[i] == x[i])
            index = index[0][0]
            Pyi_x.append(self.param_yx[i][index])
        Pyi_x = np.array(Pyi_x).T
        Pyi = []
        for i,item in enumerate(Pyi_x):
            mul = reduce(lambda a,b:a*b,item)
            Pyi.append(float(mul*self.param_y[i]))
        index = Pyi.index(max(Pyi))
        return self.labels[index]

if __name__ == "__main__":
    x, y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)
    x_train = featureTransfer(x_train)
    x_test = featureTransfer(x_test)
    trainDATA = zipDATA(x_train, y_train)
    testDATA = zipDATA(x_test, y_test)

    f1 = np.array([4,5,6,7,8])
    f2 = np.array([2,3,4,5])
    f3 = np.array([1,2,3,4,5,6,7])
    f4 = np.array([1,2,3])
    y = np.array([1,2,3])

    t1 = time.time()
    NB = NaiveBayes([f1,f2,f3,f4],y)
    NB.train(trainDATA)
    t2 = time.time()

    correct_num = 0
    for i,item in enumerate(x_test):
        label = NB.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))
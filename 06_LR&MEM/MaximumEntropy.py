#coding=utf-8
#Author:wangzy
#Date:2020-04-24
#Email:wangzycloud@163.com

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

def loadDATA():
    '''
    加载数据
    '''
    iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    #iris = iris[0:100]
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
    '''
    划分训练集、测试集
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test
def rebuild_features(features):
    new_features = []
    for fea in features:
        temp = []
        for idx,item in enumerate(fea):
            temp.append(str(idx)+'_'+str(item))
        new_features.append(temp)
    return new_features

class MaximumEntropy(object):
    def __init__(self,xTrainDATA,yTrainDATA):
        self.xDATA = xTrainDATA
        self.yDATA = yTrainDATA
        # 定义类别集合(训练集共有多少个类别)
        self.labels = set()

        self.N = len(xTrainDATA)
        self.M = 10000

    def get_EPxy(self):
        # 计算特征函数f(x,y)关于经验分布P(X,Y)的期望
        # 计算经验分布P(X,Y)、P(X)，实际上只是得到“可能会出现的情况”的频数
        self.cal_Pxy_Px(self.xDATA, self.yDATA)
        # 得到特征函数的个数
        self.n = len(self.Pxy)
        self.build_dict()
        # 定义一个“期望”字典，存放每个特征函数关于经验分布P(X,Y)期望
        self.EPxy = defaultdict(float)
        self.cal_EPxy()
    def build_dict(self):
        # 将(分量，类别)表示的各个特征函数按序号进行标记，便于后面的计算
        self.id2xy = {}
        self.xy2id = {}

        for i,(x,y) in enumerate(self.Pxy):
            self.id2xy[i] = (x,y)
            self.xy2id[(x,y)] = i
    def cal_Pxy_Px(self,xTrainDATA,yTrainDATA):
        # 定义两个字典，存放经验分布P(X,Y)、P(X)可能会出现的情况的频数
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)
        for i in range(len(xTrainDATA)):
            x_,y = xTrainDATA[i],yTrainDATA[i]
            # 记录训练集数据有多少个类别（集合元素不重复）
            self.labels.add(y)
            # 统计x,y共同出现的情况。将x各个维度拆开来表示特征函数：(分量，类别)
            # 例如属于类别y1的样本(x1,x2,x3,x4)，拆分为(x1,y1),(x2,y1),(x3,y1),(x4,y1)
            for x in x_:
                # 统计每种情况出现的频数
                self.Pxy[(x,y)] += 1
                self.Px[x] +=1
    def cal_EPxy(self):
        # 计算特征函数f(x,y)关于经验分布P(X,Y)的期望
        # 该函数并没有得出最终的期望，而是通过频数得到各个特征的概率
        for id in range(self.n):
            (x,y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x,y)])/ float(self.N)

    def get_EPx(self):
        # 计算特征函数f(x,y)关于模型P(Y|X)与经验分布P(X)的期望
        self.EPx = [0]*self.n
        for i,X in enumerate(self.xDATA):
            # 对训练集中的每个样本，累加得到符合各个特征函数的相应P(X)的概率
            Pyxs = self.cal_pyxs(X)
            # P(X) = ∑p(x1,yi) + ∑p(x2,yi) + ∑p(x3,yi) + ∑p(x4,yi)
            # ∑对各个yi求和，也就是求x的边缘分布，同时各个分量累加得到P(X)的概率
            for x in X:
                # ∑p(x,yi)
                for Pyx,y in Pyxs:
                    # 符合第id个特征函数的"特征(x,y)"，将相应的P(X)概率，累加到EPx[id]
                    # 这句话隐含了一个事实，不同的P(Y|X)，对应不同的P(X)
                    if self.fxy(x,y):
                        id = self.xy2id[(x,y)]
                        self.EPx[id] += Pyx*(1.0/self.N)
    def fxy(self, x, y):
        # 特征函数的表示。如果x,y满足已知事实(联合分布中可能会出现的情况)，取值为True
        return (x, y) in self.xy2id
    def cal_pyix(self,X,y):
        # 计算给定样本X下，某个类别yi的P(yi|X)的条件概率。X为样本，需要拆分成分量(x1,x2,x3,x4)，该函数主要是针对各个分量
        # 逐个得到各个(yi|x1),(yi|x2),(yi|x3),(yi|x4)的条件概率，加和表示P(yi|X)的条件概率
        res = 0.0
        for x in X:
            if self.fxy(x,y):
                # 如果该特征满足已知事实，累加得到条件概率P(yi|X)
                # 这里可以体现出build_dict(self)函数的好处
                id = self.xy2id[(x,y)]
                res += self.w[id]
        return (np.exp(res),y)
    def cal_pyxs(self,X):
        # 该函数得到P(y|X)，yi所有取值的条件概率
        # 公式(6.22)
        # Pyxs表示在不同y下的P(y|X)的条件概率
        Pyxs = [(self.cal_pyix(X, y)) for y in self.labels]
        Z = sum([prob for prob, y in Pyxs])
        # 将不同y下的P(X,y)的联合概率进行归一化
        return [(prob/Z,y) for prob,y in Pyxs]

    def trainbyGIS(self,epsilon=0.001):
        # 计算特征函数f(x,y)关于经验分布P(X,Y)的期望
        self.get_EPxy()
        # 步骤(1)初始化参数wi
        self.w = [0] * self.n
        # 步骤(2) 对参数w进行迭代更新
        max_iter = 100
        for iter in range(max_iter):
            old_w = self.w
            self.get_EPx()
            for i in range(self.n):
                # 参数更新
                lambdaDelte = 1/self.M * np.log(self.EPxy[i]/self.EPx[i])
                self.w[i] += lambdaDelte
            # 步骤(3) 检查是否达到收敛条件
            new_w = self.w
            flag = True
            for i in range(len(new_w)):
                if abs(new_w[i]-old_w[i]) > epsilon:
                    flag = False
                    break
            if flag:
                break
    def trainbyIIS(self,epsilon=0.001):
        # 计算特征函数f(x,y)关于经验分布P(X,Y)的期望
        self.get_EPxy()
        # 算法6.1步骤(1)初始化参数wi
        self.w = [0]*self.n
        # 算法6.1步骤(2)对参数w进行迭代更新
        max_iter = 100
        for iter in range(max_iter):

            old_w = self.w
            sigmas = []
            self.get_EPx()
            # 采用算法6.1改进的迭代尺度算法IIS
            for i in range(self.n):
                # 算法6.1 公式(6.34)
                sigmai = 1/self.M * np.log(self.EPxy[i]/self.EPx[i])
                sigmas.append(sigmai)
            # 参数更新
            self.w = [self.w[i] + sigmas[i] for i in range(self.n)]
            # 算法6.1步骤(3)检查是否达到收敛条件
            new_w = self.w
            flag = True
            for i in range(len(new_w)):
                if abs(new_w[i]-old_w[i]) > epsilon:
                    flag = False
                    break
            if flag:
                break

    def predict(self,x):
        t = self.cal_pyxs(x)
        lab = max(t,key=lambda a:a[0])[1]
        return lab

if __name__ == '__main__':
    x, y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)
    x_train = rebuild_features(x_train)
    x_test = rebuild_features(x_test)
    y_train = y_train.reshape(105)
    y_test = y_test.reshape(45)

    t1 = time.time()
    MET = MaximumEntropy(x_train,y_train)
    MET.trainbyGIS()

    t2 = time.time()
    correct_num = 0
    for i, item in enumerate(x_test):
        label = MET.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))


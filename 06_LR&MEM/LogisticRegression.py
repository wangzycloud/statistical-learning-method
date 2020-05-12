#coding=utf-8
#Author:wangzy
#Date:2020-04-22
#Email:wangzycloud@163.com

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def loadDATA():
    '''
    加载数据
    '''
    iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    iris = iris[0:100]
    input_vecs = iris[:, [0, 1, 2, 3]].astype('float')
    labels = iris[:, [4]]
    for i in range(labels.shape[0]):
        if labels[i][0] == 'Iris-setosa':
            labels[i][0] = 1
        elif labels[i][0] == 'Iris-versicolor':
            labels[i][0] = 0
    return input_vecs, labels
def randomDATA(x_domain, y_domain, rate=0.3):
    '''
    划分训练集、测试集
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test
def zipDATA(x_train, y_train):
    '''
    将输入数据和对应标签进行打包
    '''
    return list(zip(x_train, y_train))

class LogisticRegression(object):
    def __init__(self,x_train,y_train,alpha=0.01,epochs=500,epsilon=1e-5):
        # 输入向量加以扩充，用来表示偏置项bias
        self.x_train = np.insert(x_train,4,1,axis=1)
        self.y_train = y_train
        self.alpha = alpha
        self.epochs = epochs
        self.epsilon = epsilon

    def init_param(self):
        self.xMatrix = np.mat(self.x_train,dtype=float)
        self.yMatrix = np.mat(self.y_train,dtype=float)

        m,n = self.xMatrix.shape
        # 初始化参数向量
        self.weights = np.ones((n,1))
    def sigmoid(self,x):
        res = 1.0/(1.0 + np.exp(-x))
        return res
    def loss(self,xMatrix,weights,yMatrix):
        m,n = xMatrix.shape
        hypothesis = self.sigmoid(np.dot(xMatrix,weights))
        # 6.1.3模型参数估计。下边loss计算是化简前的对数似然函数
        # 该对数似然函数值得深入考虑。问题如下：
        # 分类问题如何计算误差？我们要通过误差大小来更新参数，但是正确率并不是参数的函数
        # 解决这个问题，需要深入理解交叉熵损失函数
        loss = (-1.0/m)*np.sum(yMatrix.T * np.log(hypothesis) + (1-yMatrix).T * np.log(1-hypothesis))
        return loss

    def train_byBGD(self):
        # 三种梯度下降方法大同小异。区别在于计算梯度时，选取的数据量大小
        # 总体(量大训练慢) -> 单一样本(单个样本并不能很好的代表全体样本) -> 小批量样本(折中选取数据)
        # 批梯度下降：每次迭代，需要全部数据的误差值来计算梯度。
        self.init_param()
        m,n = self.xMatrix.shape
        loss_list = []
        epoch_list = []
        for epoch in range(self.epochs):
            # 计算当前参数权重状态下的误差值
            loss_old = self.loss(self.xMatrix,self.weights,self.yMatrix)
            # 输入x进行分类的线性函数值w*x，通过sigmoid函数转换为概率值
            hypothesis = self.sigmoid(np.dot(self.xMatrix,self.weights))
            # 计算误差
            error = hypothesis - self.yMatrix
            # 计算梯度，这里直接应用了梯度的计算公式。对数似然函数，对不同的参数求偏导。
            grad = (1.0/m)*np.dot(self.xMatrix.T,error)
            # 参数权重更新
            self.weights = self.weights - self.alpha*grad
            # 计算参数更新后的误差值
            loss_new = self.loss(self.xMatrix,self.weights,self.yMatrix)
            # 参数更新的一种结束方式。如果更新前后的两个误差值小于阈值，停止更新。
            if abs(loss_new-loss_old) < self.epsilon:
                print('this is epoch {}'.format(epoch))
                break

            epoch_list.append(epoch)
            loss_list.append(loss_new)
        return epoch_list,loss_list
    def train_bySGD(self):
        # 随机梯度下降：每次迭代，在数据集中随机抽取一个元素来代表本数据集，计算梯度。
        self.init_param()
        m, n = self.xMatrix.shape
        loss_list = []
        epoch_list = []
        for epoch in range(self.epochs):
            rand_x = np.random.randint(m)
            loss_old = self.loss(self.xMatrix, self.weights, self.yMatrix)

            hypothesis = self.sigmoid(np.dot(self.xMatrix[rand_x:], self.weights))
            error = hypothesis - self.yMatrix[rand_x:]
            grad = (1.0/m)*np.dot(self.xMatrix[rand_x:].T, error)
            self.weights = self.weights - self.alpha * grad

            loss_new = self.loss(self.xMatrix, self.weights, self.yMatrix)
            # 检查是否收敛，结合损失值图像。
            # 多运行几次，可以看到SGD方式的劣处。某次损失值极小时，受该极端样本影响，更新结束，但此时并不是最优解
            if abs(loss_new - loss_old) < self.epsilon:
                break

            epoch_list.append(epoch)
            loss_list.append(loss_new)
        return epoch_list, loss_list
    def train_byMBGD(self,batch_size=10):
        # 小批梯度下降：每次迭代，用小批量数据的误差值来计算梯度。
        # 减少了BGD方法使用全部数据的计算量，减弱了SGD方法极端样本带来的偏差
        self.init_param()
        m, n = self.xMatrix.shape
        loss_list = []
        epoch_list = []
        for epoch in range(self.epochs):
            # np.random.choice在数组中随机抽取元素(batch_size个)
            randIndex = np.random.choice(range(len(self.xMatrix)), batch_size, replace=False)
            loss_old = self.loss(self.xMatrix,self.weights,self.yMatrix)

            hypothesis = self.sigmoid(np.dot(self.xMatrix[randIndex],self.weights))
            error = hypothesis - self.yMatrix[randIndex]
            grad = (1.0/m)*np.dot(self.xMatrix[randIndex].T,error)
            self.weights = self.weights - self.alpha*grad

            loss_new = self.loss(self.xMatrix,self.weights,self.yMatrix)
            # 检查是否收敛，结合损失值图像多观察几次，会出现和SGD同样的问题。
            if abs(loss_new-loss_old) < self.epsilon:
                break

            epoch_list.append(epoch)
            loss_list.append(loss_new)
        return epoch_list, loss_list

    def predict(self,x):
        prob = self.sigmoid(np.dot(x,self.weights))
        lab = int(0)
        if prob >= 0.5:
            lab = int(1)
        return lab

if __name__ == "__main__":
    x, y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)
    x_test = np.insert(x_test,4,1,axis=1)

    t1 = time.time()
    LR = LogisticRegression(x_train,y_train)
    #e, l = lr.train_byBGD()
    e, l = LR.train_bySGD()
    #e, l = lr.train_byMBGD()

    t2 = time.time()
    correct_num = 0
    for i, item in enumerate(x_test):
        label = LR.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))
    plt.plot(e, l)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

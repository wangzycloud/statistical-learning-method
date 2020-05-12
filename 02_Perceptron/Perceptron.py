#coding=utf-8
#Author:wangzy
#Date:2020-04-15
#Email:wangzycloud@163.com

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def loadDATA():
    iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    data = iris[0:100]
    input_vecs = data[:, [0, 1]].astype('float')
    labels = data[:,[4]]
    for i in range(labels.shape[0]):
        if labels[i][0] == 'Iris-setosa':
            labels[i][0] = 1
        else:
            labels[i][0] = -1
    return input_vecs,labels
def randomDATA(x_domain,y_domain,rate=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test

class Perceptron(object):
    def __init__(self,input_dim):
        self.weights = np.array([np.random.uniform(-1,2) for _ in range(input_dim)])
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:{0}\nbias\t:{1}\n'.format(self.weights, self.bias)

    def train(self,inputs,labels,iter=50,rate=0.003):
        for i in range(iter):
            for j in range(inputs.shape[0]):
                xj = inputs[j]
                yj = labels[j]
                if yj*(np.dot(xj,self.weights) + self.bias) <= 0:
                    self.weights = self.weights + rate*yj*xj
                    self.bias = self.bias + rate*yj
    def predict(self,x):
        y = np.dot(x,self.weights) + self.bias
        if y >= 0:
            return 1
        else:
            return -1

if __name__ == "__main__":
    # 加载数据
    x,y = loadDATA()
    # 划分数据集
    x_train,x_test,y_train,y_test = randomDATA(x,y)

    # 定义一个感知机，使用训练集进行训练，测试集进行测试
    t1 = time.time()
    P = Perceptron(2)
    P.train(x_train,y_train,50,0.003)
    print(P)

    t2 = time.time()
    correct_num = 0
    for i, item in enumerate(x_test):
        label = P.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))

    # 得到iris各个维度坐标
    y1_x1 = [x_test[i][0] for i in range(x_test.shape[0]) if y_test[i][0]==1]
    y1_x2 = [x_test[i][1] for i in range(x_test.shape[0]) if y_test[i][0]==1]
    z1 = [y_test[i][0] for i in range(x_test.shape[0]) if y_test[i][0] == 1]
    y2_x1 = [x_test[i][0] for i in range(x_test.shape[0]) if y_test[i][0]==-1]
    y2_x2 = [x_test[i][1] for i in range(x_test.shape[0]) if y_test[i][0]==-1]
    z2 = [y_test[i][0] for i in range(x_test.shape[0]) if y_test[i][0] == -1]
    # 绘制iris数据散点图
    plt.scatter(y1_x1,y1_x2,edgecolors='red')
    plt.scatter(y2_x1,y2_x2,edgecolors='green')
    # 得到分离超平面
    x2 = [3.5,4.5,5.5,6.5,7.5]
    k = P.weights[0]/P.weights[1]
    y2 = list(map(lambda b:-b*k-P.bias,x2))
    plt.plot(x2,y2,color='yellow')
    plt.show()

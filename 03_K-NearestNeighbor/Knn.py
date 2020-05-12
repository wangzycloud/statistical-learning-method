#coding=utf-8
#Author:wangzy
#Date:2020-04-16
#Email:wangzycloud@163.com

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def loadDATA():
    iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    input_vecs = iris[:, [0, 1, 2, 3]].astype('float')
    labels = iris[:,[4]]
    for i in range(labels.shape[0]):
        if labels[i][0] == 'Iris-setosa':
            labels[i][0] = 1
        elif labels[i][0] == 'Iris-versicolor':
            labels[i][0] = 2
        elif labels[i][0] == 'Iris-virginica':
            labels[i][0] = 3
    return input_vecs,labels
def randomDATA(x_domain,y_domain,rate=0.3):
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test
def zipDATA(x_train,y_train):
    return list(zip(x_train,y_train))

def EuclideanDistance(a,b):
    sum = 0
    for i in range(a.shape[0]):
        temp = np.square(np.abs(a[i]-b[i]))
        sum += temp
    dist = np.sqrt(sum)
    return dist
def ManhattanDistance(a,b):
    sum = 0
    for i in range(a.shape[0]):
        temp = np.abs(a[i]-b[i])
        sum += temp
    return sum
def ChebyshevDistance(a,b):
    dist = []
    for i in range(a.shape[0]):
        temp = np.abs(a[i] - b[i])
        dist.append(temp)
    return np.max(dist)

class Knn(object):
    def __init__(self,k,trainDATA):
        self.k = k
        self.trainDATA = trainDATA

    def predict(self,x,func):
        dist = []
        for idx in range(len(self.trainDATA)):
            temp = func(x,self.trainDATA[idx][0])
            temp = np.array([temp,self.trainDATA[idx][1][0]])
            dist.append(temp)
        dist = np.array(dist)
        dist = dist[np.lexsort(dist[:,::-1].T)]
        Nk = dist[0:self.k]
        Nk = [ k[1] for k in Nk]
        label = int(max(Nk,key=Nk.count))
        return label

if __name__ == "__main__":
    x,y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)
    trainDATA = zipDATA(x_train, y_train)
    testDATA = zipDATA(x_test, y_test)

    knn = Knn(10, trainDATA)

    correct_num = 0
    for idx in range(x_test.shape[0]):
        label = knn.predict(x_test[idx], ManhattanDistance)
        if label == y_test[idx][0]:
            correct_num += 1
    print('正确率为：{:.2f}'.format(correct_num/len(x_test)))
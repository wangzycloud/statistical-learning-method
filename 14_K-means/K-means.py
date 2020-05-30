#coding=utf-8
#Author:wangzy
#Date:2020-05-29
#Email:wangzycloud@163.com

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDATA():
    # 读取数据
    iris = pd.read_csv('../DATA/data-iris/iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    data = iris[0:150]
    inputVecs = data[:, [0, 2]].astype('float')
    return inputVecs
def EuclideanDistance(a,b):
    # 欧式距离
    sum = 0
    for i in range(a.shape[0]):
        temp = np.square(np.abs(a[i]-b[i]))
        sum += temp
    dist = np.sqrt(sum)
    return dist

class Kmeans(object):
    def __init__(self,k,data,func):
        # 参数列表
        self.k = k
        self.data = data
        self.func = func    # 距离度量函数

    def initCenters(self,data,k):
        # 算法14.2 步骤(1) 初始化
        # 随机选择k个样本点作为初始聚类中心
        return random.sample(list(data), k)
    def cluster(self,data, centers,func):
        # 算法14.2 步骤(2) 对样本进行聚类
        # 计算每个样本到类中心的距离，将每个样本指派到与其最近的中心的类中
        # 这里采用字典形式保存不同的类：key:(center);value:[属于该类的各个样本]
        k_cluster = {}
        for item in centers:
            k_cluster[tuple(item)] = []

        for item in data:
            min_dist = float('inf')
            min_idx = -1
            # 计算该样本到哪个中心点近
            for i, center in enumerate(centers):
                dist = func(item, center)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            k_cluster[tuple(centers[min_idx])].append(item)
        return k_cluster
    def getNewCenters(self,clusters):
        # 算法14.2 步骤(3) 计算新的类中心
        # 计算当前各个类中样本的均值，作为新的类中心
        centers = []
        for center in clusters.keys():
            newCenter = np.mean(np.array(clusters[center]), axis=0)
            centers.append(newCenter)
        return centers
    def getMinDistance(self,clusters):
        # 算法14.2 步骤(4) 迭代收敛条件的计算方式
        # 计算样本和其所属类的中心之间的距离总和
        # 一次迭代前后，距离总和小于阈值时停止，说明中心点变动较小
        dist = 0
        for item in clusters.keys():
            for elem in clusters[item]:
                d = self.func(np.array(item),elem)
                dist += d
        return dist
    def train(self,maxIter=100):
        # 训练过程，对应算法14.2(k均值聚类算法)
        # 初始化
        self.centers = self.initCenters(self.data,self.k)
        self.dist = 0
        for i in range(maxIter):
            old_dist = self.dist
            # 对样本进行聚类
            self.k_cluster = self.cluster(self.data,self.centers,self.func)
            self.dist = self.getMinDistance(self.k_cluster)
            # 计算新的类中心
            self.centers = self.getNewCenters(self.k_cluster)
            # 迭代停止条件
            if np.abs(self.dist-old_dist) < 0.01:
                break
        return self.k_cluster

if __name__ == '__main__':
    data = loadDATA()

    # 根据iris数据集，以划分为3类为例
    kMeans = Kmeans(3,data,EuclideanDistance)
    k_cluster = kMeans.train()
    # 取出各个簇，散点图显示
    k1 = list(k_cluster.keys())[0]
    k2 = list(k_cluster.keys())[1]
    k3 = list(k_cluster.keys())[2]

    t1 = np.array(k_cluster[k1])
    x1 = t1[:, 0]
    y1 = t1[:, 1]

    t2 = np.array(k_cluster[k2])
    x2 = t2[:, 0]
    y2 = t2[:, 1]

    t3 = np.array(k_cluster[k3])
    x3 = t3[:, 0]
    y3 = t3[:, 1]

    plt.scatter(x1, y1, c='y')
    plt.scatter(k1[0], k1[1], marker="D", c='r')
    plt.scatter(x2, y2, c='g')
    plt.scatter(k2[0], k2[1], marker="D", c='r')
    plt.scatter(x3, y3, c='m')
    plt.scatter(k3[0], k3[1], marker="D", c='r')
    plt.show()
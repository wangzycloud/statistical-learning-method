#coding=utf-8
#Author:wangzy
#Date:2020-04-18
#Email:wangzycloud@163.com

import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def loadDATA():
    '''
    加载数据
    '''
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
def featureTransfer(dataset):
    # 该函数实现输入空间特征向特征空间转换，这里只是进行了上取整，减少小数部分产生的特征。
    # 输入空间为数值型特征，特征数目太多，转换到特征空间共4*8=32个特征，共需求3*4*8=96个参数。
    # 特征空间有四个维度，每个维度可能的取值区间为[1,8]
    return np.ceil(dataset).astype(int)
def display(data, i=0):
    '''
    该函数将字典结构的树进行表示，深度相同的分支在同一列
    '''
    for k in data:
        if isinstance(k,str):
            print('   ' * i, k,':',data[k], sep='')
        else:
            print('   ' * i, k, sep='')
        if isinstance(data[k], dict):
            display(data[k], i + 1)

class DecisionTree(object):
    def __init__(self,trainDATA,epsilon=0.03):
        self.trainDATA = trainDATA
        self.epsilon = epsilon
        self.tree = None
        self.features = {}

    def init_param(self):
        '''
        统计训练数据中每个特征可能的取值，key为特征标记，本例共四个特征。
        第一个for遍历每个数据
        第二个for循环，寻找每个数据的特征值，看看是否被记录
        第一个数据创建keys，剩下的数据，寻找特征可能的各个取值
        '''
        for data in self.trainDATA:
            for idx,value in enumerate(data[0]):
                if not idx in self.features.keys():
                    self.features[idx] = [value]
                if not value in self.features[idx]:
                    self.features[idx].append(value)

    def labelCounut(self,dataSet):
        '''
        统计数据集内，各个数据的标签情况
        :param dataset: 输入数据集
        :return: 字典记录数据的标签情况，即该数据集有多少个类别(key)，每个类别有多少数据(value)
        '''
        labelCount = {}
        for item in dataSet:
            item = item[1][0]
            if item in labelCount.keys():
                labelCount[item] += 1
            else:
                labelCount[item] = 1
        return labelCount

    def getSubDATA(self,dataSet,feature,value):
        '''
        算法5.2步骤(5)中，根据最优特征的不同取值，划分子数据集
        :param dataSet:数据集
        :param feature:最优特征
        :param value:最优特征的某个取值
        :return:得到子集
        '''
        subData = []
        for data in dataSet:
            item = data[0][feature]
            if item == value:
                subData.append(data)
        return subData
    def calcHD(self,dataSet):
        '''
        算法5.1步骤(1)，计算数据集D的经验熵H(D)
        '''
        labelCount = self.labelCounut(dataSet)
        N = len(dataSet)
        HD = 0
        for i in labelCount.values():
            pi = i/N
            # 公式(5.7)
            HD -= pi*np.log2(pi)
        return HD
    def calcInforGain(self,dataSet,feature):
        '''
        算法5.1步骤(2)(3)，计算特征A对数据集D的经验条件熵H(D|A)，并得到特征A对数据集D的信息增益值
        '''
        values = self.features[feature]
        HAg = 0
        for v in values:
            subData = self.getSubDATA(dataSet,feature,v)
            # 公式(5.8)
            HAg += len(subData)/len(dataSet)*self.calcHD(subData)
        return self.calcHD(dataSet) - HAg
    def createTree(self,dataSet,features,epsilon):
        '''
        算法5.2(ID3算法)，决策树的生成算法(递归生成)
        :param dataSet: 训练数据集D
        :param features: 特征集A
        :param epsilon: 阈值ε
        :return: 决策树T
        '''
        # 统计当前数据集类别情况
        labelCount = self.labelCounut(dataSet)
        # 算法5.2步骤(1) 若D中所有实例属于同一类，则T为单节点树，并将该类作为该结点的类标记
        if len(labelCount) == 1:
            return list(labelCount.keys())[0]
        # 算法5.2步骤(2) 若特征集为空，则T为单节点树，并将D中实例数最大的类作为该结点的类标记
        if not features:
            print('not features')
            return max(list(labelCount.items()),key = lambda a:a[1])[0]
        # 算法5.2步骤(3) 否则，计算各特征对D的信息增益，选择信息增益最大的特征Ag
        Ags = list(map(lambda a: [a, self.calcInforGain(dataSet,a)], features))
        Ag, Again = max(Ags, key=lambda a: a[1])
        # 算法5.2步骤(4) 如果Ag的信息增益小于阈值ε，则T为单节点树，并将D中实例数最大的类作为该结点的类标记
        if Again < epsilon:
            return max(list(labelCount.items()),key = lambda a:a[1])[0]
        # 算法5.2步骤(5) 否则，对Ag的每一可能值，将D分割为若干非空子集Di，将Di中实例数最大的类作为该结点的类标记，构建子结点
        tree = {}
        # 获得子特征集A-{Ag}
        subFeatures = list(filter(lambda a:a != Ag,features))
        tree['feature'] = Ag
        for value in self.features[Ag]:
            # 将D分割为若干非空子集Di
            subData = self.getSubDATA(dataSet,Ag,value)
            if not subData:
                continue
            # 算法5.2步骤(6) 对第i个子结点，以Di为训练集，以A-{Ag}为特征集，递归调用(1)-(5)步，得到子树Ti
            tree[value] = self.createTree(subData,subFeatures,epsilon)
        return tree

    def train(self):
        # 训练过程是决策树生成的过程
        self.init_param()
        fs = list(self.features.keys())
        self.tree = self.createTree(trainDATA,fs,self.epsilon)

    def predict(self,x):
        '''
        根据生成的决策树，预测测试数据x的类别
        :param x: 测试数据
        :return: 类别
        '''
        def func(tree,x):
            # 定义一个递归遍历函数，根据树形的字典结构，如果当前元素不是字典，则为类别标记
            if type(tree) != dict:
                return tree
            else:
                # 如第一个元素，由'feature'提取到根结点，看根结点是哪个分量(特征)
                # 取测试数据x在该分量的值，作为下一个子树tree = tree[val]，接着继续递归
                root = tree['feature']
                val = x[root]
                # 该分支解决测试数据x在某分量，出现训练数据集在该分量不存在某个值的情况
                # 训练数据集的数据在某个分量上不存在这个取值，我的解决方法是，在该特征可能的取值中随机取一个值
                if val not in self.features[root]:
                    val = random.choice(self.features[root])
                tree = tree[val]
                return func(tree,x)
        return func(self.tree,x)

if __name__ == "__main__":
    x, y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)
    x_train = featureTransfer(x_train)
    x_test = featureTransfer(x_test)
    trainDATA = zipDATA(x_train, y_train)
    testDATA = zipDATA(x_test, y_test)

    t1 = time.time()
    DT = DecisionTree(trainDATA)
    DT.train()
    tree = DT.tree
    print(tree)
    print('决策树结构为：')
    display(tree)

    t2 = time.time()
    correct_num = 0
    for i,item in enumerate(x_test):
        label = DT.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))
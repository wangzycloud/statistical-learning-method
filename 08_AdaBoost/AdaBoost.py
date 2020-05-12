#coding=utf-8
#Author:wangzy
#Date:2020-05-08
#Email:wangzycloud@163.com

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def loadDATA():
    '''
    加载数据
    '''
    iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
    iris = iris.values
    data = iris[0:100]
    input_vecs = data[:, [0, 1]].astype('float')
    labels = data[:, [4]]
    for i in range(labels.shape[0]):
        if labels[i][0] == 'Iris-setosa':
            labels[i][0] = 1
        else:
            labels[i][0] = -1
    return input_vecs, labels
def randomDATA(x_domain, y_domain, rate=0.3):
    '''
    划分训练集、测试集
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test

class AdaBoost(object):
    def __init__(self,x_train,y_train):
        self.x_train = np.mat(x_train)
        self.y_train = np.mat(y_train)

        self.m,self.n = self.x_train.shape
        # 初始化训练数据的权值分布
        self.D = np.mat(np.ones((self.m,1))/self.m)
        self.weakClassifierArr = []
    def stumpClassify(self,data,dim,threshold,divRules):
        # 得到基本分类器的分类结果(参照8.4.1提升树模型的决策树桩)
        # 该函数对应算法8.1步骤(2)(a)
        # 实际上采用单层决策树作为分类器。该树桩用阈值threshold来决策，得到分类结果
        Gm_ClassifyRes = np.ones((data.shape[0],1))
        data_dim = data[:,dim]
        if divRules == 'lt':
            Gm_ClassifyRes[data_dim <= threshold] = -1
        else:
            Gm_ClassifyRes[data_dim > threshold] = -1
        return Gm_ClassifyRes
    def stumpBuild(self):
        # 该函数为建桩的过程，也就是考虑“怎样选取误差率最小的基本分类器？”
        stepNum = 10
        bestStump = {}
        bestClassRes = np.mat(np.zeros((self.m,1)))
        minEm = float('inf')
        # 外层循环把各个特征都考虑在内，选择所有特征上，使分类误差率最小的树桩(基本分类器)
        for i in range(self.n):
            minFeatureVal = self.x_train[:,i].min()
            maxFeatureVal = self.x_train[:,i].max()
            stepSize = (maxFeatureVal-minFeatureVal) / stepNum
            # 内层循环为了得到在该特征上，使分类误差率最小的分类阈值(参考算法5.5回归树建立过程)
            # 逐个阈值，得到不同的分类结果。找出分类误差率最小时的阈值
            for j in range(-1,int(stepNum)+1):
                for divRule in ['lt','gt']:
                    threshold = (minFeatureVal + float(j)*stepSize)
                    Gm_gotClassifyRes = self.stumpClassify(self.x_train,i,threshold,divRule)

                    # 计算分类误差率 算法8.1步骤(2)(b)
                    errArr = np.mat(np.ones((self.m,1)))
                    errArr[ Gm_gotClassifyRes == self.y_train] = 0
                    em = self.D.T * errArr  # 公式(8.1)

                    if em < minEm:
                        # 记录误差率最小的树桩
                        minEm = em
                        bestClassRes = Gm_gotClassifyRes.copy()
                        bestStump['dim'] = i
                        bestStump['threshold'] = threshold
                        bestStump['divRule'] = divRule
        return bestStump,minEm,bestClassRes

    def train(self,maxIter=50,smooth=1e-16):
        f_xs = np.mat(np.zeros((self.m,1)))
        for i in range(maxIter):
            # 根据当前参数值建桩，得到此时的最优基本分类器，加入分类器列表
            bestStump,em,Gm = self.stumpBuild()
            # 算法8.1步骤(2)(c) 公式(8.2) 计算该分类器的权重系数
            alpha_m = float(0.5*np.log((1.0-em+smooth)/(em+smooth)))    # 避免分母为0
            bestStump['alpha_m'] = alpha_m
            self.weakClassifierArr.append(bestStump)

            # 算法8.1步骤(2)(d) 公式(8.3) 更新训练数据集的权值分布
            exp = np.multiply(-1*alpha_m*self.y_train,Gm)
            exp = exp.astype(float)
            self.D = np.multiply(self.D,np.exp(exp))
            self.D = self.D / self.D.sum()

            # 算法8.1步骤(3) 构建基本分类器的线性组合
            f_xs += alpha_m*Gm
            # 计算更新之后的误差率
            errors = np.multiply(np.sign(f_xs)!=self.y_train,np.ones((self.m,1)))
            errorRate = errors.sum() / self.m
            print('iter:{} , error rate:{:.2f} '.format(i,errorRate))
            if errorRate == 0.0:
                break
    def predict(self,x):
        data = np.mat(x)
        f_x = 0.0
        for i in range(len(self.weakClassifierArr)):
            # 得到各个基本分类器的分类结果
            Gm_perClassRes = self.stumpClassify(data, self.weakClassifierArr[i]['dim'], self.weakClassifierArr[i]['threshold'],self.weakClassifierArr[i]['divRule'])
            f_x += self.weakClassifierArr[i]['alpha_m'] * Gm_perClassRes
        # 公式(8.7) 得到最终分类器分类结果
        G_x = np.sign(f_x)
        return G_x

if __name__ == '__main__':
    x, y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)

    t1 = time.time()
    AB = AdaBoost(x_train,y_train)
    AB.train()

    t2 = time.time()
    correct_num = 0
    for i, item in enumerate(x_test):
        label = AB.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('\n训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))
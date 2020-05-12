#coding=utf-8
#Author:wangzy
#Date:2020-04-28
#Email:wangzycloud@163.com

import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

class SVM(object):
    def __init__(self,x_train,y_train,C=200,slack=0.001):
        # 初始化要用到的各个变量
        self.x_train = np.mat(x_train)
        self.y_train = np.mat(y_train)

        self.m,self.n = x_train.shape
        # 惩罚参数C 松弛变量slack
        self.C = C
        self.slack = slack
        # 决策函数的偏置项b  算法7.4步骤(3)决策函数
        self.b = 0
        self.alphas = np.mat(np.zeros((self.m,1)))  # 待求解的α，每个样本点对应一个α，用alphas保存
        self.E = np.mat(np.zeros((self.m,1)))       # 求解过程中要用的Ei，是g(x)对输入xi的预测值与真实输出yi之差
        self.K = np.mat(np.zeros((self.m,self.m)))  # 核函数结果矩阵
        self.kernelCompute()                        # 提前计算核函数结果
        self.svIdx = None                           # 支持向量索引列表
    def LinearKernel(self,x1,x2):
        # 线性核函数
        res = x1*x2.T
        return res
    def GaussKernel(self,x1,x2,sigma=10):
        # 高斯核函数 公式(7.90)
        res = (x1-x2)*(x1-x2).T
        res = np.exp(-1*res/(2*sigma*2))
        return res
    def ExponentialKernel(self,x1,x2,sigma=1.3):
        # 指数核函数
        res = np.sum(x1 - x2)
        res = np.exp(-1 * res / (2 * sigma * 2))
        return res
    def LaplacianKernel(self,x1,x2,sigma=1.3):
        # 拉普拉斯核函数
        res = np.sum(x1 - x2)
        res = np.exp(-1 * res/sigma)
        return res
    def kernelCompute(self):
        # 核函数计算过程
        for i in range(self.m):
            X = self.x_train[i,:]
            for j in range(self.m):
                Z = self.x_train[j,:]
                res = self.LinearKernel(X,Z)
                self.K[i,j] = res
                self.K[j,i] = res
    def isSatisfyKKT(self,i):
        # 选择第i个变量时，检查该样本点(xi,yi)是否满足KKT条件
        gxi = self.getgxi(i)
        yi = self.y_train[i]
        # 7.4.2变量的选择方法 公式(7.111) 公式(7.112) 公式(7.113)
        if (math.fabs(self.alphas[i]) < self.slack) and (yi*gxi >= 1):
            return True
        elif (math.fabs(self.alphas[i] - self.C) < self.slack) and (yi*gxi <= 1):
            return True
        elif (self.alphas[i] > -self.slack) and (self.alphas[i] < (self.C + self.slack)) and (math.fabs(yi*gxi - 1) < self.slack):
            return True
        return False
    def getgxi(self,i):
        # 计算g(x)，公式(7.104)，两点注意：
        # 1.注意g(x)与决策函数f(x)的关系：f(x)=sign[g*(x)]。'*'表示训练好的α*和b*
        # 2.注意g(x)的计算过程中，非支持向量样本α=0，不需要参与计算
        gxi = 0
        idx = [idx for idx,alpha in enumerate(self.alphas) if alpha != 0]
        for t in idx:
            gxi += self.alphas[t] * self.y_train[t] * self.K[t,i]
        gxi += self.b
        return gxi
    def getEi(self,i):
        # 计算Ei，公式(7.105)，Ei为g(x)对输入xi的预测值与真实输出yi之差
        gxi = self.getgxi(i)
        return gxi - self.y_train[i]
    def selectJ(self,Ei,i):
        # 选择第二个变量，也就是内层循环的过程，希望能使αj有足够大的变化
        Ej = 0
        maxDeltaE = 0
        maxIndex = -1
        # 采用启发式规则，先遍历间隔边界上的支持向量样本点，依次将其对应的变量作为αj试用
        nozeroE = [i for i,Ei in enumerate(self.E) if Ei != 0]
        for j in nozeroE:
            Ej_temp = self.getEi(j)
            if math.fabs(Ei-Ej_temp) > maxDeltaE :
                maxDeltaE = math.fabs(Ei-Ej_temp)
                Ej = Ej_temp
                maxIndex = j
        # 如果在间隔边界上找不到合适的αj，那么遍历训练集，随机选择一个
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(np.random.uniform(0,self.m))
            Ej = self.getEi(maxIndex)
        return Ej,maxIndex
    def clipAlpha(self,aj,H,L):
        # 公式(7.108)，剪辑得到约束条件下的最优解
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj
    def SMOinnerCompute(self,i):
        Ei = self.getEi(i)
        # 判断样本点(xi,yi)是否违反KKT条件
        if self.isSatisfyKKT(i) == False:
            # 内层循环，选择第二个变量j
            Ej,j = self.selectJ(Ei,i)

            yi = self.y_train[i]
            yj = self.y_train[j]
            alphaOld_i = self.alphas[i]
            alphaOld_j = self.alphas[j]

            if yi != yj:
                # 图(7.8a)所示，αj的取值范围
                L = max(0,alphaOld_j-alphaOld_i)
                H = min(self.C,self.C+alphaOld_j-alphaOld_i)
            else:
                # 图(7.8b)所示，αj的取值范围
                L = max(0,alphaOld_j+alphaOld_i-self.C)
                H = min(self.C,alphaOld_j+alphaOld_i)
            if L == H:
                print("L==H")
                return 0
            # 公式(7.107)，计算η
            eta = self.K[i,i] + self.K[j,j] - 2.0*self.K[i,j]
            if eta <= 0 :
               print('eta <= 0')
               return 0
            # 得到未经剪辑的αj(new,unc)
            alphaNew_uncj = alphaOld_j + yj*(Ei-Ej)/eta
            # 得到剪辑后的αj(new)
            alphaNew_j = self.clipAlpha(alphaNew_uncj,H,L)
            # 如果αj改变量过小，不进行更新
            if (math.fabs(alphaNew_j-alphaOld_j) < self.slack):
                return 0
            # 公式(7.109)，计算αi(new)
            alphaNew_i = alphaOld_i + yi*yj*(alphaOld_j - alphaNew_j)

            # 公式(7.115)  公式(7.116)，计算b(new)
            b1 = -1*Ei - yi*self.K[i,i]*(alphaNew_i - alphaOld_i) - yj*self.K[j,i]*(alphaNew_j-alphaOld_j) + self.b
            b2 = -1*Ej - yi*self.K[i,j]*(alphaNew_i - alphaOld_i) - yj*self.K[j,j]*(alphaNew_j-alphaOld_j) + self.b
            # 根据αi和αj值的范围确定新的b
            if (alphaNew_i > 0) and (alphaNew_i < self.C):
                b = b1
            elif (alphaNew_j > 0) and (alphaNew_j < self.C):
                b = b2
            else:
                b = (b1+b2)/2.0
            # 更新两个参数αi、αj，以及参数b
            self.alphas[i] = alphaNew_i
            self.alphas[j] = alphaNew_j
            self.b = b
            # 更新对应的Ei值，并将它们保存起来
            self.E[i] = self.getEi(i)
            self.E[j] = self.getEi(j)
            return 1
        else:
            return 0

    def trainbySMO(self,maxIter = 100):
        iter = 0
        entireSet = True
        alphaPairsChanged = 1
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            # 外层循环，选择第一个变量i
            alphaPairsChanged = 0
            if entireSet:
                # 遍历所有数据(算法第一次循环的时候，在所有数据中选择变量i。这时候还没有支持向量点)
                for i in range(self.m):
                    alphaPairsChanged += self.SMOinnerCompute(i)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))  # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
                iter += 1
            else:
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                # 遍历间隔边界的数据(按照算法描述，外层循环先遍历间隔边界上的支持向量点)
                for i in nonBoundIs:
                    alphaPairsChanged += self.SMOinnerCompute(i)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif (alphaPairsChanged == 0):
                # 如果支持向量点都满足KKT条件，开始遍历整个训练集
                entireSet = True
            print("iteration number: %d" % iter)
        # 得到支持向量
        self.svIdx = np.nonzero(self.alphas)[0]
        print('----------------------------------------')
        print(self.alphas)
        print('\nthere are {} Support Vectors'.format(self.svIdx.shape[0]))
        print(self.svIdx)
    def predict(self,x):
        res = 0
        for i in self.svIdx:
            # 得到输入样本与各个支持向量的核函数值
            temp = self.LinearKernel(self.x_train[i,:],np.mat(x))
            res += self.alphas[i]*self.y_train[i]*temp
        # 算法7.4步骤(3) 决策函数f(x)
        res = np.sign(res + self.b)
        return res

if __name__ == '__main__':
    x, y = loadDATA()
    x_train, x_test, y_train, y_test = randomDATA(x, y)

    t1 = time.time()
    svm = SVM(x_train,y_train)
    svm.trainbySMO()

    t2 = time.time()
    correct_num = 0
    for i, item in enumerate(x_test):
        label = svm.predict(item)
        if label == y_test[i]:
            correct_num += 1

    t3 = time.time()
    print('\n训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))

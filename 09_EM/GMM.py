#coding=utf-8
#Author:wangzy
#Date:2020-05-11
#Email:wangzycloud@163.com

import math
import numpy as np
from functools import reduce

def loadDATA(alpha,param1,param2,num=500):
    # 两个高斯分布生成数据，将两个数据集混合
    data = []
    data1 = np.random.normal(param1[0],param1[1],int(alpha*num))
    data2 = np.random.normal(param2[0],param2[1],int((1-alpha)*num))
    data.extend(data1)
    data.extend(data2)
    np.random.shuffle(data)
    return data

class GMM(object):
    def __init__(self,data,theta):
        # 参数初始化
        self.y = np.array(data)
        self.Alpha = [theta[0],1-theta[0]]
        self.Mu = [theta[1],theta[2]]
        self.Sigma = [theta[3],theta[4]]
        self.K = 2

    def Gauss(self,y,Mu,Sigma):
        # 公式(9.25) 计算当前参数下的高斯函数值
        res = (1/(math.sqrt(2*math.pi)*Sigma)*math.exp(-(y-Mu)*(y-Mu)/2*Sigma**2))
        return res

    def E_step(self,Alpha,Mu,Sigma):
        # 算法9.2 (2)E步：依据当前模型参数，计算分模型k对观测数据yj的响应度
        # 这里是计算分子部分
        K = self.K
        resGauss = np.zeros((K,len(self.y)))
        for k in range(resGauss.shape[0]):
            for j in range(resGauss.shape[1]):
                resGauss[k][j] = Alpha[k]*self.Gauss(self.y[j],Mu[k],Sigma[k])
        # 这里是计算分母部分，转置操作是为了方便循环函数的写法，外循环对数据个数j操作，内循环对K操作
        resGauss = resGauss.T
        Gamma_jk = np.zeros((len(self.y),K))
        for j in range(Gamma_jk.shape[0]):
            denominator = reduce(lambda a,b:a+b,resGauss[j])
            for k in range(Gamma_jk.shape[1]):
                Gamma_jk[j][k] = resGauss[j][k]/denominator
        Gamma_jk = Gamma_jk.T
        return Gamma_jk
    def M_step(self,Gamma_jk):
        # 算法9.2 (3)M步：计算新一轮迭代的模型参数
        # 先得到不同k时，各个响应度γ的和
        Gamma_jk_sum = np.zeros((self.K,1))
        for k in range(Gamma_jk.shape[0]):
            Gamma_jk_sum[k][0] = reduce(lambda a,b:a+b,Gamma_jk[k])

        for k in range(Gamma_jk.shape[0]):
            # 公式(9.30) 计算参数μk
            numerator_jk = list(map(lambda a:a[0]*a[1],zip(Gamma_jk[k],self.y)))
            numerator_Mk_sum = reduce(lambda a,b:a+b,numerator_jk)
            Mu_k = float(numerator_Mk_sum/Gamma_jk_sum[k])
            # 公式(9.31) 计算参数σk
            numerator_yj_Muk = (self.y-self.Mu[k])**2
            numerator_jk1 = list(map(lambda a:a[0]*a[1],zip(Gamma_jk[k],numerator_yj_Muk)))
            numerator_Gk_sum = reduce(lambda a,b:a+b,numerator_jk1)
            Sigma_k = math.sqrt(numerator_Gk_sum/Gamma_jk_sum[k])
            # 公式(9.32)  计算参数αk
            Alpha_k = float(Gamma_jk_sum[k]/len(self.y))
            # 更新参数值
            self.Alpha[k] = Alpha_k
            self.Mu[k] = Mu_k
            self.Sigma[k] = Sigma_k

    def train(self,maxIter):
        for i in range(maxIter):
            Gamma = self.E_step(self.Alpha,self.Mu,self.Sigma)
            self.M_step(Gamma)
            # 打印每次迭代的参数更新情况，没有设置收敛条件
            print('----------iter:{}------------'.format(i))
            print(self.Alpha)
            print(self.Mu)
            print(self.Sigma)

if __name__ == '__main__':
    alpha = 0.3
    param1 = [3,0.7]
    param2 = [7, 1]
    # 生成观测数据
    data = loadDATA(alpha,param1,param2)
    theta = [0.4, 2, 8,0.4,0.8]

    gmm = GMM(data,theta)
    gmm.train(10)
#coding=utf-8
#Author:wangzy
#Date:2020-05-10
#Email:wangzycloud@163.com
from functools import reduce

class EM(object):
    def __init__(self,data,theta):
        # 参数初始化
        self.y = data
        self.pi = theta[0]
        self.p = theta[1]
        self.q = theta[2]

    def E_step(self,pi,p,q):
        # E步：公式(9.5)，计算在模型参数pi,p,q下观测数据yj来自掷硬币B的概率
        Mu = []
        for idx,yj in enumerate(self.y):
            numerator = pi*(p**yj)*(1-p)**(1-yj)
            denominator = numerator + (1-pi)*(q**yj)*(1-q)**(1-yj)
            Mu_j = numerator/denominator
            Mu.append(Mu_j)
        return Mu
    def M_step(self,Mu):
        # M步：计算模型参数的新估计值
        pi_numerator = reduce(lambda a,b:a+b,Mu)
        self.pi = pi_numerator/len(Mu)      #公式(9.6)

        Mu_py = list(map(lambda a:a[0]*a[1],zip(Mu,self.y)))
        p_numerator = reduce(lambda a,b:a+b,Mu_py)
        p_denominator = reduce(lambda a,b:a+b,Mu)
        self.p = p_numerator/p_denominator  #公式(9.7)

        Mu = list(map(lambda a:1-a, Mu))
        Mu_qy = list(map(lambda a:a[0]*a[1],zip(Mu,self.y)))
        q_numerator = reduce(lambda a, b: a + b, Mu_qy)
        q_denominator = reduce(lambda a, b: a + b, Mu)
        self.q = q_numerator/q_denominator  #公式(9.8)

    def train(self,maxIter,threshold):
        # 训练过程
        for i in range(maxIter):
            old_pi,old_p,old_q = self.pi,self.p,self.q
            Mu = self.E_step(self.pi,self.p,self.q)
            self.M_step(Mu)

            print('iter = {},pi = {},p = {},q = {}'.format(i,self.pi,self.p,self.q))
            if (abs(self.pi-old_pi) <= threshold) and (abs(self.p-old_p) <= threshold) and (abs(self.q-old_q) <= threshold):
                print('iter = {}时参数收敛，停止更新.'.format(i))
                break

if __name__ == '__main__':
    # 观测数据
    observedDate = [1,1,0,1,0,0,1,0,1,1]
    theta = [0.5,0.5,0.5]

    em = EM(observedDate,theta)
    em.train(10,0.001)
#coding=utf-8
#Author:wangzy
#Date:2020-06-03
#Email:wangzycloud@163.com
#SVD分步求解时会遇到的问题，转自知乎：https://zhuanlan.zhihu.com/p/43578482

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

def linalg_svd(A):
    # 直接调库进行SVD的过程
    m, n = A.shape
    U, Sigma, VT = linalg.svd(A)
    # 将列表形式的Sigma转换为矩阵Σ
    Sigma = np.eye(m,n)*Sigma
    res = U.dot(Sigma).dot(VT)

    print('U：\n',U)
    print('Sigma：\n',Sigma)
    print('VT：\n',VT)
    print('组合后的矩阵：\n',res)

def SVD(A):
    # 分步进行SVD的过程
    m, n = A.shape
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)
    # 分别得到左、右奇异向量(U、V)
    ATA_value, ATA_vecter = linalg.eig(ATA)
    AAT_value, AAT_vecter = linalg.eig(AAT)
    # 得到对角矩阵Σ
    Sigma = np.eye(m,n)*np.sqrt(ATA_value)

    U = AAT_vecter
    VT = ATA_vecter.T
    res = U.dot(Sigma).dot(VT)

    print('U：\n', U)
    print('Sigma：\n', Sigma)
    print('VT：\n', VT)
    print('组合后的矩阵：\n', res)

A = [[3,1],
     [2,1]]
A = np.mat(A)
print("原矩阵为：\n",A)

linalg_svd(A)
print('---------------------------------------')
SVD(A)

#######################################
# 图片压缩的例子，使用重要特征来表示图片
path = './pkq.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(256,256))
#plt.imshow(img,'gray')

# 1.图片矩阵进行奇异值分解
U,Sigma,VT = linalg.svd(img)
# 2.图片重构，取前K个特征值组成对角矩阵Σ。对图片进行压缩，用最重要的特征表示该图片
k = 30
diagonal = np.mat(np.eye(k)*Sigma[:k])
restore = U[:,:k]*diagonal*VT[:k,:]
plt.imshow(restore,'gray')

# 信息量对比
print('---------------------------------------')
info = sum(Sigma)
print('所有特征值的和：{}'.format(info))
singularK = sum(Sigma[:k])
print('前{}个特征值和：{}'.format(k,singularK))
plt.show()
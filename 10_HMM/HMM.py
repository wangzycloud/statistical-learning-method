#coding=utf-8
#Author:wangzy
#Date:2020-05-15
#Email:wangzycloud@163.com
import numpy as np

def samplingIndex(distribution):
    # 接受-拒绝采样法，根据输入分布，产生随机数据(返回一个索引)
    start = 0
    idx = 0
    u = np.random.uniform(0,1)
    for idx,scope in enumerate(distribution):
        start += scope
        if u <= start:
            break
    return idx
def generateObservation(set_sta,set_obs,Lambda,T):
    '''
    算法10.1 生成观测序列
    :param set_sta: 状态集合['box1','box2','box3','box4']
    :param set_obs: 观测集合['red','white']
    :param Lambda: λ=(pi,A,B)
    :param T: 观测序列长度
    :return: 观测序列O(o1,o2,...,oT)
    '''
    pi = np.array(Lambda[0])
    A = np.array(Lambda[1])
    B = np.array(Lambda[2])

    seq_sta =[]     # 用来存储状态序列
    seq_obs = []    # 用来存储观测序列

    # 算法10.1 步骤(1)(2)
    sta_1 = samplingIndex(pi)
    seq_sta.append(samplingIndex(pi))
    # 算法10.1 步骤(3)
    obs_1 = samplingIndex(B[sta_1])
    seq_obs.append(obs_1)
    # 算法10.1 步骤(4)(5)
    for t in range(1,T):
        prior_t = seq_sta[t-1]
        sta_t = samplingIndex(A[prior_t])
        seq_sta.append(sta_t)
        obs_t = samplingIndex(B[sta_t])
        seq_obs.append(obs_t)

    print('---------------observation:----------------')
    print([set_sta[i] for i in seq_sta])
    print([set_obs[i] for i in seq_obs])
    print('-------------------------------------------')
    return seq_sta,seq_obs

def forwardAlgorithm(Lambda,O):
    '''
    算法10.2 观测序列概率的前向算法
    :param Lambda: λ=(pi,A,B)
    :param O: 观测序列O
    :return: 观测序列概率P(O|λ)
    '''
    pi = np.mat(Lambda[0]).T
    A = np.mat(Lambda[1])
    B = np.mat(Lambda[2])
    N = A.shape[0]

    # 算法10.2 步骤(1)初值
    alphas = []
    alpha_1 = np.multiply(pi,B[:,O[0]])
    alphas.append(alpha_1)
    # 算法10.2 步骤(2)递推
    for t in range(0,len(O)-1):
        alpha_t = alphas[t]
        alpha_tPlus1 = np.zeros((N,))
        for i in range(N):
            alpha_tPlus1[i] = np.sum(np.multiply(alpha_t,A[:,i]))*B[i,O[t+1]]
        alphas.append(np.mat(alpha_tPlus1).T)
    # 算法10.2 步骤(3)终止
    p_observation = np.sum(alphas[-1])
    return p_observation,alphas
def backwardAlgorithm(Lambda,O):
    '''
    算法10.3 观测序列概率的后向算法
    :param Lambda: λ=(pi,A,B)
    :param O: 观测序列O
    :return: 观测序列概率P(O|λ)
    '''
    pi = np.mat(Lambda[0]).T
    A = np.mat(Lambda[1])
    B = np.mat(Lambda[2])
    N = pi.shape[0]
    # 算法10.3 步骤(1)初值
    betas = []
    beta_t = np.ones((N,1))
    betas.append(beta_t)
    # 算法10.3 步骤(2)递推
    for t in range(0,len(O)-1):
        beta_t = betas[t]
        beta_tMinus1 = np.zeros((N,))
        for i in range(N):
            temp = np.multiply(A[i,:].T,B[:,O[len(O)-t-1]])
            temp = np.multiply(temp,beta_t)
            beta_tMinus1[i] = np.sum(temp)
            betas.append(np.mat(beta_tMinus1).T)
    # 算法10.3 步骤(3)终止
    p_temp = np.multiply(pi,B[:,O[0]])
    p_temp = np.multiply(p_temp,betas[-1])
    p_observation = np.sum(p_temp)
    return p_observation,betas

def caclulateGamma(alphas,betas,t,i):
    '''
    10.2.4 给定模型λ和观测O，求在时刻t处于状态qi的概率
           该函数没有出现O，但αt和βt是由O求得的
           公式(10.24)
    :param alphas: 前向概率计算中每个时刻的αt
    :param betas: 后向概率计算中每个时刻的βt
    :param t: 时刻t
    :param i: 状态qi
    :return: γ(i)
    '''
    numerator = alphas[t-1][i-1]*betas[t-1][i-1]
    denominator = np.sum(np.multiply(alphas[t-1],betas[t-1]))
    gamma_t_i = numerator/denominator
    return gamma_t_i
def caclulateXi(alphas,betas,A,B,O,t,i,j):
    '''
    10.2.4 给定模型λ和观测O，求在时刻t处于状态qi，且在时刻t+1处于qi+1的概率
           公式(10.26)
    :param alphas: 前向概率计算中每个时刻的αt
    :param betas: 后向概率计算中每个时刻的βt
    :param t: 时刻t
    :param i: 状态qi
    :param j: 状态qj
    :return: ξ(i,j)
    '''
    N = len(A)
    i,j = i-1,j-1
    numerator = alphas[t-1][i]*A[i][j]*B[j][O[t]]*betas[t][j]
    denominator = 0.0
    for m in range(N):
        for n in range(N):
            denominator += alphas[t-1][m]*A[m][n]*B[n][O[t]]*betas[t][n]
    xi_t_ij = numerator/denominator
    return xi_t_ij
def getGammaTs(alphas,betas,T,N):
    # 将γ(i)和ξ(i,j)对各个时刻求和，可以得到一些有用的期望值
    # 得到所有时刻t的γ(i)
    gammas = []
    for t in range(T):
        gamma_t = np.zeros((N,))
        for i in range(N):
            gamma_t[i] = caclulateGamma(alphas,betas,t,i)
        gammas.append(gamma_t)
    return gammas
def getXiTs(alphas,betas,Lambda,O,T,N):
    # 将γ(i)和ξ(i,j)对各个时刻求和，可以得到一些有用的期望值
    # 得到所有时刻t的ξ(i,j)
    A = Lambda[1]
    B = Lambda[2]
    xis = []
    for t in range(T):
        xi_t = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                xi_t[i][j] = caclulateXi(alphas,betas,A,B,O,t,i,j)
        xis.append(xi_t)
    return xis

def BaumWelch(Lambda,O):
    '''
    算法10.4 Baum-WelCh算法
    该部分看看就好。能力有限，跑不出正常结果来，无法检验
    :param Lambda: 算法10.4 步骤(1) λ为初始化后的参数
    :param O: 观测数据O=(o1,o2,...,oT)
    :return: 隐马尔科夫模型参数
    '''
    N = len(Lambda[1])      # 状态的个数
    M = len(Lambda[2][0])   # 观测的个数
    T = len(O)              # 序列的长度
    # 先根据此时的参数情况，得到α，β，γ，ξ
    fp, alphas = forwardAlgorithm(Lambda,O)
    bp, betas = backwardAlgorithm(Lambda,O)
    gammas = getGammaTs(alphas,betas,T,N)
    xis = getXiTs(alphas,betas,Lambda,O,T,N)
    # 算法10.4 步骤(2) 递推
    pi = Lambda[0]
    A = Lambda[1]
    B = Lambda[2]
    for i in range(N):
        # pi
        pi[i] = gammas[0][i]
        # Aij
        for j in range(N):
            numerator = 0.0
            denominator = 0.0
            for t in range(T-1):
                numerator += xis[t][i][j]
                denominator += gammas[t][i]
            A[i][j] = numerator/denominator
    # Bjk
    for j in range(N):
        for k in range(M):
            numerator = 0.0
            denominator = 0.0
            for t in range(T):
                if O[t] == O[k]:
                    numerator += gammas[t][j]
                denominator += gammas[t][j]
            B[j][k] = numerator/denominator
    return (pi,A,B)
def viterbi(Lambda,O):
    '''
    算法10.5 维特比算法
    :param Lambda: λ=(pi,A,B)
    :param O: 观测O=(o1,o2,...,oT)
    :return: 最优路径I*(i1,i2,...,iT)
    '''
    pi = Lambda[0]
    A = Lambda[1]
    B = Lambda[2]
    N = len(Lambda[1])  # 状态的个数
    T = len(O)          # 序列的长度
    # 算法10.5 步骤(1) δ和Ψ初始化
    Delte = np.zeros((T,N)) # 每行存储不同状态时的最大概率
    Psi = np.zeros((N,T))   # 每行存储最大概率对应的路径
    for i in range(N):
        Delte[0][i] = pi[i]*B[i][O[0]]
        Psi[i][0] = i
    # 算法10.5 步骤(2)(3) 递推、终止
    for t in range(1,T):
        newPsi = np.zeros((N,T))
        for i in range(N):
            prob = -1
            # 找到值最大概率时的状态j
            for j in range(N):
                nProb = Delte[t-1][j]*A[j][i] * B[i][O[t]]
                if nProb > prob:
                    prob = nProb
                    state = j
                    Delte[t][i] = prob
                    # 记录值最大概率时的路径
                    for m in range(t):
                        newPsi[i][m] = Psi[state][m]
                    newPsi[i][t] = i
        Psi = newPsi
    # 算法10.5 步骤(4) 最优路径回溯
    max_prob = -1
    path_state = 0
    for i in range(N):
        if Delte[T-1][i] > max_prob:
            max_prob = Delte[T-1][i]
            path_state = i
    return Psi[path_state],Delte

if __name__ == '__main__':
    set_state = ['box1','box2','box3','box4']
    set_observation = ['red','white']
    pi = [0.2,0.4,0.4]
    A = [[0.5,0.2,0.3],
         [0.3,0.5,0.2],
         [0.2,0.3,0.5]]
    B = [[0.5,0.5],
         [0.4,0.6],
         [0.7,0.3]]
    Lambda = (pi,A,B)

    print('根据λ=(pi,A,B)生成状态序列和观测序列：')
    seq_state, seq_observation = generateObservation(set_state, set_observation, Lambda, 5)
    print('索引表示状态序列：',seq_state)
    print('索引表示观测序列：',seq_observation)

    seq_obs = [0,1,0]   # 观测序列O=(红，白，红)
    fp, alphas = forwardAlgorithm(Lambda, seq_obs)
    bp, betas = backwardAlgorithm(Lambda, seq_obs)
    print('-------------------------------------------')
    print('前向算法计算P(O|λ)：',fp)
    print('后向算法计算P(O|λ)：',bp)

    gamma_t_i = caclulateGamma(alphas, betas, 1, 0)
    xi_t_ij = caclulateXi(alphas, betas, A, B, seq_obs, 1, 0, 1)
    print('-------------------------------------------')
    print('在时刻t=2处于状态qi=box1的概率：',float(gamma_t_i))
    print('在时刻t=2处于状态qi=box1，且在时刻t=3处于qi+1=box2的概率：',float(xi_t_ij))

    optimal_path,max_p = viterbi(Lambda,seq_obs)
    print('-------------------------------------------')
    print('求解最优路径：')
    print(max_p)
    print('最优路径：',[set_state[int(optimal_path[k])] for k in range(len(optimal_path))])

    print('-------------------------------------------')
    print('测Baum-Welch算法：')
    lam = Lambda
    for i in range(10):
        lam = BaumWelch(lam, seq_obs)
    print(lam[0])
    print(lam[1])
    print(lam[2])

#coding=utf-8
#Author:wangzy
#Date:2020-05-15
#Email:wangzycloud@163.com
import numpy as np
def samplingIndex(distribution):
    # 拒绝采样
    start = 0
    idx = 0
    u = np.random.uniform(0,1)
    for idx,scope in enumerate(distribution):
        start += scope
        if u <= start:
            break
    return idx
def generateObservation(set_sta,set_obs,Lambda,T):
    pi = np.array(Lambda[0])
    A = np.array(Lambda[1])
    B = np.array(Lambda[2])

    seq_sta =[]
    seq_obs = []

    sta_1 = samplingIndex(pi)
    seq_sta.append(samplingIndex(pi))
    obs_1 = samplingIndex(B[sta_1])
    seq_obs.append(obs_1)

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
    pi = np.mat(Lambda[0]).T
    A = np.mat(Lambda[1])
    B = np.mat(Lambda[2])

    N = pi.shape[0]

    alphas = []
    alpha_1 = np.multiply(pi,B[:,O[0]])
    alphas.append(alpha_1)

    for t in range(0,len(O)-1):
        alpha_t = alphas[t]
        alpha_tPlus1 = np.zeros((N,))
        for i in range(N):
            alpha_tPlus1[i] = np.sum(np.multiply(alpha_t,A[:,i]))*B[i,O[t+1]]
        alphas.append(np.mat(alpha_tPlus1).T)

    p_observation = np.sum(alphas[-1])
    return p_observation,alphas
def backwardAlgorithm(Lambda,O):
    pi = np.mat(Lambda[0]).T
    A = np.mat(Lambda[1])
    B = np.mat(Lambda[2])

    N = pi.shape[0]
    betas = []
    beta_t = np.ones((N,1))
    betas.append(beta_t)

    for t in range(0,len(O)-1):
        beta_t = betas[t]
        beta_tMinus1 = np.zeros((N,))

        for i in range(N):
            temp = np.multiply(A[i,:].T,B[:,O[len(O)-t-1]])
            temp = np.multiply(temp,beta_t)
            beta_tMinus1[i] = np.sum(temp)
            betas.append(np.mat(beta_tMinus1).T)

    p_temp = np.multiply(pi,B[:,O[0]])
    p_temp = np.multiply(p_temp,betas[-1])
    p_observation = np.sum(p_temp)
    return p_observation,betas
def stateComputeOne(alphas,betas,t,i):
    numerator = alphas[t-1][i-1]*betas[t-1][i-1]
    denominator = np.sum(np.multiply(alphas[t-1],betas[t-1]))
    gamma_t_i = numerator/denominator
    return gamma_t_i
def stateComputeTwo(alphas,betas,A,B,O,t,i):
    numerator = alphas[t-1][i-1]*A[i-1][i]*B[i][O[t]]*betas[t][i]
    denominator = 0.0
    for m in range(len(A)):
        for n in range(len(A)):
            denominator += alphas[t-1][m]*A[m][n]*B[n][O[t]]*betas[t][n]
    xi_t_ij = numerator/denominator
    return xi_t_ij

def expectationComput(i,j):

    pass

class HMM(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    set_state = ['box1','box2','box3']
    set_observation = ['red','white']
    pi = [0.2,0.4,0.4]
    A = [[0.5,0.2,0.3],
         [0.3,0.5,0.2],
         [0.2,0.3,0.5]]
    B = [[0.5,0.5],
         [0.4,0.6],
         [0.7,0.3]]
    Lambda = (pi,A,B)
    #seq_sta,seq_obs = generateObservation(set_state,set_observation,Lambda,3)
    seq_obs = [0,1,0]
    fp,alphas = forwardAlgorithm(Lambda,seq_obs)
    bp,betas = backwardAlgorithm(Lambda,seq_obs)
    gamma_t_i = stateComputeOne(alphas,betas,1,0)
    xi_t_ij = stateComputeTwo(alphas,betas,A,B,seq_obs,1,0)
    print(fp,bp)
    print(gamma_t_i)
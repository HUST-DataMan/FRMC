
"""
This is a fast and robust imputation software based on matrix completion, 
called FRMC, for single cell RNA-Seq data .
Version 1.0.0
"""


from __future__ import division
import numpy as np
import math as ma
import scipy.io as sio
import time
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd,svd_flip
import numpy.linalg as LA
import scipy.linalg as SLA
import scipy.sparse as ss
import matplotlib.pyplot as plt
import pandas as pd


#import evaluation_gendata as myutils

def nrmse(truth, pred):
    return np.linalg.norm(pred - truth, "fro") / np.linalg.norm(truth, "fro")

#input: Matrix M, tau
#output: D_tau(M)  U*diag_shrink_S*VT
def D_tau(M, tau=None, l=5):

    if not tau:
        tau = 5 * np.sum(M.shape) / 2
    #r is rank(M)
    r = 0
    sk = r + 1
    agl = 'arpack'
    #agl = 'lobpcg'

#    (U, S, VT) = svds(M, k=min(sk, min(M.shape) - 1),solver=agl)
#    S = S[::-1]
#    U, VT = svd_flip(U[:, ::-1], VT[::-1])
    
    OK=False
    while not OK:
    #np.min(S) >= tau
        #sk = sk + l
        (U, S, VT) = svds(M, k=min(sk, min(M.shape) - 1), solver=agl)
        S = S[::-1]
        U, VT = svd_flip(U[:, ::-1], VT[::-1])
        OK = (np.min(S) < tau) or (sk == np.min(M.shape))
        sk = np.min((sk+l, np.min(M.shape)))
        print("min S:")
        print(np.min(S))
        print("sk:")
        print(sk)
        print("tau in D_tau:")
        print(tau)
    shrink_S = np.maximum(S - tau, 0)
    r = np.count_nonzero(shrink_S)
    diag_shrink_S = np.diag(shrink_S)
    res = np.linalg.multi_dot([U, diag_shrink_S, VT])
    
    '''
    s_thresh = np.maximum(S - tau, 0)
    rank = (s_thresh > 0).sum()
    s_thresh = s_thresh[:rank]
    U_thresh = U[:, :rank]
    VT_thresh = VT[:rank, :]
    S_thresh = np.diag(s_thresh)
    #res = np.dot(U_thresh, np.dot(S_thresh, VT_thresh))
    del U
    del VT
    res = np.linalg.multi_dot([U_thresh, S_thresh, VT_thresh])
    '''
    return res

def D_tau_old(M, tau=None, l=5):

    if not tau:
        tau = 5 * np.sum(M.shape) / 2
    #r is rank(M)
    r = 0
    sk = r + 1
    agl = 'arpack'
    #agl = 'lobpcg'

    (U, S, VT) = svds(M, k=min(sk, min(M.shape) - 1),solver=agl)
    S = S[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])

    while np.min(S) >= tau:
        sk = sk + l
        (U, S, VT) = svds(M, k=min(sk, min(M.shape) - 1), solver=agl)
        S = S[::-1]
        U, VT = svd_flip(U[:, ::-1], VT[::-1])
        print("min S:")
        print(np.min(S))
        print("sk:")
        print(sk)
        print("tau in D_tau:")
        print(tau)    
    shrink_S = np.maximum(S - tau, 0)
    r = np.count_nonzero(shrink_S)
    diag_shrink_S = np.diag(shrink_S)
    res = np.linalg.multi_dot([U, diag_shrink_S, VT])
    
    '''
    s_thresh = np.maximum(S - tau, 0)
    rank = (s_thresh > 0).sum()
    s_thresh = s_thresh[:rank]
    U_thresh = U[:, :rank]
    VT_thresh = VT[:rank, :]
    S_thresh = np.diag(s_thresh)
    #res = np.dot(U_thresh, np.dot(S_thresh, VT_thresh))
    del U
    del VT
    res = np.linalg.multi_dot([U_thresh, S_thresh, VT_thresh])
    '''
    return res


def D_tau_rand(M, tau=None, l=5):

    if not tau:
        tau = 5 * np.sum(M.shape) / 2
    #r is rank(M)
    r = 0
    sk = r + 1
    
    (U, S, VT) = randomized_svd(
            M, n_components=min(sk, M.shape[1]-1), n_oversamples=20)

    while np.min(S) >= tau:
        sk = sk + l
        (U, S, VT) = randomized_svd(
            M, n_components=min(sk, M.shape[1]-1), n_oversamples=20)

    shrink_S = np.maximum(S - tau, 0)
    r = np.count_nonzero(shrink_S)
    diag_shrink_S = np.diag(shrink_S)
    res = np.linalg.multi_dot([U, diag_shrink_S, VT])

    return res

#input M:
#use full SVD to matrix
#ouput  D_tau(M)  U*diag_shrink_S*VT
def fullSVT(M, tau=None):

    if not tau:
        tau = 5 * np.sum(M.shape) / 2
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    # threshold
    ss = S - tau
    s2 = np.clip(ss, 0, max(ss))
    res = np.dot(U, np.dot(np.diag(s2), VT))
    return res


def D_svt(M, sk):

    agl = 'arpack'
    #agl = 'lobpcg'

    (U, S, VT) = svds(M, k=min(sk, min(M.shape) - 1),solver=agl)
    S = S[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])    
    
    #shrink_S = np.maximum(S - tau, 0)
    diag_shrink_S = np.diag(S)
    res = np.linalg.multi_dot([U, diag_shrink_S, VT])
    print('D_svt S shape')
    print(diag_shrink_S.shape)
    
    return res

def jaccard_idx(M1,threshold=0):
    # M is cells by genes
    M = M1.copy()
    M[ M>threshold ] = 1
    M[ M<=threshold ]= 0
    #C is cells by cells
    C = 1.0 - dist.squareform(dist.pdist(M, 'jaccard'))
    return C

def FRMC(D, mu, rho, method='D_tau'):

    # thresholds
    ep1 = 1.e-4
    ep2 = 1.e-3
    Dn = np.linalg.norm(D, 'fro')

    # projector matrix
    # 这里 1 表示 D 中对应位置是缺失值0；
    #    0 表示 D 中对应位置有值
    PP = (D == 0)
    P = PP.astype(np.float)

    # initialization
    m, n = np.shape(D)
    Y = np.zeros((m, n))
    Eold = np.zeros((m, n))

    # iteration
    for i in range(1, 1000):

        # compute SVD
        tmp = D - Eold + Y / mu

        tau = 1./mu
        print('tau is %f'%(tau))
        if method =='D_tau':
            print('*Using D_tau to compute SVT:')
        ##使用 Topk + 5 步长试探性截断
            A=D_tau(tmp, tau)

        elif method=='fullSVT':
            #print('*Using full SVD to compute SVT:')
            A = fullSVT(tmp, tau)
            '''TEST code
            U, S, V = np.linalg.svd(tmp, full_matrices=False)
            ss = S - (1. / mu)
            s2 = np.clip(ss, 0, max(ss))
            A = np.dot(U, np.dot(np.diag(s2), V))
            '''
        elif method =='D_tau_rand':
            print('*Using random D_tau to compute SVT:')
            A=D_tau_rand(tmp, tau)
        elif method == 'D_svt':
            print('*Using fix k to compute SVT:')
            A=D_svt(tmp, 10)

        else:
            raise ValueError("unknown method")

        # project
        Enew = P * (D - A + Y / mu)
        DAE = D - A - Enew
        Y += mu * DAE

        # check residual and (maybe) exit
        r1 = np.linalg.norm(DAE, 'fro')
        resi = r1 / Dn
        print(i, ' residual ', resi)
        if (resi < ep1):
            break

        # adjust mu-factor
        muf = np.linalg.norm((Enew - Eold), 'fro')
        fac = min(mu, ma.sqrt(mu)) * (muf / Dn)
        if (fac < ep2):
            mu *= rho

        # update E and go back
        Eold = np.copy(Enew)

    E = np.copy(Enew)
    return A, E

import scipy.spatial.distance as dist
# input M:  cells by genes mxn
# output M: mark true zeros as 1.e-5
def getDropout(M, t=0.2):
    # mark true zeros as 1.e-5 in M
    # M is cells by genes mxn
    # J is cells by cells mxm
    m, n = np.shape(M)
    # m cells; n genes

    M0 = M.copy()
    M0[M0 > 0] = 1
    M0[M0 <= 0] = 0
    # J is cells by cells [m,m]
    print("****calculating jaccard index between cell-paires ... ")
    J = 1.0 - dist.squareform(dist.pdist(M0, 'jaccard'))
    Jp=J.copy() #keep jaccard value
    J[J >= 0.5] = 1  # this cells are similar
    J[J < 0.5] = 0

    print("****finding true zeros or dropout zeros ... ")
    # J(mxm) * M(mxn)
    for i in range(m):
        # similar cells number
        p_down = np.sum(J[i])
        if p_down < 10:
            #降序
            J[i][ np.argsort(-Jp[i])[0:10] ] = 1
            #print("#test ", np.sum(J[i]))
            p_down = np.sum(J[i])

        for j in range(n):
            #gene
            p_up = np.sum(J[i] * M0[:, j])
            prob = p_up * 1.0 / p_down
            if prob <= t and M[i, j] == 0:  # the [cell,gene] is true zeros
                M[i, j] = 1.e-5
                #M[i,j] = np.random.rand()*1.e-5
               # print('p_up, p_down, prob, i, j', p_up, p_down, prob, i, j)

    print("****end find ...")
    return M


if __name__ == '__main__':
    # matrix completion problem
    import sys

    if len(sys.argv) != 3:
        print("input: <Normalized Matrix file(cell-by-gene)> <out_prefix>\n error")
        sys.exit(-1)

    Data = pd.read_csv(sys.argv[1],index_col=0)
    #Data_random = Data.sample(n=600, random_state=1, axis=0)
    out_name=sys.argv[2]    

    stime = time.time()
    #A1=np.array(Data_random)
    A1=np.array(Data)
    A=getDropout(A1,t=0.2)
    #A=np.array(Data)
    PP = (A>0)
    # mask matrix 0, 1
    P = PP.astype(np.float)
    # number of non-zero elements
    Omega = np.count_nonzero(P)

    # 模拟需要impute的data matrix  need to be imputed matrix
    # incomplete matrix
    D = P * A
    m,n=D.shape
    fratio = float(Omega) / (m * n)
    print('fill ratio ', fratio)

    # initialize parameters
    mu = 1. / np.linalg.norm(D, 2)
    rho = 1.2172 + 1.8588 * fratio

    # call FRMC-algorithm
    #AA, EE = FRMC(D, mu, rho, method='D_tau_rand')
    #AA, EE = FRMC(D, mu, rho, method='D_tau')
    AA, EE = FRMC(D, mu, rho, method='fullSVT') 
    # compare
    elapsed_seconds = time.time() - stime
#    mse = nrmse(AA, A)
    print('\n')
    print('Running time seconds')
    print(elapsed_seconds)
    print('\n')
    
    from datetime import *
    AA[AA<=1e-4]=0
    DF_AA_2=pd.DataFrame(AA.copy(), index = Data.index, columns=Data.columns )
    #DF_AA_2=pd.DataFrame(AA.copy(), index = Data_random.index, columns=Data_random.columns )
    outfile = out_name+"imputed"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv"
    DF_AA_2.to_csv(outfile)

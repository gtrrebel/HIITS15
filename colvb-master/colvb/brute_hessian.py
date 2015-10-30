import numpy as np
from scipy.linalg import block_diag

def block_dia(C, N):
    return block_diag(*((np.ones((N, N))[:,:,None]*C[None,:]).T))

def var_change(G, H, phis):
    N, K = G.shape
    A = np.zeros((N, K, N, K))
    #First term
    #1
    for n1 in xrange(N):
        for k1 in xrange(K):
            A[n1][k1][n1][k1] += G[n1][k1]*phis[n1][k1]
    #2 & 3
    for n1 in xrange(N):
        for k1 in xrange(K):
            for k2 in xrange(K):
                A[n1][k1][n1][k2] += -(G[n1][k1]+G[n1][k2])*phis[n1][k1]*phis[n1][k2]
    #4
    for n1 in xrange(N):
        s1 = np.dot(phis[n1],G[n1])
        for k1 in xrange(K):
            A[n1][k1][n1][k1] = -phis[n1][k1]*s1
    #5
    for n1 in xrange(N):
        s1 = np.dot(phis[n1],G[n1])
        for k1 in xrange(K):
            for k2 in xrange(K):
                A[n1][k1][n1][k2] += 2*phis[n1][k1]*phis[n1][k2]*s1
    #Second term
    print A
    #1
    for n1 in xrange(N):
        for k1 in xrange(K):
            for n2 in xrange(N):
                for k2 in xrange(K):
                    A[n1][k1][n2][k2] += H[n1][k1][n2][k2]*phis[n1][k1]*phis[n2][k2]
    #2 & 3
    for n1 in xrange(N):
        for k1 in xrange(K):
            for n2 in xrange(N):
                s1 = np.dot(phis[n2],H[n1][k1][n2]) + np.dot(phis[n1],H[n2][k2][n1])
                for k2 in xrange(K):
                    A[n1][k1][n2][k2] -= s1*phis[n1][k1]*phis[n2][k2]
    #4
    for n1 in xrange(N):
        for n2 in xrange(N):
            s1 = np.dot(phis[n1],np.dot(H[n1,:,n2,:], phis[n2].T))
            for k1 in xrange(K):
                for k2 in xrange(K):
                    A[n1][k1][n2][k2] += s1*phis[n1][k1]*phis[n2][k2]
    return A.flatten().reshape((N*K,N*K))
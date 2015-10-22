import numpy as np
from scipy.linalg import block_diag

def block_dia(C, N):
    return block_diag(*((np.ones((N, N))[:,:,None]*C[None,:]).T))

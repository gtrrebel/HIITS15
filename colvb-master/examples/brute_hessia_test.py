import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from brute_hessian import var_change
import numpy as np

def test1():
	N, K = 2, 2
	a,b,c,d = 0.4, 0.6, 0.3, 0.7
	G = np.array([[c, d],[a, b]])
	H = np.zeros((N,K,N,K))
	H[1][0][0][0] = 1
	H[0][0][1][0] = 1
	H[0][1][1][1] = 1
	H[1][1][0][1] = 1
	phis = np.array([[a,b],[c,d]])
	return var_change(G,H,phis)

def test2():
	N, K = 1, 2
	a,b = 0.4, 0.6
	G = np.array([[2, 2]])
	H = np.zeros((N,K,N,K))
	H[0][0][0][0] = 2
	H[0][0][0][1] = 2
	H[0][1][0][0] = 2
	H[0][1][0][1] = 2
	phis = np.array([[a,b]])
	return var_change(G,H,phis)

print test2()
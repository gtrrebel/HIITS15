import numpy as np
import sys
import pylab as pb
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
from signlab2 import signlab2

def shuffle(A):
	return A

dia = [0.6, 0.3, 0.05]
A = np.diag(dia)

lab = signlab2()

print lab.chebyshev_index(A)

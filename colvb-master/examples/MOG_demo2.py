# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import time
real_time = time.time()
import numpy as np
import pylab as pb
import sys
sys.path.append('~/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('~/Windows/Desktop/hiit/HIITS15/colvb-master')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
#sys.path.append('..')
#sys.path.append('../colvb')

from MOG2 import MOG2
from vis1 import vis1

pb.ion()

np.random.seed(0)

#fetch input
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_input/MOG_demo2.py/1/second_test_input.txt'
inp = open(filename)
input_list = []
for line in inp:
    input_list.append(int(line.split()[0]))

inp.close()
Nclust, dim, Nmin, Nmax, MeansCoeff, Nrestarts, Nclust = input_list

#make some Data which appears in clusters:
Ndata = np.random.randint(Nmin, Nmax, Nclust)
means = np.random.randn(Nclust, dim)*MeansCoeff
aa = [np.random.randn(dim, dim+1) for i in range(Nclust)]
Sigmas = [np.dot(a, a.T) for a in aa]
X = np.vstack([np.random.multivariate_normal(mu, cov, (n,)) for mu, cov, n in zip(means, Sigmas, Ndata)])/100


m = MOG2(X, Nclust, prior_Z='DP')

#Print Stats
print 'MOG_demo2.py'
print filename
print 
print'stats:'
print 'N: ', m.N, ' K: ' , m.K, ' D: ', m.D
print 'Nclust: ', Nclust, ' dim: ', dim
print 'Nmin: ', Nmin, ' Nmax: ', Nmax
print 'MeansCoeff: ', MeansCoeff
print 'Nrestarts: ', Nrestarts
print 'NClust: ', Nclust
print

plotstart = 3

#starts = [np.random.randn(m.N*m.K) for i in range(Nrestarts)]

#The steak
from scipy.cluster import vq
starts = []
for i in range(Nrestarts):
    means = X[np.random.permutation(X.shape[0])[:Nclust]]
    dists = np.square(X[:,:,None]-means.T[None,:,:]).sum(1)
    starts.append(dists)

#mehtods: 'steepest', 'PR', 'FR', 'HS'
main_time = time.time()
for method in ['steepest']:
    for i in xrange(10):
        m.optimize(method=method, maxiter=1e4)


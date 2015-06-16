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
#sys.path.append('..')
#sys.path.append('../colvb')

from MOG2 import MOG2

pb.ion()

np.random.seed(0)

#make some Data which appears in clusters:
Nclust = 5# orig = 15
dim = 2# orig = 2
Nmin =3# orig = 25
Nmax = 5# orig = 50
Ndata = np.random.randint(Nmin, Nmax, Nclust)
MeansCoeff = 2 #orig = 5
means = np.random.randn(Nclust, dim)*MeansCoeff #orig = *5
aa = [np.random.randn(dim, dim+1) for i in range(Nclust)]
Sigmas = [np.dot(a, a.T) for a in aa]
X = np.vstack([np.random.multivariate_normal(mu, cov, (n,)) for mu, cov, n in zip(means, Sigmas, Ndata)])/100
Nrestarts= 3# orig = 3
Nclust = 5# orig = 15


m = MOG2(X, Nclust, prior_Z='DP')

#Print Stats
print 'MOG_demo2.py'
print 
print'stats:'
print 'N: ', m.N, ' K: ' , m.K, ' D: ', m.D
print 'Nclust: ', Nclust, ' dim: ', dim
print 'Nmin: ', Nmin, ' Nmax: ', Nmax
print 'MeansCoeff: ', MeansCoeff
print 'Nrestarts: ', Nrestarts
print 'NClust: ', Nclust
print

#Make functions
start_time = time.time()
m.makeFunctions()
print 'Theano-function compilation time: ', '%s seconds' % (time.time() - start_time)
print

#starts = [np.random.randn(m.N*m.K) for i in range(Nrestarts)]

#The steak
from scipy.cluster import vq
starts = []
for i in range(Nrestarts):
    means = X[np.random.permutation(X.shape[0])[:Nclust]]
    dists = np.square(X[:,:,None]-means.T[None,:,:]).sum(1)
    starts.append(dists)

main_time = time.time()
for method in ['steepest', 'PR', 'FR', 'HS']:
    for st in starts:
	print 'Start\nMethod used: ', method
	start_time = time.time()
        m.set_vb_param(st)
        m.optimize(method=method, maxiter=1e4, opt=None, index='full', tests = None)
        print 'End\nMethod used: ', method, '\nRuntime: ', '%s seconds' % (time.time() - start_time) 

print
print 'Main time: ', '%s seconds' % (time.time() - main_time)
print 'Total time: ', '%s seconds' % (time.time() - real_time)

#m.plot_tracks()
#m.plot()




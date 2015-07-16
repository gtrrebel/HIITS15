# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import time
import numpy as np
import pylab as pb
import sys
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
from MOG2 import MOG2
from data_creator import data_creator

pb.ion()

np.random.seed(0)

road_gather = []
end_gather = []
basic_data = [5, 2, 10, 20, 5]
data_type = 'basic'

j = 1
while j < len(sys.argv):
    cmd = sys.argv[j]
    j += 1
    if cmd == 'r':
        k = int(sys.argv[j])
        while k > 0:
            k, j = k - 1, j + 1
            road_gather.append(sys.argv[j])
        j += 1
    elif cmd == 'e':
        k = int(sys.argv[j])
        while k > 0:
            k, j = k - 1, j + 1
            end_gather.append(sys.argv[j])
        j += 1
    elif cmd == 'b':
        k = 0
        data_type = 'basic'
        while k < 5:
            basic_data[k] = int(sys.argv[j])
            k, j = k + 1, j + 1
    else:
        print 'oh no', j
        break

eps = 1e-14
rest = 10
X, Nclust = data_creator.mog_basic_data(*basic_data)

m = MOG2(X, Nclust, prior_Z='symmetric')
m.runspec_set('eps', eps)
m.runspec_set('restarts', rest)
m.set_invests(road_gather= road_gather, end_gather=end_gather)

from scipy.cluster import vq
starts = []
for i in range(m.runspec_get('restarts')):
    means = X[np.random.permutation(X.shape[0])[:Nclust]]
    dists = np.square(X[:,:,None]-means.T[None,:,:]).sum(1)
    starts.append(dists)

#methods: 'steepest', 'PR', 'FR', 'HS'
for method in m.runspec_get('methods'):
    for st in starts:
        m.set_vb_param(st)
        m.optimize(method=method, maxiter=1e4)
        m.end()
        m.end_print()

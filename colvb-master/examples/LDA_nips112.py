# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)
import sys
sys.path.append('../colvb')
import numpy as np
import time
import matplotlib as mlp
mlp.use('pdf')
import pylab as pb
from LDA3 import LDA3
from data_collector import data_collector

N_TOPICS = 5
NDOCS = 10
NWORDS = 50
DOCUMENT_LENGTH = 25

docs, vocab = data_collector.nips_data(N_TOPICS, NDOCS, NWORDS, DOCUMENT_LENGTH)

#build the model and optimize in parallel
m = LDA3(docs,vocab,N_TOPICS,alpha_0 = 200.)
m.optimize(hessian_freq=10000)

print 'size: ', len(m.get_vb_param()), '\n', \
            'optimize_time: ', m.optimize_time, '\n', \
            'hessian_time: ', m.hessian_time, ' - ', (100*m.hessian_time/m.optimize_time), '%,\n', \
            'pack_time: ', m.pack_time, ' - ', (100*m.pack_time/m.optimize_time), '%,\n', \
            'others: ', (m.optimize_time - m.hessian_time - m.pack_time), \
            ' - ', (100*(m.optimize_time - m.hessian_time - m.pack_time)/m.optimize_time), '%'
print 'theano-compilation time:', m.theanotime
'''
tt = time.time()
eigvals = m.eigenvalues()
print 'spectrum length: ', max(eigvals) - min(eigvals)
print time.time() - tt
'''
print 'bound: ', m.bound()

#plot the optimisation tracks for the first model
m.plot_tracks()
pb.savefig('optimisation.pdf')
pb.close()

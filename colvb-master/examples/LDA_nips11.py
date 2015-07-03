# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import matplotlib as mlp
mlp.use('pdf')
import sys
sys.path.append('../colvb')
from LDA import LDA
from IPython import parallel

cluster_err =\
"""
To run these experiments, you'll need to run an instance of ipcluster.
This will make use of as many cores as you let it. in another terminal, run\n
ipcluster start\n
before running this file. Thanks!
"""

#set up clients
try:
    c = parallel.Client()
except:
    print cluster_err
    sys.exit()
dv = c.direct_view()
with dv.sync_imports():
    import sys
    import matplotlib
dv.execute("sys.path.append('../colvb')",block=True)
dv.execute("matplotlib.use('pdf')",block=True)
with dv.sync_imports():
    from LDA import LDA

def three_opts(m):
    x = m.get_vb_param().copy()
    m.optimize('steepest',maxiter=1e4,gtol=1e-4)
    m.set_vb_param(x.copy())
    m.optimize('HS',maxiter=1e4,gtol=1e-4)
    m.set_vb_param(x.copy())
    m.optimize('FR',maxiter=1e4,gtol=1e-4)
    return m

N_TOPICS = 5
NDOCS = 20
NWORDS = 100

#load nips data
f = file('../data/nips11data/nips11_corpus')
s = f.read().split('\n')
f.close()
vocab = np.array(s[0].split())
docs = [np.asarray([int(i) for i in l.split()[1:]],dtype=np.int) for l in s[1:NDOCS]]

#cut out troublesome doc which didn;t parse (ms word user?)
docs = [d for d in docs if d.size > 1000]

#truncate the vocabulary
docs_ = np.hstack(docs)
print docs_.size
wordcounts = np.array([np.sum(docs_==i) for i in range(docs_.max())],dtype=np.int)
allowed_vocab = np.argsort(wordcounts)[::-1][:NWORDS]
docs = [np.asarray([w for w in doc if w in allowed_vocab],dtype=np.int) for doc in docs]
docs_ = np.hstack(docs)
print docs_.size

#build the model and optimize in parallel
m = LDA(docs,vocab,N_TOPICS,alpha_0 = 200.)
mm = [m.copy() for i in range(2)]
[m.randomize() for m in mm]
r = dv.map(three_opts,mm,block=True)

#plot the optimisation tracks for the first model
m[0].plot_tracks()
pb.savefig('optimisation.pdf')
pb.close()
[m.E_time_to_opt() for m in mm]






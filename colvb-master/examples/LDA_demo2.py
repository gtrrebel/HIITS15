# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator

road_gather = []
end_gather = []
basic_data = [3, 10, 5, 2]
nips_data = [5, 10, 10, 10]
data_type = 'nips'

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
        while k < 4:
            basic_data[k] = int(sys.argv[j])
            k, j = k + 1, j + 1
    elif cmd == 'n':
        k = 0
        data_type = 'nips'
        while k < 4:
            nips_data[k] = int(sys.argv[j])
            k, j = k + 1, j + 1
    else:
        print 'oh no', j
        break

alpha_0 = 1
eps = 1e-14

if data_type == 'basic':
    docs, vocab = data_creator.basic_data(*basic_data)
    N_TOPICS = basic_data[0]*basic_data[3]
elif data_type == 'nips':
    docs, vocab = data_creator.nips_data(*nips_data)
    N_TOPICS = nips_data[0]
    alpha_0 = 200


m = LDA3(docs,vocab,N_TOPICS,alpha_0=alpha_0)
m.runspec_set('eps', eps)
m.set_invests(road_gather= road_gather, end_gather=end_gather)

for method in m.runspec_get('methods'):
    for i in range(m.runspec_get('restarts')):
        m.optimize(method=method, maxiter=1e4)
        #print m.get_sums(1)/m.N
        #print m.get_param().reshape((m.D, m.N, m.K))
        #m.print_topics()
        m.end()
        m.end_print()
        m.new_param()

m.end_basic_plots

if m.runspec_get('orig_track_display'):
    pb.figure()
    m.plot_tracks()

#display learned topics
def plot_inferred_topics():
    nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
    pb.figure()
    for i,beta in enumerate(m.beta_p):
        pb.subplot(nrow,ncol,i+1)
        pb.imshow(beta.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
        pb.xticks([])
        pb.yticks([])

if m.runspec_get('orig_learned_topics'):
    plot_inferred_topics()
    pb.suptitle('inferred topics')
    pb.show()

#plot true topics
if m.runspec_get('orig_true_topics'):
    nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
    pb.figure()
    for i,topic in enumerate(topics):
        pb.subplot(nrow,ncol,i+1)
        pb.imshow(topic.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
        pb.xticks([])
        pb.yticks([])
    pb.suptitle('true topics')

print 'done'

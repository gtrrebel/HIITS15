# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator

#fetch input
if len(sys.argv) > 1:
    if 'f' == sys.argv[1]:
        filename = sys.argv[1]
        inp = open(filename)
        input_list = []
        for line in inp:
            input_list.append(int(line.split()[0]))
        inp.close()
        invest = None
    else:
        invest = sys.argv[1]
        input_list = [int(s) for s in sys.argv[2:]]
else:
    filename = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_input/LDA_demo2.py/1/first_test_input.txt'
    inp = open(filename)
    input_list = []
    for line in inp:
        input_list.append(int(line.split()[0]))
    inp.close()
    invest = None

WORDSIZE, N_DOCS, DOCUMENT_LENGTH, N_TOPIC_COEFF = input_list

print WORDSIZE, N_DOCS, DOCUMENT_LENGTH, N_TOPIC_COEFF

docs, vocab = data_creator.basic_data(WORDSIZE, N_DOCS, DOCUMENT_LENGTH, N_TOPIC_COEFF)
N_TOPICS = WORDSIZE*N_TOPIC_COEFF

eps = 1e-14

m = LDA3(docs,vocab,N_TOPICS)
m.runspec_set('eps', eps)
if invest != None:
    m.set_invests(end_gather=[invest])

for method in m.runspec_get('methods'):
    for i in range(m.runspec_get('restarts')):
        m.optimize(method=method, maxiter=1e4)
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

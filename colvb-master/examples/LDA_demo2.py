# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from vis1 import vis1
from vis2 import vis2

#fetch input
if len(sys.argv) > 1:
    if 'f' == sys.argv[1]:
        filename = sys.argv[1]
        inp = open(filename)
        input_list = []
        for line in inp:
            input_list.append(int(line.split()[0]))
        inp.close()
    else:
        input_list = [int(s) for s in sys.argv[1:]]
else:
    filename = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_input/LDA_demo2.py/1/first_test_input.txt'
    inp = open(filename)
    input_list = []
    for line in inp:
        input_list.append(int(line.split()[0]))
    inp.close()


WORDSIZE, N_DOCS, DOCUMENT_LENGTH, N_TOPIC_COEFF = input_list

restarts = 1
plotstart = 3
methods = ['steepest']
plot_specs = [('iter', 'index'), ('iter', 'bound')]
eps = 1e-14
finite_difference_check = False
hessian_freq = 1

runtime_distribution = False
distances_travelled = False
eigenvalue_histograms = False
basic_plots = False
spectrum_length = True
print_convergence = False

orig_document_display = False
orig_track_display = False
orig_learned_topics = False
orig_true_topics = False


print WORDSIZE, N_DOCS, DOCUMENT_LENGTH, N_TOPIC_COEFF

#generate some documents
DOCUMENT_LENGTHS = [np.random.randint(DOCUMENT_LENGTH, DOCUMENT_LENGTH + 1) for i in range(N_DOCS)]
N_TOPICS = WORDSIZE*N_TOPIC_COEFF # topics are horizontal or vertical bars

#here's the vocabulary
V = WORDSIZE**2
if WORDSIZE==2:
    vocab = np.array([u'\u25F0',u'\u25F3',u'\u25F1',u'\u25F2'],dtype="<U2")
else:
    vocab_ = [np.zeros((WORDSIZE,WORDSIZE)) for v in range(V)]
    [np.put(v,i,1) for i,v in enumerate(vocab_)]
    vocab = np.empty(len(vocab_),dtype=np.object)
    for i,v in enumerate(vocab_):
        vocab[i] = v

#generate the topics
topics = [np.zeros((WORDSIZE,WORDSIZE)) for i in range(N_TOPICS)]
for i in range(WORDSIZE):
    topics[i][:,i] = 1
    topics[i+WORDSIZE][i,:] = 1
topics = map(np.ravel,topics)
topics = map(lambda x: x/x.sum(),topics)

#if the docs are 2x2 square, you'll have as many topics as vocab, which won't work:
if WORDSIZE==2:
    topics = topics[:2]
    N_TOPICS = 2

#generate the documents
docs = []
doc_latents = []
doc_topic_probs = []
for d in range(N_DOCS):
    topic_probs = np.random.dirichlet(np.ones(N_TOPICS)*0.8)
    latents = np.random.multinomial(1,topic_probs,DOCUMENT_LENGTHS[d]).argmax(1)
    doc_latents.append(latents)
    doc_topic_probs.append(topic_probs)
    docs.append(np.array([np.random.multinomial(1,topics[i]).argmax() for i in latents]))
docs_visual = [np.zeros((WORDSIZE,WORDSIZE)) for d in range(N_DOCS)]
for d,dv in zip(docs, docs_visual):
    for w in d:
        dv.ravel()[w] += 1

#display the documents
if orig_document_display:
    nrow=ncol= np.ceil(np.sqrt(N_DOCS))
    vmin = np.min(map(np.min,docs_visual))
    vmax = np.max(map(np.max,docs_visual))
    for d,dv in enumerate(docs_visual):
        pb.subplot(nrow,ncol,d+1)
        pb.imshow(dv,vmin=vmin,vmax=vmax,cmap=pb.cm.gray)
        pb.xticks([])
        pb.yticks([])
    pb.suptitle('the "documents"')

m = LDA3(docs,vocab,N_TOPICS, eps=eps)
x = m.get_vb_param().copy()
v1 = vis1()
v2 = vis2()
m.makeFunctions()

def runprinter(m, v1, v2):
    if runtime_distribution:
        print 'size: ', len(m.get_vb_param()), '\n', \
            'optimize_time: ', m.optimize_time, '\n', \
            'hessian_time: ', m.hessian_time, ' - ', (100*m.hessian_time/m.optimize_time), '%,\n', \
            'pack_time: ', m.pack_time, ' - ', (100*m.pack_time/m.optimize_time), '%,\n', \
            'others: ', (m.optimize_time - m.hessian_time - m.pack_time), \
            ' - ', (100*(m.optimize_time - m.hessian_time - m.pack_time)/m.optimize_time), '%'
    if eigenvalue_histograms:
        pb.figure()
        v2.eigenvalue_histogram(m.eigenvalues())
        pb.xlabel(m.bound())
        pb.show()
    if spectrum_length:
        eigvals = m.eigenvalues()
        print max(eigvals) - min(eigvals)
    if distances_travelled:
        print 'distance_travelled: ', m.distance_travelled, ' distance_from_start: ', m.how_far()

for method in methods:
    for i in range(restarts):
        m.optimize(method=method, maxiter=1e4, opt= None, index='pack', hessian_freq=hessian_freq, \
            print_convergence=print_convergence)
        runprinter(m, v1, v2)
        v1.stack(m.info[plotstart:])
        m.new_param()

if basic_plots:
    for spec in plot_specs:
        pb.figure()
        v1.plot_stack(spec[0], spec[1])
        pb.show()

if orig_track_display:
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

if orig_learned_topics:
    plot_inferred_topics()
    pb.suptitle('inferred topics')
    pb.show()

#plot true topics
if orig_true_topics:
    nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
    pb.figure()
    for i,topic in enumerate(topics):
        pb.subplot(nrow,ncol,i+1)
        pb.imshow(topic.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
        pb.xticks([])
        pb.yticks([])
    pb.suptitle('true topics')

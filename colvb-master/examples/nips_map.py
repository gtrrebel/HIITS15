import numpy as np
import pylab as pb
from scipy import linalg
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator
from graph_vis import graph_vis
from label_switcher import label_switcher
from input_parser import input_parser
import distlab

method, run_count, basic_data, nips_data, data_type = input_parser.LDA_map_parse(sys.argv)

if data_type == 'basic':
    docs, vocab = data_creator.basic_data(*basic_data)
    N_TOPICS = basic_data[0]*basic_data[3]
elif data_type == 'nips':
    docs, vocab = data_creator.nips_data(*nips_data)
    N_TOPICS = nips_data[0]

maxs = []
m = LDA3(docs,vocab,N_TOPICS, alpha_0=200.0)

for i in xrange(run_count):
    print i
    m.new_param()
    m.optimize(method=method, maxiter=1e4)
    maxs.append((m.bound(), m.get_param()))

switch = label_switcher([ma[1].reshape(m.D, m.N, m.K) for ma in maxs])
#switch.switch_labels()
for i in xrange(run_count):
    maxs[i][1].flatten()  

maxs = distlab.merge(maxs)
distlab.print_all_dists(maxs)

graph_vis.draw(distlab.tuple_dists(maxs), distlab.all_bounds(maxs), distlab.all_counts(maxs))
'''
print min_max_dist(maxs)
for f in sorted_dists(maxs):
    print f
'''
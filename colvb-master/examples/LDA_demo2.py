# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator
from input_parser import input_parser

eps, rest, road_gather, end_gather, basic_data, nips_data, data_type = input_parser.basic_LDA_parse(sys.argv)

if data_type == 'basic':
    docs, vocab = data_creator.basic_data(*basic_data)
    N_TOPICS = basic_data[0]*basic_data[3]
    alpha_0 = 1
elif data_type == 'nips':
    docs, vocab = data_creator.nips_data(*nips_data)
    N_TOPICS = nips_data[0]
    alpha_0 = 200

m = LDA3(docs,vocab,N_TOPICS,alpha_0=alpha_0)
m.runspecs['basics']['eps'] = eps
m.runspecs['basics']['restarts'] = rest
m.set_invests(road_gather= road_gather, end_gather=end_gather)

for method in m.runspecs['basics']['methods']:
    for i in range(m.runspecs['basics']['restarts']):
        m.optimize(method=method, maxiter=1e4)
        m.end()
        m.end_print()
        m.new_param()

m.end_basic_plots()
m.end_display()

print 'done'

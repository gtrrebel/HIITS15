import sys
import numpy as np
sys.path.append('../colvb')
from LDA_creator import *
from LDA3 import LDA3
from LDA_demo4 import *
from LDA_plotter import *
from LDAtests import *

def test(alpha, beta, Ns, K, method, restarts = 10):
	docs = np.array(create_data(alpha, beta, Ns))
	voc = np.arange(len(beta))
	m = LDA3(docs, voc, K)
	res = run([m], end_gather=['bound', 'reduced_dimension', 'optimizetime'])
	for docs in res:
		for dic in docs[2]:
			dic['index'] = 0
	return res

def test2(alpha, Ko, beta, V, DN, DL, K, method, restarts = 10):
	return test([alpha for _ in xrange(Ko)], [beta for _ in xrange(V)], [DL for _ in xrange(DN)], K, method, restarts)

def test2_p(alpha, Ko, beta, V, DN, DL, K, method, restarts = 10):
	res = test2(alpha, Ko, beta, V, DN, DL, K, method, restarts)
	print_bhr_lib(res)

def plot_d_test(alpha, n):
	plot_d(create_distribution([alpha for _ in xrange(n)]))


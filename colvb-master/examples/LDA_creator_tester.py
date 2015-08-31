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
	plot_bhr_lib(res)

def test2s(alpha, Ko, beta, V, DN, DLs, K, method, restarts = 10):
	return [test2(alpha, Ko, beta, V, DN, DL, K, method, restarts)[0] for DL in DLs]

def test2s_p(alpha, Ko, beta, V, DN, DLs, K, method, restarts = 10):
	ress = test2s(alpha, Ko, beta, V, DN, DLs, K, method, restarts = 10)
	print_bhr_lib(ress)
	plot_bhr_lib(ress)

def plot_d_test(alpha, n):
	plot_d(create_distribution([alpha for _ in xrange(n)]))

def test3(topic_params, word_params, K0, V, Ns = [100 for i in xrange(20)], K = None, method = 'steepest', restarts = 10):
	if K == None:
		K = K0
	docs = np.array(create_gaussian_data(topic_params, word_params, K0, V, Ns = [100 for i in xrange(100)]))
	voc = np.arange(V)
	m = LDA3(docs, voc, K, make_fns = False)
	res = run([m], end_gather=['bound', 'reduced_dimension', 'optimizetime', 'get_vb_param', 'voc_size'], methods=[method])
	for docs in res:
		for dic in docs[2]:
			dic['index'] = 0
	return res[0]

def test4(topic_params, word_params, KVs):
	return [test3(topic_params, word_params, KV[0], KV[1]) for KV in KVs]

def test4_p(topic_params, word_params, KVs):
	res = test4(topic_params, word_params, KVs)
	plot_bhr_lib(res)
	return res

def test4s(alpha, Ko, beta, V, DN, DLs, K, method, restarts = 10):
	return [test4(alpha, Ko, beta, V, DN, DL, K, method, restarts)[0] for DL in DLs]

def test4s_p(alpha, Ko, beta, V, DN, DLs, K, method, restarts = 10):
	ress = test4s(alpha, Ko, beta, V, DN, DLs, K, method, restarts = 10)
	print_bhr_lib(ress)
	plot_bhr_lib_voc(ress)

def plot_d_test(alpha, n):
	plot_d(create_distribution([alpha for _ in xrange(n)]))

def stdvsdim(outs):
	return [np.std([dic['bound'] for dic in out[2]])/out[2][0]['reduced_dimension'] for out in outs]

def lenvsdim(outs):
	return [(max([dic['bound'] for dic in out[2]]) - min([dic['bound'] for dic in out[2]]))/out[2][0]['reduced_dimension'] for out in outs]

def KVtest(topic_params, word_params, K0, V):
	return int(np.log(stdvsdim(test3(topic_params, word_params, K0, V))))

def KVtests(topic_params, word_params, K0, V):
	for k in xrange(1, K0+1):
		for v in xrange(1, V + 1):
			print int(np.log(lenvsdim(test3(topic_params, word_params, k, v)))),
		print

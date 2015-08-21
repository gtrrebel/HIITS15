from LDA_demo4 import *
from LDA_plotter import *
from signlab3 import *


def bounds(args = [''], restarts = 10):
	ms = init(args, make_fns = False)
	outs = run(ms, restarts = restarts, end_gather=['bound'])
	bounts = [[out['bound'] for out in ous[2]] for ous in outs]
	dimensions = [(m.K -1)*m.N*m.D for m in ms]
	return bounts, dimensions

def bound_runtimes(args = [''], restarts = 10):
	ms = init(args, make_fns = False)
	return run(ms, restarts = restarts, end_gather=['bound', 'reduced_dimension', 'optimizetime'])

def plot_bounds(args = [''], restarts = 10):
	bound_plot(*bounds(args, restarts))

def plot_bound_runtimes(args =  [''], restarts = 10):
	bound_optimizetime_plot(bound_runtimes(args, restarts))

def doclenboundplot(n, restarts = 10, other= '5 10 10'):
	plot_bounds([(other + ' {0}').format(5*i) for i in xrange(2, n)], restarts=restarts)

def doclenboundruntimeplot(n, restarts = 10, other= '5 10 10'):
	plot_bound_runtimes([(other + ' {0}').format(5*i) for i in xrange(2, n)], restarts=restarts)

def boundshessians(args= [''], restarts = 10):
	ms = init(args, make_fns = False)
	outs = run(ms, restarts = restarts, end_gather=['bound', 'return_hessian'])
	hessians = [[out['return_hessian'] for out in ous[2]] for ous in outs]
	bounts = [[out['bound'] for out in ous[2]] for ous in outs]
	dimensions = [(m.K -1)*m.N*m.D for m in ms]
	data = zip(hessians, dimensions, bounts)
	data = [zip(hess, bouns - min(bouns), [dm]*restarts) for (hess, dm, bouns) in data]
	data = [sorted(dat, cmp=compare_boundshessians) for dat in data]
	return data

def plotboundshessians(args = [''], restarts = 10, indeces=1):
	data = boundshessians(args, restarts)
	good, neutral, bad = [], [], []
	indeces = min(restarts/2, indeces)
	for dat in data:
		for d in dat[indeces : -indeces]:
			neutral.append((d[1], d[2]))
		for d in dat[:indeces]:
			if largest_eigenvalue(d[0]):
				bad.append((d[1], d[2]))
			else:
				good.append((d[1], d[2]))
		for d in dat[-indeces:]:
			if largest_eigenvalue(d[0]):
				bad.append((d[1], d[2]))
			else:
				good.append((d[1], d[2]))
	hessian_bound_plot(good, bad, neutral)

def doclenhesboundplot(n, restarts=10, other = '5 10 10', indeces = 1):
	plotboundshessians([(other + ' {0}').format(5*i) for i in xrange(2, n)], restarts=restarts, indeces=indeces)

def compare_boundshessians(bh1, bh2):
	a = bh1[1] - bh2[1]
	if a > 0:
		return 1
	elif a < 0:
		return -1
	else:
		return 0

def index_check(args = [''], restarts = 10):
	ms = init(args, make_fns = False)
	outs = run(ms, restarts = restarts, end_gather=['power_largest', 'epsilon_positive'])
	for dat in outs:
		for d in dat[2]:
			print d['power_largest'], d['epsilon_positive']

def index_checks(n, restarts = 1, other='5 10 10'):
	index_check(args = [(other + ' {0}').format(i) for i in xrange(5, n)], restarts = restarts)

def bhr_lib(args = [''], restarts = 10):
	ms = init(args, make_fns = False)
	outs = run(ms, restarts = restarts, end_gather=['bound', 'return_m', 'get_vb_param', 'reduced_dimension', 'optimizetime'])
	for out in outs:
		for dic in out[2]:
			dic['index'] = 0
	return outs

def calc_dic(dic):
	m = dic['return_m']
	m.set_vb_param(dic['get_vb_param'])
	m.hessian_calc = False
	dic['index'] = largest_eigenvalue(m.get_hessian())

def calc_max(dics):
	i = 0
	for j in xrange(len(dics)):
		if dics[j]['bound'] > dics[i]['bound']:
			i = j
	calc_dic(dics[i])

def calc_min(dics):
	i = 0
	for j in xrange(len(dics)):
		if dics[j]['bound'] < dics[i]['bound']:
			i = j
	calc_dic(dics[i])

def calc_all(dics):
	for dic in dics:
		calc_dic(dic)

def print_bhr_lib(outs):
	print str_bhr_lib(outs)

def str_bhr_lib(outs):
	bhr_lib_str = ''
	for out in outs:
		minbound = min(dic['bound'] for dic in out[2])
		bhr_lib_str += str(out[2][0]['reduced_dimension']) + '\n'
		for i in xrange(len(out[2])):
			bhr_lib_str += str(i) + ': ' +out[2][i]['method']  + ' ' + \
			'{0:10}'.format('%.2e' % (out[2][i]['bound'] - minbound)) + ' ' + \
			str(out[2][i]['index']) + '\n'
	return bhr_lib_str

def index_tests(args = [''], restarts = 10):
	ms = init(args, make_fns = False)
	outs =  run(ms, restarts = restarts, end_gather = ['bound', 'return_m', 'get_vb_param', 'reduced_dimension', 'optimizetime'], methods=['steepest', 'FR'])
	for out in outs:
		for dic in out[2]:
			dic['index'] = 0
	return outs


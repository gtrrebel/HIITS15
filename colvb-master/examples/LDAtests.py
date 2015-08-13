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

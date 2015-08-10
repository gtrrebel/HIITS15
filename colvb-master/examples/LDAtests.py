from LDA_demo4 import *
from LDA_plotter import *


def bounds(args = [''], restarts = 10):
	ms = init(args, make_fns = False)
	outs = run(ms, restarts = restarts, end_gather=['bound'])
	bounts = [[out['bound'] for out in ous[2]] for ous in outs]
	dimensions = [(m.K -1)*m.N*m.D for m in ms]
	return bounts, dimensions

def plot_bounds(args = [''], restarts = 10):
	bound_plot(*bounds(args, restarts))

def doclenboundplot(n, restarts = 10, other= '5 10 10'):
	plot_bounds([(other + ' {0}').format(5*i) for i in xrange(2, n)], restarts=restarts)
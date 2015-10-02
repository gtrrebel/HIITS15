import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from MOG2 import MOG2
from data_creator import data_creator
from input_parser import input_parser
import numpy as np

def init(args=[[]], make_fns = False, ukko = False):
	ms = []
	for arg in args:
		mog_data = arg
		if arg == []:
			mog_data = [10, 2, 10, 30, 5]
		X = data_creator.mog_basic_data(*mog_data)
		N_TOPICS = mog_data[0]
		alpha = 10
		m = MOG2(X,N_TOPICS, alpha=alpha, make_fns = make_fns)
		ms.append(m)
	return ms

def run(args, out='return', restarts= 10, end_gather = ['bound'], methods=['FR'], repeat = 0, length = 1, trust_count = 0):
	data = []
	for m in args:
		m.runspecs['basics']['restarts'] = restarts
		m.runspecs['basics']['methods'] = methods
		m.set_invests(road_gather= [], end_gather=end_gather)
		end_returns = []
		for method in m.runspecs['basics']['methods']:
			for i in xrange(m.runspecs['basics']['restarts']):
				m.new_param()
				if trust_count > 0:
					m.trust_region_optimize(method=method, maxiter=1e4, trust_count=trust_count)
				else:
					m.optimize(method=method, maxiter=1e4)
				m.end()
				end_returns.append(m.end_return())
				if repeat > 0:
					diff, std, ite = do_repeat(m, repeat, length)
				else:
					diff, std, ite = 0, 0, 0
				end_returns[-1]['repeatdiff'] = diff
				end_returns[-1]['repeatstd'] = std
				end_returns[-1]['repeatite'] = ite 
		data.append( end_returns)
	if out == 'plot':
		for dat in data:
			plot(dat)
	elif out == 'save_plot':
		for dat in data:
			save_plot(dat)
	elif out == 'save_data':
		for dat in data:
			save_data(dat)
	elif out == 'return':
		return data

def do_repeat(m, repeat, length):
	starts = []
	ends = []
	iterations = []
	for i in xrange(repeat):
		m.random_jump(length)
		starts.append(m.bound())
		m.optimize()
		iterations.append(m.iteration)
		ends.append(m.bound())
	diff = sum(abs(starts[i] - ends[i]) for i in xrange(len(starts)))/len(starts)
	std = np.std(np.array(ends))
	ite = sum(iterations)/len(iterations)
	return diff, std, ite
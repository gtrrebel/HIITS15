import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from MOG2 import MOG2
from data_creator import data_creator
from input_parser import input_parser

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

def run(args, out='return', restarts= 10, end_gather = ['bound'], methods=['FR']):
	data = []
	for m in args:
		m.runspecs['basics']['restarts'] = restarts
		m.runspecs['basics']['methods'] = methods
		m.set_invests(road_gather= [], end_gather=end_gather)
		end_returns = []
		for method in m.runspecs['basics']['methods']:
			for i in xrange(m.runspecs['basics']['restarts']):
				m.new_param()
				m.optimize(method=method, maxiter=1e4)
				m.end()
				end_returns.append(m.end_return())
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

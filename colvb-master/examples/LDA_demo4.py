import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator
from input_parser import input_parser
from LDA_plotter import *

def init(args=[''], make_fns = True, ukko = False):
	ms = []
	for arg in args:
		nips_data = input_parser.LDA_parse3(arg.split())
		docs, vocab = data_creator.nips_data(*nips_data, ukko = ukko)
		N_TOPICS = nips_data[0]
		alpha_0 = 200
		m = LDA3(docs,vocab,N_TOPICS,save_specs = [arg], alpha_0=alpha_0, make_fns = make_fns)
		ms.append(m)
	return ms

def run(args, out='return', restarts= 10, end_gather = ['bound', 'epsilon_positive'], methods=['FR'], trust_region =False, trust_count=10):
	data = []
	for m in args:
		m.runspecs['basics']['restarts'] = restarts
		m.runspecs['basics']['methods'] = methods
		m.set_invests(road_gather= [], end_gather=end_gather)
		save_specs = m.save_specs
		end_returns = []
		for method in m.runspecs['basics']['methods']:
			for i in range(m.runspecs['basics']['restarts']):
				if trust_region:
					m.trust_region_optimize(method=method, maxiter=1e4, trust_count=trust_count)
				else:
					m.optimize(method=method, maxiter=1e4)
				m.end()
				end_returns.append(m.end_return())
				m.new_param()
		data.append((end_gather, save_specs, end_returns))
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


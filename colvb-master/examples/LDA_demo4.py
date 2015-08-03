import numpy as np 
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator
from input_parser import input_parser
from label_switcher import label_switcher
from LDA_plotter import plot, save

ukko = False
out = 'print'

if ukko:
	results = "/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/"
else:
	results = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/'

def init(args):
	restarts, nips_data = input_parser.LDA_parse2(args)
	docs, vocab = data_creator.nips_data(*nips_data)
	N_TOPICS = nips_data[0]
	alpha_0 = 200
	end_gather = ['bound', 'epsilon_positive']
	m = LDA3(docs,vocab,N_TOPICS,alpha_0=alpha_0)
	m.runspecs['basics']['restarts'] = restarts
	m.runspecs['basics']['methods'] = ['steepest']
	m.set_invests(road_gather= [], end_gather=end_gather)
	return m

def run(m, out):
	end_gather = m.end_gather
	end_returns = []

	for method in m.runspecs['basics']['methods']:
		for i in range(m.runspecs['basics']['restarts']):
			m.optimize(method=method, maxiter=1e4)
			m.end()
			end_returns.append(m.end_return())
			m.new_param()

	data = {}
	for spec in end_gather:
		data[spec] = []

	for run in end_returns:
		for spec in run:
			data[spec[0]].append(spec[1])

	delta = 1e-2
	randomization = delta*np.random.randn((len(data[end_gather[1]])))
	xs, ys = data[end_gather[0]], data[end_gather[1]]+randomization

	if out == 'plot':
		plot(xs, ys, end_gather)
	elif out == 'save':
		save(xs, ys, end_gather)
	elif out == 'print':
		return xs, ys, end_gather

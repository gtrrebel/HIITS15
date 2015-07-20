import numpy as np
import pylab as pb
import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from LDA3 import LDA3
from data_creator import data_creator
from input_parser import input_parser
from label_switcher import label_switcher

restarts, nips_data = input_parser.LDA_parse2(sys.argv[1:])

docs, vocab = data_creator.nips_data(*nips_data)
N_TOPICS = nips_data[0]
alpha_0 = 200
end_gather = ['bound', 'epsilon_positive']

m = LDA3(docs,vocab,N_TOPICS,alpha_0=alpha_0)
m.runspecs['basics']['restarts'] = restarts
m.runspecs['basics']['methods'] = ['HS']
m.set_invests(road_gather= [], end_gather=end_gather)

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

pb.figure()
pb.xlabel(end_gather[0])
pb.ylabel(end_gather[1])
pb.plot(data[end_gather[0]], data[end_gather[1]], '^r')
import numpy as np
from itertools import product
import os
import time
import sys
from ukko_runner import ukko_runner

def init_runner():
	runner = ukko_runner.runner()
	runner.remove_bad_hosts()
	runner.MAXLOAD = 10
	return runner

K0cl = [0.5, 1, 2]
Vcl = [2**(i) for i in xrange(-2, 11)]
Kl = [2**(i) for i in xrange(2, 9)]
Dl = [10**(i) for i in xrange(1, 4)]
Nl = [100*2**(i) for i in xrange(8)]
methodsl = ['steepest', 'FR']
lens = [len(K0cl), len(Vcl), len(Kl), len(Dl), len(Nl), len(methodsl)]	

def run_all(specs, topic_params = (1,1,0), word_params = (1,1,0), restarts = 10):
	runner = init_runner()
	basicpath = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_outputs'
	codepath = '/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/examples/LDA_big_runner.py'
	dirname = 'LDA_ukko_' + time.strftime("%H_%M_%S_%d_%m_%Y")
	outputpath = basicpath + '/' + dirname
	os.mkdir(outputpath)
	build_seeds = np.random.randint(0, (1<<32) - 1, size=2)
	param_seeds = np.random.randint(0, (1<<32) - 1, size=restarts)
	with open(outputpath + '/specs', 'w') as f:
		f.write(str(topic_params))
		f.write(str(word_params))
		f.write(str(specs))
		f.write(str(restarts))
		f.write(str(build_seeds))
		f.write(str(param_seeds))
	for tup in product(*specs):
		input_string = ' '.join([str(i) for i in topic_params]) + ' ' + ' '.join([str(i) for i in word_params]) + \
						' ' + ' '.join([str(i) for i in tup]) + ' ' + str(restarts) + ' ' + \
						' '.join([str(i) for i in build_seeds]) + ' ' + ' '.join([str(i) for i in param_seeds])
		run_string = 'python ' + codepath + ' ' + input_string + ' ' + dirname
		runner.add_jobs([(run_string,1)])
	runner.start_batches()

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

def run_all(specs, restarts = 10):
	runner = init_runner()
	basicpath = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs'
	codepath = '/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/examples/MOG_big_runner.py'
	dirname = 'MOG_ukko_' + time.strftime("%H_%M_%S_%d_%m_%Y")
	outputpath = basicpath + '/' + dirname
	os.mkdir(outputpath)
	with open(outputpath + '/specs', 'w') as f:
		f.write(str(specs))
		f.write(str(restarts))
	for tup in product(*specs):
		input_string = ' '.join([str(i) for i in tup]) + ' ' + str(restarts)
		run_string = 'python ' + codepath + ' ' + input_string + ' ' + dirname
		runner.add_jobs([(run_string,1)])
	runner.start_batches()

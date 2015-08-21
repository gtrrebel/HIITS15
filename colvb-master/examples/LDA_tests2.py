from LDAtests import *
from ukko_runner import ukko_runner
import sys
import os

ukko_test_out_path = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_ukko_tests/'
ukko_test_code_path = '/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/examples/LDA_runner1.py'

def init_runner():
	runner = ukko_runner.runner()
	runner.remove_bad_hosts()
	runner.MAXLOAD = 10
	return runner

def specs_str(args, restarts, to_calc, methods, code):
	return 'LDA3 test\n' + \
			'number of arguments: ' + str(len(args)) + '\n' + \
			'number of restarts: ' + str(restarts) + '\n' + \
			'methods: ' + ' '.join(methods) + '\n' + \
			'code: ' + code + '\n' + \
			'arguments: ' + ' '.join(args) + '\n' + \
			'end of specfile\n'

def init_index_test_dir(args, restarts, to_calc, methods, code):
	dir_name = ukko_test_out_path + code
	os.makedirs(dir_name)
	with open(dir_name + '/test_specs', 'w+') as f:
		f.write(specs_str(args, restarts, to_calc, methods, code))
	os.makedirs(dir_name + '/situations')
	os.makedirs(dir_name + '/pickled_libs')
	os.makedirs(dir_name + '/data')
	os.makedirs(dir_name + '/pics')

def run_string(arg, restarts, to_calc, methods, code):
	return 'python ' + ukko_test_code_path + ' ' + code + ' ' + arg + ' ' + str(restarts) + ' ' + \
			str(to_calc) + ' ' + str(len(methods)) + ' ' + ' '.join(methods)

def start_index_tests(args = [''], restarts = 10, to_calc = 3, methods = ['steepest', 'FR']):
	code = 'LDA_ukko_test_' + time.strftime('%H_%M_%S_%d_%m_%Y')
	init_index_test_dir(args, restarts, to_calc, methods, code)
	runner = init_runner()
	for arg in args:
		runner.add_jobs([(run_string(arg, restarts, to_calc, methods, code), 1)])
	runner.start_batches()
	return code

def situation(code):
	with open(ukko_test_out_path + code + '/test_specs', 'r') as f:
		N = int(f.readlines()[1].split()[-1])
	return len(os.listdir(ukko_test_out_path + code + '/situations')), N

def display(code):
	p = situation(code)
	if p[0] == p[1]:
		raise NotImplementedError("TODO")
	else:
		print 'calculation not yet done: ' + str(p[0]) + '/' + str(p[1])


 
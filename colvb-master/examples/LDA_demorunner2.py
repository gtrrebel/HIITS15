import sys
import os
import pylab as pb
import time
from ukko_runner import ukko_runner

runner = ukko_runner.runner()
runner.remove_bad_hosts()
runner.MAXLOAD = 10


mainpath = '/cs/fs/home/othe/Windows/Desktop/hiit'
runnerpath = mainpath + '/HIITS15/running_utils/scripts/'
coderunner = 'test_runner2.sh'
codepath = mainpath + '/HIITS15/colvb-master/examples/'
outputpath = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo3.py/'
code = 'LDA_demo3.py'

for stat in range(bound1, bound2 + 1):
	inputstring = str(invest)
	for i in range(len(basic_inputs)):
		if i != trait:
			inputstring += ' ' + str(basic_inputs[i])
		else:
			inputstring += ' ' + str(stat)
	runner.add_jobs([(runnerpath + coderunner + ' ' + codepath + code + ' ' + inputstring, 1)])

runner.start_batches()

from ukko_runner import ukko_runner
import sys
import os

code = sys.argv[1]
number = sys.argv[2]

runner = ukko_runner.runner()
runner.MAXLOAD = 10

mainpath = '/cs/fs/home/othe/Windows/Desktop/hiit'
testpath = mainpath + '/hiit_test_input/MOG_demo2.py/' + str(number) + '/'
runnerpath = mainpath + '/HIITS15/running_utils/scripts/'
coderunner = 'test_runner.sh'
codepath = mainpath + '/HIITS15/colvb-master/examples/'

for line in os.listdir('/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_input/' + code + '/' + str(number)):
	runner.add_jobs([(runnerpath + coderunner + ' ' + codepath + code + ' ' + testpath + line, 1)])

runner.start_batches()


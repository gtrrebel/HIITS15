from ukko_runner import ukko_runner
import sys

code = sys.argv[1]
runfilename = sys.argv[2]
runs = open(runfilename)

runner = ukko_runner.runner()
runner.MAXLOAD = 10
mainpath = '/cs/fs/home/othe/Windows/Desktop/hiit'
testpath = mainpath + '/hiit_test_input/MOG_demo2.py/in/'
runnerpath = mainpath + '/HIITS15/running_utils/scripts/'
coderunner = 'test_runner.sh'
codepath = mainpath + '/HIITS15/colvb-master/examples/'

for line in runs:
	runner.add_jobs([(runnerpath + coderunner + ' ' + codepath + code + ' ' +testpath + line, 1)])

runs.close()

runner.start_batches()


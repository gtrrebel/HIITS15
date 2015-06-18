from ukko_runner import ukko_runner
import sys

code = sys.argv[1]
runfilename = sys.argv[2]
runs = open(runfilename)

runner = ukko_runner.runner()
runner.MAXLOAD = 1
runnerpath = '/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/running_utils/scripts/'
coderunner= 'test_runner.sh'
codepath = '/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/examples/'

for line in runs:
	runner.add_jobs([(runnerpath + coderunner + ' ' + codepath + code + ' ' + line, 1)])

runs.close()

runner.start_batches()


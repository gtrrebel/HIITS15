import sys
import os
import pylab as pb
from ukko_runner import ukko_runner

runner = ukko_runner.runner()
runner.MAXLOAD = 10

mainpath = '/cs/fs/home/othe/Windows/Desktop/hiit'
runnerpath = mainpath + '/HIITS15/running_utils/scripts/'
coderunner = 'test_runner2.sh'
codepath = mainpath + '/HIITS15/colvb-master/examples/'
outputpath = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo2.py'
code = 'LDA_demo2.py'

trait = int(sys.argv[1])
invest = "spectrum length"
trait_names = ['WORDSIZE', 'N_DOCS', 'DOCUMENT_LENGTH', 'N_TOPIC_COEFF']
bound1, bound2 = int(sys.argv[2]), int(sys.argv[3])
basic_inputs = [3,10,5,2]
user = False
if len(sys.argv) > 4:
	user = True

outputstart = '"LDA_demo2.py\n' + \
				trait_names[trait] + ' vs. ' + invest + '\n' + \
				'basic inputs: ' + ' '.join([str(inp) for inp in basic_inputs]) + '\n' + \
				'bounds: ' + str(bound1) + ', ' + str(bound2) + '\n"'

os.system('printf ' +  outputstart +' > ' +  outputpath + '/tmp-output.txt')
for stat in range(bound1, bound2 + 1):
	inputstring = ''
	for i in range(len(basic_inputs)):
		if i != trait:
			inputstring += ' ' + str(basic_inputs[i])
		else:
			inputstring += ' ' + str(stat)
	runner.add_jobs([(runnerpath + coderunner + ' ' + codepath + code + ' ' + inputstring, 1)])

runner.start_batches()

'''
outputfiles = []
def gatheroutput():
	for (dirpath, dirnames, filenames) in os.walk(outputpath):
		outputfiles.extend(filenames)
		break
	for fil in outputfiles:
		fi = open(outputpath + fil)

inp = open(outputpath + '/tmp-output.txt')
output = []
for line in inp.readlines()[4:]:
	output.append(float(line))
inp.close()
pb.figure()
pb.title(trait_names[trait] + ' vs. ' + invest)
pb.xlabel(trait_names[trait])
pb.ylabel(invest)
pb.plot(range(bound1, bound2 + 1), output)
pb.show()
os.system('rm tmp-output.txt')
'''
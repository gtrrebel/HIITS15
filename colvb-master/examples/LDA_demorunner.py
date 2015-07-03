import sys
import os
import pylab as pb
import time
from ukko_runner import ukko_runner

runner = ukko_runner.runner()
runner.remove_bad_hosts()
runner.MAXLOAD = 10

bound1, bound2 = int(sys.argv[2]), int(sys.argv[3])
invest = int(sys.argv[4])
basic_inputs = [3,10,5,2]
user = False
max_wait_time = 12*3600

trait = int(sys.argv[1])
invest_names = ["spectrum_length", "final_bound", "distance_travelled", "distance_from_start"]
print_invest_names = ["spectrum length", "final bound", "distance travelled", "distance from start"]
trait_names = ['WORDSIZE', 'N_DOCS', 'DOCUMENT_LENGTH', 'N_TOPIC_COEFF']
print_trait_names = ['wordsize', 'number of documents', 'document length', 'topic coefficient']

mainpath = '/cs/fs/home/othe/Windows/Desktop/hiit'
runnerpath = mainpath + '/HIITS15/running_utils/scripts/'
coderunner = 'test_runner2.sh'
codepath = mainpath + '/HIITS15/colvb-master/examples/'
outputpath = '/home/tktl-csfs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_demo2.py/'
tmp_outputpath = outputpath + 'tmp-output/'
code = 'LDA_demo2.py'
outputfilename = 'LDA_demo2.py.' + trait_names[trait] + '.to.' + invest_names[invest] + '.' + \
				str(bound1) + '.' + str(bound2) + time.strftime("%H:%M:%S-%d.%m.%Y") + '.txt'



outputstart = '"LDA_demo2.py\n' + \
				print_trait_names[trait] + ' vs. ' + print_invest_names[invest] + '\n' + \
				'basic inputs: ' + ' '.join([str(inp) for inp in basic_inputs]) + '\n' + \
				'bounds: ' + str(bound1) + ', ' + str(bound2) + '\n"'

os.system('printf ' +  outputstart +' > ' +  outputpath + outputfilename)
for stat in range(bound1, bound2 + 1):
	inputstring = str(invest)
	for i in range(len(basic_inputs)):
		if i != trait:
			inputstring += ' ' + str(basic_inputs[i])
		else:
			inputstring += ' ' + str(stat)
	runner.add_jobs([(runnerpath + coderunner + ' ' + codepath + code + ' ' + inputstring, 1)])

runner.start_batches()

outputcount = bound2 + 1 - bound1
count = 0

def checkfull(filename):
	f = open(tmp_outputpath + filename)
	if os.path.getsize(tmp_outputpath + filename) == 0:
		return False
	else:
		f = open(tmp_outputpath + filename)
		ans = (f.readlines()[-1] == 'done\n')
		f.close()
		return ans


while True:
	time.sleep(1)
	if sum(1 for name in os.listdir(tmp_outputpath) if checkfull(name)) >= outputcount:
		break
	count += 1
	if count >= max_wait_time:
		break

def cmp_outputs(pair1, pair2):
	info1, info2 = [int(i) for i in pair1[0].split(' ')][trait], [int(i) for i in pair2[0].split(' ')][trait]
	if info1 > info2:
		return 1
	elif info2 > info1:
		return -1
	else: 
		return 0

def gatheroutput():
	filepairs = []
	for (dirpath, dirnames, filenames) in os.walk(tmp_outputpath):
		outputfiles.extend(filenames)
		break
	for fil in outputfiles:
		fi = open(tmp_outputpath + fil)
		lines = fi.readlines()
		filepairs.append((lines[0], lines))
		fi.close()
		os.system('rm ' + tmp_outputpath + fil)
	filepairs = sorted(filepairs, cmp=cmp_outputs)
	for filepair in filepairs:
		for line in filepair[1][1:-1]:
			os.system('printf \'%s\' "' +  line + '" >> ' +  outputpath + outputfilename)

if sum(1 for name in os.listdir(tmp_outputpath) if checkfull(name)) == outputcount:
	outputfiles = []
	gatheroutput()
	inp = open(outputpath + outputfilename)
	output = []
	lines = inp.readlines()[4:]
	#os.system('cat ' + outputpath + outputfilename)
	for line in lines:
		output.append(float(line))
	inp.close()
	pb.figure()
	pb.title(print_trait_names[trait] + ' vs. ' + print_invest_names[invest])
	pb.xlabel(print_trait_names[trait])
	pb.ylabel(print_invest_names[invest])
	pb.plot(range(bound1, bound2 + 1), output)
	pb.show()
else:
	print 'Time limit exceeded'

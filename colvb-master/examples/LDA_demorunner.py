import sys
import os
import pylab as pb

trait = int(sys.argv[1])
invest = "spectrum length"
trait_names = ['WORDSIZE', 'N_DOCS', 'DOCUMENT_LENGTH', 'N_TOPIC_COEFF']
bound1, bound2 = int(sys.argv[2]), int(sys.argv[3])
basic_inputs = [3,10,5,2]
user = True

outputstart = '"LDA_demo2.py\n' + \
				trait_names[trait] + ' vs. ' + invest + '\n' + \
				'basic inputs: ' + ' '.join([str(inp) for inp in basic_inputs]) + '\n' + \
				'bounds: ' + str(bound1) + ', ' + str(bound2) + '\n"'

os.system('printf ' +  outputstart +' > tmp-output.txt')
for stat in range(bound1, bound2 + 1):
	inputstring = ''
	for i in range(len(basic_inputs)):
		if i != trait:
			inputstring += ' ' + str(basic_inputs[i])
		else:
			inputstring += ' ' + str(stat)
	if user:
		print '\r',  stat, ': ', str(100*(stat - bound1)*1./(bound2 + 1 - bound1))[:4], '%',
		sys.stdout.flush()
	os.system('python /home/othe/Desktop/HIIT/HIITS15/colvb-master/examples/LDA_demo2.py' + inputstring + \
	 	' 1>> tmp-output.txt 2>/dev/null')
if user:
	print '\r',
	sys.stdout.flush()
	os.system('cat tmp-output.txt')
inp = open('tmp-output.txt')
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
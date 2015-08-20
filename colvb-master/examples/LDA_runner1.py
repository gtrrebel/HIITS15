import sys
import random
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/examples')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
from LDAdemo4 import *
from LDA_pickler import *

#Collect Specs
code = sys.argv[1]
arg = [int(i) for i in sys.argv[2:6]]
restarts = int(sys.argv[6])
to_calc = int(sys.argv[7])
methods = sys.argv[9:9+int(sys.argv[8])]
result_directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/LDA_ukko_tests/' + code
personal_code = '_'.join([str(a) for a in arg])

#Make lib
ms = init(args = [arg])
lib = run(ms, restarts=restarts, end_gather=['bound', 'return_m', 'get_vb_param', 'reduced_dimension', 'optimizetime'], methods=methods)
for out in lib:
	for dic in out[2]:
		dic['index'] = 0

#Do calculations
for out in lib:
	calc_min(out[2])
	calc_max(out[2])
	for i in xrange(to_calc - 2):
		calc_dic(out[2][random.randint(0, len(methods)*restarts - 1))

#Save lib
bhr_lib_pickle(lib, result_directory + '/pickled_libs/')

#Make done-file
with open(result_directory + '/situations/' + personal_code, 'w+') as f:
	f.write('done\n')


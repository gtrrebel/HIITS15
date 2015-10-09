import sys
from MOG_demo4 import *
import numpy as np
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
sys.path.append('/cs/fs/home/othe/Windows/Desktop/hiit/HIITS15/colvb-master/colvb')
sys.path.append('/Users/otteheinavaara/Desktop/HIITS15/colvb-master/colvb')
from MOG_pickler import *

arg = sys.argv[1:]
NClust = int(arg[0])
dim = int(arg[1])
Nmin = int(arg[2])
Nmax = int(arg[3])
meanscoeff = int(arg[4])
method = arg[5]
dir_name = arg[-1]
specs = arg[0:-1]

datacount = 10
restartcount = 10
jump_sizes = [0.01, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
jump_counts = 5

def MOG_trust_jump_test(args = [10, 2, 10, 30, 5], method = ['FR']):
	jump_diffs = {}
	trust_ite = []
	trust_real_ite = []
	normal_bounds = []
	trust_bounds = []
	ite = []
	for s in jump_sizes:
		jump_diffs[s] = []
	for i in xrange(datacount):
		m = MOG_demo4.init([args], make_fns = False)[0]
		for j in xrange(restartcount):
			seed = np.random.randint(0, (1 << 32) - 1)
			m.new_param(seed)
			m.trust_region_optimize(trust_count = 100, method = method)
			trust_bounds.append(m.bound())
			iteration1 = m.ite_counts
			m.new_param(seed)
			m.optimize(method=method)
			normal_bounds.append(m.bound())
			iteration2 = m.iteration
			old_phi = m.get_vb_param()
			trust_ite.append(len(iteration1))
			trust_real_ite.append(sum(iteration1))
			ite.append(iteration2)
			for s in jump_sizes:
				for c in xrange(jump_counts):
					m.set_vb_param(old_phi)
					m.random_jump(length = s)
					m.optimize(method=method)
					jump_diffs[s].append(m.bound() - normal_bounds[-1])
	output = [normal_bounds, trust_bounds, jump_diffs, trust_ite, trust_real_ite, ite]
	return output

def outstring(spec):
	return '_'.join([str(i) for i in spec])

data = MOG_trust_jump_test(args = [NClust, dim, Nmin, Nmax, meanscoeff], method = method)

outputs_pickle(data, directory = '/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_results/MOG_outputs/' + dir_name,
					ukko = True, filename = outstring(specs))

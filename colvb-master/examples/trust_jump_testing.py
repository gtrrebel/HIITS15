import numpy as np
import LDA_demo4
import MOG_demo4
from LDA3 import LDA3
from LDA_creator import *

datacount = 3
restartcount = 3
jump_sizes = [0.01, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
jump_counts = 3

def MOG_trust_jump_test(args = [10, 2, 10, 30, 5]):
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
			m.trust_region_optimize(trust_count = 100)
			trust_bounds.append(m.bound())
			iteration1 = m.ite_counts
			m.new_param(seed)
			m.optimize()
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
					m.optimize()
					jump_diffs[s].append(m.bound() - normal_bounds[-1])
	return normal_bounds, trust_bounds, jump_diffs, trust_ite, trust_real_ite, ite

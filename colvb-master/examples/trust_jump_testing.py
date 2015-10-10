import numpy as np
import pylab as pb
import LDA_demo4
import MOG_demo4
from LDA3 import LDA3
from LDA_creator import *

datacount = 5
restartcount = 5
jump_sizes = [0.01, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
jump_counts = 5

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
	output = [normal_bounds, trust_bounds, jump_diffs, trust_ite, trust_real_ite, ite]
	return output

def LDA_trust_jump_test(args = ''):
	jump_diffs = {}
	trust_ite = []
	trust_real_ite = []
	normal_bounds = []
	trust_bounds = []
	ite = []
	for s in jump_sizes:
		jump_diffs[s] = []
	for i in xrange(datacount):
		m = LDA_demo4.init([args], make_fns = False)[0]
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
	output = [normal_bounds, trust_bounds, jump_diffs, trust_ite, trust_real_ite, ite]
	return output

def plot_diffs(output):
	xs = []
	ys = []
	for i in xrange(len(jump_sizes)):
		for j in output[2][jump_sizes[i]]:
			xs.append(i)
			ys.append(j)
	pb.figure()
	pb.xlim(-1, len(jump_sizes) + 1)
	pb.plot(xs, ys, 'b*')

def plot_ite(output):
	trust_ite = output[3]
	real_ite = output[4]
	ite = output[5]
	xs = range(len(ite))
	pb.figure()
	pb.plot(xs, trust_ite, 'b')
	pb.plot(xs, real_ite, 'g')
	pb.plot(xs, ite, 'r')

def plot_bounds(output):
	normal_bounds = output[0]
	trust_bounds = output[1]
	xs = range(len(normal_bounds))
	pb.figure()
	pb.plot(xs, normal_bounds, 'r')
	pb.plot(xs, trust_bounds, 'b')

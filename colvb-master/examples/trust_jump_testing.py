import numpy as np
import pylab as pb
import LDA_demo4
import MOG_demo4
from LDA3 import LDA3
from LDA_creator import *

datacount = 5
restartcount = 5
jump_sizes = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
jump_counts = 5

def MOG_trust_jump_test(args = [10, 2, 10, 30, 5]):
	jump_diffs = []
	for i in xrange(datacount):
		m = MOG_demo4.init([args], make_fns = False)[0]
		for j in xrange(restartcount):
			m.new_param()
			m.optimize()
			bound = m.bound()
			old_phi = m.get_vb_param()
			diffs = []
			for s in jump_sizes:
				for c in xrange(jump_counts):
					m.set_vb_param(old_phi)
					m.random_jump(length = s)
					m.optimize()
					diffs.append(m.bound())
			jump_diffs.append((bound, diffs)) 
	return jump_diffs

def MOG_jumper(args = [10, 2, 10, 30, 5]):
	m = MOG_demo4.init([args])[0]
	eps = 1e-14
	m.optimize()
	N = len(m.get_vb_param())
	bound = m.bound()
	poss = []
	sposs = []
	for i in xrange(10):
		old_phi = m.get_vb_param()
		for j in xrange(100):
			l = np.random.choice(jump_sizes[3:10])
			print l
			m.random_jump(length = l)
			m.optimize()
			if m.bound() > bound:
				bound = m.bound()
				break
			m.set_vb_param(old_phi)
			if i == 99:
				return bound, sposs, poss, N
		eigs = m.eigenvalues()
		poss.append(sum(1 for i in eigs if i > 0))
		sposs.append(sum(1 for i in eigs if i > eps))
		print bound, sposs[-1], poss[-1]
	return bound, sposs, poss, N

def MOG_trust_jump_test2(args = [10, 2, 10, 30, 5]):
	jump_diffs = {}
	jump_diffs2 = {}
	trust_ite = []
	trust_real_ite = []
	normal_bounds = []
	ite = []
	for s in jump_sizes:
		jump_diffs[s] = []
		jump_diffs2[s] = []
	for i in xrange(datacount):
		m = MOG_demo4.init([args], make_fns = False)[0]
		for j in xrange(restartcount):
			seed = np.random.randint(0, (1 << 32) - 1)
			m.new_param(seed)
			m.optimize()
			normal_bounds.append(m.bound())
			old_phi = m.get_vb_param()
			for s in jump_sizes:
				for c in xrange(jump_counts):
					m.set_vb_param(old_phi)
					m.random_jump(length = s)
					m.optimize()
					jump_diffs[s].append(m.bound() - normal_bounds[-1])
			for s in jump_sizes:
				for c in xrange(jump_counts):
					m.new_param(seed)
					m.random_jump(length = s)
					m.optimize()
					jump_diffs2[s].append(m.bound() - normal_bounds[-1])
	output = [normal_bounds, jump_diffs, jump_diffs2]
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

def plot_diffs2(output):
	xs = []
	ys = []
	for i in xrange(len(jump_sizes)):
		for j in output[1][jump_sizes[i]]:
			xs.append(i)
			ys.append(j)
	pb.figure()
	pb.xlim(-1, len(jump_sizes) + 1)
	pb.plot(xs, ys, 'b*')
	xs = []
	ys = []
	for i in xrange(len(jump_sizes)):
		for j in output[2][jump_sizes[i]]:
			xs.append(i)
			ys.append(j)
	pb.figure()
	pb.xlim(-1, len(jump_sizes) + 1)
	pb.plot(xs, ys, 'r*')

def plot_diffs3(output):
	pb.figure()
	ma = -1e30
	for p in output:
		ma = max(ma, p[0])
		ma = max(ma, max(p[1]))
		pb.plot(len(p[1])*[p[0]], p[1], 'r*')
	pb.plot([0, ma*1.2], [0, ma*1.2])
	pb.show()

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

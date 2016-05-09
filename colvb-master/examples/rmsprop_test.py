from MOG_demo4 import *
from climin import RmsProp
import matplotlib.pyplot as plt

def get_m():
	return init()[0]

def setup_rms(m, step_rate = 0.1, step_adapt = 0.05):
	gradient_func = m.bound_grad()
	params = m.get_vb_param()
	rms = RmsProp(params, lambda x : -gradient_func(x), step_rate, step_adapt = 0.05, step_rate_max = 1.0)
	return rms

def run_rms(rms, m, iters = 200):
	iteration = 0
	bound_old = 0
	bound = 0
	for info in rms:
		iteration += 1
		bound_old = bound
		bound = m.autograd_bound()(rms.wrt)
		nor = np.linalg.norm(m.bound_grad()(rms.wrt))
		#print nor, bound
		if (iteration > 1):
			if abs(bound - bound_old) < 1e-6:
			#	print "no: ", bound, bound_old
				break
		if info['n_iter'] >= iters:
			#print "ite"
			break
		if abs(nor) < 1e-4:
			#print "nor: ", nor
			break
	return rms

def rms_demo(iters=200, step_rate = 0.1, step_adapt = 0.05):
	m = get_m()
	rms = setup_rms(m, step_rate=step_rate, step_adapt = step_adapt)
	b1 = m.autograd_bound()(rms.wrt)
	rms = run_rms(rms, m, iters=iters)
	b2 = m.autograd_bound()(rms.wrt)
	m.set_vb_param(rms.wrt)
	#print m.epsilon_positive()
	#print m.eigenvalues()
	#print b1, b2
	m.randomize()
	m.optimize_autograd()
	print m.autograd_bound()(m.get_vb_param())

def rms_optimize(m, iters = 200, step_rate=0.1):
	rms = setup_rms(m, step_rate=step_rate)
	rms = run_rms(rms, m, iters)
	m.set_vb_param(rms.wrt)
	return m

def try_one(iters = 200):
	m = get_m()
	params = m.get_vb_param().copy()
	m = rms_optimize(m, iters)
	b1 = m.bound()
	pos1 = m.epsilon_positive()
	m.plot()
	m.set_vb_param(params)
	m.optimize_autograd(method = "steepest")
	b2 = m.bound()
	pos2 = m.epsilon_positive()
	m.plot()
	print b1, b2
	print pos1, pos2
	return (m, params)

def compare_optimizes_rms(M = 10, iters = 200):
	m = get_m()
	bounds1 = []
	bounds2 = []
	positives1 = []
	positives2 = []
	delta_bound = 0.1
	delta_positive = 0.1
	for i in range(M):
		m.randomize()
		params = m.get_vb_param().copy()
		print i, 1
		m.set_vb_param(params)
		m = rms_optimize(m, iters)
		bounds1.append(m.bound())
		positives1.append(m.epsilon_positive())
		print i, 2
		m.set_vb_param(params)
		m.optimize_autograd()
		bounds2.append(m.bound())
		positives2.append(m.epsilon_positive())
	plt.figure()
	axes = plt.gca()
	print "bounds:"
	for b1, b2 in zip(bounds1, bounds2):
		print b2 - b1
	bounds1 = np.array(bounds1, dtype='float64')
	bounds2 = np.array(bounds2, dtype='float64')
	positives1 = np.array(positives1, dtype='float64')
	positives2 = np.array(positives2, dtype='float64')
	bounds1 += delta_bound*np.random.randn(M)
	bounds2 += delta_bound*np.random.randn(M)
	positives1 += delta_positive*np.random.randn(M)
	positives2 += delta_positive*np.random.randn(M)
	xmin, xmax = min(min(bounds1), min(bounds2)) - 10, max(max(bounds1), max(bounds2)) + 10
	ymin, ymax = min(min(positives1), min(positives2)) - 10, max(max(positives1), max(positives2)) + 10
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	plt.plot(bounds1, positives1, 'r*')
	plt.plot(bounds2, positives2, 'g.')
	plt.show()
from MOG_demo4 import *
import matplotlib.pyplot as plt

def get_m():
	return init()[0]

def get_m2(i):
	args = [[5 + i//2, 2 + i//5, 5 + i, 10 + (3*i)//2, 5]]
	return init(args)[0]

def first_demo(M = 100):
	m = get_m()
	bounds = []
	positives = []
	for i in xrange(M):
		print i, 1
		m.randomize()
		bounds.append(m.get_bound())
		positives.append(m.epsilon_positive())
		m.optimize()
		print i, 2
		bounds.append(m.get_bound())
		positives.append(m.epsilon_positive())
	plt.figure()
	axes = plt.gca()
	xmin, xmax = min(bounds) - 10, max(bounds) + 10
	ymin, ymax = min(positives) - 10, max(positives) + 10
	axes.set_xlim([xmin,xmax])
	axes.set_ylim([ymin,ymax])
	plt.plot(bounds, positives, 'r*')
	plt.show()

def compare_optimizes(M = 10):
	m = get_m()
	bounds1 = []
	bounds2 = []
	positives1 = []
	positives2 = []
	delta_bound = 0.1
	delta_positive = 0.1
	for i in range(M):
		params = m.get_vb_param().copy()
		print i, 1
		m.set_vb_param(params)
		m.optimize()
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

def bound_vs_ind(M = 100, N = 10):
	bounds = []
	positives = []
	delta_bound = 0.01
	delta_positive = 0.01
	m = get_m2(N)
	print m.dimension()
	for i in xrange(M):
		m.randomize()
		print i
		m.optimize_autograd()
		bounds.append(m.bound())
		positives.append(m.epsilon_positive())
	plt.figure()
	axes = plt.gca()
	bounds = np.array(bounds, dtype='float64')
	positives = np.array(positives, dtype='float64')
	min_bound, max_bound = np.amin(bounds), np.amax(bounds)
	plt.plot(bounds + delta_bound*np.random.randn(M), positives + delta_positive*np.random.randn(M), 'b.')
	plt.show()

def compare_bound_deviations(M = 100, N = 1):
	boundss1 = []
	boundss2 = []
	positivess1 = []
	positivess2 = []
	delta_bound = 0.01
	delta_positive = 0.01
	for ii in xrange(N):
		m = get_m()
		bounds1 = []
		bounds2 = []
		positives1 = []
		positives2 = []
		for i in xrange(M):
			m.randomize()
			params = m.get_vb_param().copy()
			print i, 1
			m.set_vb_param(params)
			m.optimize()
			bounds1.append(m.bound())
			positives1.append(m.epsilon_positive())
			print i, 2
			m.set_vb_param(params)
			m.optimize_autograd()
			bounds2.append(m.bound())
			positives2.append(m.epsilon_positive())
			print bounds1[-1], bounds2[-1]
		boundss1.append(bounds1)
		boundss2.append(bounds2)
		positivess1.append(positives1)
		positivess2.append(positives2)
	plt.figure()
	axes = plt.gca()
	bounds1 = np.array(boundss1, dtype='float64')
	bounds2 = np.array(boundss2, dtype='float64')
	positives1 = np.array(positivess1, dtype='float64')
	positives2 = np.array(positivess2, dtype='float64')
	max_bound = max(np.amax(bounds1), np.amax(bounds2))
	min_bound = min(np.amin(bounds1), np.amin(bounds2))
	for bounds1, positives1 in zip(boundss1, positivess1):
		plt.plot(bounds1 + delta_bound*np.random.randn(M), positives1 + delta_positive*np.random.randn(M), 'b.')
	for bounds2, positives2 in zip(boundss2, positivess2):
		plt.plot(bounds2 + delta_bound*np.random.randn(M), positives2 + delta_positive*np.random.randn(M), 'g.')
	plt.plot([min_bound, max_bound], [0,0])
	plt.show()

def dimension_variation(M = 10, N = 10, use_autograd=True):
	boundss1 = []
	positivess1 = []
	delta_bound = 0.01
	delta_positive = 0.01
	dimensions = []
	for ii in xrange(N):
		m = get_m2(ii)
		bounds1 = []
		positives1 = []
		dimensions.append(m.dimension())
		for i in xrange(M):
			print ii, i
			m.randomize()
			if use_autograd:
				m.optimize_autograd(method='steepest')
			else:
				m.optimize()
			bounds1.append(m.bound())
			#m.plot()
			positives1.append(m.epsilon_positive())
		boundss1.append(bounds1)
		positivess1.append(positives1)
	plt.figure()
	bounds1 = np.array(boundss1, dtype='float64')
	for dim, bounds1 in zip(dimensions,boundss1):
		plt.plot(dim*np.ones(M), bounds1 + delta_bound*np.random.randn(M), 'b.')
	positives1 = np.array(positivess1, dtype='float64')
	plt.figure()
	for dim, positives1 in zip(dimensions,positivess1):
		plt.plot(dim*np.ones(M), positives1 + delta_positive*np.random.randn(M), 'r.')
	plt.show()
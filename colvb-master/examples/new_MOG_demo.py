from MOG_demo4 import *
import matplotlib.pyplot as plt

def get_m():
	return init()[0]

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
import numpy as np
import pylab as pb

def create_distribution(gamma):
	td = np.random.dirichlet(gamma)
	return (td, np.cumsum(td))

def create_distributions(gamma, M):
	return [create_distribution(gamma) for _ in xrange(M)]

def get(d):
	x = np.random.uniform()
	n = len(d[0])
	low = 0
	high = n - 1
	while low + 1 < high:
		mid = (low+high)>>1
		if d[1][mid] > x:
			high = mid
		else:
			low = mid
	if d[1][low] < x:
		return high
	else:
		return low

def create_data(alpha, beta, Ns):
	D, K = len(Ns), len(alpha)
	tds, wds = create_distributions(alpha, D), create_distributions(beta, K)
	return [[get(wds[get(tds[i])]) for _ in xrange(Ns[i])] for i in xrange(D)]

def plot_d(d):
	fig = pb.figure()
	ax = pb.subplot(111)
	ax.bar(range(len(d[0])), d[0])
	pb.show()

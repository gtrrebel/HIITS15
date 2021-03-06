import numpy as np
import pylab as pb
import sys

def softmax(x):
    ex = np.exp(x-x.max())
    return ex/ex.sum()

def create_gaussian_distributions(params, v, n):
	alpha, beta, gamma = params
	A = np.linalg.cholesky(alpha*np.identity(v) + beta*np.ones((v,v)))
	if gamma == 0:
		b = np.zeros(v)
	else:
		b = np.random.normal(scale=gamma, size=v)
	ds = [softmax(b + np.dot(A, np.random.normal(size=v))) for _ in xrange(n)]
	return [(d, np.cumsum(d)) for d in ds]

def create_dirichlet_distribution(gamma):
	td = np.random.dirichlet(gamma)
	return (td, np.cumsum(td))

def create_dirichlet_distributions(gamma, M):
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

def create_gaussian_data(topic_params, word_params, K, V, Ns):
	tds, wds = create_gaussian_distributions(topic_params, K, len(Ns)), create_gaussian_distributions(word_params, V, K)
	return [[get(wds[get(tds[i])]) for _ in xrange(Ns[i])] for i in xrange(len(Ns))]

def new_create_gaussian_data(topic_params, word_params, K, V, Ns, topic_seed=None, word_seed=None):
	tds, topic_seed = new_create_gaussian_distributions(topic_params, K, len(Ns), seed=topic_seed)
	wds, word_seed = new_create_gaussian_distributions(word_params, V, K, seed=word_seed)
	return [[get(wds[get(tds[i])]) for _ in xrange(Ns[i])] for i in xrange(len(Ns))], [topic_seed, word_seed]

def plot_d(d):
	fig = pb.figure()
	ax = pb.subplot(111)
	ax.bar(range(len(d[0])), d[0])
	pb.show()

def covtest(params, v, n):
	alpha, beta, gamma = params
	A = np.linalg.cholesky(alpha*np.identity(v) + beta*np.ones((v,v)))
	if gamma == 0:
		b = np.zeros(v)
	else:
		b = np.random.normal(scale=gamma, size=v)
	d = [b + np.dot(A, np.random.normal(size=v)) for _ in xrange(n)]
	return np.cov(d, rowvar=0)

def covtest_both(params, v, n):
	alpha, beta, gamma = params
	A = np.linalg.cholesky(alpha*np.identity(v) + beta*np.ones((v,v)))
	if gamma == 0:
		b = np.zeros(v)
	else:
		b = np.random.normal(scale=gamma, size=v)
	d = [b + np.dot(A, np.random.normal(size=v)) for _ in xrange(n)]
	d2 = [softmax(dd) for dd in d]
	return np.cov(d, rowvar=0), np.cov(d2, rowvar=0)

def covtest2_both(params, v, n):
	alpha, beta, gamma = params
	b = beta*np.random.normal(size=(v,v))
	b = b + b.transpose()
	A = np.linalg.cholesky(alpha*np.identity(v) + b)
	if gamma == 0:
		b = np.zeros(v)
	else:
		b = np.random.normal(scale=gamma, size=v)
	d = [b + np.dot(A, np.random.normal(size=v)) for _ in xrange(n)]
	d2 = [softmax(dd) for dd in d]
	return np.cov(d, rowvar=0), np.cov(d2, rowvar=0)

def new_create_gaussian_distributions(params, v, n, seed=None):
	if seed == None:
		seed = np.random.randint(0, sys.maxint)
	np.random.seed(seed)
	alpha, beta, gamma = params
	A = np.linalg.cholesky(alpha*np.identity(n) + beta*np.ones((n,n)))
	if gamma == 0:
		b = np.zeros(n)
	else:
		b = np.random.normal(scale=gamma, size=n)
	ds = np.array([b + np.dot(A, np.random.normal(size=n)) for _ in xrange(v)])
	ds = np.array([[ds[i][j] for i in xrange(v)] for j in xrange(n)])
	ds = [softmax(d) for d in ds]
	return [(d, np.cumsum(d)) for d in ds], seed

def covtest4_both(params, v, n):
	alpha, beta, gamma = params
	A = np.linalg.cholesky(alpha*np.identity(n) + beta*np.ones((n,n)))
	if gamma == 0:
		b = np.zeros(n)
	else:
		b = np.random.normal(scale=gamma, size=n)
	ds = np.array([b + np.dot(A, np.random.normal(size=n)) for _ in xrange(v)])
	ds = np.array([[ds[i][j] for i in xrange(v)] for j in xrange(n)])
	d2 = [softmax(d) for d in ds]
	return np.cov(ds, rowvar=1), np.cov(d2, rowvar=1)

def covtest3(A, b, v, n):
	alpha, beta, gamma = params
	A = np.linalg.cholesky(alpha*np.identity(n) + beta*np.ones((n,n)))
	if gamma == 0:
		b = np.zeros(n)
	else:
		b = np.random.normal(scale=gamma, size=n)
	ds = np.array([b + np.dot(A, np.random.normal(size=n)) for _ in xrange(v)])
	ds = np.array([[ds[i][j] for i in xrange(v)] for j in xrange(n)])
	ds = [softmax(d) for d in ds]
	return np.cov(ds, rowvar=0)
	
	
	

from MOG_demo4 import *

fraction = 10
default_d = 6

def get_m():
	return init()[0]

def gradient_vs_finite_difference(phi_orig, bound_fn, gradient, d = default_d):
	M = len(phi_orig)
	h, hr = float('1e-' + str(d)), float('1e' + str(d))
	hij = h*np.eye(M)
	G = np.zeros((M))
	G2 = gradient
	for i in xrange(M):
		G[i] = hr*(bound_fn(phi_orig + hij[i]) - bound_fn(phi_orig))
	errors = []
	for i in xrange(M):
		error = 100*abs((G[i] - G2[i])/(G2[i]))
		errors.append(error)
		print G[i], ' vs ', G2[i], '------ rel. err. ', error, '%'
	errors = sorted(errors)
	errors = errors[M//fraction : M - M//fraction]
	print sum(errors)/len(errors)

def gradient_vs_gradient(gradient1, gradient2):
	M = len(gradient1)
	G = gradient1
	G2 = gradient2
	errors = []
	for i in xrange(M):
		error = 100*abs((G[i] - G2[i])/(G2[i]))
		errors.append(error)
		print G[i], ' vs ', G2[i], '------ rel. err. ', error, '%'
	errors = sorted(errors)
	errors = errors[M//fraction : M - M//fraction]
	print sum(errors)/len(errors)

def hessian_vs_finite_difference(phi_orig, bound_fn, hessian, d = default_d):
	M = len(phi_orig)
	print M
	h, hr = float('1e-' + str(d)), float('1e' + str(d))
	hij = h*np.eye(M)
	H = hessian
	H2 = np.zeros((M,M))
	for i in xrange(M):
		print i
		for j in xrange(M):
			H2[i][j] = hr*hr*(bound_fn(phi_orig + hij[i] + hij[j]) - bound_fn(phi_orig + hij[i]) \
						- bound_fn(phi_orig + hij[j]) + bound_fn(phi_orig))
	errors = []
	for i in xrange(M):
		for j in xrange(M):
			error = 100*abs((H[i][j] - H2[i][j])/(H2[i][j]))
			print H[i][j], 'vs', H2[i][j],' ------ rel. err. ', error, '%'
	errors = sorted(errors)
	errors = errors[M//fraction : M - M//fraction]
	print sum(errors)/len(errors)
			
def check_original_gradient_consistency(d = default_d):
	# 5.0 % for d = 6
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.get_brute_bound
	gradient, _ = m.vb_grad_natgrad()
	gradient_vs_finite_difference(phi_orig, bound_fn, gradient)

def check_original_gradient_vs_new_gradient(d = default_d):
	# 5.0 % for d = 6
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.autograd_bound()
	gradient, _ = m.vb_grad_natgrad()
	gradient_vs_finite_difference(phi_orig, bound_fn, gradient)

def check_original_gradient_vs_autograd(d = default_d):
	# 5.0 % for d = 6
	m = get_m()
	gradient1, _ = m.vb_grad_natgrad()
	gradient2 = m.bound_grad()(m.get_vb_param())
	gradient_vs_gradient(gradient1, gradient2)

def check_autograd_gradient_consistency(d = default_d):
	# reallu accurate << 10**(-3) % for d = 6
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.autograd_bound()
	gradient = m.bound_grad()(m.get_vb_param())
	gradient_vs_finite_difference(phi_orig, bound_fn, gradient)

def check_autograd_hessian_consistency(d = default_d):
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.autograd_bound()
	hessian = m.bound_hessian()(m.get_vb_param())
	hessian_vs_finite_difference(phi_orig, bound_fn, hessian)

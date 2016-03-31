from MOG_demo4 import *

def get_m():
	return init()[0]

def gradient_vs_finite_difference(phi_orig, bound_fn, gradient, d = 6):
	fraction = 10
	m = get_m()
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
	fraction = 10
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

def check_original_gradient_consistency(d = 4):
	# 5.0 % for d = 6
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.get_brute_bound
	gradient, _ = m.vb_grad_natgrad()
	gradient_vs_finite_difference(phi_orig, bound_fn, gradient)

def check_original_gradient_vs_new_gradient(d = 4):
	# 5.0 % for d = 6
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.autograd_bound()
	gradient, _ = m.vb_grad_natgrad()
	gradient_vs_finite_difference(phi_orig, bound_fn, gradient)

def check_original_gradient_vs_autograd(d = 4):
	# 5.0 % for d = 6
	m = get_m()
	gradient1, _ = m.vb_grad_natgrad()
	gradient2 = m.bound_grad()(m.get_vb_param())
	gradient_vs_gradient(gradient1, gradient2)

def check_autograd_gradient_consistency(d = 4):
	# reallu accurate << 10**(-3) % for d = 6
	m = get_m()
	phi_orig = m.get_vb_param().copy()
	bound_fn = m.autograd_bound()
	gradient = m.bound_grad()(m.get_vb_param())
	gradient_vs_finite_difference(phi_orig, bound_fn, gradient)

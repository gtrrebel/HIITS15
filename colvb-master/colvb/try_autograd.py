from __future__ import absolute_import
import autograd.numpy as np
import autograd.scipy as sp
import matplotlib.pyplot as plt
from autograd import (grad, elementwise_grad, jacobian, value_and_grad,
					  grad_and_aux, hessian_vector_product, hessian, multigrad,
					  jacobian, vector_jacobian_product)

default_d = 6

def finite_difference_hessian(x, fn, d = default_d):
	M = len(x)
	h, hr = float('1e-' + str(d)), float('1e' + str(d))
	hij = h*np.eye(M)
	H = np.zeros((M,M))
	for i in xrange(M):
		for j in xrange(M):
			H[i][j] = hr*hr*(fn(x + hij[i] + hij[j]) - fn(x + hij[i]) \
						- fn(x + hij[j]) + fn(x))
	return H

def test_fn(x):
	z = x
	z = z/z.sum(1, keepdims=True)
	return z[0][0]

def test_fn2(x):
	"This is the target"
	z = x.copy();
	z /= z.sum(1)[:, None]
	return z

def norm_t(a):
	return np.linalg.norm(a)**2

def sld(x):
	return np.linalg.slogdet(x)[1]

def template_test():
	d_fun	= elementwise_grad(test_fn)  # Second derivative
	dd_fun	= elementwise_grad(d_fun)
	x = np.array([[1.0,1.0, 1.0],[1.0,2.0, 3.0]])
	y = np.array([[2.0,3.0, 4.0],[5.0,6.0, 7.0]])
	#print test_fn(x)
	print "this is how it should go:\n", test_fn2(x)
	print "function:\n", test_fn(x)
	print "derivative:\n", d_fun(x)
	print "2. derivative:\n", dd_fun(x)
	print "function:\n", test_fn(y)
	print "derivative:\n", d_fun(y)
	print "2. derivative:\n", dd_fun(y)

def test_norm(d = default_d):
	x = np.array([1.0,2.0, 2.0])
	fn = norm_t
	g = elementwise_grad(fn)
	g2 = jacobian(fn)
	h = hessian(fn)
	print fn(x)
	print g(x)
	print g2(x)
	print h(x)
	print finite_difference_hessian(x, fn, d)

def test_sld():
	x = np.array([[1.0, 2.0], [3.0, 4.0]])
	f = sld
	g = jacobian(f)
	gp = elementwise_grad(f)
	h = hessian(f)
	print f(x)
	print g(x)
	print gp(x)
	print h(x)

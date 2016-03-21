from __future__ import absolute_import
import autograd.numpy as np
import autograd.scipy as sp
import matplotlib.pyplot as plt
from autograd import elementwise_grad


def test_fn(x):
	z = x
	z = z/z.sum(1, keepdims=True)
	return z[0][0]

def test_fn2(x):
	"This is the target"
	z = x.copy();
	z /= z.sum(1)[:, None]
	return z

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
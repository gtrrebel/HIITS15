from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad


def tanh(x):
	return np.linalg.cholesky(x)

def template_test():
	d_fun	= elementwise_grad(tanh)       # First derivative
	dd_fun	= elementwise_grad(d_fun)      # Second derivative
	ddd_fun	= elementwise_grad(dd_fun)      # Second derivative

	x = np.array([[5.0,2.0],[3.0,4.0]])
	x = np.array([[10.0,2.0],[3.0,4.0]])

	print ddd_fun(x)

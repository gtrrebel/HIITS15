from ad import adnumber
import numpy as np
import sys
sys.path.append('/home/othe/Desktop/HIIT/Moduleita/pyautodiff-python2-ast')
from autodiff import hessian_vector
from theano import function
import theano
import theano.tensor as T
'''
def f(x):
    return x*x + 1

def g(x):
    return x*x*x + 3

x = adnumber(3.0)
y = g(x)


def h1(x):
    return x*x + 1

@gradient
def h2(x):
    return x*x + 1



@hessian_vector
def hessianEval1(phi):
    bound = (phi**2).sum()
    return bound

a = np.ndarray(shape=(4,), buffer = np.array([1.,2.,3., 4.]))

print 'pyautodiff'

print hessianEval1(a, vectors = a)

'''
print 'theano'

d = 2

a = np.arange(d**2) + 1

def determinant(v):
	"""calculate the hessian of matrix determinant of matrix, matrix given in one row"""
	z = T.vector('z')
	w = (z**2).reshape((d,d))
	cost2 = T.nlinalg.det(w)
	input = [z]
	H2 = theano.gradient.hessian(cost2, wrt = input)
	f2 = theano.function(input, H2)
	return f2(v)[0]


def digam1(v):
	"""digamma"""
	z = T.vector('z')
	w = z.sum()
	cost = T.psi(w)
	input = [z]
	f2 = theano.function(input, cost)
	return f2(v)

def digam2(v):
	"""digamma"""
	z = T.scalar('z')
	cost = T.psi(z)
	input = [z]
	H2 = theano.gradient.jacobian(cost, wrt = input)
	f2 = theano.function(input, H2)
	return f2(v)[0].item(0)


print digam1(a)
print digam2(3.14) # vaiheessa
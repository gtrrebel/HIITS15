from ad import adnumber
import numpy as np
import sys
sys.path.append('/home/othe/Desktop/HIIT/Moduleita/pyautodiff-python2-ast')
from autodiff import hessian_vector
from theano import shared
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
	print(cost2)
	input = [z]
	H2 = theano.gradient.hessian(cost2, wrt = input)
	f2 = theano.function(input, H2)
	return f2(v)[0]

print theano1(a)
x = T.dvector('x')
y = x ** 2
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
f = function([x], H, updates=updates)

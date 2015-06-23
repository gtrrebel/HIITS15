import sys
import numpy as np
from scipy.special import gammaln, digamma
from scipy import sparse
import theano
import theano.typed_list
import theano.tensor as T

A = np.arange(9).reshape((3,3))
print A

def test1(A):
	x = T.dmatrix('x')
	w = x.reshape((3,3))
	y = theano.shared(np.arange(4).reshape((2,2)))
	z = T.set_subtensor(w[0:2, 0:2], w[0:2,0:2]**2)
	bound = z 
	input = [x]
	f1 = theano.function(input, bound)
	return f1(A)

print test1(A)
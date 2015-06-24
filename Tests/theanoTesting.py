import sys
import numpy as np
from scipy.special import gammaln, digamma
from scipy import sparse
import theano
import theano.typed_list
import theano.tensor as T

def test1(A):
	x = T.dmatrix('x')
	w = x.reshape((3,3))
	y = theano.shared(np.arange(4).reshape((2,2)))
	z = T.set_subtensor(w[0:2, 0:2], w[0:2,0:2]**2)
	bound = z 
	input = [x]
	f1 = theano.function(input, bound)
	return f1(A)

def test2():
	indeces = np.array([[0,4], [4, 8], [8, 12]], dtype=np.int8)
	inputs = T.dvector("inputs")
	outmat = T.dmatrix("outmat")

	def set_submatrix_at_position(indeces, inputs):
		A = theano.shared(np.array([1,2,3,4], dtype=np.float64))
		return T.set_subtensor(T.zeros((3,3))[:2,:2], inputs[indeces[0]: indeces[1]].reshape((2,2)))

	result, updates = theano.scan(fn=set_submatrix_at_position,
                              sequences=[indeces],
                              non_sequences=inputs
                              )
	bound = result.sum()
	grad = theano.gradient.jacobian(bound, wrt=inputs)
	f2 = theano.function(inputs=[inputs], outputs=grad)
	assign_values_at_positions = theano.function(inputs=[inputs], outputs=result)
	test_inputs = np.array([1,2, 3,4, 0.1, 0.2, 0.3, 0.4, 5, 5.2, 5.6, 6.0], dtype=np.float64)
	A = assign_values_at_positions(test_inputs)
	B = f2(test_inputs)
	print A
	print type(A)
	print A.shape
	print B
	print type(B)
	print B.shape

test2()
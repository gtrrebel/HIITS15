import numpy as np
from numpy import random
from scipy import linalg

class signlab():

	def __init__(self, eps = 1e-14):
		self.eps = eps

	def index(self, A, opt = 0, k = None, eps = None):
		if eps == None:
			eps = self.eps
		if k != None:
			return self.index(self.rand_minor(A, None), opt)
		if opt == 0:
			return self.bruteIndex(A)
		elif opt == 1:
			return self.gaussIndex(A)

	def eigenvalues(self, A):
		return linalg.eigh(A, eigvals_only=True)

	def pack(self, A, eps=None):
		if eps == None:
			eps = self.eps
		eigvals = linalg.eigh(A, eigvals_only=True)
		#print len(A), len(eigvals), sum(1 for eigval in eigvals if abs(eigval) < eps)
		return sum(1 for eigval in eigvals if eigval > eps), max(eigvals), min(eigvals), sum(1 for eigval in eigvals if abs(eigval) < eps)

	def largest(self, A):
		return linalg.eigh(A, eigvals_only=True, eigvals=(len(A) - 1, len(A) - 1))[0]

	def smallest(self, A):
		return linalg.eigh(A, eigvals_only=True, eigvals=(1, 1))[0]

	def greaterthan(self, A, eps=None):
		if eps == None:
			eps = self.eps
		eigvals = linalg.eigh(A, eigvals_only=True)
		return sum(1 for eigval in eigvals if eigval > eps)*1./len(A)

	def greaterthann(self, A, eps=None):
		if eps == None:
			eps = self.eps
		eigvals = linalg.eigh(A, eigvals_only=True)
		return sum(1 for eigval in eigvals if eigval > eps)

	def close_to_zero(self, A, eps=None):
		if eps == None:
			eps = self.eps
		eigvals = linalg.eigh(A, eigvals_only=True)
		return sum(1 for eigval in eigvals if abs(eigval) < eps)

	def col_minor(self, A, col):
		return A[col[:, np.newaxis], col]

	def rand_minor(self, A, k):
		if k == None:
			k = int(sqrt(len(A))) #default k
		col = np.array(random.choice(len(A), size=k))
		return self.col_minor(A, col)

	def bruteIndex(self, A):
		eigvals = linalg.eigh(A, eigvals_only=True)
		return sum(1 for eigval in eigvals if eigval < 0)*1./len(A)

	def gaussIndex(self, B):
		"""Determine the index (alpha) of the spectrum of the matrix A"""
		A = np.copy(B)
		n, neg= len(A), 0
		for i in range(n):
			if (A[i][i] < 0):
				neg += 1
			A[i] /= A[i][i]
			for j in range(i + 1, n):
				A[j] -= A[i]*A[j][i]
		return neg*1./n

	def printHessian(self, A, opt = 3):
		"""Print the hessian of the function. 
				opt > 0 for length of output strings for any number, 
				opt = 0 for stars (*) for nonzero and spaces otherwise
				opt < 0 for stars (*) for number of absolute value > -opt

		"""
		if opt == 0:
			help = lambda x: '  ' if (str(x)[0] == '0') else '* '
		elif opt < 0:
			help = lambda x: '  ' if (abs(x) < -opt) else '* '
		else:
			help = lambda x: (str(x) + opt*' ')[:opt] + '   '
		s = ''
		for b in A:
			for a in b:
				s += help(a)
			s += '\n'
		print s
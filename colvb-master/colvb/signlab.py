import numpy as np
from numpy import random
from scipy import linalg

class signlab():

	def __init__(self, eps = 1e-14):
		self.eps = eps
		self.build_functions(eps)

	def set_epsilon(self, *args):
		self.__init__(*args)

	def build_functions(self, eps):
		self.functions = {}
		self.functions['largest'] = lambda eigs: max(eigs)
		self.functions['smallest'] = lambda eigs: min(eigs)
		self.functions['positive'] = lambda eigs: sum(eig > 0 for eig in eigs)
		self.functions['negative'] = lambda eigs: sum(eig < 0 for eig in eigs)
		self.functions['zero'] = lambda eigs: sum(eig == 0 for eig in eigs)
		self.functions['epsilon_positive'] = lambda eigs: sum(eig > eps for eig in eigs)
		self.functions['epsilon_negative'] = lambda eigs: sum(eig < -eps for eig in eigs)
		self.functions['epsilon_zero'] = lambda eigs: sum(abs(eig) < eps for eig in eigs)
		self.functions['index'] = lambda eigs: self.functions['positive'](eigs)*1./len(eigs)
		self.functions['epsilon_index'] = lambda eigs: self.functions['epsilon_positive'](eigs)*1./len(eigs)

	def eigenvalues(self, A):
		return linalg.eigh(A, eigvals_only=True)

	def get_info(self, A, infos, eps=None):
		if (eps != None) and (eps != self.eps):
			self.set_epsilon(eps)
		eigs = self.eigenvalues(A)
		return [self.functions[info](eigs) for info in infos]

	def pack(self, A):
		return self.get_info(A, ['epsilon_positive', 'largest', 'smallest', 'epsilon_zero'])

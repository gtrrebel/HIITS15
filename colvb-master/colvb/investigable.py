from signlab import signlab
from signlab2 import signlab2
from signlab3 import *
from runspecs import runspecs
from model_display import model_display
from scipy import linalg
import scipy
import numpy as np
import time


class investigable():

	def __init__(self):
		self.lab = signlab(self.eps)
		self.lab2 = signlab2()
		self.runspecs = runspecs()
		self.road_data = {}
		self.end_data = {}
		self.road_gather = []
		self.end_gather = []
		self.md = model_display(self)

	def road(self):
		index_gathers = []
		for gather in self.road_gather:
			if hasattr(self, gather):
				self.road_data[gather].append(getattr(self, gather)())
			else:
				index_gathers.append(gather)
		if index_gathers:
			index_infos = self.lab.get_info(self.get_hessian(), index_gathers)
			for gather, info in zip(index_gathers, index_infos):
				self.road_data[gather] = info

	def end(self):
		index_gathers = []
		for gather in self.end_gather:
			if hasattr(self, gather):
				self.end_data[gather] = getattr(self, gather)()
			else:
				index_gathers.append(gather)
		if index_gathers:
			index_infos = self.lab.get_info(self.get_hessian(), index_gathers)
			for gather, info in zip(index_gathers, index_infos):
				self.end_data[gather] = info

	def get_bound(self):
		raise NotImplementedError( "Implement bound calculating method \"get_bound\"")

	def get_gradient(self):
		raise NotImplementedError( "Implement gradient calculating method \"get_gradient\"")

	def get_hessian(self):
		raise NotImplementedError( "Implement hessian calculating method \"get_hessian\"")

	def get_param(self):
		raise NotImplementedError( "Implement parameter returning method \"get_param\"")

	def get_vb_param(self):
		raise NotImplementedError( "Implement vb parameter returning method \"get_vb_param\"")

	def finite_difference_check(self, hessian_check = False, gradient_check=False, brute_hessian_check = True):
		phi_orig = self.get_vb_param().copy()
		phi_orig2 = self.get_param().copy()
		M = len(phi_orig)
		d = 3
		h, hr = float('1e-' + str(d)), float('1e' + str(d))
		hij = h*np.eye(M)
		if hessian_check:
			H = np.zeros((M,M))
			H2 = self.get_hessian()
			for i in xrange(M):
				for j in xrange(M):
					H[i][j] = hr*hr*(self.f1(phi_orig + hij[i] + hij[j]) - self.f1(phi_orig + hij[i]) - self.f1(phi_orig + hij[j]) + self.f1(phi_orig))
			for i in xrange(M):
				for j in xrange(M):
					if abs(H[i][j]) > h:
						print H[i][j], ' vs ', H2[i][j], ' ------ rel. err. ', 100*abs((H[i][j] - H2[i][j])/(H2[i][j])), '%'
		if gradient_check:
			G = np.zeros((M))
			G2, G3 = self.vb_grad_natgrad_test()
			for i in xrange(M):
				G[i] = hr*(self.f1(phi_orig + hij[i]) - self.f1(phi_orig))
			for i in xrange(M):
				if abs(G[i]) + abs(G2[i]) > h:
					print G[i], ' vs ', G2[i], ' ------ rel. err. ', 100*abs((G[i] - G2[i])/(G2[i])), '%'
		if brute_hessian_check:
			H = np.zeros((M,M))
			H2 = self.brutehessian()
			for i in xrange(M):
				for j in xrange(M):
					H[i][j] = hr*hr*(self.f1(phi_orig + hij[i] + hij[j]) - self.f1(phi_orig + hij[i]) - self.f1(phi_orig + hij[j]) + self.f1(phi_orig))
			for i in xrange(M):
				for j in xrange(M):
					if abs(H[i][j]) > h:
						print H[i][j], ' vs ', H2[i][j], ' ------ rel. err. ', 100*abs((H[i][j] - H2[i][j])/(H2[i][j])), '%'
	
	def finite_difference_check2(self, hessian_check = False, gradient_check=False, brute_hessian_check = True, terms = [1,2,3,4,5], change = True):
		phi_orig = self.get_vb_param().copy()
		phi_orig2 = self.get_param().copy()
		M = len(phi_orig)
		d = 3
		h, hr = float('1e-' + str(d)), float('1e' + str(d))
		hij = h*np.eye(M)
		if hessian_check:
			H = np.zeros((M,M))
			H2 = self.get_hessian()
			for i in xrange(M):
				for j in xrange(M):
					H[i][j] = hr*hr*(self.f1(phi_orig + hij[i] + hij[j]) - self.f1(phi_orig + hij[i]) - self.f1(phi_orig + hij[j]) + self.f1(phi_orig))
			for i in xrange(M):
				for j in xrange(M):
					if abs(H[i][j]) > h:
						print H[i][j], ' vs ', H2[i][j], ' ------ rel. err. ', 100*abs((H[i][j] - H2[i][j])/(H2[i][j])), '%'
		if gradient_check:
			G = np.zeros((M))
			G2, G3 = self.vb_grad_natgrad_test()
			for i in xrange(M):
				G[i] = hr*(self.f1(phi_orig + hij[i]) - self.f1(phi_orig))
			for i in xrange(M):
				if abs(G[i]) + abs(G2[i]) > h:
					print G[i], ' vs ', G2[i], ' ------ rel. err. ', 100*abs((G[i] - G2[i])/(G2[i])), '%'
		if brute_hessian_check:
			self.make_functions2(terms = terms, change = change)
			if change:
				H = self.f3(self.get_vb_param())
			else:
				H = self.f3(self.get_param())
			H2 = self.brutehessian(terms=terms, change=change)
			H3 = np.zeros((M,M))
			for i in xrange(M):
				for j in xrange(M):
					H3[i][j] = hr*hr*(self.f1(phi_orig + hij[i] + hij[j]) - self.f1(phi_orig + hij[i]) - self.f1(phi_orig + hij[j]) + self.f1(phi_orig))
			for i in xrange(M):
				for j in xrange(M):
					print H[i][j], 'vs', H3[i][j], ' vs ', H2[i][j], ' ------ rel. err. ', 100*abs((H3[i][j] - H2[i][j])/(H2[i][j])), '%'

	def end_print(self):
		if self.runspecs['prints']['runtime_distribution']:
			self.print_runtime_distribution()
		if self.runspecs['display']['eigenvalue_histograms']:
			self.print_eigenvalue_histograms()
		for gather in self.end_data:
			print gather + ': ' + str(self.end_data[gather])

	def end_return(self):
		end_return = {}
		end_return['method'] = self.method
		for gather in self.end_data:
			end_return[gather] = self.end_data[gather]
		return end_return

	def voc_size(self):
		return self.V

	def end_display(self):
		self.md.display()

	def end_basic_plots(self):
		self.md.display()

	def cheb_index(self):
		return self.lab2.chebyshev_index(self.get_hessian())

	def eigenvalues(self):
		return self.lab.eigenvalues(self.get_hessian())

	def dimension(self):
		return len(self.get_param())

	def get_seed(self):
		return self.seed

	def reduced_dimension(self):
		return (self.K -1)*self.N*self.D

	def negative_definite(self):
		try:
			np.linalg.cholesky(-self.get_hessian())
			return 0
		except np.linalg.linalg.LinAlgError:
			return 1

	def positive_definite(self):
		A = -self.get_hessian()
		if self.lab2.positive_definite(A):
			return 0
		else:
			return 1

	def return_m(self):
		return self

	def return_hessian(self):
		return self.get_hessian()

	def power_largest(self, tol=1E-6, maxiter=5000, sigma=0.1):
		return largest_eigenvalue(self.get_hessian(), tol, maxiter, sigma)

	def power_smallest(self):
		return largest_eigenvalue(self.get_hessian(), tol, maxiter, sigma)

	def inverse_eigenvalues(self):
		inv = np.linalg.inv(self.get_hessian())
		eigs = 1/self.lab.eigenvalues(inv)
		return sum(1 for i in eigs if abs(i) < self.eps)

	def eigenvectors(self):
		return np.linalg.eig(self.get_hessian())

	def how_far(self):
		return linalg.norm(np.array(self.get_param()) - self.orig_params)

	def distance_travelled(self):
		return self.travelled_distance

	def rank(self):
		return np.linalg.matrix_rank(self.get_hessian())

	def det(self):
		return np.linalg.det(self.get_hessian())

	def collapse(self, hessian):
		collapsed = np.zeros((self.D*self.N*(self.K - 1), self.D*self.N*(self.K - 1)))
		for i1 in xrange(self.K - 1):
			for i2 in xrange(self.K - 1):
				for j1 in xrange(self.D*self.N):
					for j2 in xrange(self.D*self.N):
						collapsed[j1*(self.K - 1)+i1][j2*(self.K - 1)+i2] = hessian[j1*self.K+i1][j2*self.K+i2] - \
																			hessian[j1*self.K+i1][j2*self.K+self.K-1] - \
																			hessian[j1*self.K+self.K-1][j2*self.K+i2] + \
																			hessian[j1*self.K+self.K-1][j2*self.K+self.K-1]
		self.collapsed = collapsed
		return collapsed

	def set_invests(self, road_gather=[], end_gather=[]):
		self.road_data = {}
		self.end_data = {}
		self.road_gather = road_gather
		for gather in self.road_gather:
			self.road_data[gather] = []
		self.end_gather = end_gather

	def print_runtime_distribution(self):
		print 'size: ', len(m.get_vb_param()), '\n', \
			'optimize_time: ', m.optimize_time, '\n', \
			'hessian_time: ', m.hessian_time, ' - ', (100*m.hessian_time/m.optimize_time), '%,\n', \
			'pack_time: ', m.pack_time, ' - ', (100*m.pack_time/m.optimize_time), '%,\n', \
			'others: ', (m.optimize_time - m.hessian_time - m.pack_time), \
			' - ', (100*(m.optimize_time - m.hessian_time - m.pack_time)/m.optimize_time), '%'

	def optimizetime(self):
		return self.optimize_time

	def print_eigenvalue_histograms(self):
		pb.figure()
		self.v2.eigenvalue_histogram(m.eigenvalues())
		pb.xlabel(m.bound())
		pb.show()

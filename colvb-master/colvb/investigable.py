from signlab import signlab
from signlab2 import signlab2
from runspecs import runspecs
from vis1 import vis1
from vis2 import vis2
from scipy import linalg
import numpy as np


class investigable():

	def __init__(self):
		self.lab = signlab(self.eps)
		self.lab2 = signlab2()
		self.runspecs = runspecs()
		self.road_data = {}
		self.end_data = {}
		self.road_gather = []
		self.end_gather = []
		v1 = vis1()
		v2 = vis2()

	def road(self):
		index_gathers = []
		for gather in self.road_gather:
			print 'fail'
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

	def finite_difference_check(self):
		phi_orig = self.get_param().copy()
		M = len(phi_orig)
		h = 1e-4
		hr = 1e4
		hij = np.zeros((M, M))
		H = np.zeros((M,M))
		G = np.zeros((M))
		H2 = self.hessian()
		G2 = self.gradient()
		for i in range(M):
			hij[i][i] = h
		for i in range(M):
			G[i] = hr*(self.f1(phi_orig + hij[i]) - self.f1(phi_orig))
		for i in range(M):
			for j in range(M):
				H[i][j] = hr*hr*(self.f1(phi_orig + hij[i] + hij[j]) - self.f1(phi_orig + hij[i]) - self.f1(phi_orig + hij[j]) + self.f1(phi_orig))
		for i in range(M):
			for j in range(M):
				print H[i][j], ' vs ', H2[i][j], ' ------ rel. err. ', 100*abs((H[i][j] - H2[i][j])/(H2[i][j])), '%'

	def end_print(self):
		if self.runspec_get('runtime_distribution'):
			self.print_runtime_distribution()
		if self.runspec_get('eigenvalue_histograms'):
			self.print_eigenvalue_histograms()
		for gather in self.end_data:
			print gather + ': ' + str(self.end_data[gather])

	def end_basic_plots(self):
		if self.runspec_get('basic_plots'):
			for pair in self.runspec_get('plot_specs'):
				self.v1.plot_stack(pair[0], pair[1])

	def cheb_index(self):
		return self.lab2.chebyshev_index(self.get_hessian())

	def eigenvalues(self):
		return self.lab.eigenvalues(self.get_hessian())

	def inverse_eigenvalues(self):
		inv = np.linalg.inv(self.get_hessian())
		eigs = 1/self.lab.eigenvalues(inv)
		return sum(1 for i in eigs if abs(i) < self.eps)

	def how_far(self):
		return linalg.norm(np.array(self.get_param()) - self.orig_params)

	def distance_travelled(self):
		return self.distance_travelled

	def rank(self):
		return np.linalg.matrix_rank(self.get_hessian())

	def det(self):
		return np.linalg.det(self.get_hessian())

	def set_invests(self, road_gather=[], end_gather=[]):
		self.road_data = {}
		self.end_data = {}
		self.road_gather = road_gather
		for gather in self.road_gather:
			self.road_data[gather] = []
		self.end_gather = end_gather

	def runspec_set(self, spec, value):
		if spec=='eps':
			self.lab.set_epsilon(value)
		self.runspecs.set(spec, value)

	def runspec_do(self, spec):
		self.runspecs.do(spec)

	def runspec_undo(self, spec):
		self.runspecs.undo(spec)

	def runspec_get(self, spec):
		return self.runspecs.__getitem__(spec)

	def print_runtime_distribution(self):
		print 'size: ', len(m.get_vb_param()), '\n', \
			'optimize_time: ', m.optimize_time, '\n', \
			'hessian_time: ', m.hessian_time, ' - ', (100*m.hessian_time/m.optimize_time), '%,\n', \
			'pack_time: ', m.pack_time, ' - ', (100*m.pack_time/m.optimize_time), '%,\n', \
			'others: ', (m.optimize_time - m.hessian_time - m.pack_time), \
			' - ', (100*(m.optimize_time - m.hessian_time - m.pack_time)/m.optimize_time), '%'

	def print_eigenvalue_histograms(self):
		pb.figure()
		self.v2.eigenvalue_histogram(m.eigenvalues())
		pb.xlabel(m.bound())
		pb.show()
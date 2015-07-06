from signlab import signlab
from runspecs import runspecs
from vis1 import vis1
from vis2 import vis2
from scipy import linalg


class investigable():

	def __init__(self, eps=1e-14, road_gather=[], end_gather=[]):
		self.lab = signlab(self.eps)
		self.runspecs = runspecs()
		v1 = vis1()
		v2 = vis2()

	def road(self):
		for gather in self.road_gather:
			self.road_data[gather].append(getattr(self, gather)())

	def end(self):
		self.end_data = {}
		for gather in self.end_gather:
			self.end_data[gather] = getattr(self, gather)()

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

	def eigenvalues(self):
		return self.lab.eigenvalues(self.get_hessian())

	def how_far(self):
		return linalg.norm(np.array(self.get_param()) - self.orig_params)

	def set_invests(self, road_gather=[], end_gather=[]):
		self.road_data = {}
		self.end_data = {}
		self.road_gather = road_gather
		for gather in self.road_gather:
			self.road_data[gather] = []
		self.end_gather = end_gather

	def runspec_set(self, spec, value):
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
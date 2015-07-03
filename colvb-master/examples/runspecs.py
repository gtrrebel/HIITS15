
class runspecs():

	def __init__(self):
		self.invest_map = ['spectrum_length', 'final_bound', 'distance_travelled', 'distance_from_start']

		self.runspecs = {}

		self.runspecs['restarts'] = 1
		self.runspecs['plotstart'] = 3
		self.runspecs['methods'] = ['steepest']
		self.runspecs['plot_specs'] = [('iter', 'index'), ('iter', 'bound')]
		self.runspecs['eps'] = 1e-14
		self.runspecs['finite_difference_check'] = False
		self.runspecs['hessian_freq'] = 1

		self.runspecs['runtime_distribution'] = False
		self.runspecs['eigenvalue_histograms'] = False
		self.runspecs['basic_plots'] = False
		self.runspecs['print_convergence'] = False

		self.runspecs['spectrum_length'] = False
		self.runspecs['final_bound'] = False
		self.runspecs['distance_travelled'] = False
		self.runspecs['distance_from_start'] = False

		self.runspecs['orig_track_display'] = False
		self.runspecs['orig_learned_topics'] = False
		self.runspecs['orig_true_topics'] = False

	def __getitem__(self, spec):
		return self.runspecs[spec]

	def set(self, spec, value):
		self.runspecs[spec] = value

	def do(self, spec):
		self.runspecs[spec] = True

	def nodo(self, spec):
		self.runspecs[spec] = False

	def default(self):
		self.__init__()

	def invest_names(self, i):
		return self.invest_map[i]

	def set_invest(self, i):
		self.do(self.invest_map[i])
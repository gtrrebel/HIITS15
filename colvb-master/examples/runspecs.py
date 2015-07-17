
class runspecs():

	def __init__(self):
		self.invest_map = ['spectrum_length', 'final_bound', 'distance_travelled', 'distance_from_start']

		self.runspecs = {}

		self.runspecs['basics'] = {}
		self.runspecs['prints'] = {}
		self.runspecs['display'] = {}
		self.runspecs['tests'] = {}

		self.runspecs['basics']['restarts'] = 1
		self.runspecs['basics']['hessian_freq'] = 1
		self.runspecs['basics']['eps'] = 1e-14
		self.runspecs['basics']['methods'] = ['steepest']

		self.runspecs['prints']['plotstart'] = 3
		self.runspecs['prints']['runtime_distribution'] = False
		self.runspecs['prints']['print_convergence'] = False
		self.runspecs['prints']['spectrum_length'] = False
		self.runspecs['prints']['final_bound'] = False
		self.runspecs['prints']['distance_travelled'] = False
		self.runspecs['prints']['distance_from_start'] = False

		self.runspecs['display']['plot_specs'] = [('iter', 'index'), ('iter', 'bound')]
		self.runspecs['display']['eigenvalue_histograms'] = False
		self.runspecs['display']['basic_plots'] = False
		self.runspecs['display']['orig_track_display'] = False
		self.runspecs['display']['orig_learned_topics'] = False
		self.runspecs['display']['orig_true_topics'] = False

		self.runspecs['tests']['finite_difference_check'] = False

	def __getitem__(self, spec):
		return self.runspecs[spec]

	def set(self, spec, value):
		self.runspecs[spec] = value

	def do(self, spec):
		self.runspecs[spec] = True

	def undo(self, spec):
		self.runspecs[spec] = False

	def default(self):
		self.__init__()

	def invest_names(self, i):
		return self.invest_map[i]

	def set_invest(self, i):
		self.do(self.invest_map[i])
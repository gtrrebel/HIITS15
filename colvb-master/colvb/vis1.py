import numpy as np
import pylab as pb

class vis1():

	def __init__(self):
		self.keys = {'index': 0, 'max': 1, 'min':2, 'close':3, 'bound': 4}
		self.infos = []

	def stack(self, info):
		self.infos.append(info)

	def empty(self):
		self.infos = []

	def plot_stack(self, x, y):
		self.infos = sorted(self.infos, cmp=lambda i1, i2: int(np.sign(i1[-1][self.keys['bound']] - i2[-1][self.keys['bound']])))
		i = 0
		for info in self.infos:
			i += 1
			self.plot(x, y, info, label=str(i) + ': ' + str(info[-1][self.keys['bound']]))

	def plot_stack_aver(self, x, y):
		ma = max([len(info) for info in self.infos])
		for info in self.infos:
			info.extend((ma - len(info))*[info[-1]])
		m, n, k = len(self.infos), len(self.infos[0]), len(self.infos[0][0])
		new_info = [[sum([info[i][j]*1.0/m for info in self.infos]) for j in range(k)] for i in range(n)]
		self.plot(x,y,new_info)


	def plot(self, x, y, pairs, label=''):
		X = self.data(x, pairs)
		Y = self.data(y, pairs)
		pb.plot(X,Y, label=label)
		pb.xlabel(x)
		pb.ylabel(y)
		pb.legend(loc='lower right')

	def data(self, what, pairs):
		if what in self.keys:
			return [pair[self.keys[what]] for pair in pairs]
		else:
			return getattr(self, what)(pairs)

	def dist(self, pairs):
		bound = pairs[-1][self.keys['bound']]
		return [bound - pair[self.keys['bound']] for pair in pairs]

	def iter(self, pairs):
		return range(len(pairs))

	def speclen(self, pairs):
		return [pair[self.keys['max']] - pair[self.keys['min']] for pair in pairs]

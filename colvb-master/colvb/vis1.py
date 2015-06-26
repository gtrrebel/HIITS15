import numpy as np
import pylab as pb

class vis1():

	def __init__(self):
		self.keys = {'index': 0, 'max': 1, 'min':2, 'bound': 3}

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

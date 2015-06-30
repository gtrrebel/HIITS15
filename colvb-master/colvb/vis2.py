import numpy as np
import pylab as pb

class vis2():

	def __init__(self):
		pass

	def eigenvalue_histogram(self, eigs, bins = 20):
		eigs = sorted(eigs)
		pb.hist(eigs, bins=bins)
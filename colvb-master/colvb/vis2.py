import numpy as np
import pylab as pb

class vis2():

	def eigenvalue_histogram(self, eigs, bins = 20):
		eigs = sorted(eigs)
		pb.hist(eigs, bins=bins)
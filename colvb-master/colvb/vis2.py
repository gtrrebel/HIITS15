import numpy as np
import pylab as pb

class vis2():

	@staticmethod
	def eigenvalue_histogram(eigs, bins = 20):
		eigs = sorted(eigs)
		pb.hist(eigs, bins=bins)

	@staticmethod
	def dist_histogram(dists, bins = 20):
		pb.hist(dists, bins=bins)
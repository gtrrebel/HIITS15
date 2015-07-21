import numpy as np
import random

class label_switcher():

	def __init__(self, ms):
		self.T = 100
		self.ms = ms
		self.M = len(ms)
		self.D, self.N, self.K = list(ms[0].shape)
		self.perms = np.array([range(self.K) for _ in xrange(self.M)])

	def calc_average(self):
		self.aver = np.zeros((self.D, self.N, self.K))
		for i1 in xrange(self.D):
			for i2 in xrange(self.N):
				for i3 in xrange(self.K):
					ent = 0
					for i4 in xrange(self.M):
						ent += self.ms[i4][i1][i2][self.perms[i4][i3]]
					self.aver[i1][i2][i3] = ent/self.M

	def KL_div(self, i, perm):
		div = 0
		for i1 in xrange(self.D):
			for i2 in xrange(self.N):
				for i3 in xrange(self.K):
					div += self.ms[i][i1][i2][perm[i3]] * np.log(self.ms[i][i1][i2][perm[i3]]/(self.aver[i1][i2][i3]))
		return div

	def shuffle(self, i):
		perm = self.perms[i].copy()
		ss = random.randint(2, 3)
		ran = random.sample(xrange(self.K), ss)
		prev = perm[ran]
		for j in xrange(ss):
			perm[ran[j]] = prev[(j + 1) % ss]
		return perm

	def change(self, i):
		for j in xrange(self.T):
			shuf = self.shuffle(i)
			if self.KL_div(i, shuf) < self.KL_divs[i]:
				self.perms[i] = shuf

	def change_all(self):
		self.calc_average()
		self.KL_divs = [self.KL_div(i, self.perms[i]) for i in xrange(self.M)]
		self.total_bound = sum(self.KL_divs)
		for i in xrange(self.M):
			self.change(i)

	def find_labels(self):
		self.change_all()
		old = self.total_bound
		while True:
			self.change_all()
			if old <= self.total_bound:
				break
			old = self.total_bound

	def switch_labels(self):
		self.find_labels()
		for i in xrange(self.M):
			self.switch(i)

	def switch(self, i):
		old = self.ms[i].copy()
		for i1 in xrange(self.D):
			for i2 in xrange(self.N):
				old = self.ms[i][i1][i2].copy()
				for i3 in xrange(self.K):
					self.ms[i][i1][i2][i3] = old[self.perms[i][i3]]

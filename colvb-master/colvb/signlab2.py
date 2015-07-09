import numpy as np

class signlab2():

	def __init__(self, a=0.0, b=0.5, deg= 80, jack=False):
		self.a, self.b, self.deg, self.jack = a, b, deg, jack
		self.chebyshev_coeff()

	def chebyshev_coeff(self):
		v = np.arange(1, self.deg + 1)
		v = 2.0/np.pi*(np.sin(v*np.arccos(self.a)) - np.sin(v*np.arccos(self.b)))/v
		gamma0 = np.array([1./np.pi*(np.arccos(self.a) - np.arccos(self.b))])
		gammas = np.concatenate((gamma0, v))
		if self.jack:
			alpha = np.pi/(self.deg + 2)
			jackcoeff = np.sin(alpha*(np.arange(1, self.deg + 2)))/((self.deg + 2)*np.sin(alpha))+ \
				(1-np.arange(1, self.deg + 2)/(self.deg + 2))*np.cos(alpha*np.arange(self.deg + 1))
			gammas *= jackcoeff
		self.gammas = gammas

	def calc(self, coeff, A, v):
		deg = len(coeff) - 1
		vs = [v, A.dot(v)]
		for i in xrange(2, deg + 1):
			vs.append(2*A.dot(vs[-1]) - vs[-2])
		return sum(coeff[i]*v.transpose().dot(vs[i]) for i in xrange(deg + 1))

	def chebyshev_index(self, A, m=500, deg = None, a = None, b = None, jack=None):
		change = False
		if a != None and abs(a - self.a) > 1e-2:
			change = True
			self.a = a
		if b != None and abs(b - self.b) > 1e-2:
			change = True
			self.b = b
		if deg != None and (deg != self.deg):
			change = True
			self.deg = deg
		if jack != None and (jack != self.jack):
			change = True
			self.jack = jack
		if change:
			self.chebyshev_coeff()
		count = 0
		n = len(A)
		for i in xrange(m):
			v = 2*np.random.randint(2, size=n)-1
			count += self.calc(self.gammas, A, v)
		return int(round(count/m))

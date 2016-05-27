# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)
from __future__ import absolute_import

import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/Moduleita/pyautodiff-python2-ast')
sys.path.append('/home/fs/othe/Windows/Desktop/hiit/HIITS15/Moduleita/pyautodiff-python2-ast')
import numpy as np
import pylab as pb
from scipy import optimize, linalg
from utilities import softmax, multiple_pdinv, lngammad, ln_dirichlet_C
from scipy.special import gammaln, digamma, polygamma
from brute_hessian import var_change
from scipy import stats
from scipy import linalg
from col_mix2 import collapsed_mixture2
import theano.tensor as T
from theano.tensor import nlinalg, slinalg
from theano import *
import time
import random
import matplotlib.pyplot as plt

import autograd.numpy as np_a
import autograd.scipy as sp_a
import matplotlib.pyplot as plt
from autograd.util import *
from autograd import (grad, elementwise_grad, jacobian, value_and_grad,
					  grad_and_aux, hessian_vector_product, hessian, multigrad,
					  jacobian, vector_jacobian_product)

class MOG2(collapsed_mixture2):
	"""
	A Mixture of Gaussians
	"""
	def __init__(self, X, K=2, prior_Z='symmetric', alpha=10., eps=1e-9, finite_difference_checks=False, make_fns = False, seed=None):
		self.eps = eps
		self.finite_difference_checks = finite_difference_checks
		self.X = X
		self.N, self.D = X.shape

		#prior cluster parameters
		self.m0 = self.X.mean(0) # priors on the Gaussian components
		self.k0 = 1e-6
		self.S0 = np.eye(self.D)*1e-3
		self.S0_halflogdet = np.sum(np.log(np.sqrt(np.diag(self.S0))))
		self.v0 = self.D+1.

		#precomputed stuff
		self.k0m0m0T = self.k0*self.m0[:,np.newaxis]*self.m0[np.newaxis,:]
		self.XXT = self.X[:,:,np.newaxis]*self.X[:,np.newaxis,:]
		collapsed_mixture2.__init__(self, self.N, K, prior_Z, alpha)
		self.make_fns = make_fns
		if make_fns:
			self.make_functions()

	def do_computations(self):
		#computations needed for bound, gradient and predictions
		self.kNs = self.phi_hat + self.k0
		self.vNs = self.phi_hat + self.v0
		self.Xsumk = np.tensordot(self.X,self.phi,((0),(0))) #D x K
		Ck = np.tensordot(self.phi, self.XXT,((0),(0))).T# D x D x K
		self.mun = (self.k0*self.m0[:,None] + self.Xsumk)/self.kNs[None,:] # D x K
		self.munmunT = self.mun[:,None,:]*self.mun[None,:,:]
		self.Sns = self.S0[:,:,None] + Ck + self.k0m0m0T[:,:,None] - self.kNs[None,None,:]*self.munmunT
		self.Sns_inv, self.Sns_halflogdet = multiple_pdinv(self.Sns)

	def bound(self):
		"""Compute the lower bound on the model evidence.  """
		return -0.5*self.D*np.sum(np.log(self.kNs/self.k0))\
			+self.K*self.v0*self.S0_halflogdet - np.sum(self.vNs*self.Sns_halflogdet)\
			+np.sum(lngammad(self.vNs, self.D))- self.K*lngammad(self.v0, self.D)\
			+self.mixing_prop_bound()\
			+self.H\
			-0.5*self.N*self.D*np.log(np.pi)

	def vb_grad_natgrad(self):
		"""Gradients of the bound"""
		x_m = self.X[:,:,None]-self.mun[None,:,:]
		dS = x_m[:,:,None,:]*x_m[:,None,:,:]
		SnidS = self.Sns_inv[None,:,:,:]*dS
		dlndtS_dphi = np.dot(np.ones(self.D), np.dot(np.ones(self.D), SnidS))

		grad_phi =  (-0.5*self.D/self.kNs + 0.5*digamma((self.vNs-np.arange(self.D)[:,None])/2.).sum(0) + self.mixing_prop_bound_grad() - self.Sns_halflogdet -1.) + (self.Hgrad-0.5*dlndtS_dphi*self.vNs)

		natgrad = grad_phi - np.sum(self.phi*grad_phi, 1)[:,None] # corrects for softmax (over) parameterisation
		grad = natgrad*self.phi

		return grad.flatten(), natgrad.flatten()

	def vb_grad_natgrad_test(self, terms = [1,2,3,4,5], change = True):
		"""Gradients of the bound"""
		#print self.K, self.D, self.N
		x_m = self.X[:,:,None]-self.mun[None,:,:]
		dS = x_m[:,:,None,:]*x_m[:,None,:,:]
		SnidS = self.Sns_inv[None,:,:,:]*dS
		dlndtS_dphi = np.dot(np.ones(self.D), np.dot(np.ones(self.D), SnidS))
		grad_phi = np.zeros((self.N, self.K))
		if 1 in terms:
			grad_phi += -1 + self.Hgrad
		if 2 in terms:
			grad_phi += -0.5*self.D/self.kNs
		if 3 in terms:
			grad_phi += self.mixing_prop_bound_grad()
		if 4 in terms:
			grad_phi += 0.5*digamma((self.vNs-np.arange(self.D)[:,None])/2.).sum(0)
		if 5 in terms:
			grad_phi += -0.5*dlndtS_dphi*self.vNs - self.Sns_halflogdet
		
		if change:
			natgrad = grad_phi - np.sum(self.phi*grad_phi, 1)[:,None] # corrects for softmax (over) parameterisation
			grad = natgrad*self.phi
			return grad.flatten(), natgrad.flatten()
		return grad_phi

	def make_functions(self):
		"""Initializes the theano-functions for
			f1 = regular theano evaluation of the bound
			f2 = gradient of the bound wrt r_nk:s (phi_:s)
			f3 = hessian of the bound wrt r_nk:s (phi_:s)
		"""

		#Gather the bound
		x = T.dvector('x')
		phi_ = x.reshape((self.N, self.K))
		phi = T.exp(phi_)
		phi /= phi.sum(1)[:, None]
		phi_hats = phi.sum(0)
		alphas = self.alpha + phi_hats
		kappas = self.k0 + phi_hats
		nus = self.v0 + phi_hats
		Ybars = T.tensordot(self.X, phi,((0),(0)))
		Cks = T.tensordot(phi, self.XXT,((0),(0))).T
		mks = (self.k0*self.m0[:,None] + Ybars)/kappas[None,:]
		mkprods = mks[:,None,:]*mks[None,:,:] #product of mk and it's transpose
		Sks = self.S0[:,:,None] + Cks + self.k0m0m0T[:,:,None] - kappas[None,None,:]*mkprods
		bound = 0
		bound += -T.tensordot(T.log(phi + 1e-10), phi) #entropy H_L                                 #toimii
		bound += -self.D/2. * T.log(kappas).sum()                                                   #toimii
		bound += T.gammaln(alphas).sum()                                                            #toimii (ainakin symmetric)
		bound += T.gammaln((nus-T.arange(self.D)[:,None])/2.).sum()                                 #toimii
		boundH = 0                                                                                  #melko hyvin
		for k in range(self.K):
			boundH += 0.5*T.log(T.nlinalg.det(Sks[:, :, k]))*nus[k]
		bound -= boundH
		
		#Make the functions
		input = [x]
		self.f1 = theano.function(input, bound)
		grad = theano.gradient.jacobian(bound, wrt=input)[0]
		self.f2 = theano.function(input, grad)
		hess = theano.gradient.hessian(bound, wrt=input)[0]
		self.f3 = theano.function(input, hess)

	def make_functions2(self, terms = [1,2,3,4,5], change = True):
		"""Initializes the theano-functions for
			f1 = regular theano evaluation of the bound
			f2 = gradient of the bound wrt r_nk:s (phi_:s)
			f3 = hessian of the bound wrt r_nk:s (phi_:s)
		"""

		#Gather the bound
		x = T.dvector('x')
		if change:
			phi_ = x.reshape((self.N, self.K))
			phi = T.exp(phi_)
			phi /= phi.sum(1)[:, None]
		else:
			phi = x.reshape((self.N, self.K))
		phi_hats = phi.sum(0)
		alphas = self.alpha + phi_hats
		kappas = self.k0 + phi_hats
		nus = self.v0 + phi_hats
		Ybars = T.tensordot(self.X, phi,((0),(0)))
		Cks = T.tensordot(phi, self.XXT,((0),(0))).T
		mks = (self.k0*self.m0[:,None] + Ybars)/kappas[None,:]
		mkprods = mks[:,None,:]*mks[None,:,:] #product of mk and it's transpose
		Sks = self.S0[:,:,None] + Cks + self.k0m0m0T[:,:,None] - kappas[None,None,:]*mkprods
		bound = 0
		if 1 in terms:
			bound += -T.tensordot(T.log(phi + 1e-10), phi) #entropy H_L                                 #1
		if 2 in terms:
			bound += -self.D/2. * T.log(kappas).sum()                                                   #2
		if 3 in terms:
			bound += T.gammaln(alphas).sum()                                                            #3
		if 4 in terms:
			bound += T.gammaln((nus-T.arange(self.D)[:,None])/2.).sum()                                 #4
		if 5 in terms:
			boundH = 0                                                                                  #5
			for k in range(self.K):
				boundH += 0.5*T.log(T.nlinalg.det(Sks[:, :, k]))*nus[k]
			bound -= boundH
		
		#Make the functions
		input = [x]
		self.f1 = theano.function(input, bound)
		grad = theano.gradient.jacobian(bound, wrt=input)[0]
		self.f2 = theano.function(input, grad)
		hess = theano.gradient.hessian(bound, wrt=input)[0]
		self.f3 = theano.function(input, hess)

	def autograd_bound_helper(self, alpha, k0, v0, m0, X, XXT, S0, k0m0m0T, N, K, D, terms = [1,2,3,4,5], change = True):
		def autograd_bound_help2(x):
			if change:
				phi_ = x.reshape((N, K))
				phi_m = np_a.amax(phi_)
				phi_ = phi_ - phi_m
				phi = np_a.exp(phi_)
				phi /= phi.sum(axis=1, keepdims=True)
			else:
				phi = x.reshape((N, K))
			phi_hats = phi.sum(0)
			alphas = alpha + phi_hats
			kappas = k0 + phi_hats
			nus = v0 + phi_hats
			Ybars = np_a.tensordot(X, phi, axes=([0,],[0,]))
			Cks = np_a.tensordot(phi, XXT, axes=([0,],[0,])).T
			mks = (k0*m0[:,np.newaxis] + Ybars)/(np_a.expand_dims(kappas, axis=0))
			mkprods = np_a.expand_dims(mks, axis=1)*np_a.expand_dims(mks, axis=0) #product of mk and it's transpose
			Sks = S0[:,:,None] + Cks + k0m0m0T[:,:,None] - np_a.expand_dims(np_a.expand_dims(kappas, axis=0), axis=0)*mkprods
			bound = 0
			if 1 in terms:
				bound = bound - np_a.tensordot(np_a.log(phi + 1e-10), phi) #entropy H_L											#1
			if 2 in terms:
				bound = bound - D/2. * np_a.log(kappas).sum()																	#2
			if 3 in terms:
				bound = bound + sp_a.special.gammaln(alphas).sum()																#3
			if 4 in terms:
				bound = bound + sp_a.special.gammaln((nus-np_a.arange(self.D)[:,None])/2.).sum()								#4
			if 5 in terms:																										#5
				for k in range(K):
					bound = bound - 0.5*nus[k]*np_a.linalg.slogdet(Sks[:, :, k])[1]
			return bound
		return autograd_bound_help2

	def autograd_bound(self, terms=[1,2,3,4,5], change = True):
		autograd_bound_h = self.autograd_bound_helper(self.alpha, self.k0, self.v0, self.m0, self.X, self.XXT, self.S0, \
								self.k0m0m0T, self.N, self.K, self.D, terms=terms, change=change)
		return autograd_bound_h

	def bound_grad(self, terms=[1,2,3,4,5], change=True):
		return jacobian(self.autograd_bound(terms=terms, change=change))

	def bound_hessian(self, terms=[1,2,3,4,5], change=True):
		return hessian(self.autograd_bound(terms=terms, change=change))

	def get_brute_bound(self, vb_param):
		copy_params = self.get_vb_param().copy()
		self.set_vb_param(vb_param)
		b = self.bound()
		self.set_vb_param(copy_params)
		return b


	def brutehessian(self, terms = [1,2,3,4,5], change = True):
		"""Calculate hessian matrix (without automatic differentiation)"""
		phis = self.phi.copy()
		d = self.N*self.K
		H = np.zeros((self.N, self.K, self.N, self.K))
		if 1 in terms:
			for n1 in xrange(self.N):
				for k1 in xrange(self.K):
					H[n1][k1][n1][k1] += -1/phis[n1][k1]
		#Gather common terms for Hessian
		Hk = np.zeros(self.K)
		if 2 in terms:
			Hk += 0.5*self.D/(self.kNs)**2
		if 3 in terms:
			Hk += polygamma(1, self.alpha + self.phi_hat)
		if 4 in terms:
			Hk += 0.25*polygamma(1, (self.vNs-np.arange(self.D)[:,None])/2.).sum(0)
		for n1 in xrange(self.N):
			for n2 in xrange(self.N):
				for k1 in xrange(self.K):
					H[n1][k1][n2][k1] += Hk[k1]
		#Logdet thingies

		x_m = self.X[:,:,None]-self.mun[None,:,:]
		dS = x_m[:,:,None,:]*x_m[:,None,:,:]
		SnidS = self.Sns_inv[None,:,:,:]*dS
		Ank = np.dot(np.ones(self.D), np.dot(np.ones(self.D), SnidS))

		if 5 in terms:
			for k in xrange(self.K):
				for n1 in xrange(self.N):
					for n2 in xrange(self.N):
						a11 = Ank[n1][k]
						a12 = Ank[n2][k]
						a2 = np.dot(np.dot((self.X[n1]-self.mun[:,k]).T,self.Sns_inv[:,:,k]),(self.X[n2]-self.mun[:,k]))
						H[n1][k][n2][k] += -0.5*(a11+a12)+0.5*self.vNs[k]*a2*a2 + self.vNs[k]*a2/self.kNs[k]
		G = self.vb_grad_natgrad_test(terms = terms, change = False)
		if change:
			return var_change(G, H, phis)
		return H.flatten().reshape((self.N*self.K,self.N*self.K))

	def predict_components_ln(self, Xnew):
		"""The predictive density under each component"""
		Dist = Xnew[:,:,np.newaxis]-self.mun[np.newaxis,:,:] # Nnew x D x K
		tmp = np.sum(Dist[:,:,None,:]*self.Sns_inv[None,:,:,:],1)#*(kn+1.)/(kn*(vn-self.D+1.))
		mahalanobis = np.sum(tmp*Dist, 1)/(self.kNs+1.)*self.kNs*(self.vNs-self.D+1.)
		halflndetSigma = self.Sns_halflogdet + 0.5*self.D*np.log((self.kNs+1.)/(self.kNs*(self.vNs-self.D+1.)))

		Z  = gammaln(0.5*(self.vNs[np.newaxis,:]+1.))\
			-gammaln(0.5*(self.vNs[np.newaxis,:]-self.D+1.))\
			-(0.5*self.D)*(np.log(self.vNs[np.newaxis,:]-self.D+1.) + np.log(np.pi))\
			- halflndetSigma \
			- (0.5*(self.vNs[np.newaxis,:]+1.))*np.log(1.+mahalanobis/(self.vNs[np.newaxis,:]-self.D+1.))
		return Z

	def predict_components(self, Xnew):
		return np.exp(self.predict_components_ln(Xnew))

	def predict(self, Xnew):
		Z = self.predict_components(Xnew)
		#calculate the weights for each component
		phi_hat = self.phi.sum(0)
		pi = phi_hat+self.alpha
		pi /= pi.sum()
		Z *= pi[np.newaxis,:]
		return Z.sum(1)

	def test_mgrid(self):
		pass

	def plot(self, newfig=True):
		import numpy as npp
		if self.X.shape[1]==2:
			if newfig:pb.figure()
			xmin, ymin = self.X.min(0)
			xmax, ymax = self.X.max(0)
			xmin, xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
			ymin, ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
			xx, yy = npp.mgrid[xmin:xmax:100j, ymin:ymax:100j]
			Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
			zz = self.predict(Xgrid).reshape(100, 100)
			zz_data = self.predict(self.X)
			pb.contour(xx, yy, zz, [stats.scoreatpercentile(zz_data, 5)], colors='k', linewidths=3)
			pb.scatter(self.X[:,0], self.X[:,1], 30, np.argmax(self.phi, 1), linewidth=0, cmap=pb.cm.gist_rainbow)

			zz_components = self.predict_components(Xgrid)
			phi_hat = self.phi.sum(0)
			pi = phi_hat+self.alpha
			pi /= pi.sum()
			zz_components *= pi[np.newaxis,:]
			[pb.contour(xx, yy, zz.reshape(100, 100), [stats.scoreatpercentile(zz_data, 5.)], colors='k', linewidths=1) for zz in zz_components.T]


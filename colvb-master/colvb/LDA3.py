# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)
import sys
sys.path.append('/home/othe/Desktop/HIIT/Moduleita/pyautodiff-python2-ast')
import numpy as np
#import pylab as pb
import matplotlib as mlp
#mlp.use('cairo.pdf')
from scipy.special import gammaln, digamma
from scipy import sparse
from col_vb2 import col_vb2
from weave_fns import LDA_mult
import theano
import theano.tensor as T
import itertools

def softmax(x):
    ex = np.exp(x-x.max(1)[:,None])
    return ex/ex.sum(1)[:,None]

class LDA3(col_vb2):
    """Collapsed Latent Dirichlet Allocation"""

    def __init__(self, documents,vocabulary, K,alpha_0=1.,beta_0=1.):
        """
        Arguments
        ---------
        :documents: @todo
        :vocabulary: @todo
        :K: number of topics

        """

        col_vb2.__init__(self)
        assert len(vocabulary.shape)==1
        assert np.max(map(np.max,documents)) <= vocabulary.size
        for doc in documents:
            assert len(doc.shape)==1

        self.documents = documents
        self.vocabulary = vocabulary
        self.D = len(documents)
        self.Nd = map(np.size,documents)
        self.V = vocabulary.size
        self.N = self.Nd[0]

        self.K = K


        #this is used when packing/unpacking  the model, to reshape a vector into our document shapes

        self.document_index = np.vstack((np.hstack((0,np.cumsum(self.Nd)[:-1])),np.cumsum(self.Nd))).T*self.K

        #reversing the way that words are counted. In the data's form
        #(self.documents), each doc is represented by a vector of integers of
        #length N_d: the integer indexes  the vocab. In this form, each doc is
        #represented by a binary matrix of size Nd x V, each +ve entry
        #indicating membership: only one positive entry allowed per row
        self.word_mats = [sparse.coo_matrix((np.ones(Nd_),(np.arange(Nd_),doc)),shape=(Nd_,self.V)) for Nd_,doc in zip(self.Nd,self.documents)]
        self.word_mats_T = [wm.T for wm in self.word_mats]
        self.word_mats_csr = [wm.tocsr() for wm in self.word_mats] # a faster sparse matrix

        #TODO: optimize these?
        self.alpha_0 = np.ones(self.K)*alpha_0
        self.beta_0 = np.ones(self.V)*beta_0

        self.set_vb_param(np.random.randn(sum(self.Nd)*self.K))

    def get_vb_param(self):
        return np.vstack(self.phi_).flatten()

    def set_vb_param(self,x):
        self.phi_ = [x[start:stop].reshape(Ndi,self.K) for Ndi,(start,stop) in zip(self.Nd, self.document_index)]
        self.phi = map(softmax,self.phi_)
        self.log_phi = [np.log(phi) for phi in self.phi]
        self.alpha_p = [self.alpha_0 + phi_d.sum(0) for phi_d in self.phi]
        #self.beta_p = self.beta_0 + np.sum([wm.T.dot(phi).T for phi,wm in zip(self.phi, self.word_mats_csr)],0)
        self.beta_p = self.beta_0 + np.sum([LDA_mult(wm,phi).T for phi,wm in zip(self.phi, self.word_mats_T)],0)

        self.phi_flat = np.vstack(self.phi).flatten()
        self.log_phi_flat = np.vstack(self.log_phi).flatten()

    def bound(self):
        """Lower bound on the marginal likelihood"""
        entropy = -np.dot(self.phi_flat,self.log_phi_flat)
        const1 = self.D*(gammaln(self.alpha_0.sum())-gammaln(self.alpha_0).sum())
        const2 = self.K*(gammaln(self.beta_0.sum())-gammaln(self.beta_0).sum())
        alpha_part = gammaln(np.sum(self.alpha_p,1)).sum() - gammaln(self.alpha_p).sum()
        beta_part = gammaln(np.sum(self.beta_p,1)).sum() - gammaln(self.beta_p).sum()
        return const1 + const2 - alpha_part - beta_part +entropy

    def vb_grad_natgrad(self):
        """The gradient of the bound with respect to $r_{dnk}$"""
        #first compute the gradient wrt r
        alpha_part1 = [digamma(alpha.sum()) for alpha in self.alpha_p]
        alpha_part2 = [digamma(alpha) for alpha in self.alpha_p]
        beta_part1 =  digamma(self.beta_p.sum(1))
        digamma_beta = digamma(self.beta_p)
        #beta_part2 = [x.dot(digamma_beta.T) for x in self.word_mats]
        beta_part2 = [LDA_mult(x,digamma_beta.T) for x in self.word_mats]
        entropy_part = [log_phi+1. for log_phi in self.log_phi]
        grad_phi = [-ap1+ap2-beta_part1+ bp2-ep for ap1,ap2,bp2,ep in zip(alpha_part1,alpha_part2,beta_part2,entropy_part)]

        #transform...
        natgrad = [grad_phi_d - np.sum(phi*grad_phi_d,1)[:,None] for grad_phi_d,phi in zip(grad_phi,self.phi)]
        natgrad = np.hstack(map(np.ravel,natgrad))
        grad = natgrad*np.hstack(map(np.ravel,self.phi))
        return grad,natgrad

    def testGrad(self):
        """The gradient of the bound with respect to $r_{dnk}$"""
        #first compute the gradient wrt r
        alpha_part1 = [digamma(alpha.sum()) for alpha in self.alpha_p]
        alpha_part2 = [digamma(alpha) for alpha in self.alpha_p]
        beta_part1 =  digamma(self.beta_p.sum(1))
        digamma_beta = digamma(self.beta_p)
        #beta_part2 = [x.dot(digamma_beta.T) for x in self.word_mats] -ap1+ap2-beta_part1+ bp2-ep
        beta_part2 = [LDA_mult(x,digamma_beta.T) for x in self.word_mats]
        entropy_part = [log_phi+1. for log_phi in self.log_phi]
        grad_phi = np.array([-ap1+ap2-beta_part1+ bp2-ep for ap1,ap2,bp2,ep in zip(alpha_part1,alpha_part2,beta_part2,entropy_part)])
        natgrad = [grad_phi_d - np.sum(phi*grad_phi_d,1)[:,None] for grad_phi_d,phi in zip(grad_phi,self.phi)]
        natgrad = np.hstack(map(np.ravel,natgrad))
        grad = natgrad*np.hstack(map(np.ravel,self.phi))
        return grad

    '''
    def makeFunctions(self):


        # Segfault on Hessian, fixes possible?


        #theanize words
        words = [np.array(word.todense()) for word in self.word_mats]
        words = np.array([np.concatenate((word, np.zeros((20 - word.shape[0],self.V))), axis=0) for word in words])
        words = theano.shared(words)

        x = T.dvector('x')
        #make phis
        def phis(Ndi, indeces):
            sub = x[indeces[0] : indeces[1]].reshape((Ndi,self.K))
            sub = T.exp(sub-sub.max(1)[:,None])
            sub = sub/sub.sum(1)[:,None]
            return T.set_subtensor(T.zeros((20,self.K))[:Ndi,:self.K], sub)
        phi, updates = theano.scan(fn=phis, sequences=[theano.shared(np.array(self.Nd)), theano.shared(self.document_index)])
        
        #make alphas
        alpha_p, updates = theano.scan(fn=lambda phirow: self.alpha_0+phirow.sum(0), sequences=[phi])
        
        #make betas
        beta_p, updates = theano.scan(fn=lambda phirow, word: T.transpose(phirow).dot(word), sequences=[phi, words])
        
        #gather bound
        bound = 0
        bound += -T.gammaln(T.sum(alpha_p,1)).sum() - T.gammaln(alpha_p).sum()
        bound += -T.gammaln(T.sum(beta_p,1)).sum() - T.gammaln(beta_p).sum()
        bound += (phi.flatten()*T.log(phi.flatten()+1e-100)).sum()
        
        bound = x.sum()
        self.f1 = theano.function([x], bound)
        grad = theano.gradient.jacobian(bound, wrt=[x])[0]
        self.f2 = theano.function([x], grad)
        hess = theano.gradient.hessian(bound, wrt=[x])[0]
        self.f3 = theano.function([x], hess)
    '''

    def makeFunctions(self):
        
        #theanize words
        words = theano.shared(np.array([np.array(word.todense()) for word in self.word_mats]))

        x = T.dvector('x')

        #make phis
        phi_ = x.reshape((self.D, self.N, self.K))
        phi = T.exp(phi_-phi_.max(2)[:,:,None])
        phi = phi/phi.sum(2)[:,:, None]
        
        alpha_p = self.alpha_0 + phi.sum(1)
        beta_p = self.beta_0 + T.tensordot(phi,words, axes=[(1,0), (1,0)])
        #gather bound
        bound = 0
        bound += -T.gammaln(T.sum(alpha_p,1)).sum() + T.gammaln(alpha_p).sum()
        bound += -T.gammaln(T.sum(beta_p,1)).sum() + T.gammaln(beta_p).sum()
        bound += -(phi.flatten()*T.log(phi.flatten()+1e-100)).sum()
        
        #self.f0 = theano.function([x], phi[3][3][3])

        self.f1 = theano.function([x], bound)
        grad = theano.gradient.jacobian(bound, wrt=[x])[0]
        self.f2 = theano.function([x], grad)
        hess = theano.gradient.hessian(bound, wrt=[x])[0]
        self.f3 = theano.function([x], hess)

    def tester(self):
        g1 = self.testGrad()
        g2 = self.newGradient()
        print 'eka'
        print g1
        print 'toka'
        print g2
        print '\n'

    def print_topics(self,wordlim=10):
        vocab_indexes = [np.argsort(b)[::-1] for b in self.beta_p]
        for i in np.argsort(sum(self.alpha_p))[::-1]:
            ii = vocab_indexes[i]
            print ' '.join(self.vocabulary[ii][:wordlim])

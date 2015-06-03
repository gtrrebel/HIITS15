import numpy as np
import pylab as pb
from scipy import optimize, linalg
from scipy.special import gammaln, digamma
import sys
from col_vb import col_vb
from col_mix import collapsed_mixture
import GPy
from GPy.util.linalg import mdot, pdinv, backsub_both_sides, dpotrs, jitchol, dtrtrs
from scipy import weave

class MOHGP(collapsed_mixture):
    """
    A Hierarchical Mixture of Gaussian Processes
    A hierarchy is formed by using a GP prior for the cluster function values and another for the likelihood
    """
    def __init__(self, X, kernF, kernY, Y, K=2, alpha=1., prior_Z='symmetric'):
        N,self.D = Y.shape
        self.Y = Y
        self.X = X
        assert X.shape[0]==self.D, "input data don't match observations"

        #initialize kernels
        self.kernF = kernF
        self.kernY = kernY
        self.Sf = self.kernF.K(self.X)
        self.Sy = self.kernY.K(self.X)
        self.Sy_inv, self.Sy_chol, self.Sy_chol_inv, self.Sy_logdet = pdinv(self.Sy)


        #Computations that can be done outside the optimisation loop
        self.YYT = self.Y[:,:,np.newaxis]*self.Y[:,np.newaxis,:]
        self.YTY = np.dot(self.Y.T,self.Y)

        collapsed_mixture.__init__(self, N, K, prior_Z, alpha)

    def _set_params(self,x):
        """ Set the kernel parameters. Note that the variational parameters are handled separately."""
        #st the kernels with their parameters
        self.kernF._set_params_transformed(x[:self.kernF.num_params])
        self.kernY._set_params_transformed(x[self.kernF.num_params:])

        #get the latest kernel matrices, decompose
        self.Sf = self.kernF.K(self.X)
        self.Sy = self.kernY.K(self.X)
        self.Sy_inv, self.Sy_chol, self.Sy_chol_inv, self.Sy_logdet = pdinv(self.Sy)

        #update everything
        self.do_computations()

    def _get_params(self):
        """ returns the kernel parameters """
        return np.hstack([self.kernF._get_params_transformed(), self.kernY._get_params_transformed()])

    def _get_param_names(self):
        return ['kernF_'+ n for n in self.kernF._get_param_names_transformed()] + ['kernY_' + n for n in self.kernY._get_param_names_transformed()]

    def _log_likelihood_gradients(self):
        """
        The derivative of the lower bound wrt the (kernel) parameters
        """

        tmp = [dtrtrs(L, self.Sy_chol_inv, lower=1)[0] for L in self._C_chols]
        B_invs = [phi_hat_i*np.dot(tmp_i.T, tmp_i) for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)]
        #B_invs = [phi_hat_i*mdot(self.Sy_chol_inv.T,Ci,self.Sy_chol_inv) for phi_hat_i, Ci in zip(self.phi_hat,self.C_invs)]

        #heres the mukmukT*Lambda term
        LiSfi = [np.eye(self.D)-np.dot(self.Sf,Bi) for Bi in B_invs]#seems okay
        tmp1 = [np.dot(LiSfik.T,Sy_inv_ybark_k) for LiSfik, Sy_inv_ybark_k in zip(LiSfi,self.Syi_ybark.T)]
        tmp = 0.5*sum([np.dot(tmpi[:,None],tmpi[None,:]) for tmpi in tmp1])

        #here's the difference in log determinants term
        tmp += -0.5*sum(B_invs)

        #kernF_grads = np.array([np.sum(tmp*g) for g in self.kernF.extract_gradients()]) # OKAY!
        kernF_grads = self.kernF.dK_dtheta(tmp,self.X)

        #gradient wrt Sigma_Y
        ybarkybarkT = self.ybark.T[:,None,:]*self.ybark.T[:,:,None]
        Byks = [np.dot(Bi,yk) for Bi,yk in zip(B_invs,self.ybark.T)]
        tmp = sum([np.dot(Byk[:,None],Byk[None,:])/np.power(ph_k,3)\
                -Syi_ybarkybarkT_Syi/ph_k -Bi/ph_k for Bi, Byk, yyT, ph_k, Syi_ybarkybarkT_Syi in zip(B_invs, Byks, ybarkybarkT, self.phi_hat, self.Syi_ybarkybarkT_Syi) if ph_k >1e-6])
        tmp += (self.K-self.N)*self.Sy_inv
        tmp += mdot(self.Sy_inv,self.YTY,self.Sy_inv)
        tmp /= 2.

        #kernY_grads = np.array([np.sum(tmp*g) for g in self.kernY.extract_gradients()])
        kernY_grads = self.kernY.dK_dtheta(tmp,self.X)

        return np.hstack((kernF_grads, kernY_grads))

    def do_computations(self):
        """
        Here we do all the computations that are required whenever the kernels or the varaitional parameters are changed
        """
        #sufficient stats. speed bottleneck?
        self.ybark = np.dot(self.phi.T,self.Y).T
        #self.Ck = np.dstack([np.dot(self.Y.T,phi[:,None]*self.Y) for phi in self.phi.T])

        # compute posterior variances of each cluster (lambda_inv)
        tmp = backsub_both_sides(self.Sy_chol, self.Sf, transpose='right')
        self.Cs = [np.eye(self.D) + tmp*phi_hat_i for phi_hat_i in self.phi_hat]

        self._C_chols = [jitchol(C) for C in self.Cs]
        self.log_det_diff = np.array([2.*np.sum(np.log(np.diag(L))) for L in self._C_chols])
        tmp = [dtrtrs(L, self.Sy_chol.T, lower=1)[0] for L in self._C_chols]
        self.Lambda_inv = np.array([( self.Sy - np.dot(tmp_i.T, tmp_i) )/phi_hat_i if (phi_hat_i>1e-6) else self.Sf for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)])

        #posterior mean and other useful quantities
        self.Syi_ybark, _ = dpotrs(self.Sy_chol, self.ybark, lower=1)
        self.Syi_ybarkybarkT_Syi = self.Syi_ybark.T[:,None,:]*self.Syi_ybark.T[:,:,None]
        self.muk = (self.Lambda_inv*self.Syi_ybark.T[:,:,None]).sum(1).T

    def bound(self):
        """Compute the lower bound on the marginal likelihood (conditioned on the GP hyper parameters). """
        return -0.5*(self.N*self.D*np.log(2.*np.pi) + self.log_det_diff.sum() + self.N*self.Sy_logdet + np.sum(self.YTY*self.Sy_inv))\
            + 0.5*np.sum(self.Syi_ybarkybarkT_Syi*self.Lambda_inv)\
            + self.mixing_prop_bound() + self.H

    def vb_grad_natgrad(self):
        """Gradients of the bound"""
        #yn_mk = self.Y[:,:,None]-self.muk[None,:,:]
        #ynmk2 = np.sum(np.dot(self.Sy_inv,yn_mk)*np.rollaxis(yn_mk,0,2),0)
        ynmk2 = multiple_mahalanobis(self.Y, self.muk.T, self.Sy_chol)

        grad_phi = (self.mixing_prop_bound_grad() -
                    0.5*np.sum(np.sum(self.Lambda_inv*self.Sy_inv[None,:,:],1),1)) + \
                   ( self.Hgrad - 0.5*ynmk2 ) # parentheses are for operation ordering!
        natgrad = grad_phi - np.sum(self.phi*grad_phi,1)[:,None]
        grad = natgrad*self.phi

        return grad.flatten(), natgrad.flatten()


    def predict_components(self,Xnew):
        """The predictive density under each component"""

        tmp = [dtrtrs(L, self.Sy_chol_inv, lower=1)[0] for L in self._C_chols]
        B_invs = [phi_hat_i*np.dot(tmp_i.T, tmp_i) for phi_hat_i, tmp_i in zip(self.phi_hat, tmp)]
        kx= self.kernF.K(self.X,Xnew)
        try:
            kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew)
        except TypeError:
            #kernY has a hierarchical structure that we should deal with
            con = np.ones((Xnew.shape[0],self.kernY.connections.shape[1]))
            kxx = self.kernF.K(Xnew) + self.kernY.K(Xnew,con)

        #prediction as per my notes
        tmp = [np.eye(self.D) - np.dot(Bi,self.Sf) for Bi in B_invs]
        mu = [mdot(kx.T,tmpi,self.Sy_inv,ybark) for tmpi,ybark in zip(tmp,self.ybark.T)]
        var = [kxx - mdot(kx.T,Bi,kx) for Bi in B_invs]

        return mu,var

    def plot_simple(self):
        assert self.X.shape[1]==1, "can only plot mixtures of 1D functions"
        #pb.figure()
        pb.plot(self.Y.T,'k',linewidth=0.5,alpha=0.4)
        pb.plot(self.muk[:,self.phi_hat>1e-3],'k',linewidth=2)

    def plot(self, on_subplots=False,colour=False,newfig=True,errorbars=False, in_a_row=False,joined=True, gpplot=True,min_in_cluster=1e-3, data_in_grey=False, numbered=True, data_in_duplicate=False, fixed_inputs=[]):

        #work out what input dimensions to plot
        fixed_dims = np.array([i for i,v in fixed_inputs])
        free_dims = np.setdiff1d(np.arange(self.X.shape[1]),fixed_dims)
        assert len(free_dims)==1, "can only plot mixtures of 1D functions"

        #figure, subplots
        if newfig:
            f = pb.figure()
        else:
            f = pb.gcf()
        GPy.util.plot.Tango.reset()

        if data_in_duplicate:
            X = self.X[::2, free_dims]
            Y = 0.5*(self.Y[:,:-1:2] + self.Y[:,1::2])
        else:
            Y = self.Y
            X = self.X[:,free_dims]

        #work out how many clusters we're going to plot.
        Ntotal = np.sum(self.phi_hat > min_in_cluster)
        if on_subplots:
            if in_a_row:
                Nx = 1
                Ny = Ntotal
            else:
                Nx = np.floor(np.sqrt(Ntotal))
                Ny = int(np.ceil(Ntotal/Nx))
                Nx = int(Nx)
        else:
            ax = pb.gca() # this seems to make new ax if needed

        #limits of GPs
        xmin,xmax = X.min(), X.max()
        ymin,ymax = Y.min(), Y.max()
        xmin,xmax = xmin-0.1*(xmax-xmin), xmax+0.1*(xmax-xmin)
        ymin,ymax = ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)
        Xgrid = np.empty((300,self.X.shape[1]))
        Xgrid[:,free_dims] = np.linspace(xmin,xmax,300)[:,None]
        for i,v in fixed_inputs:
            Xgrid[:,i] = v


        subplot_count = 0
        for i, ph, mu, var in zip(range(self.K), self.phi_hat, *self.predict_components(Xgrid)):
            if ph>(min_in_cluster):
                ii = np.argmax(self.phi,1)==i
                num_in_clust = np.sum(ii)
                if not np.any(ii):
                    continue
                if on_subplots:
                    ax = pb.subplot(Nx,Ny,subplot_count+1)
                    subplot_count += 1
                if colour:
                    col = GPy.util.plot.Tango.nextMedium()
                else:
                    col='k'
                if joined:
                    if data_in_grey:
                        ax.plot(X,Y[ii].T,'k',marker=None, linewidth=0.2,alpha=0.4)
                    else:
                        ax.plot(X,Y[ii].T,col,marker=None, linewidth=0.2,alpha=1)
                else:
                    if data_in_grey:
                        ax.plot(X,Y[ii].T,'k',marker='.', linewidth=0.0,alpha=0.4)
                    else:
                        ax.plot(X,Y[ii].T,col,marker='.', linewidth=0.0,alpha=1)

                if gpplot: GPy.util.plot.gpplot(Xgrid[:,free_dims].flatten(),mu.flatten(),mu- 2.*np.sqrt(np.diag(var)),mu+2.*np.sqrt(np.diag(var)),col,col,axes=ax,alpha=0.1)

                if numbered and on_subplots:
                    ax.text(1,1,str(int(num_in_clust)),transform=ax.transAxes,ha='right',va='top',bbox={'ec':'k','lw':1.3,'fc':'w'})

                err = 2*np.sqrt(np.diag(self.Lambda_inv[i,:,:]))
                if errorbars:ax.errorbar(self.X.flatten(), self.muk[:,i], yerr=err,ecolor=col, elinewidth=2, linewidth=0)

        if on_subplots:
            GPy.util.plot.align_subplots(Nx,Ny,xlim=(xmin,xmax))
        else:
            ax.set_xlim(xmin,xmax)
def multiple_mahalanobis(X1, X2, L):
    """
    X1 is a N1 x D array
    X2 is a N2 x D array
    L is a D x D array, lower triangular

    compute (x1_n - x2_m).T * (L * L.T)^-1 * (x1_n - x2_m)

    for each pair of the D_vectors x1_n, x2_m

    Returns: a N1 x N2 array of each distance
    """
    N1,D = X1.shape
    N2,D = X2.shape
    assert L.shape == (D,D)
    result = np.zeros(shape=(N1,N2), dtype=np.float64)

    #configure weave for parallel (or not)
    weave_options_openmp = {'headers'           : ['<omp.h>'],
                            'extra_compile_args': ['-fopenmp -O3'],
                            'extra_link_args'   : ['-lgomp'],
                            'libraries': ['gomp']}
    weave_options_noopenmp = {'extra_compile_args': ['-O3']}

    if GPy.util.config.config.getboolean('parallel', 'openmp'):
        weave_options = weave_options_openmp
        weave_support_code =  """
        #include <omp.h>
        #include <math.h>
        """
    else:
        weave_options = weave_options_noopenmp
        weave_support_code = "#include <math.h>"

    if GPy.util.config.config.getboolean('parallel', 'openmp'):
        pragma_string = '#pragma omp parallel for private(n,m,i,j,tmp)'
    else:
        pragma_string = ''

    code = """
    double tmp [D];
    //two loops over the N1 x N2 vectors
    int n, m, i, j;
    {pragma}
    for(n=0; n<N1; n++){{
      for(m=0; m<N2; m++){{

        //a double loop to solve the cholesy problem into tmp (should really use blas?)
        for(i=0; i<D; i++){{
          tmp[i] = X1(n,i) - X2(m,i);
          for(j=0; j<i; j++){{
            tmp[i] -= L(i,j)*tmp[j];
          }}
          tmp[i] /= L(i,i);
        }}

        //loop over tmp to get the result: tmp.T * tmp (should really use blas again)
        for(i=0; i<D; i++){{
          result(n,m) += tmp[i]*tmp[i];
        }}
      }}
    }}
    """.format(pragma=pragma_string)
    weave.inline(code, arg_names=["X1", "X2", "L", "N1", "N2", "D", "result"], type_converters=weave.converters.blitz, support_code=weave_support_code, **weave_options)
    return result



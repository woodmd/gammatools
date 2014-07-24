"""
@author Matthew Wood <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood <mdwood@slac.stanford.edu>"
__date__     = "$Date: 2013/10/20 23:53:52 $"

import numpy as np
import copy
from scipy.interpolate import UnivariateSpline
import scipy.optimize as opt
import matplotlib.pyplot as plt
from gammatools.core.histogram import Histogram
from scipy.stats import norm
from gammatools.core.util import find_root, find_fn_root
from gammatools.core.nonlinear_fitting import BFGSFitter

def pval_to_sigma(p):
    """Convert the pval of a one-sided confidence interval to sigma."""
    return norm().isf(p)

def sigma_to_pval(s):
    return norm().cdf(s)

def gauss_pval_to_sigma(p):
    """Convert the pval of a two-sided confidence interval to sigma."""
    return norm().isf(0.5+p*0.5)

def gauss_sigma_to_pval(s):
    return 2.0*(norm().cdf(s)-0.5)

def poisson_lnl(nc,mu):
    """Log-likelihood function for a poisson distribution with nc
    observed counts and expectation value mu.  Note that this function
    can accept arguments with different lengths along each dimension
    and will apply the standard numpy broadcasting rules during
    evaluation."""

    nc = np.array(nc,ndmin=1)
    mu = np.array(mu,ndmin=1)

    shape = max(nc.shape,mu.shape)

    lnl = np.zeros(shape)
    mu = mu*np.ones(shape)
    nc = nc*np.ones(shape)

    msk = nc>0

    lnl[msk] = nc[msk]*np.log(mu[msk])-mu[msk]
    lnl[~msk] = -mu[~msk]
    return lnl

def poisson_delta_lnl(nc,mu0,mu1):
    """Compute the log-likelihood ratio for a binned counts
    distribution given two models."""
    return poisson_lnl(nc,mu0) - poisson_lnl(nc,mu1)


class OnOffExperiment(object):
    """Evaluate the sensitivity of an on-off counting experiment.  If
    alpha = None then the background will be assumed to be known."""

    def __init__(self,mus,mub,alpha=None,known_background=False):
        self._mus = np.array(mus,ndmin=1)
        self._mub = np.array(mub,ndmin=1)
        if not alpha is None: self._alpha = np.array(alpha,ndmin=1)
        else: self._alpha = None
        self._data_axes = [0]

    def mc_ts(self,mu,ntrial):
        """Simulate a set of TS values."""

        shape = (ntrial,len(self._mus))

        ns = np.random.poisson(self._mus*mu,shape).T
        nb = np.random.poisson(self._mub,shape).T
        nc = np.random.poisson(np.sum(self._mub)/self._alpha,(ntrial,1)).T

        ns = np.array(ns,dtype='float')
        nb = np.array(nb,dtype='float')
        nc = np.array(nc,dtype='float')

        tsv = []

        for i in range(ntrial):

            # Fit for signal lnl
            fn0 = lambda x,y: -OnOffExperiment.lnl_signal(ns[:,i]+nb[:,i],nc[:,i],
                                                         x*self._mus,y*self._mub,
                                                         self._alpha)

            # Fit for signal null lnl
            fn1 = lambda x: -OnOffExperiment.lnl_null(ns[:,i]+nb[:,i],nc[:,i],
                                                      x*self._mub,self._alpha)

            p0 = BFGSFitter.fit(fn0,[1.0,1.0],bounds=[[0.01,None],[0.01,None]])
            p1 = BFGSFitter.fit(fn1,[1.0],bounds=[[0.01,None]])

            ts = OnOffExperiment.ts(ns[:,i]+nb[:,i],nc[:,i],
                                    p0[0].value*self._mus,
                                    p0[1].value*self._mub,
                                    p1[0].value*self._mub,
                                    self._alpha)
                                 
            ts = max(ts,0)
            tsv.append(ts)

        return np.array(tsv)

    def asimov_mu_ts0(self,ts):
        """Return the value of the signal strength parameter for which
        the TS (-2*lnL) for discovery is equal to the given value."""

        smin = 1E-3
        smax = 1E3

        while self.asimov_ts0_signal(smax) < ts: smax *= 10
        while self.asimov_ts0_signal(smin) > ts: smin *= 0.1

        mu = find_fn_root(self.asimov_ts0_signal,smin,smax,ts)
        mu_err = np.sqrt(mu**2/ts)

        return (mu,mu_err)

    def asimov_mu_p0(self,alpha):
        """Return the value of the signal strength parameter for which
        the p-value for discovery is equal to the given value."""
        
        ts = pval_to_sigma(alpha)**2        
        return self.asimov_mu_ts0(ts)

    def asimov_ts0_signal(self,s,sum_lnl=True):
        """Compute the median discovery test statistic for a signal
        strength parameter s using the asimov method."""

        s = np.array(s,ndmin=1)[np.newaxis,...]

        mub = self._mub[:,np.newaxis]
        mus = self._mus[:,np.newaxis]

        wb = mub/np.apply_over_axes(np.sum,mub,self._data_axes)

        # model amplitude for signal counts in signal region under
        # signal hypothesis
        s1 = s*mus

        # nb of counts in signal region
        ns = mub + s1

        if self._alpha is None:        

            b0 = wb*np.apply_over_axes(np.sum,ns,self._data_axes)
            lnls1 = poisson_lnl(ns,ns)
            lnls0 = poisson_lnl(ns,b0)
            ts = 2*np.apply_over_axes(np.sum,(lnls1-lnls0),self._data_axes)

            return ts

        alpha  = self._alpha[:,np.newaxis]

        # nb of counts in control region
        nc = np.apply_over_axes(np.sum,mub/alpha,self._data_axes)

        # model amplitude for background counts in signal region under
        # null hypothesis
        b0 = wb*(nc+np.apply_over_axes(np.sum,ns,
                                       self._data_axes))*alpha/(1+alpha)

        lnl1 = OnOffExperiment.lnl_signal(ns,nc,s1,mub,alpha,
                                          self._data_axes,sum_lnl)
        lnl0 = OnOffExperiment.lnl_null(ns,nc,b0,alpha,
                                        self._data_axes,sum_lnl)

        return 2*(lnl1-lnl0)

    @staticmethod
    def lnl_signal(ns,nc,mus,mub,alpha=None,data_axes=0,sum_lnl=True):
        """
        Log-likelihood for signal hypothesis.

        Parameters
        ----------
        ns: Vector of observed counts in signal region.

        nc: Vector of observed counts in control region(s).
        """       

        lnls = poisson_lnl(ns,mus+mub)
        lnlc = np.zeros(nc.shape)

        if alpha: 
            # model amplitude for counts in control region
            muc = np.apply_over_axes(np.sum,mub,data_axes)/alpha
            lnlc = poisson_lnl(nc,muc)

        if sum_lnl: 
            lnls = np.apply_over_axes(np.sum,lnls,data_axes)
            lnls = np.squeeze(lnls,data_axes)

            lnlc = np.apply_over_axes(np.sum,lnlc,data_axes)
            lnlc = np.squeeze(lnlc,data_axes)

            return lnls+lnlc
        else:
            return lnls

    @staticmethod
    def lnl_null(ns,nc,mub,alpha=None,data_axes=0,sum_lnl=True):
        """
        Log-likelihood for null hypothesis.

        Parameters
        ----------
        ns: Vector of observed counts in signal region.

        nc: Vector of observed counts in control region(s).
        """       
        lnls = poisson_lnl(ns,mub)
        lnlc = np.zeros(nc.shape)

        if alpha: 
            # model amplitude for counts in control region
            muc = np.apply_over_axes(np.sum,mub,data_axes)/alpha
            lnlc = poisson_lnl(nc,muc)

        if sum_lnl: 
            lnls = np.apply_over_axes(np.sum,lnls,data_axes)
            lnls = np.squeeze(lnls,data_axes)

            lnlc = np.apply_over_axes(np.sum,lnlc,data_axes)
            lnlc = np.squeeze(lnlc,data_axes)
            return lnls+lnlc
        else:
            return lnls


    @staticmethod
    def ts(ns,nc,mus1,mub1,mub0,alpha,data_axis=0,sum_lnl=True):
        """
        Compute the TS (2 x delta log likelihood) between signal and
        null hypotheses given a number of counts and model amplitude
        in the signal/control regions.

        Parameters
        ----------
        sc: Observed counts in signal region.

        nc: Observed counts in control region.
        """
        
        lnl1 = OnOffExperiment.lnl_signal(ns,nc,mus1,mub1,alpha,data_axis,sum_lnl)
        lnl0 = OnOffExperiment.lnl_null(ns,nc,mub0,alpha,data_axis,sum_lnl)

        return 2*(lnl1-lnl0)

def poisson_median_ts(sc,bc,alpha):
    """Compute the median TS."""

    # total counts in each bin
    nc = sc + bc        

    # number of counts in control region
    cc = bc/alpha

    # model for total background counts in null hypothesis
    mub0 = (nc+cc)/(1.0+alpha)*alpha

    # model for total background counts in signal hypothesis
    mub1 = bc

    # model for signal counts
    mus = sc

    lnl0 = nc*np.log(mub0)-mub0 + cc*np.log(mub0/alpha) - mub0/alpha
    lnl1 = nc*np.log(mub1+mus) - mub1 - mus + \
        cc*np.log(mub1/alpha) - mub1/alpha                

    return 2*(lnl1-lnl0)

def poisson_median_ul(sc,bc,alpha):
    """Compute the median UL."""

    # total counts in each bin
    nc = bc        

    # number of counts in control region
    cc = bc/alpha

    # model for total background counts in null hypothesis
    mub0 = (nc+cc)/(1.0+alpha)*alpha

    # model for total background counts in signal hypothesis
    mub1 = bc

    # model for signal counts
    mus = sc

    lnl0 = nc*np.log(mub0)-mub0 + cc*np.log(mub0/alpha) - mub0/alpha
    lnl1 = nc*np.log(mub1+mus) - mub1 - mus + \
        cc*np.log(mub1/alpha) - mub1/alpha
                

    return 2*(lnl1-lnl0)

def poisson_ts(nc,mus,mub,data_axes=1):
    """Test statistic for discovery with known background."""

    # MLE for signal norm under signal hypothesis
    snorm = np.apply_over_axes(np.sum,nc-mub,data_axes)

    lnl0 = nc*np.log(mub) - mub 
    lnl1 = nc*np.log(mus*snorm+mub) - (mus*snorm+mub) 

    dlnl = 2*(lnl1-lnl0)
                
    return dlnl

def poisson_ul(nc,mus,mub,data_axes=1):
    """Test statistic for discovery with known background."""


    # MLE for signal norm under signal hypothesis
    snorm = np.apply_over_axes(np.sum,nc-mub,data_axes)
    snorm[snorm<0] = 0

    x = np.linspace(-3,3,50)

    mutot = snorm*mus+mub

    deltas = 10**x#*np.sum(mub)

    smutot = deltas[np.newaxis,np.newaxis,:]*mus[...,np.newaxis] + mutot[...,np.newaxis]

    lnl = nc[...,np.newaxis]*np.log(smutot) - smutot

    lnl = np.sum(lnl,axis=data_axes)

    ul = np.zeros(lnl.shape[0])

    for i in range(lnl.shape[0]):
        
        dlnl = -2*(lnl[i]-lnl[i][0])

        deltas_root = find_root(deltas,dlnl,2.72)

        ul[i] = snorm[i][0] + deltas_root


        continue

        print i, snorm[i][0], deltas_root

        z0 = 2*np.sum(poisson_lnl(nc[i],mutot[i]))
        z1 = 2*np.sum(poisson_lnl(nc[i],mutot[i]+deltas_root*mus))

        print z0, z1, z1-z0

        continue

        ul = snorm[i][0] + find_root(deltas,dlnl,2.72)

        print '------------'

        z0 = 2*np.sum(poisson_lnl(nc[i],mutot[i]))
        z1 = 2*np.sum(poisson_lnl(nc[i],mub+ul*mus))

        print z0, z1, z1-z0

        print snorm[i][0], ul

#        plt.figure()
#        plt.plot(x,dlnl)
#        plt.show()
    return ul


if __name__ == '__main__':

    from gammatools.core.histogram import *

    fn_qmu = lambda n, mu0, mu1: -2*(poisson_lnl(n,mu1) - poisson_lnl(n,mu0))
    np.random.seed(1)

    ntrial = 1000

    mub = np.array([100.0,50.0])    
    mus = 10.*np.array([3.0,1.0])
    alpha = np.array([1.0])

    scalc = OnOffExperiment(mus,mub,alpha)


    print scalc.asimov_ts0_signal(1.0)
    print scalc.asimov_ts0_signal(np.linspace(0.1,100,10))


    ts = scalc.mc_ts(1.0,1000)

    print np.median(ts)

    sys.exit(0)

    s = np.linspace(0.1,100,10)

    print 'ts0_signal ', scalc.asimov_ts0_signal(s)
    print 'ts0_signal[5] ', s[5], scalc.asimov_ts0_signal(s[5])
    mu, muerr =  scalc.asimov_mu_ts0(25.0)

    print 'TS(mu): ', scalc.asimov_ts0_signal(mu)
    print 'TS(mu+muerr): ', scalc.asimov_ts0_signal(mu+muerr)

    ns = np.random.poisson(mus,(ntrial,len(mus))).T
    nb = np.random.poisson(mub,(ntrial,len(mub))).T

    mub = mub[:,np.newaxis]
    mus = mus[:,np.newaxis]

    nexcess = ns+nb-mub
    nexcess[nexcess<=0] = 0

    ts_mc = -2*(poisson_lnl(ns+nb,mub) - poisson_lnl(ns+nb,ns+nb))
    ts_asimov = fn_qmu(mus+mub,mus+mub,mub)
    
    ul_mc = np.zeros(ntrial)

    alpha = 0.05
    dlnl = pval_to_sigma(alpha)**2

    for i in range(ntrial):

        xroot = find_fn_root(lambda t: fn_qmu((ns+nb)[:,i],
                                              (mub+nexcess)[:,i],
                                              (mub+nexcess)[:,i]+t),0,100,dlnl)
        ul_mc[i] = nexcess[i]+xroot


    sigma_mu_fn = lambda t: np.sqrt(t**2/fn_qmu(mub,mub,mub+t))

    ul_asimov_qmu = find_fn_root(lambda t: fn_qmu(mub,mub,mub+t),0,100,dlnl)
    sigma = np.sqrt(ul_asimov_qmu**2/dlnl)

    ul_asimov = 1.64*sigma
    ul_asimov_upper = (1.64+1)*sigma
    ul_asimov_lower = (1.64-1)*sigma


    print 'SIGMA ', sigma
    print 'SIGMA ', sigma_mu_fn(1.0)
    print 'SIGMA ', sigma_mu_fn(10.0)
    print 'Asimov q0 UL  ', find_fn_root(lambda t: fn_qmu(mub+t,mub+t,mub),0,100,dlnl)
    print 'Asimov qmu UL ', ul_asimov_qmu, sigma
    print 'Asimov UL ', ul_asimov, ul_asimov_upper-ul_asimov, ul_asimov_lower-ul_asimov

    print -2*poisson_delta_lnl(mub,mub,mub+ul_asimov)
    print -2*poisson_delta_lnl(mub,mub,mub+ul_asimov_qmu)
    print fn_qmu(mub,mub,mub+ul_asimov_qmu)
    print dlnl

    qmu = -2*poisson_delta_lnl(mub,mub,mub+ul_asimov_qmu)

    h = Histogram(Axis.create(0,100,100))

    h.fill(ul_mc)

    h.normalize().cumulative().plot()

    plt.axhline(0.5-0.34,color='g')

    plt.axhline(0.5+0.34,color='g')

    plt.axvline(ul_asimov_qmu,color='k')

    plt.axvline(ul_asimov_qmu+sigma,color='k',linestyle='--')
    plt.axvline(ul_asimov_qmu-sigma,color='k',linestyle='--')
#    plt.axvline(ul_asimov_qmu-sigma/(ul_asimov_qmu+sigma)*ul_asimov_qmu,
#                color='k',linestyle='--')

    plt.gca().grid(True)

    plt.show()

    print 'Median UL ', np.median(ul_mc)

    print 'Median TS ', np.median(ts_mc)
    print 'Asimov TS ', ts_asimov
#    print fn_qmu(mus+mub,mus+mub,mub)
#    print fn_qmu(mub,mub,mus+mub)

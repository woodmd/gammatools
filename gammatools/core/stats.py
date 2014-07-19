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

    def __init__(self,mus,mub,alpha=None):
        self._mus = np.array(mus,ndmin=1)
        self._mub = np.array(mub,ndmin=1)
        if not alpha is None: self._alpha = np.array(alpha,ndmin=1)
        else: self._alpha = None

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

    def asimov_ts0_signal(self,s):
        """Compute the discovery test statistic for a signal strength
        parameter s for an asimov data set matching the signal hypothesis."""

        s = np.array(s,ndmin=1)[np.newaxis,...]

        mub = self._mub[:,np.newaxis]
        mus = self._mus[:,np.newaxis]

        wb = mub/np.apply_over_axes(np.sum,mub,0)

        # model amplitude for signal counts in signal region under
        # signal hypothesis
        s1 = s*mus

        # nb of counts in signal region
        sc = mub + s1

        if self._alpha is None:        

            b0 = wb*np.apply_over_axes(np.sum,sc,0)
            lnls1 = poisson_lnl(sc,sc)
            lnls0 = poisson_lnl(sc,b0)
            return 2*np.sum((lnls1-lnls0),axis=0)

        alpha  = self._alpha[:,np.newaxis]

        # nb of counts in control region
        cc = np.apply_over_axes(np.sum,mub/alpha,0)

        # model amplitude for background counts in signal region under
        # null hypothesis
        b0 = wb*(cc+np.apply_over_axes(np.sum,sc,0))*alpha/(1+alpha)

        # model amplitude for background counts in signal region under
        # signal hypothesis
        b1 = mub

        # model amplitude for counts in control region under null hypothesis
        c0 = np.apply_over_axes(np.sum,b0,0)/alpha

        # model amplitude for counts in control region under signal hypothesis
        c1 = np.apply_over_axes(np.sum,b1,0)/alpha
        
        return OnOffExperiment.ts(sc,cc,s1+b1,b0,c1,c0)

    def median_ts(self,s):
        return self.plnl_signal(s)

    

    @staticmethod
    def ts(ns,nc,mus1,mus0,muc1,muc0,data_axis=0):
        """
        Compute the TS (2 x delta log likelihood) between two
        hypotheses given a number of counts in the signal/control
        regions.

        Parameters
        ----------
        sc: Observed counts in signal region.

        cc: Observed counts in control region.
        """

        lnls1 = np.sum(poisson_lnl(ns,mus1),axis=data_axis)
        lnlc1 = np.sum(poisson_lnl(nc,muc1),axis=data_axis)

        lnls0 = np.sum(poisson_lnl(ns,mus0),axis=data_axis)
        lnlc0 = np.sum(poisson_lnl(nc,muc0),axis=data_axis)
 
        return 2*(lnls1+lnlc1-lnls0-lnlc0)


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

    

    mub = np.array([101.3,51.3])
    
    mus = 1.*np.array([3.1,1.0])
    alpha = np.array([1.0])

    scalc = OnOffExperiment(mus,mub,alpha)

    s = np.linspace(0.1,100,10)

    print 'ts0_signal ', scalc.ts0_signal(s)
    print 'ts0_signal[5] ', s[5], scalc.ts0_signal(s[5])
    mu, muerr =  scalc.mu_ts0(25.0)

    print 'TS(mu): ', scalc.ts0_signal(mu)
    print 'TS(mu+muerr): ', scalc.ts0_signal(mu+muerr)

    ns = np.random.poisson(mus,(ntrial,len(mus))).T
    nb = np.random.poisson(mub,(ntrial,len(mub))).T

    nexcess = ns+nb-mub
    nexcess[nexcess<=0] = 0

    ts_mc = -2*(poisson_lnl(ns+nb,mub) - poisson_lnl(ns+nb,ns+nb))
    ts_asimov = fn_qmu(mus+mub,mus+mub,mub)
    
    ul_mc = np.zeros(ntrial)

    alpha = 0.05
    dlnl = pval_to_sigma(alpha)**2

    print pval_to_sigma(0.05)
    print gauss_pval_to_sigma(0.68)

    for i in range(ntrial):

        xroot = find_fn_root(lambda t: fn_qmu((ns+nb)[i],
                                              (mub+nexcess)[i],
                                              (mub+nexcess)[i]+t),0,100,dlnl)
        ul_mc[i] = nexcess[i]+xroot


#        print i, nexcess[i], xroot, fn_qmu((ns+nb)[i],(mub+nexcess)[i],(mub+nexcess)[i]+xroot)


#print ts_mc
#print ns+nb
#print poisson_lnl(0,0)

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

#!/usr/bin/env python

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

def gauss_pval_to_sigma(p):

    return norm().isf(0.5+p*0.5)

def gauss_sigma_to_pval(s):

    return 2.0*(norm().cdf(s)-0.5)


class HistBootstrap(object):
    def __init__(self,hist,fn):

        self._fn = fn
        self._hist = hist
        self._x = np.array(hist.edges(),copy=True)
        self._ncounts = copy.copy(hist._counts)

    def bootstrap(self,niter=1000,**kwargs):

        nbin = len(self._ncounts)
        ncounts_tmp = np.zeros((nbin,niter))

        for i in range(nbin):

            if self._ncounts[i] > 0:
                ncounts_tmp[i,:] += np.random.poisson(self._ncounts[i],niter)

        fval = []
        for i in range(niter):
            self._hist._counts = ncounts_tmp[:,i]
            #fn = self._fn(self._hist)            
            fval.append(self._fn(self._hist,**kwargs))

            
        fval_mean = np.mean(np.array(fval))
        fval_rms = np.std(np.array(fval))
        
        return fval_mean, fval_rms


class HistQuantileBkgFn(object):
    """
    HistQuantileBkgFn(hcounts,bkg_fn)

    Class that computes quantiles of a histogram using a user-provided
    background function which is normalized by a Poisson-distributed
    random variate.

    Parameters
    ----------
    hcounts : histogram

       Histogram object containg the counts data for which the
       quantile will be estimated.

    bkg_fn : function

       Function that returns the cumulative background.
       
    """
    def __init__(self,hcounts,bkg_fn,nbkg):

        self._xedge = np.array(hcounts.axis().edges())
        self._ncounts = copy.copy(hcounts.counts())
        self._ncounts = np.concatenate(([0],self._ncounts))
        self._bkg_fn = bkg_fn
        self._nbkg = nbkg

    def quantile(self,fraction=0.68):
        return self._quantile(self._nbkg,self._xedge,self._ncounts,fraction)
        
    def _quantile(self,nbkg=None,xedge=None,ncounts=None,fraction=0.68):

        if nbkg is None: nbkg = self._nbkg        
        if ncounts is None: ncounts = self._ncounts
        if xedge is None: xedge = self._xedge
        
        ncounts_cum = np.cumsum(ncounts)
        nbkg_cum = self._bkg_fn(xedge)*nbkg

        nex_cum = copy.copy(ncounts_cum)
        nex_cum -= nbkg_cum
        nex_tot = nex_cum[-1]

        fn_nexcdf = UnivariateSpline(xedge,nex_cum,s=0,k=1)
        
        # Find the first instance of crossing 1

        r = (nex_cum-nex_tot)        
        idx = np.where(r>=0)[0][0]
        xmax = xedge[idx]
        
        q = opt.brentq(lambda t: fn_nexcdf(t)-nex_tot*fraction,xedge[0],xmax)

        return q
        
        
    def bootstrap(self,fraction=0.68,niter=100,xmax=None):

        nedge = len(self._ncounts[self._xedge<=xmax])
        xedge = self._xedge[:nedge]
        
        h = Histogram.createHistModel(xedge,self._ncounts[1:nedge])
        nbkg = np.random.poisson(self._nbkg,niter)
        ncounts = np.random.poisson(np.concatenate(([0],h._counts)),
                                    (niter,nedge))
        
        xq = []

        for i in range(niter):
            xq.append(self._quantile(nbkg[i],self._xedge[:nedge],
                                     ncounts[i],fraction))

        xq_mean = np.mean(np.array(xq))
        xq_rms = np.std(np.array(xq))
        
        return xq_mean, xq_rms


class HistGOF(object):

    def __init__(self,h,hmodel):

        self._h = h
        self._hmodel = hmodel

        self.chi2 = 0

#        for b in self._h.iterbins():
#            self.chi2 += np.power(
        
    def chi2(self):
        return self.chi2
        

class HistQuantileBkgHist(object):

    def __init__(self,hon,hoff,alpha):

        self._xedges = np.array(hon.axis().edges())

        self._non = copy.copy(hon.counts())
        self._noff = copy.copy(hoff.counts())
        self._alpha = alpha
        self._non = np.concatenate(([0],self._non))
        self._noff = np.concatenate(([0],self._noff))


    def eval(self,fraction):

        return self.binomial(self._non,self._noff,fraction)


    def bootstrap(self,fraction=0.68,niter=1000,xmax=None):

        nedge = len(self._non)
        hon = Histogram.createHistModel(self._xedges,self._non[1:])
        hoff = Histogram.createHistModel(self._xedges,self._noff[1:])

        non = np.random.poisson(np.concatenate(([0],hon._counts)),
                                (niter,nedge))
        noff = np.random.poisson(np.concatenate(([0],hoff._counts)),
                                 (niter,nedge))

        xq = []

        for i in range(niter):

            xq.append(self._quantile(non[i],noff[i],fraction))

            

        xq_mean = np.mean(np.array(xq))
        xq_rms = np.std(np.array(xq))

        return xq_mean, xq_rms

    def quantile(self,fraction=0.68):
        return self._quantile(self._non,self._noff,fraction)

    def _quantile(self,non=None,noff=None,fraction=0.68):

        if non is None: non = self._non        
        if noff is None: noff = self._noff
        
        non_cum = np.cumsum(non)
        noff_cum = np.cumsum(noff)
        nex_cum = copy.copy(non_cum)
        nex_cum -= self._alpha*noff_cum
        
        non_tot = non_cum[-1]
        noff_tot = noff_cum[-1]
        nex_tot = non_tot-self._alpha*noff_tot
        
        fn_nexcdf = UnivariateSpline(self._xedges,nex_cum,s=0,k=1)

        return opt.brentq(lambda t: fn_nexcdf(t)-nex_tot*fraction,
                          self._xedges[0],self._xedges[-1])

    def binomial(self,non,noff,fraction=0.68):

        non_cum = np.cumsum(non)
        noff_cum = np.cumsum(noff)
        nex_cum = copy.copy(non_cum)
        nex_cum -= self._alpha*noff_cum
        
        non_tot = non_cum[-1]
        noff_tot = noff_cum[-1]
        nex_tot = non_tot-self._alpha*noff_tot

        fn_noncdf = UnivariateSpline(self._xedges,non_cum,s=0)
        fn_noffcdf = UnivariateSpline(self._xedges,noff_cum,s=0,k=1)
        fn_nexcdf = UnivariateSpline(self._xedges,nex_cum,s=0)

        xq = opt.brentq(lambda t: fn_nexcdf(t)-nex_tot*fraction,
                        self._xedges[0],self._xedges[-1])

        eff_on  = fn_noncdf(xq)/non_tot
        eff_off = fn_noffcdf(xq)/noff_tot

        nerr_on = np.sqrt(non_tot*eff_on*(1-eff_on))
        nerr_off = np.sqrt(noff_tot*eff_off*(1-eff_off))

        nerr = np.sqrt(nerr_on**2 + nerr_off**2)

        nerr_hi = nex_tot*fraction+nerr
        nerr_lo = nex_tot*fraction-nerr

        xq_hi = self._xedges[-1]
        xq_lo = self._xedges[0]

        if nerr_hi < nex_tot:
            xq_hi = opt.brentq(lambda t: fn_nexcdf(t)-nerr_hi,
                               self._xedges[0],self._xedges[-1])

        if nerr_lo > 0:
            xq_lo = opt.brentq(lambda t: fn_nexcdf(t)-nerr_lo,
                               self._xedges[0],self._xedges[-1])

        xq_err = 0.5*(xq_hi-xq_lo)
        
        return xq, xq_err


class HistQuantile(object):

    def __init__(self,hist):

        self._h = copy.deepcopy(hist)
        self._x = np.array(hist.edges(),copy=True)
        self._ncounts = copy.copy(hist._counts)
        self._ncounts = np.concatenate(([0],self._ncounts))


    def eval(self,fraction,method='binomial',**kwargs):

        if method == 'binomial':
            return self.binomial(self._ncounts,fraction,**kwargs)
        elif method == 'mc':
            return self.bootstrap(fraction,**kwargs)
        elif method is None:
            return [HistQuantile.quantile(self._h,fraction),0]
        else:
            print 'Unknown method ', method
            sys.exit(1)


    def bootstrap(self,fraction=0.68,niter=1000):

        nbin = len(self._ncounts)
        ncounts_tmp = np.random.poisson(self._ncounts,(niter,nbin))
       
        xq = []
        for i in range(niter):
            xq.append(HistQuantile.array_quantile(self._x,
                                                  ncounts_tmp[i],fraction))

        xq_mean = np.mean(np.array(xq))
        xq_rms = np.std(np.array(xq))

        return xq_mean, xq_rms

    @staticmethod
    def quantile(h,fraction=0.68):
        counts = np.concatenate(([0],h._counts))
        return HistQuantile.array_quantile(h.axis().edges(),counts,fraction)
        
    @staticmethod
    def array_quantile(edges,ncounts,fraction=0.68):
        """Find the value of X which contains the given fraction of counts
        in the histogram."""
        ncounts_cum = np.cumsum(ncounts)
        ncounts_tot = ncounts_cum[-1]

        fn_ncdf = UnivariateSpline(edges,ncounts_cum,s=0,k=1)

        return opt.brentq(lambda t: fn_ncdf(t)-ncounts_tot*fraction,
                          edges[0],edges[-1])

    @staticmethod
    def cumulative(h,x):

        if x <= h._xedges[0]: return 0
        elif x >= h._xedges[-1]: return h.sum()
        
        counts = np.concatenate(([0],h._counts))
        counts_cum = np.cumsum(counts)
        fn_ncdf = UnivariateSpline(h.axis().edges(),counts_cum,s=0,k=1)
        return fn_ncdf(x)
        
    def binomial(self,ncounts,fraction=0.68):

        ncounts_cum = np.cumsum(ncounts)
        ncounts_tot = ncounts_cum[-1]

        fn_ncdf = UnivariateSpline(self._x,ncounts_cum,s=0,k=1)        
        xq = opt.brentq(lambda t: fn_ncdf(t)-ncounts_tot*fraction,
                        self._x[0],self._x[-1])

        eff  = fn_ncdf(xq)/ncounts_tot
        nerr = np.sqrt(ncounts_tot*eff*(1-eff))

        nerr_hi = ncounts_tot*fraction+nerr
        nerr_lo = ncounts_tot*fraction-nerr

        xq_hi = self._x[-1]
        xq_lo = self._x[0]

        if nerr_hi < ncounts_tot:
            xq_hi = opt.brentq(lambda t: fn_ncdf(t)-nerr_hi,
                               self._x[0],self._x[-1])

        if nerr_lo > 0:
            xq_lo = opt.brentq(lambda t: fn_ncdf(t)-nerr_lo,
                               self._x[0],self._x[-1])

        xq_err = 0.5*(xq_hi-xq_lo)
        
        return xq, xq_err


    

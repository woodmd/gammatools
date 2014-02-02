"""
@file  likelihood.py

@brief Python classes related to fitting/calculation of likelihoods.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""
__source__   = "$Source: /nfs/slac/g/glast/ground/cvs/users/mdwood/python/likelihood.py,v $"
__author__   = "Matthew Wood <mdwood@slac.stanford.edu>"
__date__     = "$Date: 2013/08/15 20:48:09 $"
__revision__ = "$Revision: 1.8 $, $Author: mdwood $"

import numpy as np
import copy
import re
from scipy.interpolate import UnivariateSpline
from histogram import Histogram
from util import expand_aliases, get_parameters
from minuit import Minuit
from parameter_set import Parameter, ParameterSet
import matplotlib.pyplot as plt
from model_fn import ParamFn, Model

class CompProdModel(Model):

    def __init__(self):
        Model.__init__(self)
        self._models = []

    def addModel(self,m):
        self._models.append(m)
        self._param.addSet(m.param())

    def eval(self,x,p=None):
        s = None
        for i, m in enumerate(self._models):            
            v = m.eval(x,p)
            if i == 0: s = v
            else: s *= v
        return s
            
    def integrate(self,xlo,xhi,p=None):
        s = None
        for i, m in enumerate(self._models):
            v = m.integrate(xlo,xhi,p)
            if i == 0: s = v
            else: s *= v
        return s

class CompositeParameter(ParamFn):

    def __init__(self,expr,pset):
        ParamFn.__init__(self,pset)

        par_names = get_parameters(expr)
        for p in par_names:            
            self.addParameter(pset.getParByName(p))

        aliases = {}
        for k, p in self._param._pars.iteritems():
            aliases[p.name()] = 'pset[%i]'%(p.pid())
        expr = expand_aliases(aliases,expr)
        self._expr = expr

    def eval(self,x,p=None):
        pset = self.param(True)
        pset.update(p)
        return eval(self._expr)

class JointLnL(ParamFn):

    def __init__(self,lnlfn=None):
        ParamFn.__init__(self)
        self._lnlfn = []
        if not lnlfn is None:     
            for m in lnlfn: self.add(m)
        
    def add(self,lnl):
        self._lnlfn.append(lnl)
        self._param.addSet(lnl.param())

    def eval(self,p=None):

        pset = self.param(True)
        pset.update(p)

        s = None
        for i, m in enumerate(self._lnlfn):
            if i == 0: s = m.eval(pset)
            else: s += m.eval(pset)

        return s

class FitResults(ParameterSet):

    def __init__(self,pset,fval,cov=None):
        ParameterSet.__init__(self,pset)

        if cov is None: cov=np.zeros(shape=(pset.npar(),pset.npar()))
        else: self._cov = cov
        
        self._err = np.sqrt(np.diag(cov))
        self._fval = fval

    def fval(self):
        return self._fval

    def getParError(self,pid):

        if isinstance(pid,str):
            pid = self.getParByName(pid).pid()

        return self._err[pid]

    def __str__(self):

        os = ''
        for i, p in enumerate(self._pars):
            os += '%s %.6g\n'%(p,self._err[i])

        os += 'fval: %.3f\n'%(self._fval)
#        os += 'cov:\n %s'%(str(self._cov))

        return os

class Chi2Fn(ParamFn):

    def __init__(self,x,y,yerr,model):
        ParamFn.__init__(self,model.param())
        self._x = x
        self._y = y
        self._yerr = yerr
        self._model = model

    def __call__(self,p):
        return self.eval(p)
            
    def eval(self,p):

        pset = self._model.param(True)
        pset.update(p)

        fv = self._model(self._x,pset)

        var = self._yerr**2
        delta2 = (self._y-fv)**2
        v = delta2/var
        
        if v.ndim == 2:
            s = np.sum(v,axis=1)
        else:
            s = np.sum(v)

        return s


class Chi2HistFn(ParamFn):

    def __init__(self,h,model):
        ParamFn.__init__(self,model.param())
        self._h = h
        self._model = model

    def __call__(self,p):

        return self.eval(p)
        
    def eval(self,p):

        pset = self._model.param(True)
        pset.update(p)

        fv = self._model.histogram(self._h.edges(),pset)

#        print 'fv ', fv.shape
#        print fv

        var = self._h.var()
        ivar = np.zeros(shape=var.shape)
        ivar[var>0] = 1./var[var>0]
        
        delta2 = (self._h.counts()-fv)**2
        v = delta2*ivar

#        print 'v.shape ', v.shape
        
        if v.ndim == 2:
            return np.sum(v,axis=1)
        else:
            return np.sum(v)



        
class MinuitFitter(object):
    """Wrapper class for performing function minimization with minuit."""

    def __init__(self,objfn,tol=1E-3):
        self._objfn = objfn
        self._fval = 0
        self._tol = tol
        self._maxcalls = 1000

    def rnd_scan(self,par_index=None,nscan=100,scale=1.3):

        pset = copy.copy(self._objfn.param())
        p = pset.array()

        if par_index is None: par_index = range(pset.npar())

        prnd = np.ones((p.shape[0],nscan,1))
        prnd *= p

        for i in par_index:
            rnd = np.random.uniform(0.0,1.0,(nscan,1))
            prnd[i] = p[i] - p[i]/scale + rnd*(scale*p[i] - (p[i] - p[i]/scale))

        lnl = self._objfn.eval(prnd)

        imin = np.argmin(lnl)

        pset.update(prnd[:,imin])

        print self._objfn.eval(pset)

        return pset

    def profile(self,pset,pname,pval,refit=True):

#        pset = copy.deepcopy(self._objfn.param())
        pset = copy.deepcopy(pset)

        fval = []
        pset.getParByName(pname).fix(True)

        if refit is True:

            for p in pval:  
         
                pset.setParByName(pname,p)
                pset_fit = self.fit(pset)
                fval.append(pset_fit.fval())
        else:
            for p in pval:           
                pset.setParByName(pname,p)

                v = self._objfn.eval(pset)

#                print p, v, pset.getParByName('agn_norm').value()

                fval.append(v)
            
        return np.array(fval)

    def fit(self,pset=None):

        if pset is None: pset = self._objfn.param(True)

        npar = pset.npar()

        fixed = pset.fixed()
        lo_lims = npar*[None]
        hi_lims = npar*[None]
        lims = []
        
        for i, p in enumerate(pset):
            if not p.lims() is None: lims.append(p.lims())
            else: lims.append([0.0,0.0])
                            
        print pset.array()

        minuit = Minuit(lambda x: self._objfn.eval(x),
                        pset.array(),fixed=fixed,limits=lims,
                        tolerance=self._tol,strategy=1,
                        printMode=-1,
                        maxcalls=self._maxcalls)
        (pars,fval) = minuit.minimize()

        cov = minuit.errors()
        pset.update(pars)
            
        return FitResults(pset,fval,cov)

    def plot_lnl_scan(self,pset):
        print pset

        fig = plt.figure(figsize=(12,8))
        for i in range(9):

            j = i+4
            
            p = pset.makeParameterArray(j,
                                        np.linspace((pset[j].flat[0]*0.5),
                                                    (pset[j].flat[0]*2),50))
            y = self._objfn.eval(p)
            ax = fig.add_subplot(3,3,i+1)
            ax.set_title(p.getParByIndex(j).name())
            plt.plot(p[j],y-pset.fval())
            plt.axvline(pset[j])

    @staticmethod
    def fit2(objfn):
        """Convenience method for fitting."""
        fitter = Fitter(objfn)
        return fitter.fit()


class BFGSFitter(object):

    def __init__(self,objfn,tol=1E-3):

        self._objfn=objfn
    
    def fit(self,pset=None):

        if pset is None: pset = self._objfn.param(True)

        bounds = []
        for p in pset:
            if p.fixed():
                bounds.append([p.value().flat[0],p.value().flat[0]])
            else:
                bounds.append(p.lims())

        from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs

        res = fmin_bfgs(self._objfn,
                        pset.array(),None,bounds=bounds,
                        approx_grad=1)

        pset.update(res[0])        
        self._fit_results = FitResults(pset,res[1])

        # How to compute errors?
        
        return copy.deepcopy(self._fit_results)
        
#,epsilon=1E-10,
#                        iprint=0,pgtol=1E-10)



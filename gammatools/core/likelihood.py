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
        
class ParamFn(object):
    """Class for a parameterized function."""

    def __init__(self, param = None, name = None):

        if param is None: self._param = ParameterSet()
        else: self._param = param
        self._name = name

    def setName(self,name):
        """Set the name of the function."""
        self._name = name

    def name(self):
        """Return the name of the function."""
        return self._name

    def npar(self):
        return self._param.npar()

    def param(self,p=None):
        """Get the parameter set of this function.  If the optional input
        argument set is defined then return a copy of the model
        parameter set with values updated from this set."""

        if p is None: pset = self._param
        else:
            pset = copy.deepcopy(self._param)
            pset.setParam(p)

        return pset

    def setParam(self,pset):
        """Update the parameters."""
        self._param.setParam(pset)

    def setParamByIndex(self,index,value):
        """Set the value of one of the function parameters given its index."""

        self._param.getParByIndex(index).set(value)

    def setParamByName(self,name,value):
        """Set the value of one of the function parameters given its name."""

        self._param.getParByName(name).set(value)

    def addParameter(self,p):
        self._param.addParameter(p)


class Model(ParamFn):
    
    def __init__(self, pset=None, name=None, cname=None):
        ParamFn.__init__(self,pset,name)
        self._cname = cname

    def eval(self,x,p=None):
        
        pset = self.param(p)

        if isinstance(x,dict):
            x = np.array(x[self._cname],ndmin=1)
        else:
            x = np.array(x,ndmin=1)

        return self._eval(x,pset)

    def integrate(self,xlo,xhi,p=None):
        
        pset = self.param(p)

        if isinstance(xlo,dict):
            xlo = np.array(xlo[self._cname],ndmin=1)
            xhi = np.array(xhi[self._cname],ndmin=1)
        else:
            xlo = np.array(xlo,ndmin=1)
            xhi = np.array(xhi,ndmin=1)

        return self._integrate(xlo,xhi,pset)

    def histogram(self,edges,p=None):
        
        pset = self.param(p)

        if isinstance(edges,dict):
            edges = np.array(edges[self._cname],ndmin=1)
        else:
            edges = np.array(edges,ndmin=1)

        h = Histogram(edges,label=self.name())
        h._counts = self._integrate(edges[:-1],edges[1:],p)[0]
        return h

    def rnd(self,n,xmax,p=None):

        x = np.linspace(0,xmax,1000)
        cdf = self.cdf(x,p)
        cdf /= cdf[-1]

        fn = UnivariateSpline(cdf,x,s=0,k=1)

        p = np.random.uniform(0.0,1.0,n)

        return fn(p)
    
    def cdf(self,x,p=None):    
        return self.integrate(np.zeros(shape=x.shape),x,p)


class ScaledModel(Model):
    def __init__(self,model,pset,expr,name=None):
        Model.__init__(self,name=name)

#        pset = model.param()
        par_names = get_parameters(expr)
        for p in par_names:            
            self._param.addParameter(pset.getParByName(p))
        self._param.addSet(model.param())
        self._model = model

        aliases = {}
        for k, p in self._param._pars.iteritems():
            aliases[p.name()] = 'pset[%i]'%(p.pid())
        expr = expand_aliases(aliases,expr)
        self._expr = expr

    def eval(self,x,p=None):
        pset = self.param(p)

        if self._expr is None: return self._model.eval(x,pset)
        else: return self._model.eval(x,pset)*eval(self._expr)

    def integrate(self,xlo,xhi,p=None):        
        pset = self.param(p)

        if self._expr is None: return self._model.integrate(xlo,xhi,pset)
        else: return self._model.integrate(xlo,xhi,pset)*eval(self._expr)


class CompositeModel(Model):

    def __init__(self):
        Model.__init__(self)
        self._models = []

    def addModel(self,m):
        self._models.append(m)
        self._param.addSet(m.param())

    def _eval(self,x,pset=None):
        s = None
        for i, m in enumerate(self._models):
            
            v = m.eval(x,pset)
            if i == 0: s = v
            else: s += v
        return s
            
    def _integrate(self,xlo,xhi,pset=None):

        s = None
        for i, m in enumerate(self._models):

            v = m.integrate(xlo,xhi,pset)
            if i == 0: s = v
            else: s += v
        return s

    def histogramComponents(self,edges,p=None):

        hists = []
        for i, m in enumerate(self._models):
            hists.append(m.histogram(edges,p))
        return hists

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
        pset = self.param(p)
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

        p = self.param(p)
        s = None
        for i, m in enumerate(self._lnlfn):
            if i == 0: s = m.eval(p)
            else: s += m.eval(p)

        return s

class FitResults(ParameterSet):

    def __init__(self,pset,cov,fval):
        ParameterSet.__init__(self,pset)
        self._err = np.sqrt(np.diag(cov))
        self._cov = cov
        self._fval = fval

    def fval(self):
        return self._fval

    def getParError(self,pid):

        if isinstance(pid,str):
            pid = self.getParByName(pid).pid()

        return self._err[pid]

    def __str__(self):

        os = ''
        for i, k in enumerate(sorted(self._pars.keys())):
            os += '%s %.6g\n'%(self._pars[k],self._err[i])

        os += 'fval: %.3f\n'%(self._fval)
#        os += 'cov:\n %s'%(str(self._cov))

        return os






class Fitter(object):
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

        pset.setParam(prnd[:,imin])

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

        if pset is None:
            pset = copy.copy(self._objfn.param())

        npar = pset.npar()

        fixed = pset.fixed()
        lo_lims = npar*[None]
        hi_lims = npar*[None]
        lims = []
        
        for i, pid in enumerate(pset.pids()):
            if not pset.getParByIndex(pid).lims() is None:
                lims.append(pset.getParByIndex(pid).lims())
            else: lims.append([0.0,0.0])
                            
        minuit = Minuit(lambda x: self._objfn.eval(x),
                        pset.array(),fixed=fixed,limits=lims,
                        tolerance=self._tol,strategy=1,
                        printMode=-1,
                        maxcalls=self._maxcalls)
        (pars,fval) = minuit.minimize()

        cov = minuit.errors()
        pset.setParam(pars)
            
        return FitResults(pset,cov,fval)

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

    def fit(self,pset=None):

        from scipy.optimize import minimize

        bounds = []
        for p in pset:
            if p.fixed():
                bounds.append([p.value().flat[0],p.value().flat[0]])
            else:
                bounds.append([None,None])

        bounds[4] = [0,None]
        bounds[5] = [0,None]
        bounds[8] = [0,None]

        print bounds
        print pset.array()

        from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
#        from scipy.optimize import fmin_bfgs
#       from scipy.optimize import anneal

#        res = minimize(lambda x: self._objfn.eval(x),
#                       pset.array(),method='L-BFGS-B',bounds=bounds,
#                       factr=10.0)

        print 'Performing BFGS Fit'

        class TestFn(object):

            def __init__(self,fn):
                self._fn = fn
                self._ncall = 0

            def __call__(self,p):
                self._ncall +=1

                print self._ncall
                print p

                return self._fn.eval(p)

            

        testfn = TestFn(self._objfn)

#lambda x: self._objfn.eval(x),
        res = fmin_bfgs(testfn,
                       pset.array(),None,bounds=bounds,
                       approx_grad=1)
#,epsilon=1E-10,
#                        iprint=0,pgtol=1E-10)

#        res = anneal(lambda x: self._objfn.eval(x),
#                       pset.array(),None,bounds=bounds)

        print res

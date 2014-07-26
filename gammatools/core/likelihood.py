"""
@file  likelihood.py

@brief Python classes related to calculation of likelihoods.

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
from model_fn import ParamFnBase, PDF

class CompProdModel(PDF):

    def __init__(self):
        PDF.__init__(self)
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

class CompositeParameter(ParamFnBase):

    def __init__(self,expr,pset):
        ParamFnBase.__init__(self,pset)

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

class JointLnL(ParamFnBase):

    def __init__(self,lnlfn=None):
        ParamFnBase.__init__(self)
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



def chi2(y,var,fy,fvar=None):
    tvar = var 
    if not fvar is None: tvar += fvar
    ivar = np.zeros(shape=var.shape)
    ivar[var>0] = 1./tvar[tvar>0]
        
    delta2 = (y-fy)**2
    return delta2*ivar

class BinnedChi2Fn(ParamFnBase):
    """Objective function for binned chi2."""
    def __init__(self,h,model):
        ParamFnBase.__init__(self,model.param())
        self._h = h
        self._model = model

    def __call__(self,p):
        return self.eval(p)
            
    def eval(self,p):

        pset = self._model.param(True)
        pset.update(p)

        fv = self._model.histogram(self._h.axis().edges(),pset)
        v = chi2(self._h.counts,self._h.var,fv)

        if v.ndim == 2:
            s = np.sum(v,axis=1)
        else:
            s = np.sum(v)

        return s

class Chi2Fn(ParamFnBase):

    def __init__(self,x,y,yerr,model):
        ParamFnBase.__init__(self,model.param())
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


class Chi2HistFn(ParamFnBase):

    def __init__(self,h,model):
        ParamFnBase.__init__(self,model.param())
        self._h = h
        self._model = model

    def __call__(self,p):

        return self.eval(p)
        
    def eval(self,p):

        pset = self._model.param(True)
        pset.update(p)

        fv = self._model.counts(pset)
        fvar = self._model.var(pset)

        var = self._h.var + fvar
        ivar = np.zeros(shape=var.shape)
        ivar[var>0] = 1./var[var>0]
        
        delta2 = (self._h.counts-fv)**2
        v = delta2*ivar
        
        if v.ndim == 2:
            return np.sum(v,axis=1)
        else:
            return np.sum(v)



        

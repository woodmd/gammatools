"""
@file  model_fn.py

@brief Python classes related to fitting/calculation of likelihoods.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood <mdwood@slac.stanford.edu>"

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

    def param(self,make_copy=False):
        """Get the parameter set of this function.  If the optional input
        argument set is defined then return a copy of the model
        parameter set with values updated from this set."""

        if make_copy: return copy.deepcopy(self._param)
        else: return self._param

    def update(self,pset):
        """Update the parameters."""
        self._param.update(pset)

class Model(ParamFn):
    
    def __init__(self, pset=None, name=None, cname=None):
        ParamFn.__init__(self,pset,name)
        self._cname = cname

    def __call__(self,x,p=None):
        return self.eval(x,p)

    def eval(self,x,p=None):
        
        pset = self.param(True)
        pset.update(p)

        if isinstance(x,dict):
            x = np.array(x[self._cname],ndmin=1)
        else:
            x = np.array(x,ndmin=1)

        return self._eval(x,pset)

    def set_norm(self,norm,xlo,xhi):

        n = self.integrate(xlo,xhi)

        for i, p in enumerate(self._param):
            self._param[i].set(self._param[i].value()*norm/n)

    def integrate(self,xlo,xhi,p=None):
        
        pset = self.param(True)
        pset.update(p)

        if isinstance(xlo,dict):
            xlo = np.array(xlo[self._cname],ndmin=1)
            xhi = np.array(xhi[self._cname],ndmin=1)
        else:
            xlo = np.array(xlo,ndmin=1)
            xhi = np.array(xhi,ndmin=1)

        return self._integrate(xlo,xhi,pset)

    def histogram(self,edges,p=None):
        
        pset = self.param(True)
        pset.update(p)

        if isinstance(edges,dict):
            edges = np.array(edges[self._cname],ndmin=1)
        else:
            edges = np.array(edges,ndmin=1)

        return self._integrate(edges[:-1],edges[1:],p)

    def rnd(self,n,xmin,xmax,p=None):

        x = np.linspace(xmin,xmax,1000)
        cdf = self.cdf(x,p)
        cdf /= cdf[-1]

        fn = UnivariateSpline(cdf,x,s=0,k=1)

        p = np.random.uniform(0.0,1.0,n)

        return fn(p)
    
    def cdf(self,x,p=None):    
        return self.integrate(np.zeros(shape=x.shape),x,p)

def polyval(c,x):

    c = np.array(c, ndmin=2, copy=True)
    x = np.array(x, ndmin=1, copy=True)

#    print 'x ', x.shape, x
#    print 'c ', c.shape, c
    
    x.shape = c.ndim*(1,) + x.shape 
    c.shape += (1,)

    c0 = c[-1] 
    for i in range(2, len(c) + 1) :
        c0 = c[-i] + c0*x

#    print 'c0 ', c0.shape, len(x)

    if c.shape[1] == 1:
        return c0.reshape(c0.shape[-1])
    else:
        return c0.reshape(c0.shape[1:])

class GaussFn(Model):


    @staticmethod
    def create(norm,mu,sigma):
        
        pass


class LogParabola(Model):

    def __init__(self,pset,name=None):
        Model.__init__(self,pset,name)
        
    def _eval(self,x,pset):

        a = pset.array()
        es = 10**(x-a[3])
        return 10**a[0]*np.power(es,-(a[1]+a[2]*np.log(es)))

    @staticmethod
    def create(norm,alpha,beta,eb):

        pset = ParameterSet()
        pset.createParameter(np.log10(norm),'norm')
        pset.createParameter(alpha,'alpha')
        pset.createParameter(beta,'beta')
        pset.createParameter(eb,'eb')
        return LogParabola(pset)

    
class PolyFn(Model):
    def __init__(self,pset,name=None):
        Model.__init__(self,pset,name)
        self._nc = pset.npar()

    @staticmethod
    def create(norder,coeff=None,offset=0):

        pset = ParameterSet()
        if coeff is None: coeff = np.zeros(norder)
        for i in range(norder):
            pset.addParameter(Parameter(offset+i,coeff[i],'a%i'%i))

        return PolyFn(pset)

    def _eval(self,x,pset):
        
        a = pset.array()
        return polyval(a,x)

    def _integrate(self,dlo,dhi,pset):

        a = pset.array()

        if a.ndim == 1:
            aint = np.zeros(self._nc+1)
            aint[1:] = a/np.linspace(1,self._nc,self._nc)
            return polyval(aint,dhi) - polyval(aint,dlo)
        else:
            aint = np.zeros(shape=(self._nc+1,a.shape[1]))
            c = np.linspace(1,self._nc,self._nc)
            c = c.reshape(c.shape + (1,))

            aint[1:] = a/c
            v = polyval(aint,dhi) - polyval(aint,dlo)
            return v

class PolarPolyFn(PolyFn):

    @staticmethod
    def create(norder,coeff=None,offset=0):

        pset = ParameterSet()
        if coeff is None: coeff = np.zeros(norder)
        for i in range(norder):
            pset.addParameter(Parameter(offset+i,coeff[i],'a%i'%i))

        return PolarPolyFn(pset)

    

    def _integrate(self,dlo,dhi,pset):

        a = pset.array()

        if a.ndim == 1:
            aint = np.zeros(self._nc+2)
            aint[2:] = a/np.linspace(1,self._nc,self._nc)
            return np.pi*(polyval(aint,dhi) - polyval(aint,dlo))
        else:
            aint = np.zeros(shape=(self._nc+2,) + a.shape[1:])
            c = np.linspace(2,self._nc+1,self._nc)
            c = c.reshape(c.shape + (1,))

#            print 'integrate ', aint.shape, c.shape

            aint[2:] = a/c
            v = np.pi*(polyval(aint,dhi) - polyval(aint,dlo))
            return v

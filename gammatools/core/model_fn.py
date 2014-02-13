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
    
    def __init__(self, pset=None, name=None):
        ParamFn.__init__(self,pset,name)

    def __call__(self,x,p=None):
        return self.eval(x,p)

    def eval(self,x,p=None):
        
        pset = self.param(True)
        pset.update(p)

        x = np.array(x,ndmin=1)

        return self._eval(x,pset)

    def set_norm(self,norm,xlo,xhi):

        n = self.integrate(xlo,xhi)

        for i, p in enumerate(self._param):
            self._param[i].set(self._param[i].value()*norm/n)

    def integrate(self,xlo,xhi,p=None):
        
        pset = self.param(True)
        pset.update(p)

        xlo = np.array(xlo,ndmin=1)
        xhi = np.array(xhi,ndmin=1)

        return self._integrate(xlo,xhi,pset)

    def histogram(self,edges,p=None):
        
        pset = self.param(True)
        pset.update(p)
        edges = np.array(edges,ndmin=1)

        return self._integrate(edges[:-1],edges[1:],pset)

    def rnd(self,n,xmin,xmax,p=None):

        x = np.linspace(xmin,xmax,1000)
        cdf = self.cdf(x,p)
        cdf /= cdf[-1]

        fn = UnivariateSpline(cdf,x,s=0,k=1)

        pv = np.random.uniform(0.0,1.0,n)

        return fn(pv)
    
    def cdf(self,x,p=None):    
        return self.integrate(np.zeros(shape=x.shape),x,p)

class ScaledHistogramModel(Model):

    def __init__(self,h,pset,name=None):
        Model.__init__(self,pset,name)
        self._h = copy.deepcopy(h)
    
    @staticmethod
    def create(h,norm=1.0,pset=None,name=None,prefix=''):

        if pset is None: pset = ParameterSet()
        p0 = pset.createParameter(norm,prefix + 'norm')
        return ScaledHistogramModel(h,ParameterSet([p0]),name)        

    def var(self,p):

        pset = self.param(True)
        pset.update(p)
        
        a = pset.array()
        if a.shape[1] > 1: a = a[...,np.newaxis]        
        return a[0]**2*self._h.var()

    def counts(self,p):

        pset = self.param(True)
        pset.update(p)
        
        a = pset.array()
        if a.shape[1] > 1: a = a[...,np.newaxis]        
        return a[0]*self._h.counts()
    
    def _eval(self,x,pset):
        
        a = pset.array()
        if a.shape[1] > 1: a = a[...,np.newaxis]        
        return a[0]*self._h.interpolate(x)

    def _integrate(self,xlo,xhi,pset):
        
        a = pset.array()
        if a.shape[1] > 1: a = a[...,np.newaxis]        
        return a[0]*self._h.counts()
    
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
        pset = self.param(True)
        pset.update(p)

        if self._expr is None: return self._model.eval(x,pset)
        else: return self._model.eval(x,pset)*eval(self._expr)

    def integrate(self,xlo,xhi,p=None):        
        pset = self.param(True)
        pset.update(p)

        if self._expr is None: return self._model.integrate(xlo,xhi,pset)
        else: return self._model.integrate(xlo,xhi,pset)*eval(self._expr)

class CompositeSumModel(Model):

    def __init__(self,models=None):
        Model.__init__(self)
        self._models = []

        if not models is None:
            for m in models: self.addModel(m)
        
    def addModel(self,m):
        self._models.append(copy.deepcopy(m))
        self._param.addSet(m.param())

    def counts(self,pset=None):

        s = None
        for i, m in enumerate(self._models):

            v = m.counts(pset)
            
            if i == 0: s = v
            else: s += v
        return s

    def var(self,pset=None):

        s = None
        for i, m in enumerate(self._models):

            v = m.var(pset)
            
            if i == 0: s = v
            else: s += v
        return s
        
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

            c = m.histogram(edges,p) 
            h = Histogram(edges,label=m.name(),counts=c,var=0)
            
            hists.append(h)
        return hists

    
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
    def create(norm,mu,sigma,pset=None):
        
        if pset is None: pset = ParameterSet()
        p0 = pset.createParameter(norm,'norm')
        p1 = pset.createParameter(mu,'mu')
        p2 = pset.createParameter(sigma,'sigma')
        return GaussFn(ParameterSet([p0,p1,p2]))

    def _eval(self,x,pset):

        a = pset.array()
        sig2 = a[2]        
        return a[0]/np.sqrt(2.*np.pi*sig2)*np.exp(-(x-a[1])**2/(2.0*sig2))

    
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
        pset.createParameter(np.log10(eb),'eb')
        return LogParabola(pset)

class PowerLawExp(Model):

    def __init__(self,pset,name=None):
        Model.__init__(self,pset,name)
        
    def _eval(self,x,pset):

        a = pset.array()
        es = 10**(x-a[3])
        return 10**a[0]*np.power(es,-a[1])*np.exp(-10**(x-a[2]))

    @staticmethod
    def create(norm,alpha,ecut,eb):

        pset = ParameterSet()
        pset.createParameter(np.log10(norm),'norm')
        pset.createParameter(alpha,'alpha')
        pset.createParameter(np.log10(ecut),'ecut')
        pset.createParameter(np.log10(eb),'eb')
        return PowerLawExp(pset)

class PowerLaw(Model):

    def __init__(self,pset,name=None):
        Model.__init__(self,pset,name)      

    def _eval(self,x,pset):

        a = pset.array()
        es = 10**(x-a[2])
        return 10**a[0]*np.power(es,-a[1])

    def _integrate(self,xlo,xhi,p):

        x = 0.5*(xhi+xlo)
        dx = xhi-xlo

        a = pset.array()
        
        norm = 10**a[0]
        gamma = a[1]
        enorm = a[2]

        g1 = -gamma+1
        return norm/g1*10**(gamma*enorm)*(10**(xhi*g1) - 10**(xlo*g1))

    @staticmethod
    def create(norm,gamma,eb,pset=None,name=None,prefix=''):

        if pset is None: pset = ParameterSet()
        p0 = pset.createParameter(np.log10(norm),prefix+'norm')
        p1 = pset.createParameter(gamma,prefix+'gamma')
        p2 = pset.createParameter(np.log10(eb),prefix+'eb')
        return PowerLaw(ParameterSet([p0,p1,p2]),name)
        
class PolyFn(Model):
    def __init__(self,pset,name=None):
        Model.__init__(self,pset,name)
        self._nc = pset.npar()

    @staticmethod
    def create(norder,coeff=None,pset=None,name=None,prefix=''):

        if pset is None: pset = ParameterSet()
        if coeff is None: coeff = np.zeros(norder)

        pars = []
        for i in range(norder):
            p = pset.createParameter(coeff[i],prefix+'a%i'%i)
            pars.append(p)

        return PolyFn(ParameterSet(pars),name)

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



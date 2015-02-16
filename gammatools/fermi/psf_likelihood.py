#!/usr/bin/env python

import copy
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

from gammatools.core.histogram import Histogram

from gammatools.core.model_fn import PDF, ParamFnBase

from gammatools.core.util import convolve2d_gauss
import scipy.special as spfn


class ConvolvedGaussFn(PDF):
    def __init__(self,pset,psf_model):   
        PDF.__init__(self,pset)
        self._psf_model = copy.deepcopy(psf_model)
#        self._pid = [pnorm.pid(),psigma.pid()]

        x = np.linspace(-4,4,800)
        self._ive = UnivariateSpline(x,spfn.ive(0,10**x),s=0,k=2)

    @staticmethod
    def create(norm,sigma,psf_model,pset=None,prefix=''):

        if pset is None: pset = ParameterSet()
#        
        p0 = pset.createParameter(norm,prefix + 'norm')
        p1 = pset.createParameter(sigma,prefix + 'sigma')
        pset.addSet(psf_model.param()) 

        return ConvolvedGaussFn(pset,psf_model)

    def _eval(self,dtheta,pset):

        a = pset.array()

#        norm = pset[self._pid[0]]
#        sig = pset[self._pid[1]]

#        self._psf_model.setParam(pset)
        v = self._psf_model.eval(dtheta,pset)

        return a[0]*self.convolve(lambda x: self._psf_model.eval(x),
                                  dtheta,a[1],3.0,nstep=200)

    def _integrate(self,xlo,xhi,pset):

        xlo = np.array(xlo,ndmin=1,copy=True)
        xhi = np.array(xhi,ndmin=1,copy=True)

        xlo.shape += (1,)
        xhi.shape += (1,)

        nbin = 3

        xedge = np.linspace(0,1,nbin+1)
        x = 0.5*(xedge[1:]+xedge[:-1])
        x.shape = (1,) + x.shape

        xp = np.zeros(shape=xlo.shape + (nbin,))

        xp = xlo + x*(xhi-xlo)
        dx = (xhi-xlo)/float(nbin)
        dx.shape = (1,) + dx.shape

        v = self.eval(xp.flat,pset)
        v = v.reshape((v.shape[0],) + xp.shape)
        v *= 2*dx*xp*np.pi

        return np.sum(v,axis=2)

    def convolve(self,fn,r,sig,rmax,nstep=200):
        r = np.array(r,ndmin=1,copy=True)
        sig = np.array(sig,ndmin=1,copy=True)

        rp = np.ones(shape=(1,1,nstep))
        rp *= np.linspace(0,rmax,nstep)
             
        r = r.reshape((1,r.shape[0],1))
        sig = sig.reshape(sig.shape + (1,))
        
        dr = rmax/float(nstep)

        sig2 = sig*sig
        x = r*rp/(sig2)

        x[x==0] = 1E-4
        je = self._ive(np.log10(x.flat))
        je = je.reshape(x.shape)

#        je2 = spfn.ive(0,x)
#        plt.hist(((je-je2)/je2).flat)
#        plt.show()

        fnrp = fn(rp.flat)
        fnrp = fnrp.reshape((sig.shape[0],) + (1,) + (rp.shape[2],))
        s = np.sum(rp*fnrp/(sig2)*
                   np.exp(np.log(je)+x-(r*r+rp*rp)/(2*sig2)),axis=2)*dr

        return s

        

class KingFn(PDF):
    def __init__(self,psigma,pgamma,pnorm=None):

        pset = ParameterSet([psigma,pgamma])
        self._pid = [psigma.pid(),pgamma.pid()]
        if not pnorm is None: 
            pset.addParameter(pnorm)
            self._pid += [pnorm.pid()]
        PDF.__init__(self,pset,cname='dtheta')      

    @staticmethod
    def create(sigma,gamma,norm=1.0,offset=0):

        return KingFn(Parameter(offset+0,sigma,'sigma'),
                      Parameter(offset+1,gamma,'gamma'),
                      Parameter(offset+2,norm,'norm'))

#    def setSigma(self,sigma):
#        self._param.getParByID(self._pid[1]).set(sigma)

#    def setNorm(self,norm):
#        self._param.getParByID(self._pid[0]).set(norm)

    def norm(self):
        return self._param.getParByID(self._pid[2])

    def _eval(self,x,pset):

        sig = pset[self._pid[0]]
        g = pset[self._pid[1]]        
        if len(self._pid) == 3: norm = pset[self._pid[2]]
        else: norm = 1.0

#        if len(norm) > 1:
#            norm = norm.reshape(norm.shape + (1,)*dtheta.ndim)
#            sig = sig.reshape(sig.shape + (1,)*dtheta.ndim)
#            g = g.reshape(g.shape + (1,)*dtheta.ndim)

        g[g<=1.1] = 1.1

        n = 2*np.pi*sig*sig            
        u = np.power(x,2)/(2*sig*sig)
        
        return norm*(1-1/g)*np.power(1+u/g,-g)/n
    
    def integrate(self,dlo,dhi,pset):

        sig = pset[self._pid[0]]
        g = pset[self._pid[1]]        
        if len(self._pid) == 3: norm = pset[self._pid[2]]
        else: norm = 1.0

        g[g<=1.1] = 1.1

        um = dlo*dlo/(2.*sig*sig)
        ua = dhi*dhi/(2.*sig*sig)
        f0 = (1+um/g)**(-g+1)
        f1 = (1+ua/g)**(-g+1)
        return norm*(f0-f1)

    def cdf(self,dtheta,p=None):
    
        return self.integrate(0,dtheta,p)

class PulsarOnFn(PDF):

    def __init__(self,non,noff,alpha,psf_model):
        self._non = copy.copy(non)
        self._noff = copy.copy(noff)
        self._alpha = alpha
        self._model = model
        self._xedge = xedge
        self._mub0 = (self._non+self._noff)/(1+alpha)
        

    def eval(dtheta,p=None):

        alpha = self._alpha

        mus = self._psf_model.integrate(xlo,xhi,p)

        mub = ((self._mub0/2. - mus/(2.*alpha) + 
                np.sqrt(noff*mus/(alpha*(1+alpha)) + 
                        (self._mub0/2.-mus/(2.*alpha))**2)))

class BinnedPLFluxModel(PDF):

    def __init__(self,spectral_model,spatial_model,ebin_edges,exp):
        pset = ParameterSet()
        pset.addSet(spectral_model.param())
        pset.addSet(spatial_model.param())
        PDF.__init__(self,pset)      
        self._ebin_edges = ebin_edges
        self._exp = exp
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

    def _eval(self,x,p):

        v0 = self.spatial_model._eval(x,p)
        v1 = self.spectral_model._eval(self._ebin_edges[0],
                                       self._ebin_edges[1],p)
        return v0*v1*self._exp

    def _integrate(self,xlo,xhi,p):

        v0 = self.spatial_model._integrate(xlo,xhi,p)
        v1 = self.spectral_model._integrate(self._ebin_edges[0],
                                            self._ebin_edges[1],p)
        return v0*v1*self._exp

class PowerlawFn(PDF):

    def __init__(self,pnorm,pgamma):
        pset = ParameterSet([pnorm,pgamma])
        self._pid = [pnorm.pid(),pgamma.pid()]
        self._enorm = 3.0
        PDF.__init__(self,pset,cname='energy')      

    def _eval(self,x,p):

        norm = p[self._pid[0]]
        gamma = p[self._pid[1]]        

        return norm*10**(-gamma*(x-self._enorm))

    def _integrate(self,xlo,xhi,p):

        x = 0.5*(xhi+xlo)
        dx = xhi-xlo

        norm = p[self._pid[0]]
        gamma = p[self._pid[1]]  

        g1 = -gamma+1
        return norm/g1*10**(gamma*self._enorm)*(10**(xhi*g1) - 10**(xlo*g1))

class BinnedLnL(ParamFnBase):

    def __init__(self,non,xedge,model):
        ParamFnBase.__init__(self,model.param())
        self._non = non
        self._model = model
        self._xedge = xedge        

    @staticmethod
    def createFromHist(hon,model):
        return BinnedLnL(hon.counts(),hon.edges(),model)

    def eval(self,p):

        pset = self._model.param(True).update(p)

        nbin = len(self._xedge)-1
        xlo = self._xedge[:-1]
        xhi = self._xedge[1:]


        non = (np.ones(shape=(pset.size(),nbin))*self._non)

        mus = self._model.integrate(xlo,xhi,pset)
        msk_on = non > 0

        lnl = (-mus)
        lnl[msk_on] += non[msk_on]*np.log(mus[msk_on])

        if pset.size() > 1: return -np.sum(lnl,axis=1)
        else: return -np.sum(lnl)

class Binned2DLnL(ParamFnBase):

    def __init__(self,non,xedge,yedge,model):
        ParamFnBase.__init__(self,model.param())
        self._non = non.flat
        self._model = model
        self._xedge = xedge        
        self._yedge = yedge 

        xlo, ylo = np.meshgrid(self._xedge[:-1],self._yedge[:-1])
        xhi, yhi = np.meshgrid(self._xedge[1:],self._yedge[1:])

        self._xlo = xlo.T.flat
        self._ylo = ylo.T.flat

        self._xhi = xhi.T.flat
        self._yhi = yhi.T.flat

    @staticmethod
    def createFromHist(hon,model):
        return Binned2DLnL(hon.counts(),hon.xedges(),hon.yedges(),model)

    def eval(self,p):

        pset = self._model.param(True).update(p)

        nbinx = len(self._xedge)-1
        nbiny = len(self._yedge)-1

        non = (np.ones(shape=(pset.size(),nbinx*nbiny))*self._non)

        clo = { 'energy' : self._xlo, 'dtheta' : self._ylo }
        chi = { 'energy' : self._xhi, 'dtheta' : self._yhi }

        mus = self._model.integrate(clo,chi,pset)
        msk_on = non > 0

        lnl = (-mus)
        lnl[msk_on] += non[msk_on]*np.log(mus[msk_on])

        if pset.size() > 1: return -np.sum(lnl,axis=1)
        else: return -np.sum(lnl)

class OnOffBinnedLnL(ParamFnBase):

    def __init__(self,non,noff,xedge,alpha,model):
        ParamFnBase.__init__(self,model.param())
        self._non = copy.copy(non)
        self._noff = copy.copy(noff)
        self._alpha = alpha
        self._model = model
        self._xedge = xedge
        self._mub0 = (self._non+self._noff)/(1+alpha)

    @staticmethod
    def createFromHist(hon,hoff,alpha,model):
        return OnOffBinnedLnL(hon.counts(),hoff.counts(),hon.edges(),
                              alpha,model)

    def eval(self,p):

        pset = self._model.param(True)
        pset.update(p)

        alpha = self._alpha

        nbin = len(self._xedge)-1

        xlo = self._xedge[:-1]
        xhi = self._xedge[1:]

#        non = copy.deepcopy(self._non)
#        noff = copy.deepcopy(self._noff)

#        non.shape = (1,) + non.shape
#        noff.shape = (1,) + noff.shape

#        if pset.size() > 1:
        non = (np.ones(shape=(pset.size(),nbin))*self._non)
        noff = (np.ones(shape=(pset.size(),nbin))*self._noff)
#        else:            
#            non = self._non
#            noff = self._noff

        mus = self._model.integrate(xlo,xhi,pset)

        mub = ((self._mub0/2. - mus/(2.*alpha) + 
               np.sqrt(noff*mus/(alpha*(1+alpha)) + 
                       (self._mub0/2.-mus/(2.*alpha))**2)))



        msk_on = non > 0
        msk_off = noff > 0
    
        lnl = (-self._alpha*mub - mus - mub)
        lnl[msk_on] += non[msk_on]*np.log(self._alpha*mub[msk_on]+
                                          mus[msk_on])

        if np.any(msk_off):
            lnl[msk_off] += noff[msk_off]*np.log(mub[msk_off])

        if pset.size() > 1:
            return -np.sum(lnl,axis=1)
        else:
            return -np.sum(lnl)



        


if __name__ == '__main__':

    gfn = ConvolvedGaussFn.create(3.0,0.1,KingFn.create(0.1,3.0),4)

    pset = gfn.param()

    pset = pset.makeParameterArray(0,np.linspace(0.5,1,8))

    xlo = np.array([0])
    xhi = np.array([0.5])


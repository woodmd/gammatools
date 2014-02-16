#!/usr/bin/env python

import os
import sys
import re
import yaml

import numpy as np
import matplotlib.pyplot as plt
import gammatools.dm.jcalc as jcalc
from gammatools.core.util import Units
import scipy.special as spfn
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline
from gammatools.core.util import *
from gammatools.core.histogram import *

class LimitData(object):

    def __init__(self,f,label,color='k',linestyle='-'):
        d = np.load(f)
        self.data = d
        self.mass = d['mass']
        self.ul_med = []
        self.ul68_lo = []
        self.ul68_hi = []
        self.ul95_lo = []
        self.ul95_hi = []
        self.label= label
        self.color=color
        self.linestyle=linestyle

        for i in range(len(d['mass'])):

            ul = np.sort(d['ul'][:,i])
            ul = ul[ul>0]
            
            n = len(ul)
        
            m = np.median(ul)

            self.ul68_lo.append(ul[max(0,n/2.-n*0.34)])
            self.ul68_hi.append(ul[min(n-1,n/2.+n*0.34)])
            self.ul95_lo.append(ul[max(0,n/2.-n*0.95/2.)])
            self.ul95_hi.append(ul[min(n-1,n/2.+n*0.95/2.)])
            self.ul_med.append(np.median(ul))

    def plot(self):
        plt.plot(self.mass,self.ul_med,color=self.color,
                 linestyle=self.linestyle,
             linewidth=2,label=self.label)

        if len(self.data['ul'][:]) > 2:
            plt.gca().fill_between(self.mass, 
                                   self.ul68_lo,self.ul68_hi,
                                   facecolor=self.color, alpha=0.4)


            plt.gca().add_patch(plt.Rectangle((0,0),0,0,fc=self.color,alpha=0.4,
                                              zorder=0,
                                              label=self.label + " 68% Containment"))

        if len(self.data['ul'][:]) > 10:
            plt.gca().fill_between(self.mass, 
                                   self.ul95_lo,self.ul95_hi,
                                   facecolor=self.color, alpha=0.2)

            plt.gca().add_patch(plt.Rectangle((0,0),0,0,fc='black',alpha=0.2,
                                              zorder=0,
                                              label= self.label + " 95% Containment"))


class ModelPMSSM(object):

    def __init__(self,model_data,sp):

        index = model_data[0]
        model_no = model_data[1]
        model_id = int(1E5*model_data[0]) + int(model_data[1])

        self._model_id = int(1E5*index) + int(model_no)
        self._model_no = model_no
        self._model_index = index
        self._spectrum = sp
        self._model_data = model_data

class JFunction(object):
    """Class that stores the J density profile convolved with an
    energy-dependent PSF."""
    def __init__(self,fn,irf):

        self._irf = irf

        self._loge_axis = self._irf._psf.axis()
        th68 = self._irf._psf.counts()

        self._psi_axis = Axis(np.linspace(0,np.radians(3.0),401))
        self._psi = self._psi_axis.center()
        self._loge = self._loge_axis.center()+Units.log10_mev

        self._z = np.zeros(shape=(len(self._loge),len(self._psi)))

        self._h = Histogram2D(self._loge_axis,self._psi_axis)

        for i in range(len(self._loge)):
            self._z[i] = convolve2d_gauss(fn,self._psi,
                                          np.radians(th68[i]/1.50959),
                                          self._psi[-1])
        self._h._counts = self._z

    def __call__(self,loge,psi):

        return interpolate2d(self._loge,self._psi,self._z,
                             loge,psi)


def multiply(x,y,dims):
    """Mutiply x by y."""

    z = copy.copy(x)
    
    shape = list(z.shape)
    for i in range(z.ndim):
        if not i in dims: shape[i] = 1
    z *= y.reshape(shape)
    return z

def outer(x,y):
    z = np.ones(shape=(x.shape + y.shape))
    z = (z.T*x.T).T
    z *= y
    return z

def rebin(x,n):

    z = np.zeros(len(x)/n)
    for i in range(len(x)/n):
        z[i] = np.sum(x[i*n:(i+1)*n])

    return z

class DMChanSpectrum(object):
    """Class that computes the differential annihilation yield for
    different DM channels.  Interpolates a set of tabulated values
    from DarkSUSY."""
    def __init__(self,chan,mass = 100*Units.gev):

        dirname = os.path.dirname(__file__)

        d = np.loadtxt(os.path.join(dirname,'gammamc_dif.dat'),unpack=True)

        xedge = np.linspace(0,1.0,251)
        self._x = 0.5*(xedge[1:]+xedge[:-1])

        self._mass = mass
        self._xwidth = self._x[1:]-self._x[:-1]

        self._mass_bins = np.array([10.0,25.0,50.0,80.3,91.2,
                                    100.0,150.0,176.0,200.0,250.0,
                                    350.0,500.0,750.0,
                                    1000.0,1500.0,2000.0,3000.0,5000.0])

        self._mwidth = self._mass_bins[1:]-self._mass_bins[:-1]
        self._ndec = 10.0

        channel = ['cc','bb','tt','tau','ww','zz']
        offset = [0,0,7,0,3,4,0,0]

        channel_index = { 'cc' : 0,
                          'bb' : 1,
                          'tt' : 2,
                          'tau' : 3,
                          'ww' : 4,
                          'zz' : 5,
                          'mumu' : 6,
                          'gg'  : 7 }

        dndx = {}

        j = 0
        for i, c in enumerate(channel):
            dndx[c] = np.zeros(shape=(250,18))
            dndx[c][:,offset[i]:18] = d[:,j:j+18-offset[i]]
#            self._dndx[c] = d[:,j:j+18-offset[i]]
            j+= 18-offset[i]

        self._dndx = dndx[chan]

#        for c, i in channel_index.iteritems():
#            pass
#            print c, i

#            self._dndx[c] = d[:,i*18:(i+1)*18]

    def e2dnde(self,loge,mass=None,dloge=None):
        """
        Evaluate the spectral energy density.

        @param m:
        @param loge:
        @param dloge:
        @return:
        """
        e = np.power(10,loge)
        return e**2*self.dnde(loge,mass,dloge)
 
    def ednde(self,loge,mass=None,dloge=None):        
        e = np.power(10,loge)
        return e*self.dnde(loge,mass,dloge)

    def dnde(self,loge,mass=None,dloge=None):
        
        loge = np.array(loge,ndmin=1)

        if mass is None: m = self._mass
        else: m = mass

        e = np.power(10,loge)
        x = (np.log10(e/m)+self._ndec)/self._ndec
        dndx = self.dndx(m,x)
        dndx[e>=m] = 0
        return dndx*0.434/(self._ndec*(e))*250.

    def dndx(self,m,x):
        return interpolate2d(self._x,self._mass_bins,self._dndx,x,m/Units.gev)


class DMModelSpectrum(object):
    """Class that computes the differential annihilation yield for
    a specific DM model using a pretabulated spectrum.  """
    def __init__(self,egy,dnde):

        self._loge = np.log10(egy)
        self._dnde = 8.*np.pi*dnde*1E-29*np.power(Units.gev,-2)

    @staticmethod
    def create(spfile):
        d = np.loadtxt(spfile,unpack=True)
        return DMModelSpectrum(d[0],d[1])
#        self._loge = np.log10(d[0])
#        self._dnde = 8.*np.pi*d[1]*1E-29*np.power(Units.gev,-2)

    def e2dnde(self,mass,loge,dloge=None):        
        e = np.power(10,loge)
        return e**2*self.dnde(mass,loge,dloge)
 
    def ednde(self,mass,loge,dloge=None):        
        e = np.power(10,loge)
        return e*self.dnde(mass,loge,dloge)

    def dnde(self,mass,loge,dloge=None):
        """Return the differential gamma-ray rate per annihilation or
        decay."""

        loge = np.array(loge,ndmin=1)

        if dloge is not None:
            loge_edge = np.linspace(loge[0]-dloge/2.,loge[-1]+dloge/2.,
                                    len(loge)*10+1)
            logep = 0.5*(loge_edge[1:] + loge_edge[:-1])
        else:
            logep = loge

        e = np.power(10,logep)
        dnde = interpolate(self._loge,self._dnde,logep)
#        dnde[e>=m] = 0

        if dloge is None:
            return dnde
        else:
            z = np.zeros(len(loge))
            for i in range(len(z)):
                z[i] = np.sum(dnde[i*10:(i+1)*10]*e[i*10:(i+1)*10])
            z *= np.log(10.)/(np.power(10,loge+dloge/2.) - 
                              np.power(10,loge-dloge/2.))*(logep[1]-logep[0])

            return z

class DMModel(object):
    def __init__(self, sp, jp, mass = 100.0*Units.gev, sigmav = 1.0):

        self._mass = mass
        self._sigmav = sigmav
        self._sp = sp
        self._jp = jp

    @staticmethod
    def createChanModel(jp,mass,sigmav,chan):
        """Create a model with a 100% BR to a single
        annihilation/decay channel."""
        sp = DMChanSpectrum(chan)
        return DMModel(sp,jp,mass,sigmav)

    @staticmethod
    def createModel(jp,d):
        sp = DMModelSpectrum(d[0],d[1])
        return DMModel(sp,jp,1.0,1.0)

    @staticmethod
    def createModelFromFile(jp,spfile):
        sp = DMModelSpectrum.create(spfile)
        return DMModel(sp,jp,1.0,1.0)

    def e2flux(self,loge,psi):
        return np.power(10,2*loge)*self.flux(loge,psi)

    def eflux(self,loge,psi):
        return np.power(10,loge)*self.flux(loge,psi)

    def flux(self,loge,psi):
        djdomega = self._jp(loge,psi)
        flux = 1./(8.*np.pi)*self._sigmav*np.power(self._mass,-2)* \
            self._sp.dnde(loge,self._mass)

        return flux*djdomega

class DMLimitCalc(object):

    def __init__(self,irf,redge=0.0):

        self._irf = irf
#        self._th68 = self._irf._psf.counts()        
#        self._bkg_rate = (self._det.proton_wcounts_density + 
#                          self._det.electron_wcounts_density)/(50.0*Units.hr*
#                                                               Units.deg2)

        self._loge_axis = Axis.create(np.log10(Units.gev)+1.4,
                                      np.log10(Units.gev)+3.2,18)
        self._loge = self._loge_axis.center()
        
        self._dloge = self._loge_axis.width()
        self._aeff = self._irf.aeff(self._loge - Units.log10_mev)
        self._bkg_rate = irf.bkg(self._loge - Units.log10_mev)*self._dloge


        self._redge = [np.radians(redge),np.radians(0.99)]

        self._psi_axis = Axis(np.linspace(np.radians(0.0),np.radians(1.0),101))
        self._psi = self._psi_axis.center()
        self._domega = np.pi*(np.power(self._psi_axis.edges()[1:],2)-
                              np.power(self._psi_axis.edges()[:-1],2))

        self._iedge = [np.where(self._psi >= self._redge[0])[0][0],
                       np.where(self._psi >= self._redge[1])[0][0]]
               
    def counts(self,model,tau):
        """Compute the signal counts distribution as a function of
        energy and source offset."""

        x, y = np.meshgrid(self._loge,self._psi,indexing='ij')
        eflux = model.eflux(np.ravel(x),np.ravel(y))
        eflux = eflux.reshape((len(self._loge),len(self._psi)))

        exp = (self._dloge*np.log(10.)*self._aeff*tau)
        counts = eflux*exp[:,np.newaxis]*self._domega
        counts = counts[:,self._iedge[0]:]

        return counts

    def bkg_counts(self,tau):
        counts = np.zeros(shape=(len(self._loge),len(self._psi)))
        counts = self._domega[np.newaxis,:]*self._bkg_rate[:,np.newaxis]*tau
        counts = counts[:,self._iedge[0]:]
        return counts

    def lnl(self,sc,bc,alpha):
        
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
        
    def plot_lnl(self,dmmodel,mchi,sigmav,tau):

        prefix = '%04.f'%(np.log10(mchi/Units.gev)*100)
        
        bc = self.bkg_counts(tau)

        dmmodel._mass = mchi
        dmmodel._sigmav = sigmav
        sc = self.counts(dmmodel,tau)
        sc_cum = np.cumsum(sc,axis=1)        
        sc_ncum = (sc_cum.T/np.sum(sc,axis=1)).T


        dlnl = self.lnl(sc,bc,0.2)
        dlnl_cum = np.cumsum(dlnl,axis=1)

        
        ipeak = np.argmax(np.sum(dlnl,axis=1))
        i68 = np.where(sc_ncum[ipeak] > 0.68)[0][0]


        fig = plt.figure(figsize=(10,6))

        ax = fig.add_subplot(1,2,1)

        sc_density = sc/self._domega[self._iedge[0]:]

        def psf(x,s):
            return np.exp(-x**2/(2*s**2))

        plt.plot(np.degrees(self._psi),sc_density[ipeak]*(np.pi/180.)**2,
                 label='Signal')
        plt.plot(np.degrees(self._psi),
                 psf(np.degrees(self._psi),self._th68[ipeak]/1.50959)
                 *sc_density[ipeak][0]*(np.pi/180.)**2,label='PSF')

        plt.gca().axvline(self._th68[ipeak],color='k',linestyle='--',
                          label='PSF 68% Containment Radius')

        plt.grid(True)

        plt.gca().legend(prop={'size':10})

        plt.gca().set_ylabel('Counts Density [deg$^{-2}$]')
        plt.gca().set_xlabel('Offset [deg]')
        plt.gca().set_xlim(0.0,1.0)
        ax = fig.add_subplot(1,2,2)

        plt.plot(np.degrees(self._psi),dlnl[ipeak])
        
        plt.grid(True)
        plt.gca().set_xlabel('Offset [deg]')
        plt.gca().set_ylabel('TS')

        plt.gca().axvline(self._th68[ipeak],color='k',linestyle='--',
                          label='PSF 68% Containment Radius')

        plt.gca().axvline(np.degrees(self._psi[i68]),color='r',linestyle='--',
                          label='Signal 68% Containment Radius')

        plt.gca().set_xlim(0.0,0.5)
        plt.gca().legend(prop={'size':10})
        plt.savefig('%s_density.png'%(prefix))

#        dlnl_cum = dlnl_cum.reshape(dlnl.shape)

        def make_plot(z,figname):
            plt.figure()
            im = plt.imshow(z.T,interpolation='nearest', origin='lower',
                            aspect='auto',
                            extent=[self._emin[0],self._emax[-1],
                                    np.degrees(self._psi_edge[0]),
                                    np.degrees(self._psi_edge[-1])])

            plt.gca().set_xlabel('Energy [log$_{10}$(E/GeV)]')
            plt.gca().set_ylabel('Offset [deg]')
            plt.colorbar(im,ax=plt.gca(),orientation='horizontal',
                         fraction=0.05,shrink=0.7,
                         pad=0.1)

#            figname = '%04.f_lnl.png'%(np.log10(mchi/Units.gev)*100)
        
            plt.gca().grid(True)
            plt.gca().set_ylim(0.0,0.5)

            plt.savefig(figname)

        make_plot(dlnl,'%s_lnl.png'%(prefix))
        make_plot(dlnl_cum,'%s_lnl_cum.png'%(prefix))
        make_plot(sc,'%s_scounts.png'%(prefix))
        make_plot(sc_ncum,'%s_scounts_cum.png'%(prefix))

#        plt.show()

#        self._loge),len(self._psi)


    def significance(self,sc,bc,alpha):

        dlnl = self.lnl(sc,bc,alpha)
        if dlnl.ndim == 2:
            return np.sqrt(max(0,np.sum(dlnl)))
        else:
            s0 = np.sum(dlnl,axis=1)
            s1 = np.sum(s0,axis=1)
            s1[s1<=0]=0
            return np.sqrt(s1)
            

    def boost(self,model,tau,sthresh=5.0,alpha=0.2):

        b = np.linspace(-4,6,60)

        bc = self.bkg_counts(tau)
        sc = self.counts(model,tau)

        sc2 = outer(sc,np.power(10,b))

        s = self.significance(sc2.T,bc.T,alpha)
#        i = np.where(s>sthresh)[0][0]

        if s[-1] <= sthresh: return b[-1]

        fn = UnivariateSpline(b,s,k=2,s=0)
        b0 = brentq(lambda t: fn(t) - sthresh ,b[0],b[-1])
        return b0


    def limit(self,dmmodel,mchi,tau,sthresh=5.0,alpha=0.2):

        ul = []
        bc = self.bkg_counts(tau)

        for i, m in enumerate(mchi):

            dmmodel._mass = m
            dmmodel._sigmav = np.power(10.,-28.)

            sc = self.counts(dmmodel,tau)
            t = np.linspace(np.log10(10./np.sum(sc)),
                            np.log10(10./np.sum(sc))+4,20)
            sc2 = outer(sc,np.power(10,t))
            s = self.significance(sc2.T,bc.T,alpha)
            fn = UnivariateSpline(t,s,k=2,s=0)

            t0 = brentq(lambda t: fn(t) - sthresh ,t[0],t[-1])
            ul.append(np.power(10,-28.+t0))

        return ul
            


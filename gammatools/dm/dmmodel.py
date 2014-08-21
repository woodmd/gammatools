#!/usr/bin/env python

import os
import sys
import re
import yaml

import numpy as np
import matplotlib.pyplot as plt
import gammatools
import gammatools.dm.jcalc as jcalc
from gammatools.core.util import Units
import scipy.special as spfn
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline
from gammatools.core.util import *
from gammatools.core.histogram import *
from gammatools.core.stats import *

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

class ConvolvedHaloModel(object):
    """Class that stores the J density profile convolved with an
    energy-dependent PSF."""
    def __init__(self,hm,irf):

        self._irf = irf

        self._loge_axis = self._irf._psf.axis()
        th68 = self._irf._psf.counts

        self._psi_axis = Axis(np.linspace(0,np.radians(10.0),401))
        self._psi = self._psi_axis.center
        self._loge = self._loge_axis.center+Units.log10_mev

        self._z = np.zeros(shape=(len(self._loge),len(self._psi)))

        self._h = Histogram2D(self._loge_axis,self._psi_axis)

        for i in range(len(self._loge)):
            self._z[i] = convolve2d_gauss(hm._jp,self._psi,
                                          np.radians(th68[i]/1.50959),
                                          self._psi[-1]+np.radians(0.5),
                                          nstep=1000)
        self._h._counts = self._z

    def jval(self,loge,psi):

        return interpolate2d(self._loge,self._psi,self._z,
                             loge,psi)

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
    def __init__(self,chan,mass = 100*Units.gev, yield_table=None):

        if yield_table is None:
            yield_table = os.path.join(gammatools.PACKAGE_ROOT,
                                       'data/gammamc_dif.dat')

        d = np.loadtxt(yield_table,unpack=True)

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

    def e2dnde(self,loge,mass=None):
        """
        Evaluate the spectral energy density.

        @param m:
        @param loge:
        @return:
        """
        e = np.power(10,loge)
        return e**2*self.dnde(loge,mass)
 
    def ednde(self,loge,mass=None):        
        e = np.power(10,loge)
        return e*self.dnde(loge,mass)

    def dnde(self,loge,mass=None):
        
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

        self._loge = np.log10(egy) + np.log10(Units.gev)
        self._dnde = 8.*np.pi*dnde*1E-29*np.power(Units.gev,-3)

    @staticmethod
    def create(spfile):
        d = np.loadtxt(spfile,unpack=True)
        return DMModelSpectrum(d[0],d[1])
#        self._loge = np.log10(d[0])
#        self._dnde = 8.*np.pi*d[1]*1E-29*np.power(Units.gev,-2)

    def e2dnde(self,loge,mass=None):        
        e = np.power(10,loge)
        return e**2*self.dnde(loge,mass)
 
    def ednde(self,loge,mass=None):        
        e = np.power(10,loge)
        return e*self.dnde(loge,mass)

    def dnde(self,loge,mass=None):
        """Return the differential gamma-ray rate per annihilation or
        decay."""

        loge = np.array(loge,ndmin=1)
        dnde = interpolate(self._loge,self._dnde,loge)
        return dnde

class DMFluxModel(object):
    def __init__(self, sp, hm, mass = 1.0, sigmav = 1.0):

        self._mass = mass
        self._sigmav = sigmav
        self._sp = sp
        self._hm = hm

    @staticmethod
    def createChanModel(hm,mass,sigmav=3E-26*Units.cm3_s,chan='bb'):
        """Create a model with a 100% BR to a single
        annihilation/decay channel."""
        sp = DMChanSpectrum(chan)
        return DMFluxModel(sp,hm,mass,sigmav)

    @staticmethod
    def createModel(hm,d):
        sp = DMFluxModelSpectrum(d[0],d[1])
        return DMFluxModel(sp,jp,1.0,1.0)

    @staticmethod
    def createModelFromFile(hm,spfile):
        sp = DMFluxModelSpectrum.create(spfile)
        return DMFluxModel(sp,hm,1.0,1.0)

    def e2flux(self,loge,psi):
        return np.power(10,2*loge)*self.flux(loge,psi)

    def eflux(self,loge,psi):
        return np.power(10,loge)*self.flux(loge,psi)

    def flux(self,loge,psi):
        djdomega = self._hm.jval(loge,psi)
        flux = 1./(8.*np.pi)*self._sigmav*np.power(self._mass,-2)* \
            self._sp.dnde(loge,self._mass)

        return flux*djdomega

class DMLimitCalc(object):

    def __init__(self,irf,alpha,min_fsig=0.0,redge='0.0/1.0'):

        self._irf = irf
#        self._th68 = self._irf._psf.counts        
#        self._bkg_rate = (self._det.proton_wcounts_density + 
#                          self._det.electron_wcounts_density)/(50.0*Units.hr*
#                                                               Units.deg2)

        self._loge_axis = Axis.create(np.log10(Units.gev)+1.4,
                                      np.log10(Units.gev)+3.6,22)
        self._loge = self._loge_axis.center        
        self._dloge = self._loge_axis.width

        rmin, rmax = [float(t) for t in redge.split('/')]

        self._psi_axis = Axis(np.linspace(np.radians(0.0),np.radians(1.0),101))
        self._psi = self._psi_axis.center
        self._domega = np.pi*(np.power(self._psi_axis.edges()[1:],2)-
                              np.power(self._psi_axis.edges()[:-1],2))

        self._aeff = self._irf.aeff(self._loge - Units.log10_mev)
        self._bkg_rate = irf.bkg(self._loge - Units.log10_mev)*self._dloge


        self._redge = [np.radians(rmin),np.radians(rmax)]
        self._msk = (self._psi > self._redge[0]) & (self._psi < self._redge[1])
        

        self._domega_sig = np.sum(self._domega[self._msk])
        self._domega_bkg = self._domega_sig/alpha


        self._min_fsig = min_fsig
        self._alpha = alpha
        self._alpha_bin = self._domega/self._domega_bkg

#        self._iedge = [np.where(self._psi >= self._redge[0])[0][0],
#                       np.where(self._psi >= self._redge[1])[0][0]]


               
    def counts(self,model,tau):
        """Compute the signal counts distribution as a function of
        energy and source offset."""

        x, y = np.meshgrid(self._loge,self._psi,indexing='ij')
        eflux = model.eflux(np.ravel(x),np.ravel(y))
        eflux = eflux.reshape((len(self._loge),len(self._psi)))

        exp = (self._dloge*np.log(10.)*self._aeff*tau)
        counts = eflux*exp[:,np.newaxis]*self._domega
        return counts

    def bkg_counts(self,tau):
        counts = np.zeros(shape=(len(self._loge),len(self._psi)))
        counts = self._domega[np.newaxis,:]*self._bkg_rate[:,np.newaxis]*tau
        return counts
        
    def plot_lnl(self,dmmodel,mchi,sigmav,tau):

        prefix = '%04.f'%(np.log10(mchi/Units.gev)*100)
        
        bc = self.bkg_counts(tau)

        dmmodel._mass = mchi
        dmmodel._sigmav = sigmav
        sc = self.counts(dmmodel,tau)
        sc_cum = np.cumsum(sc,axis=1)        
        sc_ncum = (sc_cum.T/np.sum(sc,axis=1)).T


        dlnl = poisson_lnl(sc,bc,0.2)
        dlnl_cum = np.cumsum(dlnl,axis=1)

        
        ipeak = np.argmax(np.sum(dlnl,axis=1))
        i68 = np.where(sc_ncum[ipeak] > 0.68)[0][0]


        fig = plt.figure(figsize=(10,6))

        ax = fig.add_subplot(1,2,1)

        sc_density = sc/self._domega[self._msk]

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


    def significance(self,sc,bc,alpha=None):

        if alpha is None: alpha = self._alpha_bin[:,np.newaxis]

        dlnl = poisson_median_ts(sc,bc,alpha)[:,self._msk,...]
        if dlnl.ndim == 2:
            return np.sqrt(max(0,np.sum(dlnl)))
        else:
            s0 = np.sum(dlnl,axis=0)
            s1 = np.sum(s0,axis=0)
            s1[s1<=0]=0
            return np.sqrt(s1)
            

    def boost(self,model,tau,sthresh=5.0):

        b = np.linspace(-4,6,60)

        bc = self.bkg_counts(tau)
        sc = self.counts(model,tau)

        sc2 = outer(sc,np.power(10,b))

        s = self.significance(sc2.T,bc.T)
#        i = np.where(s>sthresh)[0][0]

        if s[-1] <= sthresh: return b[-1]

        fn = UnivariateSpline(b,s,k=2,s=0)
        b0 = brentq(lambda t: fn(t) - sthresh ,b[0],b[-1])
        return b0


    def limit(self,dmmodel,mchi,tau,sthresh=5.0):

        o = {}
        bc = self.bkg_counts(tau)

        dmmodel._mass = mchi
        dmmodel._sigmav = np.power(10.,-28.)

        sc = self.counts(dmmodel,tau)
        
        sc_msk = copy.copy(sc); sc_msk[:,~self._msk] = 0
        bc_msk = copy.copy(bc); bc_msk[:,~self._msk] = 0

        onoff = OnOffExperiment(np.ravel(sc_msk),
                                np.ravel(bc_msk),
                                self._alpha)

        t0, t0_err = onoff.asimov_mu_ts0(sthresh**2)

        sc0 = sc*t0
        ts = onoff.asimov_ts0_signal(t0,sum_lnl=False)
        ts = ts.reshape(sc.shape)

        iarg = np.argsort(np.ravel(ts))[::-1]
        ts_sort = ts.flat[iarg]

        i0 = int(percentile(np.linspace(0,len(iarg),len(iarg)),
                            np.cumsum(ts_sort)/np.sum(ts),0.5))

        msk = np.empty(shape=ts_sort.shape,dtype=bool); msk.fill(False)
        msk[iarg[:i0]] = True
        msk = msk.reshape(ts.shape)
            
        scounts = np.sum(sc0[msk])
        bcounts = np.sum(bc[msk])
        sfrac = np.sum(sc0[msk])/np.sum(bc[msk])

        if sfrac < self._min_fsig:
            t0 *= self._min_fsig/sfrac
            ts = onoff.asimov_ts0_signal(t0,sum_lnl=False)
            ts = ts.reshape(sc.shape)

        print t0, sfrac, self._min_fsig, np.sum(ts[msk])

        axis0 = Axis(self._loge_axis.edges()-np.log10(Units.gev))
        axis1 = Axis(np.degrees(self._psi_axis.edges()))

        o['sigmav_ul'] = np.power(10,-28.)*t0
        o['sc_hist'] = Histogram2D(axis0,axis1,counts=sc*t0)
        o['bc_hist'] = Histogram2D(axis0,axis1,counts=bc)
        o['ts_hist'] = Histogram2D(axis0,axis1,counts=ts)
        o['msk_sfrac'] = msk
        o['scounts_msk'] = scounts
        o['bcounts_msk'] = bcounts

        return o

        axis0 = Axis(self._loge_axis.edges()-np.log10(Units.gev))
        axis1 = Axis(np.degrees(self._psi_axis.edges()))

        dlnl_hist = Histogram2D(axis0,axis1)
        dlnl_hist._counts = copy.copy(dlnl)

        dlnl_hist_msk = copy.deepcopy(dlnl_hist)
            
        dlnl_hist_msk._counts[~msk]=0

        plt.figure()
        plt.gca().set_title('M = %.2f GeV'%(m/Units.gev))
        im = dlnl_hist.plot()

        plt.colorbar(im,label='TS')

        plt.gca().set_xlabel('Energy [log$_{10}$(E/GeV)]')
        plt.gca().set_ylabel('Offset [deg]')

        plt.figure()
        plt.gca().set_title('M = %.2f GeV'%(m/Units.gev))
        im = dlnl_hist_msk.plot()
            
        plt.colorbar(im,label='TS')

        plt.gca().set_xlabel('Energy [log$_{10}$(E/GeV)]')
        plt.gca().set_ylabel('Offset [deg]')

        plt.show()

        sh2 = Histogram2D(axis,self._psi_axis)
#            sh2._counts[:,self._msk][msk] = 1.0
        sh2._counts[msk] = 1.0
        plt.figure()

        sh2.plot()

        plt.show()

        sh = Histogram2D(axis,self._psi_axis)
        sh._counts[:,self._msk] = sc0

        bh = Histogram2D(axis,self._psi_axis)
        bh._counts[:,self._msk] = bc

        plt.figure()
        sh.project(0).plot()
        bh.project(0).plot()

        plt.gca().set_yscale('log')

        shp = sh.project(0)
        bhp = bh.project(0)

        plt.figure()
        (shp/bhp).plot()
        
        plt.figure()
        im = (sh/bh).plot(zlabel='test')
        plt.colorbar(im)
        
        plt.show()
            


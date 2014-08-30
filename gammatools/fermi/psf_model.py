import os
import sys
import re
import bisect
import pyfits
import healpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from gammatools.core.histogram import Histogram
from gammatools.fermi.irf_util import *
from gammatools.fermi.catalog import Catalog

def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return array[idx]

class PSFModel(object):
    def __init__(self,psf_file=None,model='powerlaw',sp_param=(2,1000)):

        self._dtheta_max_deg = 90.0
        
        if psf_file is not None:
            self.load_file(psf_file)

        self.set_spectrum(model=model,sp_param=sp_param)

    def set_spectrum(self,model,sp_param):
        self._sp_param = sp_param

        if model == 'powerlaw':
            self._wfn = self.powerlaw
        elif model == 'deltafn':
            self._wfn = self.deltafn
            self._deltae = find_nearest(self._energy,self._sp_param[0])
        else:
            self._wfn = self.powerlaw_exp

    def load_file(self,file):
        hdulist = pyfits.open(file)
        self._dtheta = np.array(hdulist[2].data.field(0))
        self._energy = np.array(hdulist[1].data.field(0))
        self._exps = np.array(hdulist[1].data.field(1))
        self._psf = np.array(hdulist[1].data.field(2))

    def deltafn(self,e):

        if np.fabs(e-self._deltae) < 0.01:
            return 1
        else:
            return 0

    def powerlaw(self,e):

        return e**(-self._sp_param[0])

    def powerlaw_exp(self,e):

        return e**(-self._sp_param[0])*np.exp(-e/self._sp_param[1])        

    def histogram(self,emin,emax,cthmin,cthmax,edges):
        y = self.thetasq(emin,emax,cthmin,cthmax,edges)
        return Histogram(edges,counts=y)

    def pdf(self,emin,emax,cthmin,cthmax,theta):

        x,y = self.psf(emin,emax,cthmin,cthmax)
        
        f = UnivariateSpline(x,y,s=0)
        return f(theta)
    
    def thetasq(self,emin,emax,cthmin,cthmax,x_theta):

        x,y = self.psf(emin,emax,cthmin,cthmax)

        f = UnivariateSpline(x,y,s=0)

        y_thsq = []

        for i in range(len(x_theta)-1):

            theta_lo = max(0,x_theta[i])
            theta_hi = x_theta[i+1]

            s = quad(lambda t: 2*np.pi*t*f(t),theta_lo,theta_hi)[0]
            s *= (np.pi/180.)**2
            y_thsq.append(s)


        return np.array(y_thsq)

    def quantile(self,emin,emax,cthmin,cthmax,frac=0.68,xmax=None):
        
        radii = np.logspace(-3.0,np.log10(self._dtheta_max_deg),300)
        radii = np.concatenate(([0],radii))

        x,y = self.psf(emin,emax,cthmin,cthmax)

        f = UnivariateSpline(x,y,s=0)


        rcenters = 0.5*(radii[:-1]+radii[1:])
        rwidth = np.radians(radii[1:] - radii[:-1])
        
        cdf = 2*np.pi*np.sin(np.radians(rcenters))*f(rcenters)*rwidth
        cdf = np.cumsum(cdf)
        cdf = np.concatenate(([0],cdf))
        stot = cdf[-1]

        if not xmax is None:
            fcdf = UnivariateSpline(radii,cdf,s=0)        
            stot = fcdf(xmax)

        cdf /= stot

        indx = bisect.bisect(cdf, frac) - 1
        return ((frac - cdf[indx])/(cdf[indx+1] - cdf[indx])
                *(radii[indx+1] - radii[indx]) + radii[indx])


    def psf(self,emin,emax,cthmin,cthmax):
        """Return energy- and livetime-weighted PSF density vector as
        a function of angular offset for a bin in energy and
        inclination angle."""
        
        logemin = np.log10(emin)
        logemax = np.log10(emax)

        ilo = np.argwhere(self._energy > emin)[0,0]
        ihi = np.argwhere(self._energy < emax)[-1,0]+1
        
        jlo = np.argwhere(self._ctheta_axis.center > cthmin)[0,0]
        jhi = np.argwhere(self._ctheta_axis.center < cthmax)[-1,0] +1
        
        weights = (self._energy[ilo:ihi,np.newaxis]*
                   self._exp[ilo:ihi,jlo:jhi]*
                   self._wfn(self._energy[ilo:ihi,np.newaxis]))
        
        wsum = np.sum(weights)
        psf = np.apply_over_axes(np.sum,
                                 self._psf[:,ilo:ihi,jlo:jhi]*
                                 weights[np.newaxis,...],
                                 [1,2])
        psf = np.squeeze(psf)                   
        psf *= (1./wsum)
        return self._dtheta, psf
    
    def psf2(self,emin,emax):
        """Return energy-weighted PSF density vector as a function of
        angular offset for the given energy bin."""

        idx = ((self._energy - emin) >= 0) & ((self._energy - emax) <= 0)
        weights = \
            self._energy[idx]*self._exps[idx]*self._wfn(self._energy[idx])

        if self._edisp is not None:
        
            idx = ((self._erec_center - np.log10(emin)) >= 0) & \
                ((self._erec_center - np.log10(emax)) <= 0)

            weights = (self._energy*self._exps* 
                       self._wfn(self._energy)*self._edisp.T).T
            
            # PSF vs. reconstructed energy
            psf = np.zeros(self._dtheta.shape[0])
            wsum = 0

            for i in range(len(self._erec_center)):
                if self._erec_center[i] < np.log10(emin): continue
                elif self._erec_center[i] > np.log10(emax): break
                psf += np.sum(self._psf.T*weights[:,i],axis=1)                
                wsum += np.sum(weights[:,i])
                
            psf *= 1./wsum
            return self._dtheta, psf
        
        else:
            idx = ((self._energy - emin) >= 0) & ((self._energy - emax) <= 0)
            weights = \
                self._energy[idx]*self._exps[idx]*self._wfn(self._energy[idx])
            wsum = np.sum(weights)
            psf = np.sum(self._psf[idx].T*weights,axis=1)
            psf *= (1./wsum)
            return self._dtheta, psf

class PSFModelLT(PSFModel):
    
    def __init__(self,irf,nbin=600,
                 ebins_per_decade=16,
                 src_type='iso',
                 spectrum='powerlaw',
                 spectrum_pars=[2.0],
                 build_model=True,
                 ltfile=None,
                 edisp_table=None):

        PSFModel.__init__(self,model=spectrum,sp_param=spectrum_pars)

        self._src_type = src_type
        self._nbin_dtheta = nbin
        self._irf = irf
        self._edisp_table = edisp_table

        self._lonlat = (0, 0)
        if src_type != 'iso' and src_type != 'isodec':
            cat = Catalog.get()
            src = cat.get_source_by_name(src_type)
            self._lonlat = (src['RAJ2000'], src['DEJ2000'])
        
        loge_step = 1./float(ebins_per_decade)
        emin = 1.0+loge_step/2.
        emax = 6.0-loge_step/2.
        nbin = int((emax-emin)/loge_step)+1
        self._loge_axis = Axis.create(emin,emax,nbin)
        
        self._energy = np.power(10,self._loge_axis.center)
        self._exps = np.zeros(self._loge_axis.nbins)
        
        self._psf = np.zeros((self._loge_axis.nbins,self._nbin_dtheta))

        self._dtheta = np.array([self._dtheta_max_deg*
                                 (float(i)/float(self._nbin_dtheta))**2 
                                 for i in range(self._nbin_dtheta)])


        self._dtheta_axis = Axis(self._dtheta)
        self._ctheta_axis = Axis.create(0.2,1.0,40)
        self._tau = np.zeros(self._ctheta_axis.nbins)
        
        self.loadLTCube(ltfile)
        self.fillLivetime()

        if build_model: self.buildModel()

    def buildModel(self):
        """Build a model for the exposure-weighted PSF averaged over
        instrument inclination angle."""

        gx, gy = np.meshgrid(np.log10(self._energy),
                             self._ctheta_axis.center) 

        gx = gx.T
        gy = gy.T
        
        self._psf = np.zeros((self._loge_axis.nbins,self._nbin_dtheta))
        self._edisp = None

        shape = (self._loge_axis.nbins, self._ctheta_axis.nbins)
        
        aeff = self._irf.aeff(np.ravel(gx),np.ravel(gy))
        aeff = aeff.reshape(shape)
        aeff[aeff < 0] = 0
        aeff[np.isnan(aeff)] = 0

        dtheta = self._dtheta.reshape(self._dtheta.shape + (1,))
        
        self._exp = self._tau*aeff
        self._exps = np.sum(self._exp,axis=1)
        
        psf = self._irf.psf(self._dtheta[...,np.newaxis],
                            np.ravel(gx)[np.newaxis,...],
                            np.ravel(gy)[np.newaxis,...])
        
        psf[psf<0] = 0
        psf[np.isnan(psf)] = 0

        self._psf = psf.reshape((self._nbin_dtheta,) + shape)
#        psf /= self._exps[np.newaxis,:,np.newaxis]
#        psf = np.sum(psf*self._exp,axis=2).T
#        self._psf = psf
        
        if self._edisp_table is not None:
        
            edisp_data = np.load(self._edisp_table)
            log_egy_edges = edisp_data['log_egy_edges']
            self._erec_edges = edisp_data['log_erec_edges']
            log_egy_center = 0.5*(log_egy_edges[:-1]+log_egy_edges[1:])
            self._erec_center = 0.5*(self._erec_edges[:-1]+
                                     self._erec_edges[1:]) 
            costh_edges = edisp_data['costh_edges']

            self._edisp = np.zeros((self._log_energy.shape[0],
                                    self._erec_center.shape[0]))
            self._psf_edisp = np.zeros((self._log_energy.shape[0],
                                        self._erec_center.shape[0],
                                        self._nbin_dtheta))

            self._edisp = np.sum(edisp_data['edisp'].T*self._tau*aeff,axis=2).T
            self._edisp = (self._edisp.T/self._exps).T

            
            self._edisp[np.isnan(self._edisp)] = 0
            self._edisp[np.isinf(self._edisp)] = 0
            
        return

                
    def loadLTCube(self,ltcube_file):

        if ltcube_file is None: return
        hdulist = pyfits.open(ltcube_file)

        self._ltmap = hdulist[1].data.field(0)[:,::-1]
        
        ctheta = np.array(hdulist[3].data.field(0))
        self._ctheta_axis = Axis(np.concatenate(([1],ctheta))[::-1])
        
#        self._ctheta_center = \
#            np.array([1-(0.5*(np.sqrt(1-self._ctheta[i]) + 
#                              np.sqrt(1-self._ctheta[i+1])))**2 
#                      for i in range(len(self._ctheta)-1)])
#        self._dcostheta = np.array([self._ctheta[i]-self._ctheta[i+1]
#                                    for i in range(len(self._ctheta)-1)])
#        self._theta_center = np.arccos(self._ctheta_center)*180/np.pi  
        
        self._tau = np.zeros(self._ctheta_axis.nbins)

    def fillLivetime(self):
        
        for i, cth in enumerate(self._ctheta_axis.center):

            dcostheta = self._ctheta_axis.width[i]
            
            if self._src_type == 'iso':
                self._tau[i] = dcostheta
            elif self._src_type == 'isodec':
                sinlat = np.linspace(-1,1,48)

                m = self._ltmap[:,i]

                self._tau[i] = 0
                for s in sinlat:                    
                    lat = np.arcsin(s)
                    th = np.pi/2. - lat                    
                    ipix = healpy.ang2pix(64,th,0,nest=True)
                    self._tau[i] += m[ipix]
                                       
            else:
                th = np.pi/2. - self._lonlat[1]*np.pi/180.
                phi = self._lonlat[0]*np.pi/180.
                m = self._ltmap[:,i]
                ipix = healpy.ang2pix(64,th,phi,nest=True)
#            tau = healpy.get_interp_val(m,th,phi,nest=True)
                self._tau[i] = m[ipix]


if __name__ == '__main__':

    from optparse import Option
    from optparse import OptionParser

    usage = "usage: %prog [options]"
    description = ""
    parser = OptionParser(usage=usage,description=description)

    parser.add_option('--ltfile', default = '', type='string',
                      help = 'LT file')

    parser.add_option('--irf', default = 'P6_V3_DIFFUSE', type='string',
                      help = 'LT file')

    (opts, args) = parser.parse_args()

    SourceCatalog = { 'vela' : (128.83606354, -45.17643181),
                      'geminga' : (98.475638, 17.770253),
                      'crab' : (83.63313, 22.01447) }

    logemin = 3
    logemax = 3.25

    emin = np.power(10,logemin)
    emax = np.power(10,logemax)

    ctheta_range=(0.4,1.0)

    irf = IRFManager('../custom_irfs/psf_P7SOURCE_V6MC_front.fits',
                     '../custom_irfs/aeff_P7SOURCE_V6MC_front.fits')

    np.seterr(all='raise')
    
    m = PSFModelLT(opts.ltfile, opts.irf,
                   nbin=300,
                   ctheta_range=ctheta_range,
                   src_type='src',
                   lonlat=SourceCatalog['vela'])#,irf=irf)

    print '34% ', m.quantile(emin,emax,0.34)
    print '68% ', m.quantile(emin,emax,0.68)
    print '85% ', m.quantile(emin,emax,0.85)
    print '95% ', m.quantile(emin,emax,0.95)


    sys.exit(1)

    psf_model1 = PSFModel(sys.argv[1],'powerlaw_exp',1.607,3508.6)
    psf_model2 = PSFModel(sys.argv[2],'powerlaw_exp',1.607,3508.6)
#    psf_model1 = PSFModel(sys.argv[1],'deltafn',60000)
#    psf_model2 = PSFModel(sys.argv[2],'deltafn',60000)

    print psf_model1.quantile(emin,emax,0.99)

#    psf_model3 = PSFModel(sys.argv[3],'powerlaw',2)
#    psf_model4 = PSFModel(sys.argv[1],'powerlaw_exp',1.607,3508.6)

#    x_thsq = np.linspace(0,10,100)             
#    fig = plt.figure()
#    y_thsq = psf_model1.thetasq(emin,emax,x_thsq)
#    plt.errorbar(x_thsq,y_thsq,xerr=0.5*(x_thsq[1]-x_thsq[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111)


    x1,y1 = psf_model1.psf(emin,emax)
    plt.plot(x1,y1,color='r')

    x2,y2 = psf_model2.psf(emin,emax)
    plt.plot(x2,y2,color='b')


    ax.set_xlim(0,1.0)



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')



    plt.plot(x1,y1/y2,label='gam=3')
 

    ax.legend()

    plt.show()

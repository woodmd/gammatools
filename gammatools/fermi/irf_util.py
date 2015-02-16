
import os
import copy
import re
from gammatools.core.astropy_helper import pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import bisect
from gammatools.core.histogram import *

def expand_irf(irf):
    
    irf_names = []

    if not (re.search('FRONT',irf) or re.search('BACK',irf)):
        irf_names.append(irf + '::FRONT')
        irf_names.append(irf + '::BACK')
    else:
        irf_names.append(irf)

    return irf_names


class IRFManager(object):

    load_irf = False

    defaults = {'irf_dir'         : 'custom_irfs',
                'expand_irf_name' : False,
                'load_from_file'  : True }
    
    def __init__(self,irfs=None):

        self._irfs = []
        if not irfs is None: self._irfs = irfs

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--irf_dir', default = None, 
                            help = 'Set the IRF directory.')

        parser.add_argument('--expand_irf_name', default = False,
                            action='store_true',
                            help = 'Set the IRF directory.')
        
    @staticmethod
    def create(irf_name,load_from_file=False,irf_dir=None):
        
        if load_from_file:  return IRFManager.createFromFile(irf_name,irf_dir)
        else: return IRFManager.createFromPyIRF(irf_name)

    @staticmethod
    def createFromFile(irf_name,irf_dir=None,expand_irf_name=True):

        print 'Create From File ', irf_name
        
        if expand_irf_name: irf_names = expand_irf(irf_name)
        else: irf_names = [irf_name]

        print irf_names
        
        irfset = IRFManager()
        
        for name in irf_names:     
            irf = IRF.createFromFile(name,irf_dir)
            irfset.add_irf(irf)
#        irf.loadPyIRF(irf_name)
        return irfset

    @staticmethod
    def createFromPyIRF(irf_name,expand_irf_name=True):

        if expand_irf_name: irf_names = expand_irf(irf_name)
        else: irf_names = [irf_name]
        
        irfset = IRFManager()

        for name in irf_names:

            print 'Creating ', name
            
            irf = IRF.createFromPyIRF(name)
            irfset.add_irf(irf)
        return irfset
        
        #        irf = IRFManager()
#        irf.loadPyIRF(irf_name)
#        return irf

    def add_irf(self,irf):
        self._irfs.append(irf)

    def fisheye(self,egy,cth,**kwargs):
        
        aeff_tot = self.aeff(egy,cth,**kwargs)
        
        v = None
        for i, irf in enumerate(self._irfs):

            aeff = irf.aeff(egy,cth,**kwargs)            
            psf = irf.fisheye(egy,cth,**kwargs)*aeff
            if i == 0: v  = psf
            else: v += psf

        v[aeff_tot>0] = (v/aeff_tot)[aeff_tot>0]
        return v
        
    def psf(self,dtheta,egy,cth,**kwargs):
        
        aeff_tot = self.aeff(egy,cth,**kwargs)
        
        v = None
        for i, irf in enumerate(self._irfs):

            aeff = irf.aeff(egy,cth,**kwargs)            
            psf = irf.psf(dtheta,egy,cth,**kwargs)*aeff
            if i == 0: v  = psf
            else: v += psf

        return v/aeff_tot

    def psf_quantile(self,egy,cth,frac=0.68):
        x = np.logspace(-3.0,np.log10(90.0),300)        
        x = np.concatenate(([0],x))
        xc = 0.5*(x[:-1]+x[1:])
        deltax = np.radians(x[1:] - x[:-1])
        
        xc = xc.reshape((1,300))
        deltax = deltax.reshape((1,300))
        egy = np.ravel(np.array(egy,ndmin=1))
        cth = np.ravel(np.array(cth,ndmin=1))
        egy = egy.reshape((egy.shape[0],) + (1,))
        cth = cth.reshape((cth.shape[0],) + (1,))

        y = np.zeros((egy.shape[0],300))
        for i in range(len(self._irfs)):
            y += self._irfs[i].psf(xc,egy,cth).reshape(y.shape)


        cdf = 2*np.pi*np.sin(np.radians(xc))*y*deltax
        cdf = np.cumsum(cdf,axis=1)
#        cdf = np.concatenate(([0],cdf))
#        cdf = np.vstack((np.zeros(cdf.shape[0]),cdf))
        cdf /= cdf[:,-1][:,np.newaxis]

        p = np.zeros(cdf.shape[0])
        for i in range(len(p)): 
            p[i] = percentile(x[1:],cdf[i],frac)
        return p
#        return percentile(x,cdf,frac)
    
    def aeff(self,*args,**kwargs):

        v = None
        for i in range(len(self._irfs)):
            if i == 0: v = self._irfs[i].aeff(*args,**kwargs)
            else: v += self._irfs[i].aeff(*args,**kwargs)

        return v

    def edisp(self,*args,**kwargs):

        v = None
        for i in range(len(self._irfs)):
            if i == 0: v = self._irfs[i].edisp(*args,**kwargs)
            else: v += self._irfs[i].edisp(*args,**kwargs)

        return v

    def dump(self):
        for irf in self._irfs: irf.dump()
    
    def save(self,irf_name):
        for irf in self._irfs: irf.save(irf_name)

class IRF(object):

    def __init__(self,psf,aeff,edisp):
        self._psf = psf
        self._aeff = aeff
        self._edisp = edisp

    @staticmethod
    def create(irf_name,load_from_file=False,irf_dir=None):
        
        if load_from_file:  return IRF.createFromFile(irf_name,irf_dir)
        else: return IRF.createFromIRF(irf_name)

    @staticmethod
    def createFromFile(irf_name,irf_dir=None):
        
        if irf_dir is None: irf_dir = 'custom_irfs'

        irf_name = irf_name.replace('::FRONT','_front')
        irf_name = irf_name.replace('::BACK','_back')

        psf_file = os.path.join(irf_dir,'psf_%s.fits'%(irf_name))
        aeff_file = os.path.join(irf_dir,'aeff_%s.fits'%(irf_name))
        edisp_file = os.path.join(irf_dir,'edisp_%s.fits'%(irf_name))

        psf = PSFIRF(psf_file)
        aeff = AeffIRF(aeff_file)
        edisp = EDispIRF(edisp_file)
        
        return IRF(psf,aeff,edisp)

    @staticmethod
    def createFromPyIRF(irf_name):
        """Create IRF object using pyIrf modules in Science Tools."""
        
        import pyIrfLoader
        if not IRFManager.load_irf:
            pyIrfLoader.Loader_go()
            IRFManager.load_irf = True

        irf_factory=pyIrfLoader.IrfsFactory.instance()
        irfs = irf_factory.create(irf_name)

        psf = PSFPyIRF(irfs.psf())
        aeff = AeffPyIRF(irfs.aeff())
        edisp = EDispPyIRF(irfs.edisp())

        return IRF(psf,aeff,edisp)
        
    def aeff(self,*args,**kwargs):
        return self._aeff(*args,**kwargs)

    def psf(self,*args,**kwargs):
        return self._psf(*args,**kwargs)

    def edisp(self,*args,**kwargs):
        return self._edisp(*args,**kwargs)

    def fisheye(self,*args,**kwargs):
        return self._psf.fisheye(*args,**kwargs)
    
    def dump(self):

        self._aeff.dump()
        self._psf.dump()
        self._edisp.dump()
    
    def save(self,irf_name):

        aeff_file = 'aeff_' + irf_name + '.fits'
        psf_file = 'psf_' + irf_name + '.fits'
        edisp_file = 'edisp_' + irf_name + '.fits'

        self._psf.save(psf_file)
        self._aeff.save(aeff_file)
        self._edisp.save(edisp_file)
            
class IRFComponent(object):

    @staticmethod
    def load_table_axes(data):

        elo = np.log10(np.array(data[0][0]))
        ehi = np.log10(np.array(data[0][1]))
        cthlo = np.array(data[0][2])
        cthhi = np.array(data[0][3])
        
        edges = np.concatenate((elo,np.array(ehi[-1],ndmin=1)))
        energy_axis = Axis(edges)
        edges = np.concatenate((cthlo,np.array(cthhi[-1],ndmin=1)))
        cth_axis = Axis(edges)
        
        return energy_axis, cth_axis
    
    def setup_axes(self,data):
        self._energy_axis, self._cth_axis = IRFComponent.load_table_axes(data)
                
class AeffIRF(IRFComponent):

    def __init__(self,fits_file):
        
        self._hdulist = pyfits.open(fits_file)        
        hdulist = self._hdulist
        
        self.setup_axes(hdulist[1].data)
        self._aeff = np.array(hdulist[1].data[0][4])

        nx = self._cth_axis.nbins
        ny = self._energy_axis.nbins

        self._aeff.resize((nx,ny))
        self._aeff_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                      counts=self._aeff,var=0)
#        hdulist.close()

    def dump(self):
        self._hdulist.info()

        print 'Energy Axis: ', self._energy_axis.edges
        print 'Angle Axis: ', self._cth_axis.edges
        
        for k, v in self._hdulist[0].header.iteritems():
            print '%-30s %s'%(k, v)
        
        
    def __call__(self,egy,cth,**kwargs):

        egy = np.array(egy,ndmin=1)
        cth = np.array(cth,ndmin=1)
        aeff = self._aeff_hist.interpolate(cth,egy)
        aeff[aeff<= 0.0] = 0.0
        return aeff

    def save(self,filename):

        self._hdulist[0].header['FILENAME'] = filename
        
        print 'Writing ', filename
        self._hdulist.writeto(filename,clobber=True)
        
        
class AeffPyIRF(IRFComponent):

    def __init__(self,irf):
        self._irf = irf
        self._irf.setPhiDependence(False)
    
    def __call__(self,egy,cth):

        egy = np.asarray(egy)
        if egy.ndim == 0: egy.resize((1))

        cth = np.asarray(cth)
        if cth.ndim == 0: cth.resize((1))
        
        if cth.shape[0] > 1:
            z = np.zeros(shape=cth.shape)
            for j, c in enumerate(cth):
                z[j] = self._irf.value(float(np.power(10,egy)),
                                       float(np.degrees(np.arccos(c))),0)
#            return z
        else:
            z = self._irf.value(float(np.power(10,egy)),
                               float(np.degrees(np.arccos(cth))),0)

        z *= 1E-4
        return z
            
class EDispIRF(IRFComponent):

    def __init__(self,fits_file):
        
        self._hdulist = pyfits.open(fits_file)
        hdulist = self._hdulist
#        hdulist.info()
       
        self._elo = np.log10(np.array(hdulist[1].data[0][0]))
        self._ehi = np.log10(np.array(hdulist[1].data[0][1]))
        self._cthlo = np.array(hdulist[1].data[0][2])
        self._cthhi = np.array(hdulist[1].data[0][3])

        edges = np.concatenate((self._elo,np.array(self._ehi[-1],ndmin=1)))
        self._energy_axis = Axis(edges)
        edges = np.concatenate((self._cthlo,np.array(self._cthhi[-1],ndmin=1)))
        self._cth_axis = Axis(edges)
        
        
        self._center = [0.5*(self._cthlo + self._cthhi),
                        0.5*(self._elo + self._ehi)]
        
        self._bin_width = [self._cthhi-self._cthlo,
                           self._ehi-self._elo]

    def dump(self):
        self._hdulist.info()

        print 'Energy Axis: ', self._energy_axis.edges
        print 'Angle Axis: ', self._cth_axis.edges
        
        for k, v in self._hdulist[0].header.iteritems():
            print '%-30s %s'%(k, v)
        
    def save(self,filename):

        self._hdulist[0].header['FILENAME'] = filename
        
        print 'Writing ', filename
        self._hdulist.writeto(filename,clobber=True)
        

class EDispPyIRF(IRFComponent):

    def __init__(self,irf):
        self._irf = irf
    
class PSFIRF(IRFComponent):

    def __init__(self,fits_file,interpolate_density=True):

        self._interpolate_density = interpolate_density
        self._hdulist = pyfits.open(fits_file)
        hdulist = self._hdulist
        hdulist.info()

        if re.search('front',fits_file.lower()) is not None: self._ct = 'front'
        elif re.search('back',fits_file.lower()) is not None: self._ct = 'back'
        else: self._ct = 'none'

        self._cfront = hdulist['PSF_SCALING_PARAMS'].data[0][0][0:2]
        self._cback = hdulist['PSF_SCALING_PARAMS'].data[0][0][2:4]
        self._beta = hdulist['PSF_SCALING_PARAMS'].data[0][0][4]
        
        self.setup_axes(hdulist[1].data)
        nx = self._cth_axis.nbins
        ny = self._energy_axis.nbins
        shape = (nx,ny)
        
        ncore = np.array(hdulist[1].data[0][4]).reshape(shape)
        ntail = np.array(hdulist[1].data[0][5]).reshape(shape)
        score = np.array(hdulist[1].data[0][6]).reshape(shape)
        stail = np.array(hdulist[1].data[0][7]).reshape(shape)
        gcore = np.array(hdulist[1].data[0][8]).reshape(shape)
        gtail = np.array(hdulist[1].data[0][9]).reshape(shape)        
        fcore = 1./(1.+ntail*np.power(stail/score,2))
        
        self._ncore_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=ncore,var=0)
        self._ntail_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=ntail,var=0)
        self._score_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=score,var=0)
        self._stail_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=stail,var=0)
        self._gcore_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=gcore,var=0)
        self._gtail_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=gtail,var=0)
        self._fcore_hist = Histogram2D(self._cth_axis,self._energy_axis,
                                       counts=fcore,var=0)

        try:
            
            fisheye_axes = IRFComponent.load_table_axes(hdulist['FISHEYE_CORRECTION'].data)[::-1]
            fisheye_shape = (fisheye_axes[0].nbins,fisheye_axes[1].nbins)
        
            fisheye_mean   = np.array(hdulist['FISHEYE_CORRECTION'].
                                      data[0][4]).reshape(fisheye_shape)
            fisheye_median = np.array(hdulist['FISHEYE_CORRECTION'].
                                      data[0][5]).reshape(fisheye_shape)
            fisheye_peak   = np.array(hdulist['FISHEYE_CORRECTION'].
                                      data[0][6]).reshape(fisheye_shape)
        except:
            fisheye_axes = (self._cth_axis,self._energy_axis)
            fisheye_mean   = np.zeros(shape)
            fisheye_median   = np.zeros(shape)
            fisheye_peak   = np.zeros(shape)

        self._mean_hist = Histogram2D(*fisheye_axes,counts=fisheye_mean,var=0)
        self._median_hist = Histogram2D(*fisheye_axes,counts=fisheye_median,
                                         var=0)
        self._peak_hist = Histogram2D(*fisheye_axes,counts=fisheye_peak,var=0)

        psf_scale = np.degrees(self.psf_scale(self._mean_hist.axis(1).center))
        
        self._mean_hist *= psf_scale
        self._median_hist *= psf_scale
        self._peak_hist *= psf_scale
        
        self._theta_axis = Axis(np.linspace(-3.0,np.log10(90.0),101))
        self._psf_hist = HistogramND([self._cth_axis,
                                      self._energy_axis,
                                      self._theta_axis])
                                     
        th = self._theta_axis.center

        for i in range(nx):
            for j in range(ny):
                x = self._cth_axis.center[i]
                y = self._energy_axis.center[j]                
                z = self.eval(10**th,y,x)
                self._psf_hist._counts[i,j] = self.eval(10**th,y,x)


        return
        plt.figure()

        egy0 = self._energy_axis.center[10]
        cth0 = self._cth_axis.center[5]

        y0 = self.eval2(10**th,egy0,cth0)
        y1 = self.eval(10**th,egy0,cth0)

        self._psf_hist.slice([0,1],[5,10]).plot(hist_style='line')
        plt.plot(th,y0)
        plt.plot(th,y1)
        plt.gca().set_yscale('log')

        plt.figure()
        self._psf_hist.interpolateSlice([0,1],[cth0,egy0]).plot(hist_style='line')
        self._psf_hist.slice([0,1],[5,10]).plot(hist_style='line')
        plt.gca().set_yscale('log')

        plt.figure()
        sh = self._psf_hist.interpolateSlice([0,1],[cth0+0.01,egy0+0.03])
        y0 = self.eval2(10**th,egy0+0.03,cth0+0.01)
        y1 = self.eval(10**th,egy0+0.03,cth0+0.01)
        y2 = sh.counts()
        y3 = self._psf_hist.interpolate(np.vstack(((cth0+0.01)*np.ones(100),
                                                   (egy0+0.03)*np.ones(100),
                                                   th)))

        sh.plot(hist_style='line')
        plt.plot(th,y0)
        plt.plot(th,y1)
        plt.plot(th,y3)
        plt.gca().set_yscale('log')

        plt.figure()
        plt.plot(th,y0/y2)
        plt.plot(th,y0/y3)
        plt.plot(th,y0/y1)
#        plt.plot(th,y2/y1)

        plt.figure()
        self._psf_hist.slice([0],[5]).plot(logz=True)

        plt.show()

        return

    def dump(self):
        self._hdulist.info()

        print 'Energy Axis: ', self._energy_axis.edges
        print 'Angle Axis: ', self._cth_axis.edges
        
        for k, v in self._hdulist[0].header.iteritems():
            print '%-30s %s'%(k, v)
    
    def plot(self,ft):
    
        fig = ft.create('psf_table',nax=(3,2),figscale=1.5)
        fig[0].set_title('score')
        fig[0].add_hist(self._score_hist)
        fig[1].set_title('stail')
        fig[1].add_hist(self._stail_hist)
        fig[2].set_title('gcore')
        fig[2].add_hist(self._gcore_hist)
        fig[3].set_title('gtail')
        fig[3].add_hist(self._gtail_hist)
        fig[4].set_title('fcore')
        fig[4].add_hist(self._fcore_hist)

        fig.plot()
                
    def __call__(self,dtheta,egy,cth,**kwargs):        

        if self._interpolate_density:
            return self.eval2(dtheta,egy,cth)
        else:
            return self.eval(dtheta,egy,cth)

    def fisheye(self,egy,cth,ctype='mean'):

        v = self._mean_hist.interpolate(egy,cth)
        if ctype == 'mean':
            return self._mean_hist.interpolate(cth,egy)
        elif ctype == 'median':
            return self._median_hist.interpolate(cth,egy)
        elif ctype == 'peak':
            return self._peak_hist.interpolate(cth,egy)
        else:
            raise Exception('Invalid fisheye correction.')
        
    def quantile(self,egy,cth,frac=0.68):
        x = np.logspace(-3.0,np.log10(45.0),300)
        x = np.concatenate(([0],x))
        xc = 0.5*(x[:-1]+x[1:])
        y = self(xc,egy,cth)

        deltax = np.radians(x[1:] - x[:-1])
        
        cdf = 2*np.pi*np.sin(np.radians(xc))*y*deltax
        cdf = np.cumsum(cdf)
        cdf = np.concatenate(([0],cdf))
        cdf /= cdf[-1]

        return percentile(x,cdf,frac)

    def psf_scale(self,loge):

        if self._ct == 'back': c = self._cback
        else: c = self._cfront
        
        return np.sqrt(np.power(c[0]*np.power(10,-self._beta*(2.0-loge)),2) +
                       np.power(c[1],2))
    
        
    def eval(self,dtheta,egy,cth):
        """Evaluate PSF by interpolating in PSF parameters."""
        
        if self._ct == 'back': c = self._cback
        else: c = self._cfront
        
        spx = np.sqrt(np.power(c[0]*np.power(10,-self._beta*(2.0-egy)),2) +
                      np.power(c[1],2))

        spx = np.degrees(spx)
        
        x = dtheta/spx
                
        gcore = self._gcore_hist.interpolate(cth,egy)
        score = self._score_hist.interpolate(cth,egy)
        gtail = self._gtail_hist.interpolate(cth,egy)
        stail = self._stail_hist.interpolate(cth,egy)
        fcore = self._fcore_hist.interpolate(cth,egy)

        fcore[fcore < 0.0] = 0.0  # = max(0.0,fcore)
        fcore[fcore > 1.0] = 1.0  # min(1.0,fcore)

        #gcore = max(1.2,gcore)
        #gtail = max(1.2,gtail)
        
        return (fcore*self.king(x,score,gcore) +
                (1-fcore)*self.king(x,stail,gtail))/(spx*spx)

    def eval2(self,dtheta,egy,cth):
        """Evaluate PSF by interpolating in PSF density."""
        dtheta = np.array(dtheta,ndmin=1)
        egy = np.array(egy,ndmin=1)
        cth = np.array(cth,ndmin=1)


        dtheta[dtheta <= 10**self._theta_axis.lo_edge()] =\
            10**self._theta_axis.lo_edge()
        
        return self._psf_hist.interpolate(cth,egy,np.log10(dtheta))

    def king(self,dtheta,sig,g):

        if sig.shape[0] > 1:

            if dtheta.ndim == 1:
                dtheta2 = np.empty(shape=(dtheta.shape[0],sig.shape[0]))
                dtheta2.T[:] = dtheta
            else:
                dtheta2 = dtheta

            sig2 = np.empty(shape=(dtheta.shape[0],sig.shape[0]))
            sig2[:] = sig
            
            g2 = np.empty(shape=(dtheta.shape[0],sig.shape[0]))
            g2[:] = g

            n = 1./(2*np.pi*sig2*sig2)
            u = np.power(dtheta2,2)/(2*sig2*sig2)
            
            return n*(1-1/g2)*np.power(1+u/g2,-g2)

        else:
                    
            n = 1./(2*np.pi*sig*sig)
            u = np.power(dtheta,2)/(2*sig*sig)
            
            return n*(1-1/g)*np.power(1+u/g,-g)
    
    def save(self,filename):

        self._hdulist[0].header['FILENAME'] = filename

        self._hdulist[1].data[0][4] = self._ncore
        self._hdulist[1].data[0][5] = self._ntail
        self._hdulist[1].data[0][6] = self._score
        self._hdulist[1].data[0][7] = self._stail
        self._hdulist[1].data[0][8] = self._gcore
        self._hdulist[1].data[0][9] = self._gtail
        
        self._hdulist[2].data[0][0][0:2] = self._cfront
        self._hdulist[2].data[0][0][2:4] = self._cback
        self._hdulist[2].data[0][0][4] = self._beta
        
        print 'Writing ', filename
        self._hdulist.writeto(filename,clobber=True)
        
class PSFPyIRF(PSFIRF):

    def __init__(self,irf):
        self._irf = irf

    def __call__(self,dtheta,egy,cth):

        dtheta = np.asarray(dtheta)
        cth = np.asarray(cth)
        
        if dtheta.ndim == 0:
            return self._irf.value(dtheta,float(np.power(10,egy)),
                                   float(np.degrees(np.arccos(cth))),0)

        if cth.shape[0] > 1:        
            z = np.zeros(shape=(dtheta.shape[0],cth.shape[0]))

            for i, t in enumerate(dtheta):
                for j, c in enumerate(cth):
                    z[i,j] = self._irf.value(t,float(np.power(10,egy)),
                                             float(np.degrees(np.arccos(c))),0)
        else:

            
            
            z =  np.zeros(shape=dtheta.shape)
            for i, t in enumerate(dtheta):
                z[i] = self._irf.value(t,float(np.power(10,egy)),
                                       float(np.degrees(np.arccos(cth))),0)

        return z
            
        
        

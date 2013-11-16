
import os
import copy
import re
import pyfits
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

    def __init__(self,psf_file=None,aeff_file=None,edisp_file=None):

        self._psf = []
        self._aeff = []
        self._edisp = []

        if not psf_file is None and not isinstance(psf_file,list): 
            self._psf.append(PSFIRF(psf_file))
        elif not psf_file is None and isinstance(psf_file,list): 
            for f in psf_file: self._psf.append(PSFIRF(f))

        if not aeff_file is None and not isinstance(aeff_file,list): 
            self._aeff.append(AeffIRF(aeff_file))
        elif not aeff_file is None and isinstance(aeff_file,list): 
            for f in aeff_file: self._aeff.append(AeffIRF(f))

        if not edisp_file is None and not isinstance(edisp_file,list): 
            self._edisp.append(EDispIRF(edisp_file))
        elif not edisp_file is None and isinstance(edisp_file,list): 
            for f in edisp_file: self._edisp.append(EDispIRF(f))

    @staticmethod
    def configure(parser):
        parser.add_argument('--irf_dir', default = 'custom_irfs', 
                            help = 'Set the IRF directory.')

    @staticmethod
    def create(irf_name,load_from_file=False,irf_dir=None):
        
        if load_from_file:  return IRFManager.createFromFile(irf_name,irf_dir)
        else: return IRFManager.createFromIRF(irf_name)

    @staticmethod
    def createFromFile(irf_name,irf_dir=None):
        
        if irf_dir is None: irf_dir = 'custom_irfs'

        psf_files = []
        aeff_files = []
        edisp_files = []

        irf_names = expand_irf(irf_name)
        for name in irf_names:     

            name = name.replace('::FRONT','_front')
            name = name.replace('::BACK','_back')

            psf_file = os.path.join(irf_dir,'psf_%s.fits'%(name))
            aeff_file = os.path.join(irf_dir,'aeff_%s.fits'%(name))
            edisp_file = os.path.join(irf_dir,'edisp_%s.fits'%(name))
            
            psf_files.append(psf_file)
            aeff_files.append(aeff_file)
            edisp_files.append(edisp_file)

        irf = IRFManager(psf_files,aeff_files,edisp_files)
#        irf.loadPyIRF(irf_name)
        return irf

    @staticmethod
    def createFromIRF(irf_name):
        irf = IRFManager()
        irf.loadPyIRF(irf_name)
        return irf
        
    def loadPyIRF(self,irf_name):
        """Create IRF object using pyIrf modules in Science Tools."""
        print 'loadPyIRF: ', irf_name

        import pyIrfLoader
        if not IRFManager.load_irf:
            pyIrfLoader.Loader_go()
            IRFManager.load_irf = True

        irf_factory=pyIrfLoader.IrfsFactory.instance()
        irfs = irf_factory.create(irf_name)

        self._psf = [PSFPyIRF(irfs.psf())]
        self._aeff = [AeffPyIRF(irfs.aeff())]
        self._edisp = [EDispPyIRF(irfs.edisp())]
        

    def psf(self,*args,**kwargs):

        v = None
        for i in range(len(self._psf)):
            if i == 0: v = self._psf[i](*args,**kwargs)
            else: v += self._psf[i](*args,**kwargs)

        return v

#        return self._psf(*args,**kwargs)

    def aeff(self,*args,**kwargs):

        v = None
        for i in range(len(self._aeff)):
            if i == 0: v = self._aeff[i](*args,**kwargs)
            else: v += self._aeff[i](*args,**kwargs)

        return v

#        return self._aeff(*args,**kwargs)

    def edisp(self,*args,**kwargs):

        v = None
        for i in range(len(self._edisp)):
            if i == 0: v = self._edisp[i](*args,**kwargs)
            else: v += self._edisp[i](*args,**kwargs)

        return v

#        return self._edisp(*args,**kwargs)

    def save(self,irf_name):


        for i in range(len(self._aeff)):
            aeff_file = 'aeff_' + irf_name + '.fits'
            psf_file = 'psf_' + irf_name + '.fits'
            edisp_file = 'edisp_' + irf_name + '.fits'

            self._psf[i].save(psf_file)
            self._aeff[i].save(aeff_file)
            self._edisp[i].save(edisp_file)
        
    
class IRF(object):

    def get_index(self,x,center,width):

        x = np.asarray(x)
        if x.ndim == 0: x.resize((1))

        dx = np.zeros(shape=(center.shape[0],x.shape[0]))
        dx[:] = x
        dx = np.abs(dx.T-center-0.5*width)

        ix = np.argmin(dx,axis=1)
        ix[ix > center.shape[0]-2] = center.shape[0]-2

        return ix
        
    def interpolate(self,x,y,z):

        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim == 0: x.resize((1))
        if y.ndim == 0: y.resize((1))
        
        dx = np.zeros(shape=(self._center[0].shape[0],x.shape[0]))
        dx[:] = x
        dx = np.abs(dx.T-self._center[0]-0.5*self._bin_width[0])

        dy = np.zeros(shape=(self._center[1].shape[0],y.shape[0]))
        dy[:] = y
        dy = np.abs(dy.T-self._center[1]-0.5*self._bin_width[1])
        
        ix = np.argmin(dx,axis=1)
        iy = np.argmin(dy,axis=1)

        ix[ix > self._center[0].shape[0]-2] = self._center[0].shape[0]-2
        iy[iy > self._center[1].shape[0]-2] = self._center[1].shape[0]-2        

        xs = (x - self._center[0][ix])/self._bin_width[0][ix]
        ys = (y - self._center[1][iy])/self._bin_width[1][iy]

        # Do not interpolate beyond the edge of the lowest bin in cos-theta
        xs[(ix == 0) & (xs <= 0)] = 0
        
        return (z[ix,iy]*(1-xs)*(1-ys) + z[ix+1,iy]*xs*(1-ys) +
                z[ix,iy+1]*(1-xs)*ys + z[ix+1,iy+1]*xs*ys)

    
class AeffIRF(IRF):

    def __init__(self,fits_file):
        
        self._hdulist = pyfits.open(fits_file)
        #hdulist.info()
        hdulist = self._hdulist
        
        self._elo = np.log10(np.array(hdulist[1].data[0][0]))
        self._ehi = np.log10(np.array(hdulist[1].data[0][1]))
        self._egy_edge = np.append(self._elo,[self._ehi[-1]])

        
        self._cthlo = np.array(hdulist[1].data[0][2])
        self._cthhi = np.array(hdulist[1].data[0][3])
        self._cth_edge = np.append(self._cthlo,[self._cthhi[-1]])
        
        self._center = [0.5*(self._cthlo + self._cthhi),
                        0.5*(self._elo + self._ehi)]
        
        self._bin_width = [self._cthhi-self._cthlo,
                           self._ehi-self._elo]

        self._aeff = np.array(hdulist[1].data[0][4])

        self._aeff.resize((self._cthlo.shape[0],self._elo.shape[0]))

#        hdulist.close()
        
        return


    def plot(self):
        h = Histogram2D([self._cthlo[0],self._cthhi[-1]],
                        [self._elo[0],self._ehi[-1]],
                        self._aeff.shape[1],
                        self._aeff.shape[0])

        h._counts = self._aeff
        h.plot()
#        plt.colorbar()

    def __call__(self,egy,cth):

        egy = np.array(egy,ndmin=1)
        cth = np.array(cth,ndmin=1)        
        aeff = self.interpolate(cth,egy,self._aeff)
        aeff[(aeff<= 0.0) | (cth < self._cthlo[0])] = 0.0
        return aeff

#        if aeff <= 0.0 or cth < self._cthlo[0]: return 0.0
#        else: return aeff
#        return np.max(0.0,self.interpolate(cth,egy,self._aeff))

    def save(self,filename):

        self._hdulist[0].header['FILENAME'] = filename
        
        print 'Writing ', filename
        self._hdulist.writeto(filename,clobber=True)
        
        
class AeffPyIRF(IRF):

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
            
class EDispIRF(IRF):

    def __init__(self,fits_file):
        
        self._hdulist = pyfits.open(fits_file)

        hdulist = self._hdulist
        
        self._elo = np.log10(np.array(hdulist[1].data[0][0]))
        self._ehi = np.log10(np.array(hdulist[1].data[0][1]))
        self._cthlo = np.array(hdulist[1].data[0][2])
        self._cthhi = np.array(hdulist[1].data[0][3])

        self._center = [0.5*(self._cthlo + self._cthhi),
                        0.5*(self._elo + self._ehi)]
        
        self._bin_width = [self._cthhi-self._cthlo,
                           self._ehi-self._elo]

#        print self._center
#        print self._bin_width
        
        
    def save(self,filename):

        self._hdulist[0].header['FILENAME'] = filename
        
        print 'Writing ', filename
        self._hdulist.writeto(filename,clobber=True)
        

class EDispPyIRF(IRF):

    def __init__(self,irf):
        self._irf = irf
    
class PSFIRF(IRF):

    def __init__(self,fits_file,interpolate_density=True):

        self._interpolate_density = interpolate_density
        self._hdulist = pyfits.open(fits_file)
        hdulist = self._hdulist
#        hdulist.info()

        if re.search('front',fits_file) is not None:
            self._ct = 'front'
        elif re.search('back',fits_file) is not None:
            self._ct = 'back'
        else:
            print 'Error: Could not identify conversion type.'
            sys.exit(1)
        
        self._elo = np.log10(np.array(hdulist[1].data[0][0]))
        self._ehi = np.log10(np.array(hdulist[1].data[0][1]))
        self._cthlo = np.array(hdulist[1].data[0][2])
        self._cthhi = np.array(hdulist[1].data[0][3])

        self._ncore = np.array(hdulist[1].data[0][4])
        self._ntail = np.array(hdulist[1].data[0][5])
        self._score = np.array(hdulist[1].data[0][6])
        self._stail = np.array(hdulist[1].data[0][7])
        self._gcore = np.array(hdulist[1].data[0][8])
        self._gtail = np.array(hdulist[1].data[0][9])
        
        self._ncore.resize((self._cthlo.shape[0],self._elo.shape[0]))
        self._ntail.resize((self._cthlo.shape[0],self._elo.shape[0]))
        self._score.resize((self._cthlo.shape[0],self._elo.shape[0]))
        self._stail.resize((self._cthlo.shape[0],self._elo.shape[0]))
        self._gcore.resize((self._cthlo.shape[0],self._elo.shape[0]))
        self._gtail.resize((self._cthlo.shape[0],self._elo.shape[0]))
        self._fcore = 1./(1.+self._ntail*
                          np.power(self._stail/self._score,2))
        
        self._cfront = hdulist[2].data[0][0][0:2]
        self._cback = hdulist[2].data[0][0][2:4]
        self._beta = hdulist[2].data[0][0][4]

        self._center = [0.5*(self._cthlo + self._cthhi),
                        0.5*(self._elo + self._ehi)]

        self._bin_width = [self._cthhi-self._cthlo,
                           self._ehi-self._elo]
        
        
        return
        
        x0 = np.linspace(1,6,200)        
        
        plt.figure()

        plt.plot(x0,self.interpolate(self._center[0][0],x0,self._gcore))
        plt.plot(x0,self.interpolate(0.5*(self._center[0][0]+self._center[0][1]),x0,self._gcore))
        plt.plot(x0,self.interpolate(self._center[0][1],x0,self._gcore))
        plt.plot(self._center[1],self._gcore[0],marker='o')
        plt.plot(self._center[1],self._gcore[1],marker='o')
        plt.show()
        
#        self.interpolate(np.array([0.15,0.4,1.0,0.5]),3.,self._gcore)
        return

    def plot(self):
    
        hists = []

        for i in range(6):
        
            h = Histogram2D([self._cthlo[0],self._cthhi[-1]],
                            self._score.shape[1],
                            [self._elo[0],self._ehi[-1]],
                            self._score.shape[0])

            hists.append(h)


        fig = plt.figure()

        
        
        hists[0]._counts = self._score
        hists[1]._counts = self._gcore
        hists[2]._counts = self._stail
        hists[3]._counts = self._gtail
        hists[4]._counts = self._ncore
        hists[5]._counts = self._ntail
        
        ax = fig.add_subplot(2,3,1)
        hists[0].plot(vmax=2.0)
        plt.colorbar()
        ax = fig.add_subplot(2,3,2)
        hists[1].plot(vmax=4)        
        plt.colorbar()
        ax = fig.add_subplot(2,3,3)
        hists[2].plot(vmax=2.0)
        plt.colorbar()
        ax = fig.add_subplot(2,3,4)
        hists[3].plot()        
        plt.colorbar()
        ax = fig.add_subplot(2,3,5)
        hists[4].plot()        
        plt.colorbar()

        ax = fig.add_subplot(2,3,6)
        hists[5].plot()        
        plt.colorbar()
        
        plt.show()
        
    def __call__(self,dtheta,egy,cth):        

        if self._interpolate_density:
            return self.eval2(dtheta,egy,cth)
        else:
            return self.eval(dtheta,egy,cth)
        
    def quantile(self,egy,cth,frac=0.68):
        radii = np.logspace(-3.0,np.log10(45.0),300)
        radii = np.concatenate(([0],radii))

        y = self(radii,egy,cth)

        f = UnivariateSpline(radii,y,s=0)

        rcenters = 0.5*(radii[:-1]+radii[1:])
        rwidth = np.radians(radii[1:] - radii[:-1])
        
        cdf = 2*np.pi*np.sin(np.radians(rcenters))*f(rcenters)*rwidth
        cdf = np.cumsum(cdf)
        cdf = np.concatenate(([0],cdf))

        cdf /= cdf[-1]
        
        indx = bisect.bisect(cdf, frac) - 1
        return ((frac - cdf[indx])/(cdf[indx+1] - cdf[indx])
                *(radii[indx+1] - radii[indx]) + radii[indx])
        
        
    def eval(self,dtheta,egy,cth):
        """Evaluate PSF by interpolating in PSF parameters."""
        if self._ct == 'front': c = self._cfront
        else: c = self._cback
        
        spx = np.sqrt(np.power(c[0]*np.power(10,-self._beta*(2.0-egy)),2) +
                      np.power(c[1],2))

        spx = np.degrees(spx)
        
        x = dtheta/spx

        gcore = self.interpolate(cth,egy,self._gcore)
        score = self.interpolate(cth,egy,self._score)
        gtail = self.interpolate(cth,egy,self._gtail)
        stail = self.interpolate(cth,egy,self._stail)
        fcore = self.interpolate(cth,egy,self._fcore)
        
        fcore[fcore < 0.0] = 0.0  # = max(0.0,fcore)
        fcore[fcore > 1.0] = 1.0  # min(1.0,fcore)

        #gcore = max(1.2,gcore)
        #gtail = max(1.2,gtail)
        
        return (fcore*self.king(x,score,gcore) +
                (1-fcore)*self.king(x,stail,gtail))/(spx*spx)

    def eval2(self,dtheta,egy,cth):
        """Evaluate PSF by interpolating in PSF density."""
        ix = self.get_index(cth,self._center[0],self._bin_width[0])
        iy = self.get_index(egy,self._center[1],self._bin_width[1])

        if ix.ndim > 0 and ix.shape[0] > 1:
            dtheta = dtheta.reshape(dtheta.shape + (1,))
        
        z00 = self.king2(dtheta,ix,iy)
        z10 = self.king2(dtheta,ix+1,iy)
        z01 = self.king2(dtheta,ix,iy+1)
        z11 = self.king2(dtheta,ix+1,iy+1)

        xs = (cth - self._center[0][ix])/self._bin_width[0][ix]
        ys = (egy - self._center[1][iy])/self._bin_width[1][iy]

        z = (z00*(1-xs)*(1-ys) + z10*xs*(1-ys) +
             z01*(1-xs)*ys + z11*xs*ys)
        
#        z[z<0] = 0
        return z
        
    def king2(self,dtheta,ix,iy):

        egy = self._center[1][iy]
        
        if self._ct == 'front': c = self._cfront
        else: c = self._cback
        
        spx = np.sqrt(np.power(c[0]*np.power(10,-self._beta*(2.0-egy)),2) +
                      np.power(c[1],2))

        spx = np.degrees(spx)
        
        x = dtheta/spx

        king0 = self.king(x,self._score[ix,iy],self._gcore[ix,iy])
        king1 = self.king(x,self._stail[ix,iy],self._gtail[ix,iy])
        
        return (self._fcore[ix,iy]*king0 +
                (1-self._fcore[ix,iy])*king1)/(spx*spx)

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
        
#        print self._hdulist[1].data[0][4].shape, self._ncore.shape

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
            
        
        

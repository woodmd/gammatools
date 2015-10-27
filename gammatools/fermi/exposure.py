
import glob
import numpy as np
from gammatools.core.histogram import *
from irf_util import IRFManager
from catalog import Catalog
from gammatools.core.util import eq2gal, gal2eq
import healpy
from gammatools.core.astropy_helper import pyfits

def get_src_mask(src,ra,dec,radius=5.0):
    dist = np.sqrt( ((src[0]-ra)*np.cos(src[1]))**2 + (src[1]-dec)**2)
    msk = dist > np.radians(radius)
    return msk

class LTCube(object):

    def __init__(self,ltfile=None):

        self._ltmap = None

        if ltfile is None: return
        elif isinstance(ltfile,list):
            for f in ltfile: self.load_ltfile(f)
        elif not re.search('\.txt?',ltfile) is None:
            files=np.loadtxt(ltfile,unpack=True,dtype='str')
            for f in files: self.load_ltfile(f)
        else:
            self.load_ltfile(ltfile)


    @staticmethod
    def create(ltfile):

        ltc = LTCube()
        if not isinstance(ltfile,list):
            ltfile = glob.glob(ltfile)
        for f in ltfile:  
            ltc.load_ltfile(f)

        return ltc
        

    def load_ltfile(self,ltfile):

        print 'Loading ', ltfile
        
        hdulist = pyfits.open(ltfile)
                
        if self._ltmap is None:
            self._ltmap = hdulist[1].data.field(0)
            self._tstart = hdulist[0].header['TSTART']
            self._tstop = hdulist[0].header['TSTOP']
        else:
            self._ltmap += hdulist[1].data.field(0)
            self._tstart = min(self._tstart,hdulist[0].header['TSTART'])
            self._tstop = max(self._tstop,hdulist[0].header['TSTOP'])

        cth_edges = np.array(hdulist[3].data.field(0))
        cth_edges = np.concatenate(([1],cth_edges))
        cth_edges = cth_edges[::-1]
        self._cth_axis = Axis(cth_edges)

        self._domega = (self._cth_axis.edges[1:]-
                        self._cth_axis.edges[:-1])*2*np.pi
            
    def get_src_lthist(self,ra,dec,cth_axis=None):

        if cth_axis is None:
            cth_axis = copy.deepcopy(self._cth_axis)
            new_axis = False
        else:
            tmp_cth_axis = Axis.create(cth_axis.edges[0],
                                       cth_axis.edges[-1],
                                       cth_axis.nbins*4)
            new_axis = True

        

        ipix = healpy.ang2pix(64,np.pi/2. - np.radians(dec),
                              np.radians(ra),nest=True)


        if new_axis:
            lt = interpolate(self._cth_axis.center,
                             self._ltmap[ipix,::-1]/self._cth_axis.width,
                             tmp_cth_axis.center)*tmp_cth_axis.width
            lt = np.sum(lt.reshape(-1,4),axis=1)  
        else:
            lt = self._ltmap[ipix,::-1]

        return Histogram(cth_axis,counts=lt,var=0)

    def get_allsky_lthist(self,lon_axis,slat_axis,coordsys='gal',cth_axis=None):

        h = HistogramND([lon_axis,slat_axis,self._cth_axis])

        if coordsys=='gal':
        
            lon, slat = np.meshgrid(h.axis(0).center,
                                    h.axis(1).center,
                                    indexing='ij')
        

            ra, dec = gal2eq(np.degrees(lon),
                             np.degrees(np.arcsin(slat)))
        
            ra = np.radians(ra)
            dec = np.radians(dec)

        else:
            ra, dec = np.meshgrid(h.axis(0).center,
                                  np.arcsin(h.axis(1).center),
                                  indexing='ij')


        theta = np.pi/2. - dec
        phi = ra
            
        ipix = healpy.ang2pix(64,np.ravel(theta),
                              np.ravel(phi),nest=True)

        lt = self._ltmap[ipix,::-1]

#        print lt.shape
#        print h.axes()[0].nbins, h.axes()[1].nbins, h.axes()[2].nbins
        
        lt = lt.reshape((h.axes()[0].nbins,
                         h.axes()[1].nbins,
                         h.axes()[2].nbins))

        h._counts = lt

        return h
        
        
        
    def get_hlat_ltcube(self):

        
        import healpy
        
        

        nbin = 400

        ra_edge = np.linspace(0,2*np.pi,nbin+1)
        dec_edge = np.linspace(-1,1,nbin+1)

        ra_center = 0.5*(ra_edge[1:] + ra_edge[:-1])
        dec_center = 0.5*(dec_edge[1:] + dec_edge[:-1])
        
        dec, ra = np.meshgrid(np.arcsin(dec_center),ra_center)

        lthist = pHist([ra_edge,dec_edge,self._cth_edges])
        
        srcs = np.loadtxt('src.txt',unpack=False)
        ipix = healpy.ang2pix(64,np.ravel(np.pi/2. - dec),
                              np.ravel(ra),nest=True)

        lt = self._ltmap[ipix,::-1]
        
        (l, b) = eq2gal(np.degrees(ra),np.degrees(dec))

        gal_msk = (np.abs(b) > 40.) & (np.abs(b) < 80.)
        eq_msk = (np.abs(np.degrees(dec)) < 79.9)

        msk = gal_msk & eq_msk
        for i in range(len(srcs)):
            msk &= get_src_mask(np.radians(srcs[i]),ra,dec,5.0)

        lt = lt.reshape((nbin,nbin,40))
        lt[msk==False,:] = 0
        
        
        lthist._counts = lt
        
        self._omega_tot = float(len(msk[msk==True]))/(nbin**2)*4*np.pi
        self._domega_bin = 4*np.pi/(nbin**2)
        
        h0 = lthist.slice([2],[0])
        h1 = lthist.slice([2],[20])
        h2 = lthist.slice([2],[30])
        
        
        plt.figure()
        h0.plot()
        plt.figure()
        h1.plot()
        plt.figure()
        h2.plot()
        
        plt.show()



class ExposureCalc(object):

    def __init__(self,irfm,ltc):
        self._irfm = irfm
        self._ltc = ltc

    def getExpByName(self,src_names,egy_axis,cth_axis=None,weights=None):

        exph = None
        cat = Catalog.get()
        for s in src_names:
            src = cat.get_source_by_name(s) 
            h = self.eval(src['RAJ2000'], src['DEJ2000'],egy_axis,
                          cth_axis,weights=weights)
       
            if exph is None: exph = h
            else: exph += h

        exph /= float(len(src_names))        
        return exph

    def eval(self,ra,dec,egy_axis,cth_axis=None,weights=None):

        if cth_axis is None:
            cth_axis = self._ltc._cth_axis

        cth = cth_axis.center
        egy = egy_axis.center

        x, y = np.meshgrid(egy,cth,indexing='ij')
        aeff = self._irfm.aeff(x,y)

        exph = Histogram(egy_axis)
        lthist = self._ltc.get_src_lthist(ra,dec,cth_axis)

        exp = np.sum(aeff*lthist.counts[np.newaxis,:],axis=1)


        if weights is not None:
            edisp = self._irfm.edisp(egy[:,np.newaxis,np.newaxis],
                                 egy[np.newaxis,:,np.newaxis],
                                 cth[np.newaxis,np.newaxis,:])

            wedisp = edisp*aeff[np.newaxis,:,:]*lthist.counts[np.newaxis,np.newaxis,:]            
            wedisp = np.sum(wedisp,axis=2)/exp[np.newaxis,:]
            wedisp /= np.sum(wedisp,axis=1)[:,np.newaxis]
            wexp = np.dot(wedisp,weights*exp)/weights
            exph._counts[...] = wexp
        else:
            exph._counts[...] = exp
#        plt.figure()
#        plt.imshow(wedisp,aspect='auto',interpolation='nearest')
#        


#        print exp/wexp

#        exph._counts[...] = exp#np.sum(aeff*lthist.counts[np.newaxis,:],axis=1)
 #       exph._counts = np.sum(aeff,axis=1)
        return exph

    @staticmethod
    def create(irf,ltfile,irf_dir=None):
        irfm = IRFManager.create(irf,True,irf_dir)
        ltc = LTCube(ltfile)
        return ExposureCalc(irfm,ltc)


if __name__ == '__main__':

    import sys
    import matplotlib.pyplot as plt

    ltc = LTCube(sys.argv[1])

    h = ltc.get_src_lthist(0,0)
    
    h.plot()

    egy_edges = np.linspace(1.0,5.0,4.0/0.25)


    expcalc = ExposureCalc.create('P7SOURCE_V6MC',sys.argv[1])

    expcalc.eval(0,0,egy_edges)

    plt.show()

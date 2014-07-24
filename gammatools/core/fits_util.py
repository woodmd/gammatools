"""
@file  fits_util.py

@brief Various utility classes for manipulating FITS data.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood"
__date__     = "01/01/2014"

import re
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import pywcsgrid2
import pywcsgrid2.allsky_axes
import pywcs
import numpy as np
from gammatools.core.algebra import Vector3D
from gammatools.fermi.catalog import *
from gammatools.core.util import *
from gammatools.core.histogram import *


from pywcsgrid2.allsky_axes import make_allsky_axes_from_header

def load_ds9_cmap():
    # http://tdc-www.harvard.edu/software/saoimage/saoimage.color.html
    ds9_b = {
        'red'   : [[0.0 , 0.0 , 0.0], 
                   [0.25, 0.0 , 0.0], 
                   [0.50, 1.0 , 1.0], 
                   [0.75, 1.0 , 1.0], 
                   [1.0 , 1.0 , 1.0]],
        'green' : [[0.0 , 0.0 , 0.0], 
                   [0.25, 0.0 , 0.0], 
                   [0.50, 0.0 , 0.0], 
                   [0.75, 1.0 , 1.0], 
                   [1.0 , 1.0 , 1.0]],
        'blue'  : [[0.0 , 0.0 , 0.0], 
                   [0.25, 1.0 , 1.0], 
                   [0.50, 0.0 , 0.0], 
                   [0.75, 0.0 , 0.0], 
                   [1.0 , 1.0 , 1.0]]
        }
     
    plt.register_cmap(name='ds9_b', data=ds9_b) 
    plt.cm.ds9_b = plt.cm.get_cmap('ds9_b')
    return plt.cm.ds9_b

def get_circle(ra,dec,rad_deg,n=100):
    th = np.linspace(np.radians(rad_deg),
                     np.radians(rad_deg),n)
    phi = np.linspace(0,2*np.pi,n)

    v = Vector3D.createThetaPhi(th,phi)

    v.rotatey(np.pi/2.-np.radians(dec))
    v.rotatez(np.radians(ra))

    return np.degrees(v.lon()), np.degrees(v.lat())

class FITSAxis(Axis):

    def __init__(self,ctype,crpix,crval,cdelt,naxis,logaxis=False,offset=0.0):

        self._type = ctype
        self._crpix = crpix
        self._crval = crval
        self._delta = cdelt
        self._naxis = naxis
        if logaxis:
            self._delta = np.log10((self._crval+self._delta)/self._crval)
            self._crval = np.log10(self._crval)

#        xmin = self._crval+(self._crpix-offset)*self._delta
#        xmax = self._crval+(self._crpix+self._naxis-offset)*self._delta
#        if edges is None:
            
        edges = np.linspace(0.0,self._naxis,self._naxis+1)
        
        if re.search('GLON',self._type) is None and \
                re.search('GLAT',self._type) is None:
            self._coordsys = 'cel'
        else:
            self._coordsys = 'gal'

        super(FITSAxis, self).__init__(edges,label=ctype)
            

    def naxis(self):
        return self._naxis
    
    def pix_to_coord(self,p):
        return self._crval + (p-self._crpix)*self._delta

    def coord_to_pix(self,x):
        return self._crpix + (x-self._crval)/self._delta

    def coord_to_index(self,x):
        pix = self.coord_to_pix(x)
        index = np.round(pix)
        return index

    @staticmethod
    def create_from_axis(ctype,axis):
        return FITSAxis(ctype,0,axis.lo_edge(),axis.width()[0],axis.nbins())
    
    @staticmethod
    def create_from_header(header,iaxis,logaxis=False,offset=0.0):
        return FITSAxis(header.get('CTYPE'+str(iaxis+1)),
                        header.get('CRPIX'+str(iaxis+1))-1,
                        header.get('CRVAL'+str(iaxis+1)),
                        header.get('CDELT'+str(iaxis+1)),
                        header.get('NAXIS'+str(iaxis+1)),
                        logaxis,offset)
    
    @staticmethod
    def create_axes(header):

        if 'NAXIS' in header: naxis = header.get('NAXIS')
        elif 'WCSAXES' in header: naxis = header.get('WCSAXES')
        
        axes = []
        for i in range(naxis):
            
            ctype = header.get('CTYPE'+str(i+1))
            if ctype == 'Energy' or ctype == 'photon energy':
                axis = FITSAxis.create_from_header(header,i,logaxis=True)
            else:
                axis = FITSAxis.create_from_header(header,i)

            axes.append(axis)

        return axes


class FITSImage(HistogramND):
    """Base class for SkyImage and SkyCube classes.  Handles common
    functionality for performing sky to pixel coordinate conversions."""
    
    def __init__(self,wcs,axes,counts=None,roi_radius_deg=180.,roi_msk=None):
        super(FITSImage, self).__init__(axes,counts=counts,var=counts)
        
        self._wcs = wcs
        self._roi_radius_deg = roi_radius_deg
        self._header = self._wcs.to_header(True)
        
        self._lon = self._header['CRVAL1']
        self._lat = self._header['CRVAL2']
        
        self._roi_msk = np.empty(shape=self._counts.shape[:2],dtype=bool)
        self._roi_msk.fill(False)
        
        if not roi_msk is None: self._roi_msk |= roi_msk
        
        xpix, ypix = np.meshgrid(self.axis(0).center(),self.axis(1).center())
        xpix = np.ravel(xpix)
        ypix = np.ravel(ypix)
        
        self._pix_lon, self._pix_lat = self._wcs.wcs_pix2sky(xpix,ypix, 0)

        self.add_roi_msk(self._lon,self._lat,roi_radius_deg,True,
                         self.axis(1)._coordsys)

    def __getnewargs__(self):

        self._wcs = pywcs.WCS(self._header)
        return ()
#        return (self._wcs,self._counts,self._ra,self._dec,self._roi_radius)
        
    def add_roi_msk(self,lon,lat,rad,invert=False,coordsys='cel'):
        
        v0 = Vector3D.createLatLon(np.radians(self._pix_lat),
                                   np.radians(self._pix_lon))
        
        if self._axes[0]._coordsys == 'gal' and coordsys=='cel':
            lon,lat = eq2gal(lon,lat)
        elif self._axes[0]._coordsys == 'cel' and coordsys=='gal':
            lon,lat = gal2eq(lon,lat)
            
        v1 = Vector3D.createLatLon(np.radians(lat),np.radians(lon))

        dist = np.degrees(v0.separation(v1))
        dist = dist.reshape(self._counts.shape[:2])
        
        if not invert: self._roi_msk[dist<rad] = True
        else: self._roi_msk[dist>rad] = True

    def slice(self,sdims,dim_index):

        h = HistogramND.slice(self,sdims,dim_index)
        if h.ndim() == 3:
            return SkyCube(self._wcs,h.axes(),h.counts(),
                           self._roi_radius_deg,self._roi_msk)
        elif h.ndim() == 2:        
            return SkyImage(self._wcs,h.axes(),h.counts(),
                            self._roi_radius_deg,self._roi_msk)
        else:
            return h

    def project(self,pdims,bin_range=None):

        h = HistogramND.project(self,pdims,bin_range)
        if h.ndim() == 2:
            return SkyImage(self._wcs,h.axes(),h.counts(),
                            self._roi_radius_deg,self._roi_msk)
        else:
            h._axes[0] = Axis(h.axis().pix_to_coord(h.axis().edges()))
            return h

    def marginalize(self,mdims,bin_range=None):

        mdims = np.array(mdims,ndmin=1,copy=True)
        pdims = np.setdiff1d(self._dims,mdims)
        return self.project(pdims,bin_range)

    @property
    def lat(self):
        return self._lat

    @property
    def lon(self):
        return self._lon

    @property
    def roi_radius(self):
        return self._roi_radius_deg

    @property
    def wcs(self):
        return self._wcs
    
    @staticmethod
    def createFromHDU(hdu):
        """Create an SkyCube or SkyImage object from a FITS HDU."""
        header = hdu.header

        if header['NAXIS'] == 3: return SkyCube.createFromHDU(hdu)
        elif header['NAXIS'] == 2: return SkyImage.createFromHDU(hdu)
        else:
            print 'Wrong number of axes.'
            sys.exit(1)
        
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        """ """
        hdulist = pyfits.open(fitsfile)
        return FITSImage.createFromHDU(hdulist[ihdu])
    
class SkyCube(FITSImage):
    """Container class for a FITS counts cube with two space
    dimensions and one energy dimension."""
    
    def __init__(self,wcs,axes,counts=None,roi_radius_deg=180.,roi_msk=None):
        super(SkyCube, self).__init__(wcs,axes,counts,roi_radius_deg,roi_msk)
        
    def get_spectrum(self,lon,lat):

        xy = self._wcs.wcs_sky2pix(lon,lat, 0)
        ilon = np.round(xy[0][0])
        ilat = np.round(xy[1][0])

        ilon = min(max(0,ilon),self._axes[0]._naxis-1)
        ilat = min(max(0,ilat),self._axes[1]._naxis-1)

        c = self._counts.T[ilon,ilat,:]
        edges = self._axes[2].edges()
        return Histogram.createFromArray(edges,c)

    def plot_energy_slices(self,rebin=4,logz=False):

        frame_per_fig = 1
        nx = 1
        ny = 1

        plt.figure()
        
        images = self.get_energy_slices(rebin)
        for i, im in enumerate(images):
            subplot = '%i%i%i'%(nx,ny,i%frame_per_fig+1)
            im.plot(subplot=subplot,logz=logz)
        
    def energy_slice(self,ibin):

        counts = np.sum(self._counts[ibin:ibin+1],axis=0)
        return SkyImage(self._wcs,self._axes[:2],counts)
                        
    def get_integrated_map(self,emin,emax):
        
        ebins = self._axes[2].edges()

        loge = 0.5*(ebins[1:] + ebins[:-1])
        dloge = ebins[1:] - ebins[:-1]

        imin = np.argmin(np.abs(emin-ebins))
        imax = np.argmin(np.abs(emax-ebins))
        edloge = 10**loge[imin:imax+1]*dloge[imin:imax+1]

        counts = np.sum(self._counts[imin:imax+1].T*edloge*np.log(10.),
                        axis=2)

        return SkyImage(self._wcs,self._axes[:2],counts)

    def fill(self,lon,lat,loge):

        pixcrd = self._wcs.wcs_sky2pix(lon,lat, 0)
        ecrd = self._axes[2].coord_to_pix(loge)
        super(SkyCube,self).fill(np.vstack((pixcrd[0],pixcrd[1],ecrd)))

    @staticmethod
    def createFromHDU(hdu):
        
        header = hdu.header
        wcs = pywcs.WCS(header,naxis=[1,2],relax=True)
#        wcs1 = pywcs.WCS(header,naxis=[3])
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        return SkyCube(wcs,axes,copy.deepcopy(hdu.data.astype(float).T))
        
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        
        hdulist = pyfits.open(fitsfile)        
        header = hdulist[ihdu].header
        wcs = pywcs.WCS(header,naxis=[1,2],relax=True)

        
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        return SkyCube(wcs,axes,
                       copy.deepcopy(hdulist[ihdu].data.astype(float).T))

    @staticmethod
    def createFromTree(tree,lon,lat,lon_var,lat_var,egy_var,roi_radius_deg,
                       energy_axis,cut='',bin_size_deg=0.2,coordsys='cel'):

        im = SkyCube.createROI(lon,lat,roi_radius_deg,energy_axis,
                               bin_size_deg,coordsys)        
        im.fill(get_vector(tree,lon_var,cut=cut),
                get_vector(tree,lat_var,cut=cut),
                get_vector(tree,egy_var,cut=cut))
        return im
    
    @staticmethod
    def createROI(ra,dec,roi_radius_deg,energy_axis,
                  bin_size_deg=0.2,coordsys='cel'):

        nbin = np.ceil(2.0*roi_radius_deg/bin_size_deg)
        
        wcs = SkyImage.createWCS(ra,dec,roi_radius_deg,bin_size_deg,coordsys)
        header = wcs.to_header(True)
        header['NAXIS1'] = nbin
        header['NAXIS2'] = nbin
        axes = FITSAxis.create_axes(header)
        axes.append(FITSAxis.create_from_axis('Energy',energy_axis))
        return SkyCube(wcs,axes,roi_radius_deg=roi_radius_deg)
    
class SkyImage(FITSImage):

    def __init__(self,wcs,axes,counts,roi_radius_deg=180.,roi_msk=None):
        super(SkyImage, self).__init__(wcs,axes,counts,roi_radius_deg,roi_msk)

        self._ax = None
        
    @staticmethod
    def createFromTree(tree,lon,lat,lon_var,lat_var,roi_radius_deg,cut='',
                       bin_size_deg=0.2,coordsys='cel'):

        im = SkyImage.createROI(lon,lat,roi_radius_deg,bin_size_deg,coordsys) 
        im.fill(get_vector(tree,lon_var,cut=cut),
                get_vector(tree,lat_var,cut=cut))
        return im

    @staticmethod
    def createFromHDU(hdu):
        
        header = hdu.header
        wcs = pywcs.WCS(header,relax=True)
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        
        return SkyImage(wcs,axes,copy.deepcopy(hdu.data.astype(float).T))
    
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        
        hdulist = pyfits.open(fitsfile)
        return SkyImage.createFromFITS(hdulist[ihdu])

    @staticmethod
    def createWCS(ra,dec,roi_radius_deg,bin_size_deg=0.2,coordsys='cel'):
        nbin = np.ceil(2.0*roi_radius_deg/bin_size_deg)
        deg_to_pix = bin_size_deg
        wcs = pywcs.WCS(naxis=2)

        wcs.wcs.crpix = [nbin/2.+0.5, nbin/2.+0.5]
        wcs.wcs.cdelt = np.array([-deg_to_pix,deg_to_pix])
        wcs.wcs.crval = [ra, dec]
        
        if coordsys == 'cel': wcs.wcs.ctype = ["RA---AIT", "DEC--AIT"]
        else: wcs.wcs.ctype = ["GLON-AIT", "GLAT-AIT"]            
        wcs.wcs.equinox = 2000.0
        return wcs
                
    @staticmethod
    def createROI(ra,dec,roi_radius_deg,bin_size_deg=0.2,coordsys='cel'):
        nbin = np.ceil(2.0*roi_radius_deg/bin_size_deg)
        wcs = SkyImage.createWCS(ra,dec,roi_radius_deg,bin_size_deg,coordsys)

        header = wcs.to_header(True)
        header['NAXIS1'] = nbin
        header['NAXIS2'] = nbin
        
        axes = FITSAxis.create_axes(header)
        im = SkyImage(wcs,axes,np.zeros(shape=(nbin,nbin)),roi_radius_deg)
        return im

#        lon, lat = get_circle(ra,dec,roi_radius_deg)
#        xy =  wcs.wcs_sky2pix(lon, lat, 0)

#        xmin = np.min(xy[0])
#        xmax = np.max(xy[0])

#        if roi_radius_deg >= 90.:
#            xypole0 = wcs.wcs_sky2pix(0.0, -90.0, 0)
#            xypole1 = wcs.wcs_sky2pix(0.0, 90.0, 0)
#            ymin = xypole0[1]
#            ymax = xypole1[1]
#        else:
#            ymin = np.min(xy[1])
#            ymax = np.max(xy[1])

    
    def ax(self):
        return self._ax
        
    def fill(self,lon,lat,w=1.0):

        pixcrd = self._wcs.wcs_sky2pix(lon,lat, 0)
        super(SkyImage,self).fill(np.vstack((pixcrd[0],pixcrd[1])),w)

    def interpolate(self,lon,lat):
        
        pixcrd = self._wcs.wcs_sky2pix(lon, lat, 0)
        return interpolate2d(self._xedge,self._yedge,self._counts,
                             *pixcrd)

    def center(self):
        pixcrd = super(SkyImage,self).center()
        skycrd = self._wcs.wcs_pix2sky(pixcrd[0], pixcrd[1], 0)

        return np.vstack((skycrd[0],skycrd[1]))

    def smooth(self,sigma,compute_var=False,summed=False):
        sigma /= np.abs(self._axes[0]._delta)
        
        from scipy import ndimage
        im = SkyImage(copy.deepcopy(self.wcs),
                      copy.deepcopy(self.axes()),
                      copy.deepcopy(self._counts),
                      self.roi_radius,
                      copy.deepcopy(self._roi_msk))

        # Construct a kernel
        nk =41
        fn = lambda t, s: 1./(2*np.pi*s**2)*np.exp(-t**2/(s**2*2.0))
        b = np.abs(np.linspace(0,nk-1,nk) - (nk-1)/2.)
        k = np.zeros((nk,nk)) + np.sqrt(b[np.newaxis,:]**2 +
                                        b[:,np.newaxis]**2)
        k = fn(k,sigma)
        k /= np.sum(k)

        im._counts = ndimage.convolve(self._counts,k,mode='nearest')
        
#        im._counts = ndimage.gaussian_filter(self._counts, sigma=sigma,
#                                             mode='nearest')

        if compute_var:
            var = ndimage.convolve(self._counts, k**2, mode='wrap')
            im._var = var
        else:
            im._var = np.zeros(im._counts.shape)
            
        if summed: im /= np.sum(k**2)
            
        return im

    def plot_marker(self,lonlat=None,**kwargs):

        if lonlat is None: lon, lat = (self._lon,self._lat)
        xy =  self._wcs.wcs_sky2pix(lon,lat, 0)
        self._ax.plot(xy[0],xy[1],**kwargs)

        plt.gca().set_xlim(self.axis(0).lo_edge(),self.axis(0).hi_edge())
        plt.gca().set_ylim(self.axis(1).lo_edge(),self.axis(1).hi_edge()) 
    
    def plot_circle(self,rad_deg,radec=None,**kwargs):

        if radec is None: radec = (self._lon,self._lat)

        lon,lat = get_circle(radec[0],radec[1],rad_deg)
        xy =  self._wcs.wcs_sky2pix(lon,lat, 0)
        self._ax.plot(xy[0],xy[1],**kwargs)

        self._ax.set_xlim(self.axis(0).lo_edge(),self.axis(0).hi_edge())
        self._ax.set_ylim(self.axis(1).lo_edge(),self.axis(1).hi_edge())    

        
    def plot(self,subplot=111,logz=False,catalog=None,
             cmap='jet',**kwargs):

        from matplotlib.colors import NoNorm, LogNorm, Normalize

        kwargs_contour = { 'levels' : None, 'colors' : ['k'],
                           'linewidths' : None,
                           'origin' : 'lower' }
        
        kwargs_imshow = { 'interpolation' : 'nearest',
                          'origin' : 'lower','norm' : None,
                          'vmin' : None, 'vmax' : None }
#               'extent' : [self.axis(0).edges()[0],
#                           self.axis(0).edges()[-1],
#                           self.axis(1).edges()[0],
#                           self.axis(1).edges()[-1]] }
#                   'vmin' : vmin }

        
        if logz: kwargs_imshow['norm'] = LogNorm()
        else: kwargs_imshow['norm'] = Normalize()
        
        ax = pywcsgrid2.subplot(subplot, header=self._wcs.to_header())
#        ax = pywcsgrid2.axes(header=self._wcs.to_header())

        load_ds9_cmap()
        colormap = mpl.cm.get_cmap(cmap)
        colormap.set_under('white')

        counts = copy.copy(self._counts)
        
        if np.any(self._roi_msk):        
            kwargs_imshow['vmin'] = 0.8*np.min(self._counts[~self._roi_msk.T])
            counts[self._roi_msk.T] = -np.inf
        
#        vmax = np.max(self._counts[~self._roi_msk])
#        c = self._counts[~self._roi_msk]        
#        if logz: vmin = np.min(c[c>0])

        update_dict(kwargs_imshow,kwargs)
        update_dict(kwargs_contour,kwargs)
                   
        im = ax.imshow(counts.T,**kwargs_imshow)
        im.set_cmap(colormap)

        if kwargs_contour['levels']:        
            cs = ax.contour(counts.T,**kwargs_contour)
        #        plt.clabel(cs, fontsize=5, inline=0)
        
#        im.set_clim(vmin=np.min(self._counts[~self._roi_msk]),
#                    vmax=np.max(self._counts[~self._roi_msk]))
        
        ax.set_ticklabel_type("d", "d")

        if self._axes[0]._coordsys == 'gal':
            ax.set_xlabel('GLON')
            ax.set_ylabel('GLAT')
        else:        
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')

#        plt.colorbar(im,orientation='horizontal',shrink=0.7,pad=0.15,
#                     fraction=0.05)
        ax.grid()

        if catalog:
            cat = Catalog.get(catalog)

            kwargs_cat = {'src_color' : 'k' }
            if cmap == 'ds9_b': kwargs_cat['src_color'] = 'w'

            cat.plot(self,ax=ax,**kwargs_cat)
        
#        ax.add_compass(loc=1)
#        ax.set_display_coord_system("gal")       
 #       ax.locator_params(axis="x", nbins=12)

        self._ax = ax
        
        return im

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
from pywcsgrid2.allsky_axes import make_allsky_axes_from_header
#import astropy.wcs as pywcs
from astropy_helper import pywcs
from astropy_helper import pyfits
#from astropy.io.fits.header import Header
import numpy as np
import healpy as hp
from gammatools.core.algebra import Vector3D
from gammatools.fermi.catalog import *
from gammatools.core.util import *
from gammatools.core.histogram import *

def bintable_to_array(data):

    return copy.deepcopy(data.view((data.dtype[0], len(data.dtype.names))))

def stack_images(files,output_file,hdu_index=0):

    hdulist0 = None
    for i, f in enumerate(files):
        hdulist = pyfits.open(f)
        if i == 0: hdulist0 = hdulist
        else:
            hdulist0[hdu_index].data += hdulist[hdu_index].data
    hdulist0.writeto(output_file,clobber=True)

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
        self._coordsys = None
        self._sky_coord = False
        if logaxis:
            self._delta = np.log10((self._crval+self._delta)/self._crval)
            self._crval = np.log10(self._crval)
            
        if np.fmod(crpix,1.0):
            edges = np.linspace(0.0,self._naxis,self._naxis+1) - 0.5
        else:
            edges = np.linspace(0.0,self._naxis,self._naxis+1) 

        if re.search('GLON',self._type) or re.search('GLAT',self._type):
            self._coordsys = 'gal'
            self._sky_coord = True
        elif re.search('RA',self._type) or re.search('DEC',self._type):
            self._coordsys = 'cel'
            self._sky_coord = True

        super(FITSAxis, self).__init__(edges,label=ctype)
            
    @property
    def naxis(self):
        return self._naxis
    
    @property
    def type(self):
        return self._type

    def to_axis(self,apply_crval=True):
        return Axis(self.pix_to_coord(self.edges,apply_crval))
    
    def pix_to_coord(self,p,apply_crval=True):
        """Convert from FITS pixel coordinates to projected sky
        coordinates."""
        if apply_crval:
            return self._crval + (p-self._crpix)*self._delta
        else:
            return (p-self._crpix)*self._delta

    def coord_to_pix(self,x,apply_crval=True):

        if apply_crval:
            return self._crpix + (x-self._crval)/self._delta 
        else:
            return self._crpix + x/self._delta 

    def coord_to_index(self,x):
        pix = self.coord_to_pix(x)
        index = np.round(pix)
        return index

    @staticmethod
    def create_from_axis(ctype,axis):
        return FITSAxis(ctype,0,axis.lo_edge(),axis.width[0],axis.nbins)
    
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


class HealpixImage(HistogramND):    
    """Base class for 2-D and 3-D HEALPix sky maps."""
    def __init__(self,axes,hp_axis_index=0,counts=None,var=None,
                 coordsys='CEL'):
        super(HealpixImage, self).__init__(axes,counts=counts,var=var)

        self._hp_axis_index = hp_axis_index
        self._hp_axis = self.axes()[hp_axis_index]
        self._nside = hp.npix2nside(self._hp_axis.nbins)
        self._nest=False
        self._coordsys=coordsys
        self._mask = np.empty(self.counts.shape,dtype='bool');
        self._mask.fill(True)

    @property
    def nside(self):
        return self._nside

    @property
    def nest(self):
        return self._nest

    @property
    def coordsys(self):
        return self._coordsys

    def create_mask(self,**kwargs):

        cuts = kwargs.get('cuts',[])
        mask = kwargs.get('mask',None)        
        if mask is None:
            mask = np.empty(self.counts.shape,dtype='bool'); mask.fill(True)
        
        for cut in cuts:
            self.create_lonlat_mask(mask=mask,**cut)

        return mask
        
    def create_lonlat_mask(self,**kwargs):

        norm_angle = lambda t: ( t + 180.) % (2 * 180. ) - 180.
        
        mask = kwargs.get('mask',None)
        coordsys = kwargs.get('coordsys',self.coordsys)
        complement = kwargs.get('complement',False)
        latrange = kwargs.get('latrange',None)
        lonrange = kwargs.get('lonrange',None)
        
        if mask is None:
            mask = np.empty(self.counts.shape,dtype='bool');
            mask.fill(True)
        
        c = self.center()
        
        if self.ndim() == 2:
            lon,lat = c[1],c[2]
        else:
            lon,lat = c[0],c[1]
        
        if coordsys == 'GAL' and (self.coordsys == 'EQU' or self.coordsys == 'CEL'):            
            lon, lat = eq2gal(lon,lat)
        elif coordsys == 'CEL' and self.coordsys == 'GAL':
            lon, lat = gal2eq(lon,lat)

        if latrange is not None:
            m = (lat > latrange[0])&(lat < latrange[1])
            
            if complement: mask &= ~m
            else: mask &= m
            
        if lonrange is not None:

            lon0 = norm_angle(lonrange[0])
            lon1 = norm_angle(lonrange[1])

            if lon0 > lon1:
                m = (norm_angle(lon) > lon0)|(norm_angle(lon) < lon1)
            else:
                m = (norm_angle(lon) > lon0)&(norm_angle(lon) < lon1)

            if complement: mask &= ~m
            else: mask &= m
                
        return mask        
    
    def createFromHist(self,h):
        """Take an input HistogramND object and cast it into a
        HealpixSkyImage if appropriate."""
        
        if h.ndim() == 2:
            return HealpixSkyCube(h.axes(),h.counts)
        else:
            return HealpixSkyImage(h.axes(),h.counts)

    def sliceByLatLon(self,lat,lon):
        ipix = hp.ang2pix(self.nside,np.pi/2.-lat,lon,nest=self.nest)
        return self.slice(1,ipix)

    def slice(self,sdims,dim_index):

        h = HistogramND.slice(self,sdims,dim_index)

        if h.ndim() == 2:
            return HealpixSkyCube(h.axes(),h.counts,h.var,
                                  coordsys=self.coordsys)
        elif h.ndim() == 1 and h.axis(0).nbins == self._hp_axis.nbins:        
            return HealpixSkyImage(h.axes(),h.counts,h.var,
                                   coordsys=self.coordsys)
        else:
            return h

    def project(self,pdims,bin_range=None):
        
        h = HistogramND.project(self,pdims,bin_range)
        return self.createFromHist(h)
        
    def marginalize(self,mdims,bin_range=None):

        mdims = np.array(mdims,ndmin=1,copy=True)
        pdims = np.setdiff1d(self._dims,mdims)        
        return self.project(pdims,bin_range)
        
class HealpixSkyImage(HealpixImage):
    """HEALPix representation of a 2-D sky map."""

    def __init__(self,axes,counts=None,var=None,coordsys='CEL'):
        super(HealpixSkyImage, self).__init__(axes,counts=counts,var=var,
                                              coordsys=coordsys)

    @staticmethod
    def create(nside,coordsys='CEL'):
        npix = hp.pixelfunc.nside2npix(nside)
        hp_axis = Axis.create(0,npix,npix) 
        return HealpixSkyImage([hp_axis],0,coordsys=coordsys)
        
    def fill(self,lon,lat,w=1.0):
        ipix = hp.ang2pix(self.nside,lat,lon,nest=self.nest)
        super(HealpixSkyImage,self).fill(ipix,w)

    def interpolate(self,lon,lat):

        theta = np.pi/2.-lat        
        return hp.pixelfunc.get_interp_val(self.counts,theta,
                                           lon,nest=self.nest)       
#        pixcrd = self._wcs.wcs_world2pix(lon, lat, 0)
#        return interpolate2d(self._xedge,self._yedge,self._counts,
#                             *pixcrd)

    def center(self):
        """Returns lon,lat of the pixel centers."""

        pixcrd = np.array(self.axes()[0].edges[:-1],dtype=int)
        pixang0, pixang1 = hp.pixelfunc.pix2ang(self.nside,pixcrd)
        
        pixang0 = np.ravel(pixang0)
        pixang1 = np.ravel(pixang1)
        pixang0 = np.pi/2. - pixang0
        
        return (np.degrees(pixang1),np.degrees(pixang0))

    def mask(self,**kwargs):
        
        mask = self.create_mask(**kwargs)
        self._mask = mask
        
    def integrate(self,**kwargs):
        
        mask = self.create_mask(**kwargs)
        return np.sum(self._counts[mask])        
        
    def smooth(self,sigma):

        im = HealpixSkyImage(copy.deepcopy(self.axes()),
                             counts=copy.deepcopy(self._counts),
                             var=copy.deepcopy(self._var),
                             coordsys=self.coordsys)
        
        sc = hp.sphtfunc.smoothing(im.counts,sigma=np.radians(sigma))

        im._counts = sc
        im._var = copy.deepcopy(sc)

        return im

    def draw_graticule_labels(self):

        phi = np.array([-179.999,-150.,-120.,-90.,-60.,-30.,0.0,
                         30.0,60.,90.,120.,150.,179.99])
                       
        phi_labels = ['-180','-150','-120','-90','-60','-30','0',
                      '30','60','90','120','150','180']

        phi = 180.+np.array([-179.999,-120.,-60.,0.0,
                              60.0,120.,179.99])
                       
        phi_labels = ['-180','-120','-60','0',
                      '60','120','180']
                      
  
        th = np.array([-60,-30,30,60])

        import matplotlib.patheffects as path_effects
 
        ef = path_effects.withStroke(foreground="w", linewidth=2)
#        text.set_path_effects([ef])

        for p,l in zip(phi,phi_labels):

            plt.gca().projtext(np.pi/2.,np.radians(p+180.),
                               l,color='k',ha='center',
                               fontsize=12,path_effects=[ef])

        for t in th:
            plt.gca().projtext(np.radians(t)-np.pi/2.,np.pi,
                               '%.0f'%(t),color='k',ha='center',
                               fontsize=12,path_effects=[ef])
                               

                
    def plot(self,**kwargs):

        kwargs_imshow = { 'norm' : None,
                          'vmin' : None, 'vmax' : None }

        gamma = kwargs.get('gamma',2.0)
        zscale = kwargs.get('zscale',None)
        cbar = kwargs.get('cbar',None)
        cbar_label = kwargs.get('cbar_label','')
        title = kwargs.get('title','')
        levels = kwargs.get('levels',None)
        rot = kwargs.get('rot',None)
        
        kwargs_imshow['vmin'] = kwargs.get('vmin',None)
        kwargs_imshow['vmax'] = kwargs.get('vmax',None)

        cmap = mpl.cm.get_cmap(kwargs.get('cmap','jet'))
        cmap.set_under('white')
        kwargs_imshow['cmap'] = cmap

        if zscale == 'pow':
            vmed = np.median(self.counts)
            vmax = max(self.counts)
            vmin = min(1.1*self.counts[self.counts>0])
#            vmin = max(vmed*(vmed/vmax),min(self.counts[self.counts>0]))

            kwargs_imshow['norm'] = PowerNorm(gamma=gamma,clip=True)
        elif zscale == 'log': kwargs_imshow['norm'] = LogNorm()
        else: kwargs_imshow['norm'] = Normalize(clip=True)
        
        from healpy import projaxes as PA
        
        fig = plt.gcf()


        if 'sub' in kwargs:
            sub = kwargs['sub']
            nrows, ncols, idx = sub/100, (sub%100)/10, (sub%10)
            
            c,r = (idx-1)%ncols,(idx-1)/ncols
#            if not margins:
            margins = (0.01,0.0,0.0,0.02)
            extent = (c*1./ncols+margins[0], 
                      1.-(r+1)*1./nrows+margins[1],
                      1./ncols-margins[2]-margins[0],
                      1./nrows-margins[3]-margins[1])
            extent = (extent[0]+margins[0],
                      extent[1]+margins[1],
                      extent[2]-margins[2]-margins[0],
                      extent[3]-margins[3]-margins[1])
        else:
            extent = (0.02,0.05,0.96,0.9)

        ax=PA.HpxMollweideAxes(fig,extent,coord=None,rot=rot,
                               format='%g',flipconv='astro')

        ax.set_title(title)
        fig.add_axes(ax)

        img0 = ax.projmap(self.counts,nest=self.nest,xsize=1600,coord='C',
                          **kwargs_imshow)

        if levels:
            cs = ax.contour(img0,extent=ax.proj.get_extent(),
                            levels=levels,colors=['k'],
                            interpolation='nearest')

        hp.visufunc.graticule(verbose=False,lw=0.5,color='k')

        if cbar is not None:

            im = ax.get_images()[0]

            cb_kw = dict(orientation = 'vertical', 
                         shrink=.6, pad=0.05)
            cb_kw.update(cbar)

            cb = fig.colorbar(im, **cb_kw)#,format='%.3g')

            cb.set_label(cbar_label)
            #, ticks=[min, max])
#            cb.ax.xaxis.set_label_text(cbar_label)

#            if zscale=='pow':
#                gamma = 1./zscale_power
#                ticks = np.linspace(vmin**gamma,
#                                    vmax**gamma,6)**(1./gamma)
#                cb.set_ticks(ticks)

#            cb.ax.xaxis.labelpad = -8
            # workaround for issue with viewers, see colorbar docstring
#            cb.solids.set_edgecolor("face")
    
    def save(self,fitsfile):

        hdu_image = pyfits.PrimaryHDU()
        col = pyfits.Column(name='CHANNEL1', format='D', array=self._counts)
        hdu = pyfits.BinTableHDU.from_columns([col],name='SKYMAP')
        hdulist = pyfits.HDUList([hdu_image,hdu])
        hdulist.writeto(fitsfile,clobber=True)
    
class HealpixSkyCube(HealpixImage):

    def __init__(self,axes,hp_axis_index=1,
                 counts=None,var=None,coordsys='CEL'):
        super(HealpixSkyCube, self).__init__(axes,hp_axis_index,counts,
                                             var,coordsys)

#    def fill(self,lon,lat,loge,w=1.0):
#        ipix = hp.ang2pix(self.nside,lat,lon,nest=self.nest)
#        super(HealpixSkyImage,self).fill(ipix,w)

#    def interpolate(self,lon,lat,loge):
#        ipix = hp.ang2pix(self.nside,lat,lon,nest=self.nest)
#        epix = self.axis(0).valToBin(loge)
#        v0 = hp.pixelfunc.get_interp_val()
#            hp.pixelfunc.get_interp_val()    
#        pixcrd = self._wcs.wcs_world2pix(lon, lat, 0)
#        return interpolate2d(self._xedge,self._yedge,self._counts,
#                             *pixcrd)

    def integrate(self,**kwargs):

        mask = self.create_mask(**kwargs)
                
#        c = self.center()
#        msk = np.empty(len(c[0]),dtype='bool'); msk.fill(True)
#        msk &= (c[2] > latrange[0])&(c[2] < latrange[1])
#        mask = mask.reshape(self.counts.shape)
        
        c = copy.copy(self._counts); c[~mask] = 0
        v = copy.copy(self._var); v[~mask] = 0
        return Histogram(self.axis(0),counts=np.sum(c,axis=1),
                         var=np.sum(v,axis=1))     
        
    def center(self):
        pixcrd = np.array(self.axes()[1].edges[:-1],dtype=int)
        pixang0, pixang1 = hp.pixelfunc.pix2ang(self.nside,pixcrd)

        pixloge = self.axis(0).center[:,np.newaxis]
        pixang0 = pixang0[np.newaxis,:]
        pixang1 = pixang1[np.newaxis,:]
        
        
#        pixloge = np.repeat(pixloge[:,np.newaxis],len(pixang0),axis=1)
#        pixang0 = np.repeat(pixang0[np.newaxis,:],len(pixloge),axis=0)
#        pixang1 = np.repeat(pixang1[np.newaxis,:],len(pixloge),axis=0)

#        pixloge = np.ravel(pixloge)
#        pixang0 = np.ravel(pixang0)
#        pixang1 = np.ravel(pixang1)
        pixang0 = np.pi/2. - pixang0
       
        return (pixloge,np.degrees(pixang1),np.degrees(pixang0))
         
    def ud_grade(self,nside):

        counts = np.vsplit(self._counts,self.shape[0])

        for i in range(len(counts)):
            counts[i] = np.squeeze(counts[i])

        counts = hp.pixelfunc.ud_grade(counts,nside)
#        var = hp.pixelfunc.ud_grade(np.vsplit(self._var,self.shape[0]),nside)
        counts = np.vstack(counts)
        
        npix0 = hp.pixelfunc.nside2npix(nside)
        npix1 = hp.pixelfunc.nside2npix(self.nside)

        counts *= npix1/npix0
        

        hp_axis = Axis.create(0,counts.shape[1],counts.shape[1])
        return HealpixSkyCube([self.axis(0),hp_axis],1,counts)

    def smooth(self,sigma):

        sc = np.zeros(self.counts.shape)

        for i in range(self.axis(0).nbins):
            sc[i,:] = hp.sphtfunc.smoothing(self.counts[i,:],sigma=np.radians(sigma))
            
#        im = HealpixSkyImage(copy.deepcopy(self.axes()),
#                             counts=copy.deepcopy(self._counts),
#                             var=copy.deepcopy(self._var),
#                             coordsys=self.coordsys)
        
        return HealpixSkyCube(copy.deepcopy(self.axes()),1,counts=sc,coordsys=self.coordsys)
    
    @staticmethod
    def create(energy_axis,nside):

        npix = hp.pixelfunc.nside2npix(nside)
        hp_axis = Axis.create(0,npix,npix)        
        return HealpixSkyCube([energy_axis,hp_axis],1)
        
    @staticmethod
    def createFromFITS(fitsfile,image_hdu='SKYMAP',energy_hdu='EBOUNDS'):
        """ """

        hdulist = pyfits.open(fitsfile)        

#        hdulist.info()

        header = hdulist[image_hdu].header
        v = hdulist[image_hdu].data
        coordsys = header.get('COORDSYS','GAL')            
        
        if image_hdu == 'SKYMAP':
            dtype = v.dtype[0]
            image_data = copy.deepcopy(v.view((dtype, len(v.dtype.names)))).T
        else:
            image_data = copy.deepcopy(v)
        #np.array(hdulist[image_hdu].data).astype(float)

        if energy_hdu == 'EBOUNDS':
            ebounds = hdulist[energy_hdu].data
            nbin = len(ebounds)        
            emin = ebounds[0][1]/1E3
            emax = ebounds[-1][2]/1E3
            delta = np.log10(emax/emin)/nbin
            energy_axis = Axis.create(np.log10(emin),np.log10(emax),nbin)
        elif energy_hdu == 'ENERGIES':
            energies = bintable_to_array(hdulist[energy_hdu].data)
            energy_axis = Axis.createFromArray(np.log10(energies))
        else:
            raise Exception('Unknown HDU name.')

        hp_axis = Axis.create(0,image_data.shape[1],image_data.shape[1])
        return HealpixSkyCube([energy_axis,hp_axis],1,image_data,
                              coordsys=coordsys)

    def save(self,fitsfile):

        hdu_image = pyfits.PrimaryHDU(self._counts)

        ecol = pyfits.Column(name='ENERGIES', format='D', 
                             array=10**self.axis(0).center)
#        cols = pyfits.ColDefs([ecol])
        hdu_energies = pyfits.BinTableHDU.from_columns([ecol],name='ENERGIES')
        hdulist = pyfits.HDUList([hdu_image,hdu_energies])

        hdulist.writeto(fitsfile,clobber=True)

class FITSImage(HistogramND):
    """Base class for SkyImage and SkyCube classes.  Handles common
    functionality for performing sky to pixel coordinate conversions."""
    
    def __init__(self,wcs,axes,counts=None,roi_radius_deg=180.,roi_msk=None):
        super(FITSImage, self).__init__(axes,counts=counts,
                                        var=copy.deepcopy(counts))
        
        self._wcs = wcs
        self._roi_radius_deg = roi_radius_deg
        self._header = self._wcs.to_header(True)
        
        self._lon = self._header['CRVAL1']
        self._lat = self._header['CRVAL2']
        
        self._roi_msk = np.empty(shape=self._counts.shape[:2],dtype=bool)
        self._roi_msk.fill(False)
        
        if not roi_msk is None: self._roi_msk |= roi_msk
        
        xpix, ypix = np.meshgrid(self.axis(0).center,self.axis(1).center)
        xpix = np.ravel(xpix)
        ypix = np.ravel(ypix)
        
#        self._pix_lon, self._pix_lat = self._wcs.wcs_pix2sky(xpix,ypix, 0)
        self._pix_lon, self._pix_lat = self._wcs.wcs_pix2world(xpix,ypix, 0)

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

        h = HistogramND.sliceByValue(self,sdims,dim_index)
        if h.ndim() == 3:
            return SkyCube(self._wcs,h.axes(),h.counts,
                           self._roi_radius_deg,self._roi_msk)
        elif h.ndim() == 2:        
            return SkyImage(self._wcs,h.axes(),h.counts,
                            self._roi_radius_deg,self._roi_msk)
        else:
            h._axes[0] = Axis(h.axis().pix_to_coord(h.axis().edges))
            return h
        
    def sliceByValue(self,sdims,dim_coord):

        sdims = np.array(sdims,ndmin=1,copy=True)
        dim_coord = np.array(dim_coord,ndmin=1,copy=True)

        if 0 in sdims and 1 in sdims:        
            dim_coord = self._wcs.wcs_world2pix(dim_coord[0],dim_coord[1],0)
            
        h = HistogramND.slice(self,sdims,dim_coord)
        if h.ndim() == 3:
            return SkyCube(self._wcs,h.axes(),h.counts,
                           self._roi_radius_deg,self._roi_msk)
        elif h.ndim() == 2:        
            return SkyImage(self._wcs,h.axes(),h.counts,
                            self._roi_radius_deg,self._roi_msk)
        else:
            h._axes[0] = Axis(h.axis().pix_to_coord(h.axis().edges))
            return h

    def project(self,pdims,bin_range=None,offset_coord=False):

        h = HistogramND.project(self,pdims,bin_range)
        return self.createFromHist(h,offset_coord=offset_coord)
        
    def marginalize(self,mdims,bin_range=None,offset_coord=False):

        mdims = np.array(mdims,ndmin=1,copy=True)
        pdims = np.setdiff1d(self._dims,mdims)
        return self.project(pdims,bin_range,offset_coord=offset_coord)

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
    
    def createFromHist(self,h,offset_coord=False):
        """Take an input HistogramND object and cast it into a
        SkyImage if appropriate."""
        
        if h.ndim() == 2:

            if h.axis(0)._sky_coord and h.axis(1)._sky_coord:
                return SkyImage(self._wcs,h.axes(),h.counts,
                                self._roi_radius_deg,self._roi_msk)
            else:
                axis0 = Axis(h.axis(0).pix_to_coord(h.axis(0).edges,not offset_coord))
                axis1 = Axis(h.axis(1).pix_to_coord(h.axis(1).edges,not offset_coord))
                
                h._axes[0] = axis0
                h._axes[1] = axis1
                return h
        else:
            h._axes[0] = Axis(h.axis().pix_to_coord(h.axis().edges, not offset_coord))
            return h

    @staticmethod
    def createFromHDU(hdu):
        """Create an SkyCube or SkyImage object from a FITS HDU."""
        header = hdu.header

        if header['NAXIS'] == 3: return SkyCube.createFromHDU(hdu)
        elif header['NAXIS'] == 2: return SkyImage.createFromHDU(hdu)
        else:
            raise Exception('Wrong number of axes.')
        
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        """ """
        hdu = pyfits.open(fitsfile)[ihdu]
        header = hdu.header

        if header['NAXIS'] == 3: return SkyCube.createFromFITS(fitsfile,ihdu)
        elif header['NAXIS'] == 2: return SkyImage.createFromFITS(fitsfile,ihdu)
        else:
            raise Exception('Wrong number of axes.')
    
class SkyCube(FITSImage):
    """Container class for a FITS counts cube with two space
    dimensions and one energy dimension."""
    
    def __init__(self,wcs,axes,counts=None,roi_radius_deg=180.,roi_msk=None):
        super(SkyCube, self).__init__(wcs,axes,counts,roi_radius_deg,roi_msk)
        
    def get_spectrum(self,lon,lat):

        xy = self._wcs.wcs_world2pix(lon,lat, 0)
        ilon = np.round(xy[0][0])
        ilat = np.round(xy[1][0])

        ilon = min(max(0,ilon),self._axes[0]._naxis-1)
        ilat = min(max(0,ilat),self._axes[1]._naxis-1)

        c = self._counts.T[ilon,ilat,:]
        edges = self._axes[2].edges
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
        
        ebins = self._axes[2].edges

        loge = 0.5*(ebins[1:] + ebins[:-1])
        dloge = ebins[1:] - ebins[:-1]

        imin = np.argmin(np.abs(emin-ebins))
        imax = np.argmin(np.abs(emax-ebins))
        edloge = 10**loge[imin:imax+1]*dloge[imin:imax+1]

        counts = np.sum(self._counts[imin:imax+1].T*edloge*np.log(10.),
                        axis=2)

        return SkyImage(self._wcs,self._axes[:2],counts)

    def fill(self,lon,lat,loge):

        pixcrd = self._wcs.wcs_world2pix(lon,lat, 0)
        ecrd = self._axes[2].coord_to_pix(loge)
        super(SkyCube,self).fill(np.vstack((pixcrd[0],pixcrd[1],ecrd)))

    def interpolate(self,lon,lat,loge):
        pixcrd = self._wcs.wcs_world2pix(lon,lat, 0)
        ecrd = np.array(self._axes[2].coord_to_pix(loge),ndmin=1)
        return super(SkyCube,self).interpolate(pixcrd[0],pixcrd[1],ecrd)
        
    def createHEALPixMap(self,nside=4,energy_axis=None):

        # Convert this projection to HP map
#        energy_axis = Axis.create(np.log10(emin),np.log10(emax),nbin)

        if energy_axis is None:
            energy_axis = Axis(self.axis(2).pix_to_coord(self.axis(2).edges,True))

        hp = HealpixSkyCube.create(energy_axis,nside)
        c = hp.center()
        counts = self.interpolate(c[1],c[2],c[0]).reshape(hp.counts.shape)
        hp._counts = counts

        return hp

    @staticmethod
    def createFromHDU(hdu):
        
        header = pyfits.Header.fromstring(hdu.header.tostring())
#        header = hdu.header

        wcs = pywcs.WCS(header,naxis=[1,2])#,relax=True)
#        wcs1 = pywcs.WCS(header,naxis=[3])
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        return SkyCube(wcs,axes,copy.deepcopy(hdu.data.astype(float).T))
        
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        
        hdulist = pyfits.open(fitsfile)        
        header = hdulist[ihdu].header
        wcs = pywcs.WCS(header,naxis=[1,2],relax=True)

        if hdulist[1].name == 'ENERGIES':
            v = bintable_to_array(hdulist[1].data)
            v = np.log10(v)
            energy_axis = Axis.createFromArray(v)
            axes = copy.deepcopy(FITSAxis.create_axes(header))
            axes[2]._crval = energy_axis.edges[0]
            axes[2]._delta = energy_axis.width[0]
            axes[2]._crpix = 0.0
        else:        
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
#        xy =  wcs.wcs_world2pix(lon, lat, 0)

#        xmin = np.min(xy[0])
#        xmax = np.max(xy[0])

#        if roi_radius_deg >= 90.:
#            xypole0 = wcs.wcs_world2pix(0.0, -90.0, 0)
#            xypole1 = wcs.wcs_world2pix(0.0, 90.0, 0)
#            ymin = xypole0[1]
#            ymax = xypole1[1]
#        else:
#            ymin = np.min(xy[1])
#            ymax = np.max(xy[1])

    
    def ax(self):
        return self._ax
        
    def fill(self,lon,lat,w=1.0):

        pixcrd = self._wcs.wcs_world2pix(lon,lat, 0)
        super(SkyImage,self).fill(np.vstack((pixcrd[0],pixcrd[1])),w)

    def interpolate(self,lon,lat):
        
        pixcrd = self._wcs.wcs_world2pix(lon, lat, 0)
        return interpolate2d(self._xedge,self._yedge,self._counts,
                             *pixcrd)

    def center(self):
        pixcrd = super(SkyImage,self).center()
        skycrd = self._wcs.wcs_pix2sky(pixcrd[0], pixcrd[1], 0)

        return np.vstack((skycrd[0],skycrd[1]))

    def smooth(self,sigma,compute_var=False,summed=False):

        sigma /= 1.5095921854516636        
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
        xy =  self._wcs.wcs_world2pix(lon,lat, 0)
        self._ax.plot(xy[0],xy[1],**kwargs)

        plt.gca().set_xlim(self.axis(0).lo_edge(),self.axis(0).hi_edge())
        plt.gca().set_ylim(self.axis(1).lo_edge(),self.axis(1).hi_edge()) 
    
    def plot_circle(self,rad_deg,radec=None,**kwargs):

        if radec is None: radec = (self._lon,self._lat)

        lon,lat = get_circle(radec[0],radec[1],rad_deg)
        xy =  self._wcs.wcs_world2pix(lon,lat, 0)
        self._ax.plot(xy[0],xy[1],**kwargs)

        self._ax.set_xlim(self.axis(0).lo_edge(),self.axis(0).hi_edge())
        self._ax.set_ylim(self.axis(1).lo_edge(),self.axis(1).hi_edge())    

        
    def plot(self,subplot=111,catalog=None,cmap='jet',**kwargs):

        from matplotlib.colors import NoNorm, LogNorm, Normalize

        kwargs_contour = { 'levels' : None, 'colors' : ['k'],
                           'linewidths' : None,
                           'origin' : 'lower' }
        
        kwargs_imshow = { 'interpolation' : 'nearest',
                          'origin' : 'lower','norm' : None,
                          'vmin' : None, 'vmax' : None }

        zscale = kwargs.get('zscale',None)
        gamma = kwargs.get('gamma',0.5)
        beam_size = kwargs.get('beam_size',None)
        
        if zscale == 'pow':
            kwargs_imshow['norm'] = PowerNorm(gamma=gamma)
        elif zscale == 'sqrt': 
            kwargs_imshow['norm'] = PowerNorm(gamma=0.5)
        elif zscale == 'log': kwargs_imshow['norm'] = LogNorm()
        else: kwargs_imshow['norm'] = Normalize()

#        ax = kwargs.get('ax',None)        
#        if ax is None:
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

        xlabel = kwargs.get('xlabel',None)
        ylabel = kwargs.get('ylabel',None)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)

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

        ax.add_size_bar(1./self._axes[0]._delta, # 30' in in pixel
                        r"$1^{\circ}$",loc=3,color='w')
            
        if beam_size is not None:
            ax.add_beam_size(2.0*beam_size[0]/self._axes[0]._delta,
                             2.0*beam_size[1]/self._axes[1]._delta,
                             beam_size[2],beam_size[3],
                             patch_props={'fc' : "none", 'ec' : "w"})
            
        self._ax = ax
        
        return im

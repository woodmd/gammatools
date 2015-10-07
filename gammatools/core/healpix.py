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
from gammatools.core.fits_util import *


def fill_header(hdu,nside,galactic=True,scheme='RING'):

    ORDERDICT = {1:0,2:1,4:2,8:3,16:4,32:5,64:6,
                 128:7,256:8,512:9,1024:10,2048:11,4096:12,8192:13}
        
    hdu.header.set("PIXTYPE", "HEALPIX");
    hdu.header.set("ORDERING", scheme)
        
    try:
        hdu.header.set("ORDER", ORDERDICT[nside] )
    except:
        hdu.header.set("ORDER", -1)
    hdu.header.set("NSIDE", nside )
    
    if galactic:
        hdu.header.set("COORDSYS", "GAL")
    else:
        hdu.header.set("EQUINOX", 2000.0,"","Equinox of RA & DEC specifications")
        hdu.header.set("COORDSYS", "EQU")
        hdu.header.set("RADECSYS","FK5")

    hdu.header.set("FIRSTPIX", 0);
    hdu.header.set("LASTPIX", (12*nside*nside)-1)
    hdu.header.set("INDXSCHM","IMPLICIT")

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
    def scheme(self):
        if self._nest: return 'NEST'
        else: return 'RING'

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

        fill_header(hdu,self.nside,scheme=self.scheme)
                
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

        #        hdu_image = pyfits.PrimaryHDU(self._counts)
        hdu_image = pyfits.PrimaryHDU()

        cols = []
        for i in range(len(self._counts)):
            col = pyfits.Column(name='CHANNEL%i'%(i+1), format='D', array=self._counts[i])
            cols += [col]
        hdu = pyfits.BinTableHDU.from_columns(cols,name='SKYMAP')

        fill_header(hdu,self.nside,scheme=self.scheme)

        ecols = []
        ecols += [pyfits.Column(name='CHANNEL', format='I',
                                array=1+np.arange(len(self._counts)))]

        ecols += [pyfits.Column(name='E_MIN', format='1E',
                                array=self.axis(0).edges[:-1])]

        ecols += [pyfits.Column(name='E_MAX', format='1E',
                                array=self.axis(0).edges[1:])]
        
#        ecol = pyfits.Column(name='ENERGIES', format='D', 
#                             array=10**self.axis(0).center)
#        hdu_energies = pyfits.BinTableHDU.from_columns([ecol],name='ENERGIES')
        hdu_energies = pyfits.BinTableHDU.from_columns(ecols,name='EBOUNDS')
        hdulist = pyfits.HDUList([hdu_image,hdu,hdu_energies])

        hdulist.writeto(fitsfile,clobber=True)

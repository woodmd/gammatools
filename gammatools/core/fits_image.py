"""
@file  fits_image.py

@brief Various utility classes for manipulating FITS data.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood"
__date__     = "01/01/2014"

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import PowerNorm
import wcsaxes
from astropy.wcs import WCS
from astropy.io import fits
import numpy as np
import healpy as hp
from gammatools.core.algebra import Vector3D
from gammatools.fermi.catalog import *
from gammatools.core.histogram import HistogramND, Histogram, get_vector, Axis
from gammatools.core.fits_util import bintable_to_array, FITSAxis, get_circle
from gammatools.core.fits_util import load_ds9_cmap
from gammatools.core.util import update_dict, interpolate2d
from gammatools.core.healpix import HealpixSkyCube


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

        self._wcs = WCS(self._header)
        return ()
#        return (self._wcs,self._counts,self._ra,self._dec,self._roi_radius)
        
    def add_roi_msk(self,lon,lat,rad,invert=False,coordsys='cel'):

        if rad == 180.:
            self._roi_msk[:] = False
            return
        
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
        hdu = fits.open(fitsfile)[ihdu]
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
        
        header = fits.Header.fromstring(hdu.header.tostring())
#        header = hdu.header

        wcs = WCS(header,naxis=[1,2])#,relax=True)
#        wcs1 = WCS(header,naxis=[3])
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        return SkyCube(wcs,axes,copy.deepcopy(hdu.data.astype(float).T))
        
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        
        hdulist = fits.open(fitsfile)        
        header = hdulist[ihdu].header
        wcs = WCS(header,naxis=[1,2],relax=True)

        hdunames = [t.name for t in hdulist]
        
        if 'ENERGIES' in hdunames and hdulist[1].name == 'ENERGIES':

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
        wcs = WCS(header,relax=True)
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        
        return SkyImage(wcs,axes,copy.deepcopy(hdu.data.astype(float).T))
    
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        
        hdulist = fits.open(fitsfile)
        return SkyImage.createFromHDU(hdulist[ihdu])



    @staticmethod
    def createWCS(ra,dec,roi_radius_deg,bin_size_deg=0.2,coordsys='cel'):
        nbin = np.ceil(2.0*roi_radius_deg/bin_size_deg)
        deg_to_pix = bin_size_deg
        wcs = WCS(naxis=2)

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

        fig = plt.gcf()
        ax = fig.add_subplot(subplot,
                             projection=wcsaxes.WCS(self._wcs.to_header()))

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
#        ax.set_ticklabel_type("d", "d")



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

#        ax.add_size_bar(1./self._axes[0]._delta, 
#                        r"$1^{\circ}$",loc=3,color='w')
            
        if beam_size is not None:
            ax.add_beam_size(2.0*beam_size[0]/self._axes[0]._delta,
                             2.0*beam_size[1]/self._axes[1]._delta,
                             beam_size[2],beam_size[3],
                             patch_props={'fc' : "none", 'ec' : "w"})
            
        self._ax = ax
        
        return im

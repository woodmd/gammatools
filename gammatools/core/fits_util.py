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
#from gammatools.fermi.catalog import *
#from gammatools.core.util import *
from gammatools.core.histogram import HistogramND, Axis

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


"""
FITS Image Plotting 
=================== 

This script demonstrates how to load and plot a FITS file using the
SkyImage and SkyCube classes.
"""

#!/usr/bin/env python

import os
import sys
import copy
import argparse
import numpy as np
from gammatools.core.astropy_helper import pyfits

import matplotlib.pyplot as plt

from gammatools.core.fits_util import FITSImage, SkyCube, SkyImage
from gammatools.fermi.catalog import Catalog

usage = "usage: %(prog)s [options] [FT1 file ...]"
description = """Plot the contents of a FITS image file."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--hdu', default = 0, type=int,
                    help = 'Set the HDU number to plot.')
 
args = parser.parse_args()

hdulist = pyfits.open(args.files[0])

im = FITSImage.createFromFITS(args.files[0],args.hdu)
if isinstance(im,SkyCube):

    # Integrate over 3rd (energy) dimension
    im = im.marginalize(2)


plt.figure()

im.plot(cmap='ds9_b')

plt.figure()

# Smooth by 0.2 deg
im.smooth(0.2).plot(cmap='ds9_b')

# Draw an arbitrary contour in Galactic Coordinates
phi = np.linspace(0,2*np.pi,10000)
r = np.sqrt(2*np.cos(2*phi))
x = im.lon + r*np.cos(phi)
y = im.lat + r*np.sin(phi)
im.ax()['gal'].plot(x,y,color='w')

cat = Catalog.get('2fgl')
cat.plot(im,ax=plt.gca(),label_threshold=5,src_color='w')

# Make 1D projection on LON axis
plt.figure()
pim = im.project(0,offset_coord=True)
pim.plot()
plt.gca().grid(True)

plt.show()

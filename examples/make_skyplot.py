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
import pyfits

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

im = FITSImage.createFromFITS(args.files[0],args.hdu)
if isinstance(im,SkyCube):

    # Integrate over 3rd (energy) dimension
    im = im.marginalize(2)

plt.figure()

im.plot(cmap='ds9_b')

plt.figure()

# Smooth by 0.2 deg
im.smooth(0.2).plot(cmap='ds9_b')

cat = Catalog.get('2fgl')
cat.plot(im,ax=plt.gca(),label_threshold=5,src_color='w')

plt.show()

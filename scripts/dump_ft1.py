#!/usr/bin/env python


import os
import sys
import copy
import argparse

import pyfits
import skymaps
import pointlike
import numpy as np
from gammatools.core.algebra import Vector3D
from gammatools.fermi.catalog import Catalog
from gammatools.fermi.data import PhotonData

usage = "usage: %prog [options] [FT1 file ...]"
description = """Inspect the contents of an FT1 file."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--zenith_cut', default = 105, type=float,
                  help = 'Set the zenith angle cut.')

args = parser.parse_args()

hdulist = pyfits.open(args.files[0])

print hdulist.info()
print hdulist[1].columns.names

for c in hdulist[1].columns.names:
    print '%-25s %-6s %-10s'%(c, hdulist[1].data[c].dtype,
                              hdulist[1].data[c].shape),
    print hdulist[1].data[c]


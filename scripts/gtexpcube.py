#!/usr/bin/env python

import os, sys
import tempfile
import re
from GtApp import GtApp
import shutil
import pyfits
import argparse

from gammatools.fermi.task import BExpTask


usage = "usage: %(prog)s [options] [ft1file]"
description = "Produce a binned counts map."
parser = argparse.ArgumentParser(usage=usage,description=description)

#parser.add_option('--ra', default = None, type='float',
#                  help = 'Source RA')

#parser.add_option('--dec', default = None, type='float',
#                  help = 'Source Dec')

#parser.add_option('--emin', default = 2, type='float',
#                  help = 'Minimum event energy')

#parser.add_option('--emax', default = 5, type='float',
#                  help = 'Maximum event energy')

#parser.add_option('--bin_size', default = 0.5, type='float',
#                  help = 'Bin size in degrees')

#parser.add_option('--nbin', default = '720/360', type='string',
#                  help = 'Number of bins.')

#parser.add_option('--proj', default = 'AIT', type='string',
#                  help = 'Projection scheme\n'                  
#                  'Aitoff [AIT]\n'
#                  'Zenithal equal-area [ZEA]\n'
#                  'Zenithal equidistant [ARC]\n'
#                  'Plate Carree [CAR]\n'
#                  'Sanson-Flamsteed [GLS]\n'
#                  'Mercator [MER]\n'
#                  'North-Celestial-Pole [NCP]\n'
#                  'Slant orthographic [SIN]\n'
#                  'Stereographic [STG]\n'
#                  'Gnomonic [TAN]\n')

#parser.add_option('--alg', default = 'CMAP', choices=['CMAP','CCUBE'],
#                  help = 'Choose binning algorithm')

#parser.add_option('--coordsys', default = 'CEL', type='string',
#                  help = 'Choose coordinate system')

parser.add_argument('files', nargs='+')

parser.add_argument('--output', default = None, 
                    help = 'Output file')

BExpTask.add_arguments(parser)

args = parser.parse_args()


for f in args.files:

    outfile = args.output    
    if outfile is None:
        outfile = os.path.splitext(os.path.basename(f))[0] + '_bexpmap.fits'
    
        
    gtexp = BExpTask(outfile,infile=os.path.abspath(f),opts=args)
    
    gtexp.run()


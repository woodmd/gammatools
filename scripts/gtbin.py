#!/usr/bin/env python

import os, sys
import tempfile
import re
from GtApp import GtApp
import shutil
import pyfits
from optparse import Option
from optparse import OptionParser


usage = "usage: %prog [options] [ft1file]"
description = "Produce a binned counts map."
parser = OptionParser(usage=usage,description=description)

parser.add_option('--ra', default = None, type='float',
                  help = 'Source RA')

parser.add_option('--dec', default = None, type='float',
                  help = 'Source Dec')

parser.add_option('--emin', default = 2, type='float',
                  help = 'Minimum event energy')

parser.add_option('--emax', default = 5, type='float',
                  help = 'Maximum event energy')

parser.add_option('--bin_size', default = 0.5, type='float',
                  help = 'Bin size in degrees')

parser.add_option('--nbin', default = '720/360', type='string',
                  help = 'Number of bins.')

parser.add_option('--proj', default = 'AIT', type='string',
                  help = 'Projection scheme\n'                  
                  'Aitoff [AIT]\n'
                  'Zenithal equal-area [ZEA]\n'
                  'Zenithal equidistant [ARC]\n'
                  'Plate Carree [CAR]\n'
                  'Sanson-Flamsteed [GLS]\n'
                  'Mercator [MER]\n'
                  'North-Celestial-Pole [NCP]\n'
                  'Slant orthographic [SIN]\n'
                  'Stereographic [STG]\n'
                  'Gnomonic [TAN]\n')

parser.add_option('--alg', default = 'CMAP', choices=['CMAP','CCUBE'],
                  help = 'Choose binning algorithm')

parser.add_option('--coordsys', default = 'CEL', type='string',
                  help = 'Choose coordinate system')

parser.add_option('--output', default = None, type='string',
                  help = 'Output file')

(opts, args) = parser.parse_args()

if len(args) != 1:
    parser.error("Incorrect number of arguments.")


nbin = [int(t) for t in opts.nbin.split('/')]
    
hdulist = pyfits.open(args[0])
#hdulist.info()
#print hdulist[0].header.ascardlist()
#print hdulist[1].header.ascardlist()

ra = 0.0
dec = 0.0

# Find RA/DEC from FITS file
if opts.ra is None or opts.dec is None:
    m = re.search("CIRCLE\(([0-9\.]+),([0-9\.]+)",hdulist[1].header['DSVAL2'])
    if not m is None:
        ra = float(m.group(1))
        dec = float(m.group(2))
else:
    ra = opts.ra
    dec = opts.dec

outfile = 'binned.fits'
    
if opts.output is None:
    outfile = os.path.splitext(os.path.basename(args[0]))[0] + '_binned.fits'

evfile = os.path.abspath(args[0])

cwd = os.getcwd()
tmpdir = tempfile.mkdtemp(prefix=os.environ['USER'] + '.', dir='/scratch')
os.chdir(tmpdir)
    
evtbin = GtApp('gtbin', 'evtbin')

evtbin['algorithm'] = opts.alg
evtbin['evfile'] = evfile
evtbin['outfile'] = outfile
evtbin['tbinalg'] = 'LIN'
evtbin['tstart'] = 0.
evtbin['tstop'] = 0.
evtbin['dtime'] = 0.0
evtbin['tbinfile'] = 'NONE'

if len(nbin) == 1:
    evtbin['nxpix'] = nbin[0]
    evtbin['nypix'] = nbin[0]
else:
    evtbin['nxpix'] = nbin[0]
    evtbin['nypix'] = nbin[1]
    
evtbin['binsz'] = opts.bin_size
evtbin['coordsys'] = opts.coordsys
evtbin['xref'] = ra
evtbin['yref'] = dec
evtbin['axisrot'] = 0.0
evtbin['ebinalg'] = 'LOG'
evtbin['emin'] = 10**opts.emin
evtbin['emax'] = 10**opts.emax
evtbin['enumbins'] = 16
evtbin['denergy'] = 0.0
evtbin['proj'] = opts.proj
evtbin['snratio'] = 1
evtbin['lcemin'] = 0
evtbin['lcemax'] = 0


evtbin.run()

#os.system('mv gtltcube.log ' + cwd + '/' + logfile)
os.system('mv ' + outfile + ' ' + cwd)

shutil.rmtree(tmpdir)

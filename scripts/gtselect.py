#!/usr/bin/env python

import os, sys
import re
import tempfile
import logging
from LogFile import LogFile
import shutil
import pyfits
from GtApp import GtApp
from pySimbad import pySimbad
import numpy as np
from optparse import Option
from optparse import OptionParser
from gtwrap import GtAnalysis

class Main(GtAnalysis):

    def __init__(self,scfile,evfile):

        output_prefix = 'gtselect'
        logfile = 'gtselect.log'
        m = re.search("(.+)_(ft1|ls1)",os.path.basename(args[1]))
        if m is not None:
            outfile = m.group(1) + '_sel_' + m.group(2) + '.fits'
            logfile = output_prefix + '.log'
        
        GtAnalysis.__init__(self,'gtmktime.log')

usage = "usage: %prog [options] <scfile> <ft1file>"
description = "Run both gtmktime and gtselect on an FT1 file."
parser = OptionParser(usage=usage,description=description)

parser.add_option('--ra', default = None, type='float',
                  help = 'Source RA')

parser.add_option('--dec', default = None, type='float',
                  help = 'Source Dec')

parser.add_option('--rad', default = 180, type='float',
                  help = 'Radius of ROI')

parser.add_option('--emin', default = 1.5, type='float',
                  help = 'Minimum event energy in log10(E/MeV)')

parser.add_option('--emax', default = 5.5, type='float',
                  help = 'Maximum event energy in log10(E/MeV)')

parser.add_option('--zmax', default = 105, type='float',
                  help = 'Maximum zenith angle')

parser.add_option('--source', default = None, type='string',
                  help = 'Source name')

parser.add_option('--output', default = None, type='string',
                  help = 'Output file')

(opts, args) = parser.parse_args()

if len(args) < 2:
    parser.error("Incorrect number of arguments.")

output_prefix = 'gtselect'
logfile = 'gtselect.log'
m = re.search("(.+)_(ft1|ls1).fits",os.path.basename(args[1]))
if m is not None:
    outfile = m.group(1) + '_sel_' + m.group(2) + '.fits'
    logfile = output_prefix + '.log'


    
# Redirect stdout and stderr
sys.stdout = LogFile('stdout',logfile,quiet=False)
sys.stderr = LogFile('stderr',logfile,quiet=False)

source_ra = opts.ra
source_dec = opts.dec
if opts.source is not None:
    pysim = pySimbad()
    pysim.findSource(opts.source)
    source_ra = pysim.ra
    source_dec = pysim.dec
elif source_ra is None or source_dec is None:
    # Find RA/DEC from FITS file
    hdulist = pyfits.open(args[1])
    print hdulist[1].header['DSVAL2']
    m = re.search("circle\(([0-9\.]+),([0-9\.]+)",hdulist[1].header['DSVAL2'])
    if m is not None:
        source_ra = float(m.group(1))
        source_dec = float(m.group(2))
    else:
        source_ra = 'INDEF'
        source_dec = 'INDEF'
        
#print 'RA:  %s'%(source_ra)
#print 'DEC: %s'%(source_dec)

scfile = os.path.abspath(args[0])

files = []
for i in range(1,len(args)):
    files.append(os.path.abspath(args[i]))

    
cwd = os.getcwd()
tmpdir = tempfile.mkdtemp(prefix=os.environ['USER'] + '.', dir='/scratch')

os.chdir(tmpdir)


fd, file_list = tempfile.mkstemp(dir=tmpdir)
for file in files:
    os.write(fd,file + '\n')

presel_outfile = 'presel.fits'
sel_outfile = outfile

filter = GtApp('gtselect', 'dataSubselector')
maketime = GtApp('gtmktime', 'dataSubselector')

filter['ra'] = source_ra
filter['dec'] = source_dec
filter['rad'] = opts.rad

if opts.emin is not None:
    filter['emin'] = np.power(10,opts.emin)

if opts.emax is not None:
    filter['emax'] = np.power(10,opts.emax)


if opts.zmax is not None:
    filter['zmax'] = opts.zmax

filter['outfile'] = presel_outfile
filter['infile'] = '@' + file_list

maketime['scfile'] = scfile
maketime['evfile'] = presel_outfile
maketime['filter'] = 'IN_SAA!=T&&DATA_QUAL==1&&LAT_CONFIG==1&&ABS(ROCK_ANGLE)<52'
maketime['outfile'] = sel_outfile
maketime['roicut'] = 'no'

try:
    filter.run()
    maketime.run()
except:
    print logging.getLogger('stderr').exception(sys.exc_info()[0])

os.system('mv ' + sel_outfile + ' ' + cwd)


shutil.rmtree(tmpdir)


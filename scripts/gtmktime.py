#!/usr/bin/env python

import os, sys
import re
import tempfile
import logging
import shutil
from GtApp import GtApp
import numpy as np
import argparse
from gammatools.fermi.task import MkTimeTask
from gammatools.core.util import dispatch_jobs

usage = "%(prog)s [options] [pickle file ...]"
description = """Run gtmktime."""
parser = argparse.ArgumentParser(usage=usage, description=description)

parser.add_argument('files', nargs='+')
parser.add_argument('--scfile', default = None, required=True,
                    help = 'Spacecraft file.')
parser.add_argument('--output', default = None, 
                    help = 'Set the output filename.')
parser.add_argument('--outdir', default = None, 
                    help = 'Set the output directory.')
parser.add_argument('--queue', default = None, 
                    help = 'Set queue name.')
parser.add_argument('--filter', default = 'default_r52', 
                    help = 'Set the mktime filter.')

args = parser.parse_args()

gtidir = '/u/gl/mdwood/ki20/mdwood/fermi/data'

grb_gticut = "gtifilter('%s/nogrb.gti',START) && gtifilter('%s/nogrb.gti',STOP)"%(gtidir,gtidir)
sfr_gticut = "gtifilter('%s/nosolarflares.gti',(START+STOP)/2)"%(gtidir)
sun_gticut = "ANGSEP(RA_SUN,DEC_SUN,RA_ZENITH,DEC_ZENITH)>115"
default_gticut = 'IN_SAA!=T&&DATA_QUAL==1&&LAT_CONFIG==1'

filters = {'default' : default_gticut,
           'default_r52' : '%s && ABS(ROCK_ANGLE)<52'%default_gticut,
           'limb' : '%s && ABS(ROCK_ANGLE)>52'%default_gticut,
           'gticut0' : '%s && %s'%(grb_gticut,sfr_gticut),
           'gticut1' : '%s && (%s || %s)'%(grb_gticut,sun_gticut,sfr_gticut),
           'catalog' : '%s && %s && ABS(ROCK_ANGLE)<52 && (%s || %s)'%(default_gticut, grb_gticut,
                                                                       sun_gticut, sfr_gticut),
           }

filter_expr = []
for t in args.filter.split(','):

    filter_expr.append(filters[t])

#mktime_filter = filters[args.filter]
mktime_filter = '&&'.join(filter_expr)
    
if args.outdir is not None:
    args.outdir = os.path.abspath(args.outdir)

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue)
    sys.exit(0)

for f in args.files:

    if args.outdir is not None:
        outfile = os.path.basename(f)
        outfile = os.path.join(args.outdir,outfile)
    elif args.output is None:

        m = re.search('(.+)\.fits?',f)
        if not m is None:
            outfile = m.group(1) + '_sel.fits'
        else:
            outfile = os.path.splitext(f)[0] + '_sel.fits'
    else:
        outfile = args.output

    gt_task = MkTimeTask(f,outfile,filter=mktime_filter,scfile=args.scfile)
    gt_task.run()



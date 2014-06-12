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
parser.add_argument('--queue', default = None, 
                    help = 'Set queue name.')
parser.add_argument('--filter', default = 'default', 
                    help = 'Set the mktime filter.')

args = parser.parse_args()

filters = {'default' : 'IN_SAA!=T&&DATA_QUAL==1&&LAT_CONFIG==1&&ABS(ROCK_ANGLE)<52' }

mktime_filter = filters[args.filter]

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue)
    sys.exit(0)

for f in args.files:

    if args.output is None:

        m = re.search('(.+)\.fits?',f)
        if not m is None:
            outfile = m.group(1) + '_sel.fits'
        else:
            outfile = os.path.splitext(f)[0] + '_sel.fits'
    else:
        outfile = args.output

    gt_task = MkTimeTask(f,outfile,filter=mktime_filter,scfile=args.scfile)
    gt_task.run()



#!/usr/bin/env python

import os, sys
import re
import tempfile
import shutil
from gammatools.fermi.task import *
from gammatools.core.util import dispatch_jobs
import argparse

usage = "%(prog)s [options] [ft1file ...]"
description = """Create a LT cube."""
parser = argparse.ArgumentParser(usage=usage, description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--output', default = None, 
                  help = 'Output file')

parser.add_argument('--queue', default = None,
                    help='Set the batch queue.')

args = parser.parse_args()

if len(args.files) < 1:
    parser.error("At least one argument required.")

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue)
    sys.exit(0)
 
    
if args.output is None:

#        m = re.search('(.+)_ft1(.*)\.fits?',f)
#        if not m is None:
#            outfile = m.group(1) + '_gtltcube.fits'
#        else:
    outfile = os.path.splitext(f)[0] + '_gtltcube.fits'%(args.zmax)
else:
    outfile = args.output


if len(args.files) > 1:
    
    ltlist = 'ltlist.txt'
    fh = open(ltlist,'w')
    for f in args.files:
        fh.write('%s\n'%os.path.abspath(f))
    fh.close()
    ltlist = os.path.abspath(ltlist)
else:
    ltlist = os.path.abspath(args.files[0])
    
gt_task = LTSumTask(outfile,infile1='@' + ltlist)

gt_task.run()
    


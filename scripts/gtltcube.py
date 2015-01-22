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

#parser.add_argument("-l", "--logdir", dest="logdir", default=".", 
#                  help="log DIRECTORY (default ./)")

#parser.add_argument("-v", "--loglevel", dest="loglevel", default="debug", 
#                  help="logging level (debug, info, error)")

#parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
#                  help="do not log to console")

parser.add_argument('--queue', default = None,
                    help='Set the batch queue.')

LTCubeTask.add_arguments(parser)

args = parser.parse_args()

if len(args.files) < 1:
    parser.error("At least one argument required.")

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue)
    sys.exit(0)
 
for f in args.files:

    f = os.path.abspath(f)
    
    if args.output is None:

#        m = re.search('(.+)_ft1(.*)\.fits?',f)
#        if not m is None:
#            outfile = m.group(1) + '_gtltcube.fits'
#        else:
        outfile = os.path.splitext(f)[0] + '_gtltcube_z%03.f.fits'%(args.zmax)
    else:
        outfile = args.output

    gt_task = LTCubeTask(outfile,opts=args,evfile=f)

    gt_task.run()
    


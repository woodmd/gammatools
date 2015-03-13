#!/usr/bin/env python

import os
import sys
import copy
import argparse

usage = "usage: %(prog)s [options] [FT1 file ...]"
description = """Generate a pickle file containing a list of all photons within
max_dist_deg of a source defined in src_list.  The script accepts as input
a list of FT1 files."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--zenith_cut', default = 105, type=float,
                  help = 'Set the zenith angle cut.')

parser.add_argument('--conversion_type', default = None, 
                  help = 'Set the conversion type.')

parser.add_argument('--event_class_id', default = None, 
                  help = 'Set the event class bit.')

parser.add_argument('--event_type_id', default = None, 
                  help = 'Set the event type bit.')

parser.add_argument('--output', default = None, 
                  help = 'Set the output filename.')

parser.add_argument('--src_list',default = None,                  
                    help = 'Set the list of sources.')

parser.add_argument('--srcs',
                    default = None,
                    help = 'Set a comma-delimited list of sources.')

parser.add_argument('--sc_file', default = None, 
                  help = 'Set the spacecraft (FT2) file.')

parser.add_argument('--max_events', default = None, type=int,
                  help = 'Set the maximum number of events that will be '
                  'read from each file.')

parser.add_argument('--erange', default = None, 
                  help = 'Set the energy range in log10(E/MeV).')

parser.add_argument('--max_dist_deg', default = 25.0, type=float,
                  help = 'Set the maximum distance.')

parser.add_argument('--phase', default = None, 
                    help = 'Select the pulsar phase selection (on/off).')

parser.add_argument("--queue",default=None,
                    help='Set the batch queue on which to run this job.')

args = parser.parse_args()

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue)
    sys.exit(0)

if args.output is None:
    args.output = os.path.basename(os.path.splitext(args.files[0])[0] + '.P')
    
ft1_files = args.files
ft2_file = args.sc_file

pl = FT1Loader(args.zenith_cut,
               args.conversion_type,
               args.event_class_id,
               args.event_type_id,
               args.max_events,args.max_dist_deg,
               args.phase,args.erange)

pl.loadsrclist(args.src_list,args.srcs)

pl.setFT2File(args.sc_file)

for f in ft1_files:
    pl.load_photons(f)

pl.save(args.output)

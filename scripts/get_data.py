#! /bin/env python

import os
import sys
from optparse import OptionParser
import numpy as np
#from readTXT import readTXT

event_samples = { 'pass6' : 'P6_public_v3',
                  'pass7' : 'P7.6_P120_BASE',
                  'pass7r' : 'P7_P202_BASE',
                  'pass8'  : 'P8_P301_BASE' }

usage = "usage: %prog [options] "
description = "Download data with astro server tool."
parser = OptionParser(usage=usage,description=description)
parser.add_option('-s','--source', default = None, type = "string", 
                  dest="source",
                  help = 'Source identifier')
parser.add_option('-o','--output', default = None, type = "string", 
                  help = 'Output file prefix')

parser.add_option('--ra', default = 0, type = float, 
                  help = 'Minimum energy')
parser.add_option('--dec', default = 0, type = float, 
                  help = 'Maximum energy')

parser.add_option('--event_sample', default='pass7r',
                  choices=event_samples.keys(),
                  help='Event Sample (pass6,pass7,pass7r,pass8)')

parser.add_option('--event_class', default='Source', type = "string",
                  help='Event Class (Diffuse,Source,Clean)')

parser.add_option('--minEnergy', default = 1.5, type = float, 
                  help = 'Minimum energy')
parser.add_option('--maxEnergy', default = 5.5, type = float, 
                  help = 'Maximum energy')

parser.add_option('--minZenith', default = None, type = float, 
                  help = 'Minimum energy')
parser.add_option('--maxZenith', default = None, type = float, 
                  help = 'Maximum energy')

parser.add_option('--data_type', default = 'ft1',
                  choices=['ft1','ls1'],
                  help = 'Choose between standard and extended data formats.')

parser.add_option('--years', default = 0.0, type = float, 
                  help = 'Number of years since mission start time.')

parser.add_option('--days', default = 0.0, type = float, 
                  help = 'Number of days since mission start time.')

parser.add_option('--minTimestamp', default = None, type = float, 
                  help = 'Minimum MET timestamp (default: 239557417)')
parser.add_option('--maxTimestamp', default = None, type = float, 
                  help = 'Maximum MET timestamp')
parser.add_option('--radius', default = None, type = float, 
                  help = 'Angular radius in deg.')

parser.add_option('--max_file_size', default = 0.5, type = float, 
                  help = 'Maximum file size in GB.')

(opts, arguments) = parser.parse_args()


if opts.source is None:
    ra = opts.ra
    dec = opts.dec


min_timestamp = 239557417
if opts.minTimestamp is not None:
    min_timestamp = opts.minTimestamp


max_timestamp = min_timestamp + opts.years*365*86400 + opts.days*86400

if opts.maxTimestamp is not None:
    max_timestamp = opts.maxTimestamp

ft1_suffix = opts.data_type
    
if opts.output is None:
    output_ft1 = '%9.0f_%9.0f_%s.fits' %(min_timestamp,max_timestamp,ft1_suffix)
    output_ft2 = '%9.0f_%9.0f_ft2-30s.fits' %(min_timestamp,max_timestamp)
else:
    output_ft1 = '%s_%9.0f_%9.0f_%s.fits' %(opts.output,
                                            min_timestamp,max_timestamp,
                                            ft1_suffix)
    output_ft2 = '%s_%9.0f_%9.0f_ft2-30s.fits' %(opts.output,min_timestamp,max_timestamp)

astroserv = '~glast/astroserver/prod/astro'

if not opts.event_sample in event_samples:
    sys.exit(1)

command  = astroserv
command += ' --event-sample %s  '%(event_samples[opts.event_sample])
command += ' --event-class-name %s '%(opts.event_class)
command += ' --ra %9.5f --dec %9.5f ' %(ra,dec)

if opts.radius is not None:
    command += ' --radius %5.2f' %(opts.radius)

if opts.minZenith is not None:
    command += ' --minZenith %9.3f '%(opts.minZenith)

if opts.maxZenith is not None:
    command += ' --maxZenith %9.3f '%(opts.maxZenith)
        
command += ' --minEnergy %9.2f --maxEnergy %9.2f' %(np.power(10,opts.minEnergy),np.power(10,opts.maxEnergy))
command += ' --output-ft1-max-bytes-per-file %i'%(opts.max_file_size*1E9)
command += ' --output-ls1-max-bytes-per-file %i'%(opts.max_file_size*1E9)

command += ' --minTimestamp %.1f' %(min_timestamp)
command += ' --maxTimestamp %.1f' %(max_timestamp)

if opts.data_type == 'ft1':
    command_ft1 = command + ' --output-ft1 ' + output_ft1 + ' store'
else:
    command_ft1 = command + ' --output-ls1 ' + output_ft1 + ' store'
    
command_ft2 = command + ' --output-ft2-30s ' + output_ft2 + ' storeft2'

print command_ft1
print command_ft2

os.system(command_ft1)
#os.system(command_ft2)


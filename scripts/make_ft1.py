#!/usr/bin/env python

import os
import sys
from optparse import OptionParser
import tempfile
import re
import shutil

usage = "usage: %prog [options] "
description = "Run the makeft1 application on a Merit file."
parser = OptionParser(usage=usage,description=description)

parser.add_option('--xml_classifier', default = None, type = "string", 
                  help = 'FT2 file')

parser.add_option('--dict_file', default = None, type = "string", 
                  help = 'Set the file that defines the mapping from Merit '
                  'to FT1 variables.')

parser.add_option("--batch",action="store_true",
                  help="Split this job into several batch jobs.")

parser.add_option('--queue', default = None,
                  type='string',help='Set the batch queue.')

(opts, args) = parser.parse_args()

if opts.batch:
    
    for x in args:
        cmd = 'make_ft1.py %s '%(x)

        fitsFile = os.path.splitext(x)[0] + '_ft1.fits'

        if os.path.isfile(fitsFile):
            print 'Skipping ', fitsFile
            continue
            
        for k, v in opts.__dict__.iteritems():
            if not v is None and k != 'batch': cmd += ' --%s=%s '%(k,v)

        print 'bsub -q %s -R rhel60 %s'%(opts.queue,cmd)
        os.system('bsub -q %s -R rhel60 %s'%(opts.queue,cmd))

    sys.exit(0)

xml_classifier = os.path.abspath(opts.xml_classifier)
dict_file = os.path.abspath(opts.dict_file)
    
input_files = []
for x in args: input_files.append(os.path.abspath(x))

cwd = os.getcwd()
user = os.environ['USER']
tmpdir = tempfile.mkdtemp(prefix=user + '.', dir='/scratch')

print 'tmpdir ', tmpdir
os.chdir(tmpdir)

for x in input_files:

    fitsFile = os.path.splitext(x)[0] + '_ft1.fits'
    inFile = os.path.basename(x)

    print 'cp %s %s'%(x,inFile)
    os.system('cp %s %s'%(x,inFile))
    
    cmd = 'makeFT1 '
    options = { 'rootFile' : inFile,
                'xml_classifier' : xml_classifier,
                'fitsFile' : fitsFile,
                'dict_file' : dict_file,
                'TCuts' : '1' }

    for k, v in options.iteritems(): cmd += ' %s=%s '%(k,v)
        
    print cmd
    os.system(cmd)

os.chdir(cwd)
shutil.rmtree(tmpdir)

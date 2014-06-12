#!/usr/bin/env python

import os
import sys
import argparse
import tempfile
import re
import shutil
from gammatools.core.util import dispatch_jobs

usage = "usage: %(prog)s [options] [files]"
description = "Run the makeft1 application on a Merit file."
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--xml_classifier', default = None,
                    required=True,
                    help = 'Set the XML cut definition file.')

parser.add_argument('--dict_file', default = None,
                    required=True,
                    help = 'Set the file that defines the mapping from Merit '
                    'to FT1 variables.')

parser.add_argument('--queue', default = None,
                    help='Set the batch queue name.')

args = parser.parse_args()

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue)
    sys.exit(0)

xml_classifier = os.path.abspath(args.xml_classifier)
dict_file = os.path.abspath(args.dict_file)
    
input_files = []
for x in args.files: input_files.append(os.path.abspath(x))

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

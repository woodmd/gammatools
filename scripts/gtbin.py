#!/usr/bin/env python

import os, sys
import copy
import tempfile
import re
from GtApp import GtApp
import shutil
import pyfits
import argparse
import pprint
from gammatools.fermi.task import BinTask, SelectorTask, Task
from gammatools.core.config import *

class TestObject(Configurable):

    default_config = {'testoption' : 'test', 'testoption2' : 'val2'}
    
    def __init__(self,opts,**kwargs):
        super(TestObject,self).__init__(register_defaults=False)

        self.update_default_config(TestObject)
        self.update_default_config(BinTask,group='gtbin')
        
        self.configure(opts=opts)

    @classmethod
    def add_arguments(cls,parser):

        config = TestObject.get_default_config()
        config.update(BinTask.get_class_config(group='gtbin'))
        
        Configurable.add_arguments(parser,config=config)
        
class BinSelectTask(Task):

    default_config = {'select' : False}#'run_gtselect' : 'test'}

    def __init__(self,infile,outfile,opts,**kwargs):
        super(BinSelectTask,self).__init__()

        self._infile = infile
        self._outfile = outfile
        
#        self.update_default_config(SelectorTask,group='gtselect')
        self.update_default_config(BinTask)
        self.update_default_config(SelectorTask,group='gtselect')

        self.configure(opts=opts)

    @classmethod
    def add_arguments(cls,parser):

        config = copy.deepcopy(BinSelectTask.get_default_config().values())
        config += BinTask.get_class_config().values()
        config += SelectorTask.get_class_config(group='gtselect').values()

        pprint.pprint(config)
        
        Configurable.add_arguments(parser,config=config)

    def prepare_file(self,infile):

        outfile = os.path.splitext(os.path.basename(self._infile))[0]
        outfile += '_sel.fits'
        outfile = os.path.join(self._workdir,outfile)
            
        print outfile
        print self._workdir

#        if re.search('^(?!\@)(.+)(\.txt|\.lst)$',infile):
#            infile = '@'+infile
        
        gtselect = SelectorTask(infile,outfile,
                                config=self.config['gtselect'])
#                                    workdir=self._workdir,savedata=True)

        pprint.pprint(gtselect.config)            
        gtselect.run()

        return outfile

         
    def run_task(self):

        if self.config['select']:            
            infile = self.prepare_file(self._infile)
        else:
            infile = self._infile
            
        gtbin = BinTask(infile,self._outfile,config=self.config)
        print 'gtbin config '
        pprint.pprint(gtbin.config)

        gtbin.run()
        

usage = "usage: %(prog)s [options] [ft1file]"
description = "Produce a binned counts map."
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--output', default = None, 
                    help = 'Output file')

#parser.add_argument('--select', default = False, action='store_true',
#                    help = 'Run gtselect.')

#keys = SelectorTask.default_config.keys()
#for k in keys:
#    SelectorTask.default_config['gtselect_' + k] = SelectorTask.default_config[k]
#    del SelectorTask.default_config[k]

#SelectorTask.add_arguments(parser,skip=['verbose']+Task.default_config.keys())
#BinTask.add_arguments(parser,skip=['verbose'])

#BinSelectTask.add_arguments(parser,dict(BinTask.get_default_config(groupname='gtselect.').items() +
#                                        BinSelectTask.get_default_config().items()))

BinSelectTask.add_arguments(parser)

args = parser.parse_args()

for f in args.files:

    outfile = args.output    
    if outfile is None:
        outfile = os.path.splitext(os.path.basename(f))[0] + '_binned.fits'

    bselect = BinSelectTask(os.path.abspath(f),outfile,args)

    pprint.pprint(bselect.config)
    bselect.run()

sys.exit(0)

# First run gtselect


for f in args.files:

#    hdulist = pyfits.open(f)
#hdulist.info()

#    ra = 0.0
#    dec = 0.0

    # Find RA/DEC from FITS file
#    if args.ra is None or args.dec is None:
#        m = re.search("CIRCLE\(([0-9\.]+),([0-9\.]+)",
#                      hdulist[1].header['DSVAL2'])
#        if not m is None:
#            ra = float(m.group(1))
#            dec = float(m.group(2))
#    else:
#        ra = args.ra
#        dec = args.dec

    outfile = args.output    
    if outfile is None:
        outfile = os.path.splitext(os.path.basename(f))[0] + '_binned.fits'
    
        
    gtbin = BinTask(os.path.abspath(f),outfile,opts=args)
    
    gtbin.run()


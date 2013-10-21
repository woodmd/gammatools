#!/usr/bin/env python

import os, sys
import tempfile
import logging
from LogFile import LogFile
import shutil
from GtApp import GtApp
from pySimbad import pySimbad
import numpy as np
from optparse import Option
from optparse import OptionParser
from gtwrap import GtAnalysis

class Main(GtAnalysis):

    def __init__(self):
        GtAnalysis.__init__(self,'gtmktime.log')
    
    def run_gtmktime(self,evfile,scfile,output_file=None):

        if output_file is None and self.opts.output_suffix is None:
            output_file = os.path.basename(evfile)
        elif output_file is None:            
            output_file = os.path.basename(evfile).split('.')[0] + \
                '_' + self.opts.output_suffix + '.fits'
            
            
        
#        logfile = output_file.split('.')[0] + '_gtmktime.log'

        maketime = GtApp('gtmktime', 'dataSubselector')
        maketime['scfile'] = scfile
        maketime['evfile'] = evfile
        maketime['filter'] = \
            'IN_SAA!=T&&DATA_QUAL==1&&LAT_CONFIG==1&&ABS(ROCK_ANGLE)<52'
        maketime['outfile'] = output_file
        maketime['roicut'] = 'no'

        try:
            maketime.run()
        except:
            print logging.getLogger('stderr').exception(sys.exc_info()[0])

        if self.opts.overwrite is not True and \
                os.path.isfile(os.path.join(output_file,self._dest_dir)):
            print 'ERROR: File exists: ', output_file
        else:
            print 'Overwriting'
            os.system('mv ' + output_file + ' ' + self._dest_dir)



    def run(self,*argv):
        
        usage = "usage: %prog [options] <scfile> <ft1file>"
        description = "Select a subset of the data"
        parser = OptionParser(usage=usage,description=description)
        
        parser.add_option('--output', default = None, type='string',
                          help = 'Output file')

        parser.add_option('--output_suffix',
                          default = None, type='string',
                          help = 'Output file')
        
        parser.add_option('--overwrite', default = False, action='store_true',
                          help = 'Output file')

        (opts, args) = parser.parse_args(list(argv))
        self.opts = opts
        
        if len(args) < 2:
            parser.error("Incorrect number of arguments.")
            
        scfile = os.path.abspath(args[0])
        evfiles = [os.path.abspath(t) for t in args[1:]]

        self.setup()
        
        for f in evfiles:
            self.run_gtmktime(f,scfile)

        self.cleanup()


if __name__ == '__main__':

    main = Main()
    main.run(*sys.argv[1:])

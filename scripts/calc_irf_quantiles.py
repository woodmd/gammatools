#!/usr/bin/env python



import os

os.environ['CUSTOM_IRF_DIR']='/u/gl/mdwood/ki10/analysis/custom_irfs/'
os.environ['CUSTOM_IRF_NAMES']='P6_v11_diff,P7CLEAN_V4,P7CLEAN_V4MIX,P7CLEAN_V4PSF,P7SOURCE_V4,P7SOURCE_V4MIX,P7SOURCE_V4PSF,P7ULTRACLEAN_V4,P7ULTRACLEAN_V4MIX,P7ULTRACLEAN_V4PSF,P7SOURCE_V11,P7SOURCE_V6,P7SOURCE_V11A,P7SOURCE_V11B,P7SOURCE_V11C,P7SOURCE_V6MC,P7SOURCE_V6MCPSFC'

import sys
import copy
import re
import pickle
import argparse

import math
import numpy as np
import matplotlib.pyplot as plt
#from skymaps import SkyDir
from gammatools.core.histogram import Histogram, Histogram2D
from matplotlib import font_manager

from gammatools.fermi.psf_model import *
from gammatools.fermi.irf_util import *
from gammatools.fermi.validate import PSFData
from gammatools.fermi.catalog import Catalog
        
class Main(object):

    def __init__(self):
        self.irf_colors = ['green','red','magenta','gray','orange']
    
    def main(self,*argv):
        usage = "usage: %(prog)s [options]"
        description = """Generates PSF model."""
        parser = argparse.ArgumentParser(usage=usage,description=description)

        IRFManager.configure(parser)

        parser.add_argument('--ltfile', default = None, 
                          help = 'Set the livetime cube which will be used '
                          'to generate the exposure-weighted PSF model.')

        parser.add_argument('--src', default = 'Vela', 
                          help = '')
        
        parser.add_argument('--irf', default = None, 
                          help = 'Set the names of one or more IRF models.')
                
        parser.add_argument('--output_dir', default = None, 
                            help = 'Set the output directory name.')
        
        parser.add_argument('--cth_bin_edge', default = '0.4,1.0', 
                            help = 'Edges of cos(theta) bins '
                            '(e.g. 0.2,0.5,1.0).')

        parser.add_argument('--egy_bin_edge', default = None, 
                            help = 'Edges of energy bins.')

        parser.add_argument('--egy_bin', default = '1.25/5.0/0.25', 
                            help = 'Set min/max energy.')
        
        parser.add_argument('--quantiles', default = '0.34,0.68,0.90,0.95', 
                            help = 'Define the set of quantiles to compute.')
        
        parser.add_argument('--conversion_type', default = 'front', 
                            help = 'Draw plots.')

        parser.add_argument('--spectrum', default = 'powerlaw/2',
                            help = 'Draw plots.')

        parser.add_argument('--edisp', default = None,
                            help = 'Set the energy dispersion lookup table.')
        
        parser.add_argument('-o', '--output', default = None, 
                            help = 'Set the output file.')
        
        parser.add_argument('--load_from_file', default = False, 
                            action='store_true',
                            help = 'Load IRFs from FITS.')

        opts = parser.parse_args(list(argv))

        irfs = opts.irf.split(',')

        [elo, ehi, ebin] = [float(t) for t in opts.egy_bin.split('/')]
        egy_bin_edge = np.linspace(elo, ehi, 1 + int((ehi - elo) / ebin))
        cth_bin_edge = [float(t) for t in opts.cth_bin_edge.split(',')]
        quantiles = [float(t) for t in opts.quantiles.split(',')]
        
        for irf in irfs:

            if opts.output is None:

                output_file = re.sub('\:\:',r'_',irf)
                
                output_file += '_%03.f%03.f'%(100*cth_bin_edge[0],
                                              100*cth_bin_edge[1]) 
                output_file += '_psfdata.P'
            else:
                output_file = opts.output

            irfm = IRFManager.create(irf,opts.load_from_file,opts.irf_dir)

            lonlat = (0,0)
            if opts.src != 'iso' and opts.src != 'iso2':
                cat = Catalog()
                src = cat.get_source_by_name(opts.src)
                lonlat = (src['RAJ2000'], src['DEJ2000'])
            
            m = PSFModelLT(opts.ltfile, irfm,
                           nbin=400,
                           cth_range=cth_bin_edge,
                           psf_type=opts.src,
                           lonlat=lonlat,
                           edisp_table=opts.edisp)

            spectrum = opts.spectrum.split('/')            
            pars = [ float(t) for t in spectrum[1].split(',')]            
            m.set_spectrum(spectrum[0],pars)
#            m.set_spectrum('powerlaw_exp',(1.607,3508.6))

            psf_data = PSFData(egy_bin_edge,cth_bin_edge,'model')

#            f = open(opts.o,'w')

            for i in range(len(psf_data.quantiles)):

                ql = psf_data.quantile_labels[i]
                q = psf_data.quantiles[i]
                for iegy in range(len(egy_bin_edge)-1):

                    elo = egy_bin_edge[iegy]
                    ehi = egy_bin_edge[iegy+1]
                    radius = m.quantile(10**elo,10**ehi,q)
#                    print elo, ehi, radius
                    psf_data.qdata[i].set(iegy,0,radius)

#                    line = '%6.3f '%(q)
#                    line += '%6.3f %6.3f '%(cth_range[0],cth_range[1])
#                    line += '%6.3f %6.3f %8.4f %8.4f'%(elo,ehi,radius,0.0)
                    
#                    f.write(line + '\n')
            
#                m.set_spectrum('powerlaw_exp',(1.607,3508.6))
#                m.set_spectrum('powerlaw',(2.0))
#            psf_data.print_quantiles('test')
            psf_data.save(output_file)
            
        # Compute results 
    
 
        
        


if __name__ == '__main__':

    main = Main()
    main.main(*sys.argv[1:])

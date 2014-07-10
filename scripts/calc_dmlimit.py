#!/usr/bin/env python

import os
import sys
import re

import yaml

import numpy as np
import matplotlib.pyplot as plt
#import gammatools.dm.jcalc as jcalc
from gammatools.dm.jcalc import *
from gammatools.dm.irf_model import *
from gammatools.dm.dmmodel import *
from scipy.interpolate import UnivariateSpline
from gammatools.dm.halo_model import *
from gammatools.core.stats import limit_pval_to_sigma

import argparse

if __name__ == "__main__":

    usage = "usage: %(prog)s [options]"
    description = """Calculate sensitivity."""

    parser = argparse.ArgumentParser(usage=usage,description=description)
        
    parser.add_argument('--sthresh', default=None, type=float,
                        help = '')
    
    parser.add_argument('--ulthresh', default=None, type=float,
                        help = '')

    parser.add_argument('--livetime', default=50.0, type=float,
                      help = 'Set the exposure in hours.')

    parser.add_argument('--rmax', default=None, type=float,
                      help = '')
    
    parser.add_argument('--output', default='limits.P', 
                      help = 'Output file to which limit data will be written.')
    
    parser.add_argument('--source', default=None, 
                        help = '',required=True)

    parser.add_argument('--chan', default='bb', 
                        help = 'Set the annihilation channel.')

    parser.add_argument('--median', default=False, action='store_true',
                      help = '')

    parser.add_argument('--redge', default=0.0, type=float,
                      help = '')

    parser.add_argument('--alpha', default=0.2, type=float,
                        help = '')

    parser.add_argument('--min_fsig', default=0.0, type=float,
                        help = '')

    parser.add_argument('--plot_lnl', default=False, action='store_true',
                        help = '')

    parser.add_argument('--irf', default=None, 
                        help = 'Set the input IRF file.',
                        required=True)

    parser.add_argument('files', nargs='*')

    args = parser.parse_args()

    if not args.ulthresh is None:
        sthresh = limit_pval_to_sigma(1.0-args.ulthresh)
    else:
        sthresh = args.sthresh


    hm = HaloModelFactory.create(args.source)
    irf = IRFModel.createCTAIRF(args.irf)
    chm = ConvolvedHaloModel(hm,irf)


    src_model = DMFluxModel.createChanModel(chm,1000.0*Units.gev,
                                            1E-24*Units.cm3_s,
                                            args.chan)

    dm = DMLimitCalc(irf,args.alpha,args.min_fsig,args.redge)


    if args.chan == 'ww': mass = np.linspace(2.0,4.0,20)
    elif args.chan == 'zz': mass = np.linspace(2.1,4.0,20)
    else: mass = np.linspace(1.75,4.0,19)

    livetime = args.livetime*Units.hr

    srcs = []

    if len(args.files) == 1:
        d = np.loadtxt(args.files[0],unpack=True)

        if args.median:
            rs = np.median(d[1])
            rhos = np.median(d[2])
            d = np.array([[0.],[rs],[rhos]])

            s = copy.deepcopy(src)            
            s['rhos'] = np.median(d[2])
            s['rs'] = np.median(d[1])
            srcs.append(s)
        else:

            for i in range(len(d[0])):
                s = copy.deepcopy(src)
                s['rhos'] = d[2][i]
                s['rs'] = d[1][i]
                if len(d) == 4: s['alpha'] = d[3][i]
                srcs.append(s)
    else:
        srcs.append(hm)
            

    o = { 'ul' : np.zeros(shape=(len(srcs),len(mass))), 
          'mass' : mass }

    jval = []
    rs = []

    for i, s in enumerate(srcs):

        dp = s._dp

#        jval = dp.jval()/(s['dist']*Units.kpc)**2
#        print dp._rs/Units.kpc, dp._rhos/Units.gev_cm3, jval/Units.gev2_cm5
#        sys.exit(0)

        jval.append(dp.jval()/(s.dist*Units.kpc)**2)
        rs.append(dp.rs)

    print np.median(np.array(jval))/Units.gev2_cm5
    print np.median(np.array(rs))/Units.kpc


    for i, s in enumerate(srcs):

        src_model._hm = ConvolvedHaloModel(hm,irf)
        o['ul'][i] = dm.limit(src_model,np.power(10,mass)*Units.gev,
                              livetime,sthresh)

        
        
#        if args.plot_lnl:
#            for j, x in enumerate(mass):
#                dm.plot_lnl(src_model,np.power(10,x)*Units.gev,ul[0][j],tau)

    save_object(o,args.output,True)


#!/usr/bin/env python


import os
import sys
import copy
import argparse
import pyfits

import numpy as np

import matplotlib

#try:             os.environ['DISPLAY']
#except KeyError: matplotlib.use('Agg')

matplotlib.interactive(False)
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.pyplot import gcf, setp

from gammatools.core.fits_util import SkyImage, SkyCube, FITSImage
from gammatools.core.fits_viewer import *
from gammatools.core.util import mkdir
from gammatools.core.mpl_util import PowerNormalize
from gammatools.fermi.irf_util import *
from gammatools.fermi.psf_model import *


def get_event_types(event_type):

    bits = {0b000100 : 'PSF0',
            0b001000 : 'PSF1',
            0b010000 : 'PSF2',
            0b100000 : 'PSF3'}

    o = []
    
    for k,v in bits.items():
        if (event_type&k): o.append(v)
    
    return o

def get_dss_keywords(header):

    o = {}
    for k, v in header.iteritems():

        m = re.search('DSTYP(\d)',k)
        if m is None: continue
        o[v] = header['DSVAL%s'%(m.group(1))]

    return o
        

def get_irf_version(header):

    kwds = get_dss_keywords(header)

    event_types = None
    event_class = None
    
    for k,v in kwds.items():

        print k
        m = re.search('BIT_MASK\(EVENT_TYPE,([0-9]+),(.+)\)',k)
        if m is not None:
            event_types = get_event_types(int(m.groups()[0]))

        if k == 'IRF_VERSION':
            m = re.search('([\S]+)',v)
            event_class = m.groups()[0]

    if 'PSF' in event_class: event_types = None
                        
    return [event_class, event_types]

usage = "usage: %(prog)s [options] [FT1 file ...]"
description = """Plot the contents of a FITS image or counts cube file."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--rsmooth', default=0.3, type=float,
                    help='Set the radius (68% containment) of the smoothing '
                    'kernel in degrees.')
parser.add_argument('--slice_per_fig', default=4, type=int)

parser.add_argument('--model_file', default=None,
                    help='Set the name of the model cube file.')
parser.add_argument('--prefix', default=None)
parser.add_argument('--irf', default=None)
parser.add_argument('--title', default=None, help='')
parser.add_argument('--nosrclabels', default=False,action='store_true')
parser.add_argument('--srclabels_thresh', default=10.0,type=float)
parser.add_argument('--format', default='png')

parser.add_argument('--hdu', default = 0, type=int,
                    help = 'Set the HDU of the input FITS file from which '
                    'the image data will be extracted.')


args = parser.parse_args()

hdulist = pyfits.open(args.files[0])
model_hdu = None
irf_version = None
im_mdl = None

im = FITSImage.createFromHDU(hdulist[args.hdu])

if args.model_file:
    model_hdu = pyfits.open(args.model_file)[0]

    if args.irf:
        irf_version = [args.irf,None]
    else:
        irf_version = get_irf_version(model_hdu.header)
        
    im_mdl = FITSImage.createFromHDU(model_hdu)
    m = re.search('(.+)V5_(.+)',irf_version[0])
    if m and 'stacked' in args.model_file:
        irf_version[0] = m.groups()[0]+'V5'
    else:
        irf_version[1] = None
        
            
irf_dir = '/u/gl/mdwood/ki20/mdwood/fermi/custom_irfs'
if 'CUSTOM_IRF_DIR' in os.environ:
    irf_dir = os.environ['CUSTOM_IRF_DIR']


irf = None
m = None

if irf_version:
    irf = IRFManager.create(irf_version[0],type_names=irf_version[1],
                            load_from_file=True,irf_dir=irf_dir)
    ltfile = '/Users/mdwood/fermi/data/p301/ltcube_5years_zmax100.fits'
    m = PSFModelLT(irf,src_type='iso')
    
    
ccube = FITSImage.createFromHDU(hdulist[args.hdu])

im_mdl = None
if model_hdu:
    im_mdl = FITSImage.createFromHDU(model_hdu)

mkdir('plots')
    
fp = FITSPlotter(ccube,im_mdl,m,args.prefix,rsmooth=args.rsmooth,
                 title=args.title,format=args.format,
                 nosrclabels=args.nosrclabels,
                 srclabels_thresh=args.srclabels_thresh)

if isinstance(im,SkyImage):
    fp.make_projection_plots_skyimage(ccube)
elif isinstance(ccube,SkyCube):

    # All Energies        
    fp.make_energy_residual(suffix='eresid')

    fp.make_plots_skycube(smooth=True,resid_type='significance',
                          suffix='data_map_resid_sigma',plots_per_fig=1)

    fp.make_mdl_plots_skycube(suffix='mdl_map',plots_per_fig=1)

    fp.make_mdl_plots_skycube(suffix='mdl_map_pnorm',plots_per_fig=1,
                              zscale='pow',zscale_power=3.0)

    fp.make_plots_skycube(suffix='data_map',plots_per_fig=1,
                          make_projection=True,projection=0.5)

    fp.make_plots_skycube(suffix='data_map_pnorm',plots_per_fig=1,
                          zscale='pow',zscale_power=3.0)
    
    fp.make_plots_skycube(suffix='data_map_smooth',plots_per_fig=1,
                          make_projection=True,projection=0.5,smooth=True)


    delta_bin = [2,2,4,16]

    # Slices
    fp.make_mdl_plots_skycube(suffix='mdl_map_slice',
                              plots_per_fig=args.slice_per_fig,
                              delta_bin=delta_bin)

    fp.make_mdl_plots_skycube(suffix='mdl_map_slice_pnorm',
                              plots_per_fig=args.slice_per_fig,
                              zscale='pow',zscale_power=3.0,
                              delta_bin=delta_bin)

    fp.make_plots_skycube(suffix='data_map_slice',
                          plots_per_fig=args.slice_per_fig,
                          delta_bin=delta_bin,
                          make_projection=True,projection=0.5)

    fp.make_plots_skycube(suffix='data_map_slice_smooth',
                          plots_per_fig=args.slice_per_fig,
                          delta_bin=delta_bin,
                          make_projection=True,projection=0.5,smooth=True)

    fp.make_plots_skycube(smooth=True,resid_type='significance',
                          suffix='data_map_slice_resid_sigma',
                          plots_per_fig=args.slice_per_fig,
                          delta_bin=delta_bin)

    sys.exit(0)

    fp.make_plots_skycube(smooth=True,resid_type='fractional',
                          suffix='_data_map_resid_frac',plots_per_fig=1)

    fp.make_plots_skycube(smooth=True,resid_type='fractional',
                          suffix='_data_map_slice_resid_frac')




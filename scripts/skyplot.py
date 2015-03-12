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


from matplotlib.widgets import Slider, Button, RadioButtons
from gammatools.core.fits_util import SkyImage, SkyCube, FITSImage
from gammatools.core.fits_viewer import *
from gammatools.core.mpl_util import PowerNormalize
from gammatools.fermi.irf_util import *
from gammatools.fermi.psf_model import *




def get_irf_version(header):

    for k, v in header.iteritems():

        m = re.search('DSTYP(\d)',k)        
        if m is None or v != 'IRF_VERSION': continue

        return header['DSVAL%s'%(m.group(1))]

    return None

usage = "usage: %(prog)s [options] [FT1 file ...]"
description = """Plot the contents of a FITS image file."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--gui', action='store_true')

parser.add_argument('--rsmooth', default=0.3, type=float)
parser.add_argument('--slice_per_fig', default=4, type=int)

parser.add_argument('--model_file', default=None)
parser.add_argument('--prefix', default=None)

parser.add_argument('--hdu', default = 0, type=int,
                    help = 'Set the HDU to plot.')


args = parser.parse_args()

hdulist = pyfits.open(args.files[0])
model_hdu = None
irf_version = None
im_mdl = None


im = FITSImage.createFromHDU(hdulist[args.hdu])


if args.model_file:
    model_hdu = pyfits.open(args.model_file)[0]
    irf_version = get_irf_version(model_hdu.header)
    im_mdl = FITSImage.createFromHDU(model_hdu)

    m = re.search('(.+)V5_(.+)',irf_version)
    if m and 'stacked' in args.model_file:
        irf_version = m.groups()[0]+'V5'

    
#fp = FITSPlotter(im,im_mdl,None,args.prefix)

#fp.make_plots_skycube(None,smooth=True,resid_type='fractional',
#                              suffix='_data_map_resid_frac')

#fp.make_plots_skycube(4,smooth=True,resid_type='fractional',
#                              suffix='_data_map_slice_resid_frac')

#sys.exit(0)

    
irf_dir = '/u/gl/mdwood/ki10/analysis/custom_irfs/'
if 'CUSTOM_IRF_DIR' in os.environ:
    irf_dir = os.environ['CUSTOM_IRF_DIR']


irf = None
m = None

if irf_version:
    irf = IRFManager.create(irf_version, True,irf_dir=irf_dir)
    ltfile = '/Users/mdwood/fermi/data/p301/ltcube_5years_zmax100.fits'
    m = PSFModelLT(irf,src_type='iso')
    
#for k, v in hdulist[0].header.iteritems():
#    print k, v
    
#viewer = FITSImageViewer(im)
#viewer.plot()



if args.gui:
    
    app = wx.App()

    frame = FITSViewerFrame(args.files,hdu=args.hdu,parent=None,
                            title="FITS Viewer",
                            size=(2.0*640, 1.5*480))

    frame.Show()


        
#frame = Frame(im)



#frame.Show(True)


    app.MainLoop()
    plt.show()



else:
    
    ccube = FITSImage.createFromHDU(hdulist[args.hdu])

    im_mdl = None
    if model_hdu:
        im_mdl = FITSImage.createFromHDU(model_hdu)
    

    fp = FITSPlotter(ccube,im_mdl,m,args.prefix,rsmooth=args.rsmooth)
        
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
                              plots_per_fig=args.slice_per_fig,delta_bin=delta_bin,
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

        
        

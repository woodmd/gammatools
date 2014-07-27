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

parser.add_argument('--model_file', default=None)
parser.add_argument('--prefix', default=None)

parser.add_argument('--hdu', default = 0, type=int,
                    help = 'Set the HDU to plot.')


args = parser.parse_args()

hdulist = pyfits.open(args.files[0])
model_hdu = None
irf_version = None


if args.model_file:
    model_hdu = pyfits.open(args.model_file)[0]
    irf_version = get_irf_version(model_hdu.header)

    
irf = IRFManager.create(irf_version, True,
                        irf_dir='/u/gl/mdwood/ki10/analysis/custom_irfs/')

ltfile = '/nfs/slac/g/ki/ki20/cta/mdwood/fermi/data/P8_P301/ltcube_5years_zmax100.fits'

m = PSFModelLT(ltfile, irf,
               cth_range=[0.2,1.0],
               src_type='iso')



    
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
    
    im = FITSImage.createFromHDU(hdulist[args.hdu])

    im_mdl = None
    if model_hdu:
        im_mdl = FITSImage.createFromHDU(model_hdu)
    

    fp = FITSPlotter(im,im_mdl,m,args.prefix)
        
    if isinstance(im,SkyImage):
        fp.make_projection_plots_skyimage(im)
    elif isinstance(im,SkyCube):
#        make_plots_skycube(im,4,paxis=0,suffix='_projx')
#        make_plots_skycube(im,4,paxis=1,suffix='_projy')

        #        make_plots_skycube(im,4,logz=True)

        fp.make_plots_skycube(smooth=True,
                              suffix='_data_map_smooth')
        
#        make_plots_skycube(im,4,smooth=True,
#                           im_mdl=im_mdl,suffix='_data_map_slice_smooth')

        fp.make_mdl_plots_skycube(suffix='_mdl_map')
        
        fp.make_plots_skycube(None,smooth=True,residual=True,
                              suffix='_data_map_resid')

        fp.make_plots_skycube(4,smooth=True,residual=True,
                              suffix='_data_map_slice_resid')
        
        sys.exit(0)
        
        make_plots_skycube(im,4,residual=True,
                           im_mdl=im_mdl,suffix='_data_map_resid2')
        
        make_plots_skycube(im_mdl,4,suffix='_mdl_map')
    

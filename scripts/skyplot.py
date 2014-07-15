#!/usr/bin/env python


import os
import sys
import copy
import argparse
import pyfits

import numpy as np

import matplotlib

matplotlib.interactive(False)
matplotlib.use('WXAgg')

import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.pyplot import gcf, setp


from matplotlib.widgets import Slider, Button, RadioButtons
from gammatools.core.fits_util import SkyImage, SkyCube, FITSImage
from gammatools.core.fits_viewer import *
        
usage = "usage: %(prog)s [options] [FT1 file ...]"
description = """Plot the contents of a FITS image file."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--gui', action='store_true')

parser.add_argument('--model_file', default=None)

parser.add_argument('--hdu', default = 0, type=int,
                    help = 'Set the HDU to plot.')


args = parser.parse_args()

hdulist = pyfits.open(args.files[0])
model_hdu = None
if args.model_file:
    model_hdu = pyfits.open(args.model_file)[0]
    
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
    
    
    if isinstance(im,SkyImage):
        make_projection_plots_skyimage(im)
    elif isinstance(im,SkyCube):
#        make_plots_skycube(im,4,paxis=0,suffix='_projx')
#        make_plots_skycube(im,4,paxis=1,suffix='_projy')

        #        make_plots_skycube(im,4,logz=True)

        make_plots_skycube(im,None,smooth=True,
                           im_mdl=im_mdl,suffix='_data_map_smooth')
        
        make_plots_skycube(im,4,smooth=True,
                           im_mdl=im_mdl,suffix='_data_map_slice_smooth')

        make_plots_skycube(im_mdl,4,smooth=True,
                           suffix='_mdl_map_slice_smooth')
        
        make_plots_skycube(im,None,smooth=True,residual=True,
                           im_mdl=im_mdl,suffix='_data_map_resid')

        plt.show()
        sys.exit(0)
        
        make_plots_skycube(im,4,residual=True,
                           im_mdl=im_mdl,suffix='_data_map_resid2')
        
        make_plots_skycube(im_mdl,4,suffix='_mdl_map')
    

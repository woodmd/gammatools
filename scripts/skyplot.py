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




class FITSImageViewer(object):

    def __init__(self,im):

        self._im = im
        self._fig, self._ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)

        axcolor = 'lightgoldenrodyellow'
#        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        axamp  = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

#        self._sfreq = Slider(axfreq, 'Freq', 0, 30.0, valinit=0)
        self._samp = Slider(axamp, 'Slice', 0, im.axis(2).nbins(), valinit=0)

#        self._sfreq.on_changed(self.update)
        self._samp.on_changed(self.update)
        
    def update(self,val):

        print val
        islice = np.round(val)
#        freq = sfreq.val
#        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        self._im.slice(2,islice).plot()
        self._fig.canvas.draw_idle()

    def plot(self):

        self._im.slice(2,0).plot()
        
        
usage = "usage: %prog [options] [FT1 file ...]"
description = """Plot the contents of a FITS image file."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')

parser.add_argument('--gui', action='store_true')

parser.add_argument('--model_file', default=None)

parser.add_argument('--hdu', default = 0, type=int,
                    help = 'Set the HDU to plot.')


args = parser.parse_args()

hdulist = pyfits.open(args.files[0])
if not args.model_file is None:
    model_hdu = pyfits.open(args.model_file)[0]
    
for k, v in hdulist[0].header.iteritems():
    print k, v
    
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

    if model_hdu:
        im_mdl = FITSImage.createFromHDU(model_hdu)
        im = im - im_mdl
    
    
    if isinstance(im,SkyImage):
        make_projection_plots_skyimage(im)
    elif isinstance(im,SkyCube):
        make_plots_skycube(im,4,paxis=0)
        make_plots_skycube(im,4,paxis=1)
#        make_plots_skycube(im,4,logz=True)
        make_plots_skycube(im,4,smooth=True)
    

    plt.show()

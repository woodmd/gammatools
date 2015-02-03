#!/usr/bin/env python

import healpy as hp
from healpy import projaxes as PA
import numpy as np
import matplotlib.pyplot as plt
from gammatools.core.astropy_helper import pyfits
from gammatools.core.mpl_util import *
from gammatools.core.fits_util import *
from gammatools.core.util import save_object
import argparse

usage = "usage: %(prog)s [options] [ft1file]"
description = "Produce a binned counts map."
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')
parser.add_argument('--prefix', default = '')
parser.add_argument('--rsmooth', default = 3.0, type=float)

args = parser.parse_args()

bexpmap_file = args.files[0]
ccube_file = args.files[1]

bexpcube = SkyCube.createFromFITS(bexpmap_file)

hdulist = pyfits.open(bexpmap_file)
ccube_healpix = HealpixSkyCube.createFromFITS(ccube_file)

# Convert CAR projection to HP map
bexpcube_hp = HealpixSkyCube.create(ccube_healpix.axes()[0],ccube_healpix.nside)
c = bexpcube_hp.center()

print 'Interpolating'
exp = bexpcube.interpolate(c[1],c[2],c[0]).reshape(bexpcube_hp.counts.shape)

print 'Copying'
bexpcube_hp._counts = exp

print 'Plotting'

energy_index = [6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24]
#energy_index = [10,12,14,16]

o = {}

for index in energy_index:

    title = 'log$_{10}$(E/MeV) = [%.3f, %.3f]'%(bexpcube_hp.axis(0).edges[index],
                                                bexpcube_hp.axis(0).edges[index+1])

    index2 = ccube_healpix.axis(0).valToBin(bexpcube_hp.axis(0).center[index])
    ecenter = bexpcube_hp.axis(0).center[index]
    deltae = 10**bexpcube_hp.axis(0).edges[index+1] - 10**bexpcube_hp.axis(0).edges[index]
    # Get pixel area
    domega = hp.pixelfunc.nside2pixarea(ccube_healpix.nside)
    
    print 'index ', index, index2, ecenter
    bexpmap = bexpcube_hp.marginalize(0,[index,index+1])
    bexpmap_norm = bexpmap/np.median(bexpmap.counts)

    ccube_smooth = ccube_healpix.marginalize(0,[index,index+1]).smooth(args.rsmooth)
    hflux = ccube_smooth/bexpcube_hp.counts[index]
    hflux /= domega*deltae

    o[index] = {'flux' : hflux, 'ecenter' : ecenter, 'title' : title,
                'bexp' : bexpmap }

    continue
    plt.figure()
    
#    bexpmap_hp.marginalize(0,[index,index+1]).plot(title=title)
    bexpmap.plot(title=title)
    plt.savefig('%sbexpmap_healpix_%02i.png'%(args.prefix,index))

    plt.figure()
    bexpmap_norm.plot(title=title)
    plt.savefig('%sbexpmap_norm_healpix_%02i.png'%(args.prefix,index))

    plt.figure()
    bexpcube.marginalize(2,[index,index+1]).plot(title=title)
    plt.savefig('%sbexpmap_car_%02i.png'%(args.prefix,index))

    plt.figure()    
    ccube_smooth.plot(zscale='pow',zscale_power=4.0,title=title)
    plt.savefig('%scounts_healpix_%02i.png'%(args.prefix,index))

    plt.figure()    
    hflux.plot(zscale='pow',zscale_power=4.0,title=title,
               cbar_label='Flux [cm$^{-2}$ MeV$^{-1}$ s$^{-1}$ sr$^{-1}$]')
    plt.savefig('%sflux_healpix_%02i.png'%(args.prefix,index))

    plt.show()

save_object(o,'%sflux_healpix.P'%(args.prefix),compress=True)



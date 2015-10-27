#!/usr/bin/env python

import matplotlib
matplotlib.interactive(False)
matplotlib.use('Agg')

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
parser.add_argument('--bexpmap', default = None)
parser.add_argument('--rsmooth', default = 3.0, type=float)
parser.add_argument('--scale', default = 1.0, type=float)

args = parser.parse_args()


bexpcube = SkyCube.createFromFITS(args.bexpmap)

ccube_hp = None

for f in args.files:
    
    if ccube_hp is None:
        ccube_hp = HealpixSkyCube.createFromFITS(f)
    else:
        ccube_hp += HealpixSkyCube.createFromFITS(f)
        

# Convert CAR projection to HP map
bexpcube_hp = HealpixSkyCube.create(ccube_hp.axes()[0],ccube_hp.nside)


c = bexpcube_hp.center()

print 'Interpolating'
exp = bexpcube.interpolate(c[1],c[2],c[0]).reshape(bexpcube_hp.counts.shape)

print 'Copying'
bexpcube_hp._counts = exp
bexpcube_hp *= args.scale

flux_hp = ccube_hp/bexpcube_hp

print 'Plotting'


o = {}

gplane_cut = {'latrange' : [-20.,20.],'coordsys' : 'GAL', 'complement' : True}

dec_cuts = [{'latrange' : [-90.,90.],'coordsys' : 'CEL'},
            {'latrange' : [-90.,-60.],'coordsys' : 'CEL'},
            {'latrange' : [-60.,-30.],'coordsys' : 'CEL'},
            {'latrange' : [-30.,-10.],'coordsys' : 'CEL'},
            {'latrange' : [-10.,10.],'coordsys' : 'CEL'},
            {'latrange' : [10.,30.],'coordsys' : 'CEL'},
            {'latrange' : [30.,60.],'coordsys' : 'CEL'},
            {'latrange' : [60.,90.],'coordsys' : 'CEL'}]

glat_cuts = [{'latrange' : [-90.,90.],'coordsys' : 'GAL'},
             {'latrange' : [-90.,-60.],'coordsys' : 'GAL'},
             {'latrange' : [-60.,-30.],'coordsys' : 'GAL'},
             {'latrange' : [-30.,-10.],'coordsys' : 'GAL'},
             {'latrange' : [-10.,10.],'coordsys' : 'GAL'},
             {'latrange' : [10.,30.],'coordsys' : 'GAL'},
             {'latrange' : [30.,60.],'coordsys' : 'GAL'},
             {'latrange' : [60.,90.],'coordsys' : 'GAL'}]

eaxis = flux_hp.axis(0)
deltae = 10**eaxis.edges[1:] - 10**eaxis.edges[:-1]

for c in dec_cuts:
    
    exp_sum = bexpcube_hp.integrate(cuts=[gplane_cut] + [c])
    flux_sum = flux_hp.integrate(cuts=[gplane_cut] + [c])
    ccube_sum = ccube_hp.integrate(cuts=[gplane_cut] + [c])
    
    m = ccube_hp.slice(0,0).create_mask(cuts=[gplane_cut] + [c])
    npix = float(np.sum(m))
    domega = 4.0*np.pi*npix/float(m.size)    
    expavg = exp_sum/npix
    
    flux_sum /= (domega*deltae)
    flux_sum2 = (ccube_sum/exp_sum)/(domega*npix*deltae)


    name = 'flux_cel_colat_%05.1f_%05.1f'%(90.-c['latrange'][1],
                                           90.-c['latrange'][0])
    
    o[name] = flux_sum

for c in glat_cuts:
    
    exp_sum = bexpcube_hp.integrate(cuts=[c])
    flux_sum = flux_hp.integrate(cuts=[c])
    ccube_sum = ccube_hp.integrate(cuts=[c])
    
    m = ccube_hp.slice(0,0).create_mask(cuts=[c])
    npix = float(np.sum(m))
    domega = 4.0*np.pi*npix/float(m.size)    
    expavg = exp_sum/npix
    
    flux_sum /= (domega*deltae)
    flux_sum2 = (ccube_sum/exp_sum)/(domega*npix*deltae)


    name = 'flux_gal_colat_%05.1f_%05.1f'%(90.-c['latrange'][1],
                                           90.-c['latrange'][0])
    
    o[name] = flux_sum
    

energy_index = np.arange(0,24)
#energy_index = [6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24]
#energy_index = [10,12,14,16]

domega = hp.pixelfunc.nside2pixarea(ccube_hp.nside)

ccube_smooth_hp = ccube_hp.smooth(args.rsmooth)
flux_smooth_hp = ccube_smooth_hp/bexpcube_hp
flux_smooth_hp /= domega*deltae[:,np.newaxis]

o['flux_allsky'] = flux_smooth_hp

save_object(o,'%sflux_healpix.P'%(args.prefix),compress=True)

sys.exit(0)

for index in energy_index:

    continue
    
    title = 'log$_{10}$(E/MeV) = [%.3f, %.3f]'%(eaxis.edges[index],
                                                eaxis.edges[index+1])

    index2 = ccube_hp.axis(0).valToBin(eaxis.center[index])
    ecenter = eaxis.center[index]
    deltae = 10**eaxis.edges[index+1] - 10**eaxis.edges[index]
    # Get pixel area
    domega = hp.pixelfunc.nside2pixarea(ccube_hp.nside)
    
    print 'index ', index, index2, ecenter
    bexpmap = bexpcube_hp.marginalize(0,[index,index+1])
    bexpmap_norm = bexpmap/np.median(bexpmap.counts)

    ccube_smooth = ccube_hp.marginalize(0,[index,index+1]).smooth(args.rsmooth)
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





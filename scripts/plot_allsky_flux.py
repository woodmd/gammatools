#!/usr/bin/env python


import healpy as hp
from healpy import projaxes as PA
import numpy as np
import matplotlib.pyplot as plt
from gammatools.core.astropy_helper import pyfits
from gammatools.core.mpl_util import *
from gammatools.core.fits_util import *
from gammatools.core.util import save_object, load_object
from scipy.interpolate import UnivariateSpline
import argparse

usage = "usage: %(prog)s [options] [ft1file]"
description = "Produce a binned counts map."
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('files', nargs='+')
parser.add_argument('--prefix', default = 'p301_source_5yr_zmax100')
parser.add_argument('--normalize', default = False,
                    action='store_true')

parser.add_argument('--iso0', default = None)
parser.add_argument('--iso1', default = None)


args = parser.parse_args()


iso = np.loadtxt('isotropic_source_4years_P8V3.txt',unpack=True)


if args.iso0 is not None:
    iso0 = np.loadtxt(args.iso0,unpack=True)
else:
    iso0 = iso

if args.iso1 is not None:
    iso1 = np.loadtxt(args.iso1,unpack=True)
else:
    iso1 = iso

iso_spline = UnivariateSpline(np.log10(iso[0]),iso[1],k=1,s=0)
iso0_spline = UnivariateSpline(np.log10(iso0[0]),iso0[1],k=1,s=0)
iso1_spline = UnivariateSpline(np.log10(iso1[0]),iso1[1],k=1,s=0)


obj0 = load_object(args.files[0])
obj1 = load_object(args.files[1])

    

for k in sorted(obj0):
    
    m0 = obj0[k]['flux']
    ecenter = obj0[k]['ecenter']
    m1 = obj1[k]['flux']

    exp0 = obj0[k]['bexp']
    exp1 = obj1[k]['bexp']

    ms0 = m0.integrate(latrange=[-5.0,5.0])
    ms1 = m1.integrate(latrange=[-5.0,5.0])


    
    
    sys.exit(0)
    
    if ms0 == 0: continue
    
    gplane_ratio = ms0/ms1
    
    print m0.integrate(latrange=[-5.0,5.0])
    print gplane_ratio
    
    if args.normalize:    
        m1 *= gplane_ratio
    
    mc0 = copy.deepcopy(m0)
    mc1 = copy.deepcopy(m1)

    iso_flux = iso_spline(ecenter)
    iso0_flux = iso0_spline(ecenter)
    iso1_flux = iso1_spline(ecenter)

    mc0 -= iso0_flux
    mc1 -= iso1_flux
    
    print 'iso_flux ', ecenter, iso0_flux, iso1_flux

    title = obj0[k]['title']

    # Exposure Ratio
    expratio = exp0/exp1
    
    # Fractional Residual
    resid0 = (m0 - m1)/m0

    # Absolute Residual (in ISO units)
    resid1 = (m0-m1)/iso_flux

    # Absolute Residual (in ISO units w/ ISO subtracted)
    residc1 = (mc0-mc1)/iso_flux

#    plt.figure()
#    m0 /= iso_flux
#    m0.plot(zscale='pow',zscale_power=4.0,title=title)
#    plt.figure()
#    m1.plot(zscale='pow',zscale_power=4.0,title=title)

    plt.figure()
#    m0.mask(latrange=[-5,5])    
    m0.plot(title=title,zscale='pow',zscale_power=4.0,
            cbar_label='Flux [cm$^{-2}$ MeV$^{-1}$ s$^{-1}$ sr$^{-1}$]')
    plt.savefig('%s_flux0_%02i.png'%(args.prefix,k))

    plt.figure()
#    m1.mask(latrange=[-5,5])    
    m1.plot(title=title,zscale='pow',zscale_power=4.0,
            cbar_label='Flux [cm$^{-2}$ MeV$^{-1}$ s$^{-1}$ sr$^{-1}$]')
    plt.savefig('%s_flux1_%02i.png'%(args.prefix,k))
    
    plt.figure()
    expratio.plot(title=title,cbar_label='Exposure Ratio')
    plt.savefig('%s_expratio_%02i.png'%(args.prefix,k))
    
    plt.figure()
    resid0.mask(latrange=[-5,5])
#    resid0.plot(vmin=-0.2,vmax=0.2,levels=[-0.1,-0.05,0.05,0.1],
    resid0.plot(vmin=-0.1,vmax=0.1,levels=[-0.07,-0.03,0.03,0.07],
                title=title,
               cbar_label='Fractional Residual')

    plt.savefig('%s_flux_residual_%02i.png'%(args.prefix,k))
    
    plt.figure()
    resid1.mask(latrange=[-5,5])    
    resid1.plot(vmin=-0.5,vmax=0.5,levels=[-0.4,-0.2,0.2,0.4],title=title,
               cbar_label='Fractional Residual (Isotropic Flux Units)')

    plt.savefig('%s_flux_residual_isotropic_%02i.png'%(args.prefix,k))

    plt.figure()
    residc1.mask(latrange=[-5,5])    
    residc1.plot(vmin=-0.5,vmax=0.5,levels=[-0.4,-0.2,0.2,0.4],title=title,
                 cbar_label='Fractional Residual (Isotropic Flux Units)')

    plt.savefig('%s_flux_residual_isotropic_corr_%02i.png'%(args.prefix,k))
    
sys.exit(0)

bexpmap = SkyCube.createFromFITS(bexpmap_file)
bexpmap2 = SkyCube.createFromFITS('bexpmap.fits')

print bexpmap.interpolate(0.0,0.0,3.0)
print bexpmap2.interpolate(0.0,0.0,3.0)

hdulist = pyfits.open(bexpmap_file)
print hdulist[1].data
print bexpmap.axis(2)


ccube_healpix = HealpixSkyCube.createFromFITS(ccube_file)

bexpmap_hp = HealpixSkyCube.create(ccube_healpix.axes()[0],ccube_healpix.nside)
c = bexpmap_hp.center()

print 'Interpolating'
exp = bexpmap.interpolate(c[1],c[2],c[0]).reshape(bexpmap_hp.counts.shape)

print 'Copying'
bexpmap_hp._counts = exp

print 'Plotting'

energy_index = [8,12,16,20,24]

#ebin = him2.axes()[0].binToVal(index)
#index = bexpmap.axes()[2].valToBin(ebin)[0]
#print 'ebin ', ebin
#print 'index ', index

o = {}

for index in energy_index:

    title = 'log$_{10}$(E/MeV) = [%.3f, %.3f]'%(bexpmap_hp.axis(0).edges[index],
                                                bexpmap_hp.axis(0).edges[index+1])

    index2 = ccube_healpix.axis(0).valToBin(bexpmap_hp.axis(0).center[index])
    ecenter = bexpmap_hp.axis(0).center[index]

    print 'index ', index, index2, ecenter

    plt.figure()

    expmap = bexpmap_hp.marginalize(0,[index,index+1])
    expmap /= np.median(expmap.counts)
    bexpmap_hp.marginalize(0,[index,index+1]).plot(title=title)
    plt.savefig('%sbexpmap_healpix_%02i.png'%(args.prefix,index))

    plt.figure()
    expmap.plot(title=title)
    plt.savefig('%sbexpmap_norm_healpix_%02i.png'%(args.prefix,index))

    plt.figure()
    bexpmap.marginalize(2,[index,index+1]).plot(title=title)
    plt.savefig('%sbexpmap_car_%02i.png'%(args.prefix,index))

    plt.figure()
    hims = ccube_healpix.marginalize(0,[index,index+1]).smooth(2.0)
    hims.plot(zscale='pow',zscale_power=4.0,title=title)
    plt.savefig('%scounts_healpix_%02i.png'%(args.prefix,index))

    # Get pixel area
    domega = hp.pixelfunc.nside2pixarea(hims.nside)

    print 'domega ', domega

    plt.figure()
    hims /= bexpmap_hp.counts[index]
    hims /= domega
    hims.plot(zscale='pow',zscale_power=4.0,title=title,
              cbar_label='Flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
    plt.savefig('%sflux_healpix_%02i.png'%(args.prefix,index))

    o[index] = hims

save_object(o,'%sflux_healpix.P'%(args.prefix),compress=True)


sys.exit(0)

print him.axes()[0]
print him.axes()[1]


him = him.marginalize(0,[20,28]).smooth(0.5)


print him.counts


him.plot(zscale='log',zscale_power=5.0)


plt.show()

sys.exit(0)



print type(hdulist[1].data)

#print hdulist[1].data.field(0)

hpm = hdulist[1].data.field(24)
hpm = hp.sphtfunc.smoothing(np.array(hpm,copy=True),sigma=np.radians(0.3))

#hpm = hp.read_map('test.fits')

print type(hpm)
print hpm.shape

fig = plt.figure()


extent = (0.02,0.05,0.96,0.9)
ax=PA.HpxMollweideAxes(fig,extent,coord=None,rot=None,
                       format='%g',flipconv='astro')

fig.add_axes(ax)
img0 = ax.projmap(hpm,nest=False,xsize=1600,coord=None)



#hp.visufunc.graticule()

#                 norm=PowerNormalize(power=2))#,#vmin=min,vmax=max,
#                 cmap=cmap,norm=norm)

#plt.figure()
#hp.mollview(hpm)

#plt.figure()
#hp.mollview(hdulist[1].data.field(0))

fig = plt.figure()

print type(hpm), hpm.shape

print np.isnan(hpm)




extent = (0.02,0.05,0.96,0.9)
ax=PA.HpxMollweideAxes(fig,extent,coord=None,rot=None,
                       format='%g',flipconv='astro')

fig.add_axes(ax)
img0 = ax.projmap(hpm,nest=False,xsize=1600,coord=None,
                  norm=PowerNormalize(power=4.,vmin=1E-3,clip=True))

plt.show()

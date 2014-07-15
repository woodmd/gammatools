#! /usr/bin/env python

import sys
import pyfits
import healpy
import matplotlib.pyplot as plt
import numpy as np
from gammatools.core.histogram import *
from gammatools.core.algebra import Vector3D
from gammatools.core.util import integrate
from gammatools.dm.jcalc import *
from scipy.interpolate import UnivariateSpline


usage = "usage: %prog [options] [FT1 file ...]"
description = """Description."""
parser = OptionParser(usage=usage,description=description)

parser.add_option('--energy_range', default = '4.0/4.5', type='string',
                  help = 'Set the energy range in GeV.')

parser.add_option('--profile', default = 'nfw', type='string',
                  help = 'Set the profile name.')

parser.add_option('--prefix', default = 'nfw', type='string',
                  help = 'Set the output file prefix.')

parser.add_option('--alpha', default = 0.17, type='float',
                  help = 'Set the alpha parameter of the DM halo profile.')

parser.add_option('--gamma', default = 1.0, type='float',
                  help = 'Set the gamma parameter of the DM halo profile.')

parser.add_option('--lon_cut', default = 6.0, type='float',
                  help = 'Set the longitude cut value.')

parser.add_option('--lat_cut', default = 5.0, type='float',
                  help = 'Set the latitude cut value.')

parser.add_option('--rgc_cut', default = None, type='float',
                  help = 'Set the latitude cut value.')

parser.add_option('--rmin', default = 0.1, type='float',
                  help = 'Set the profile name.')

parser.add_option('--decay', default = False, action='store_true',
                  help = 'Set the profile name.')

parser.add_option('--source_list', default = None, type='string',
                  help = 'Set the profile name.')

(opts, args) = parser.parse_args()

# Density Profiles

profile_opts = {'rs'    : 20*Units.kpc,
                'rhos'  : 0.1*Units.gev_cm3,
                'rmin'  : opts.rmin*Units.pc,
                'alpha' : opts.alpha,
                'gamma' : opts.gamma }

if opts.profile == 'isothermal': 
    profile_opts['rs'] = 5*Units.kpc

dp = DensityProfile.create(opts.profile,profile_opts)
dp.set_rho(0.4*Units.gev_cm3,8.5*Units.kpc)

#f = LoSIntegralFnFast(dp,rmax=100*Units.kpc,alpha=2.0,ann=(not opts.decay),   
#                      nstep=800)

f = LoSIntegralFn(dp,8.5*Units.kpc,
                  rmax=100*Units.kpc,alpha=3.0,ann=(not opts.decay))

log_psi = np.linspace(np.log10(np.radians(0.01)),
                      np.log10(np.radians(179.9)),400)
psi = np.power(10,log_psi)
jpsi = f(psi)
jspline = UnivariateSpline(psi,jpsi,s=0,k=1)
jint = JIntegrator(jspline,opts.lat_cut,opts.lon_cut,opts.source_list)
#jint.print_profile(opts.decay)

if not opts.rgc_cut is None:
    jint.eval(opts.rgc_cut,opts.decay)

jint.compute()

sys.exit(0)

#z = np.ones(shape=(360,360))
#hc = Histogram2D(theta_edges,phi_edges)

for i0, th in enumerate(theta):

    jtot = integrate(lambda t: jspline(t)*np.sin(t),
                     np.radians(theta_edges[i0]),
                     np.radians(theta_edges[i0+1]),100)

#    jval = jspline(np.radians(th))*costh_width[i0]
    v = Vector3D.createThetaPhi(np.radians(th),np.radians(phi))
    v.rotate(yaxis)

    lat = np.degrees(v.lat())
    lon = np.degrees(v.phi())

    src_msk = len(lat)*[True]

    if not sources is None:

        for k in range(len(v.lat())):
            p = Vector3D(v._x[:,k])

            sep = np.degrees(p.separation(sources))
            imin = np.argmin(sep)
            minsep = sep[imin]

            if minsep < 0.62: src_msk[k] = False



    msk = ((np.abs(lat)>=lat_cut) |
           ((np.abs(lat)<=lat_cut)&(np.abs(lon)<lon_cut)))

    dphi2 = 2.*np.pi*float(len(lat[msk]))/float(len(phi))

    msk &= src_msk
    dphi = 2.*np.pi*float(len(lat[msk]))/float(len(phi))

    hc._counts[i0,msk] = 1

    jtot *= dphi
    jsum += jtot
    domegasum += costh_width[i0]*dphi

    jv.append(jtot)
    domega.append(costh_width[i0]*dphi)

    


#plt.imshow(z.T,origin='lower')
#hc.plot()

#plt.show()

#    plt.plot(theta,np.array(jv))
#    plt.gca().set_yscale('log')
    
#z    plt.show()


#    plt.gca().set_xscale('log')



#        print th, s

#        hs.fill(th,jval*s)



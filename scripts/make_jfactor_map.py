from gammatools.dm.jcalc import *
from gammatools.core.fits_util import *
from gammatools.core.units import Units

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import UnivariateSpline

import argparse

usage = "usage: %(prog)s [options] [model file]"
description = """Compute J-factor and save it to a healpix map.
Accepts a file with a list of parameters for N halos.  The output map
will contain the J-factor (GeV^2 cm^-5 sr^-1) summed over all
halos in the input model file."""
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('--order', default = 9, type=int,
                  help = 'Set the order of the healpix map.')

parser.add_argument('--output', default = 'jfactor_map.fits', type=str,
                  help = 'Set the output file.')

parser.add_argument('files', nargs='+', help='List of halo model parameters.')

args = parser.parse_args()

im = HealpixSkyImage.create(2**args.order)
ra, dec = im.center()
skydir = SkyCoord(ra=ra,dec=dec,unit=u.deg)

hm = None
for f in args.files:
    
    d = np.loadtxt(sys.argv[1])

    if hm is None: hm = d
    else: hm = np.vstack((hm,d))

for i, h in enumerate(hm):

    print i, h

    d = SkyCoord(ra=h[0],dec=h[1],unit=u.deg)
    dp = NFWProfile(rhos=h[2]*Units.msun_kpc3,rs=h[3]*Units.kpc)
    psi = np.linspace(-3,np.log10(h[6]),400)

    fn0 = LoSIntegralFnFast(dp,h[4]*Units.kpc,rmax=h[5]*Units.kpc)    
    jval = fn0(np.radians(10**psi))

    sp = UnivariateSpline(psi,jval,k=2,s=0)

    dtheta = skydir.separation(d).deg
    msk = dtheta < h[6]

    jv = sp(np.log10(dtheta[msk]))
    jv[jv<0] = 0
    im._counts[msk] += jv/Units.gev2_cm5

im.plot(zscale='log')

im.save(args.output)

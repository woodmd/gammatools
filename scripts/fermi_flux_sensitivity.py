#!/usr/bin/env python

import matplotlib

#try:             os.environ['DISPLAY']
#except KeyError: matplotlib.use('Agg')

matplotlib.interactive(False)
matplotlib.use('Agg')

from gammatools.core.fits_util import *
from gammatools.fermi.irf_util import *
from gammatools.fermi.psf_model import *
from gammatools.fermi.fermi_tools import *
from gammatools.fermi.exposure import *
from gammatools.core.stats import poisson_lnl
from gammatools.core.util import *
from gammatools.core.bspline import BSpline, PolyFn
import matplotlib.pyplot as plt
import healpy as hp
import argparse
import glob

def poisson_ts(sig,bkg):
    return 2*(poisson_lnl(sig+bkg,sig+bkg) - poisson_lnl(sig+bkg,bkg))
    
usage = "usage: %(prog)s [options] [ft1file]"
description = "Run both gtmktime and gtselect on an FT1 file."
parser = argparse.ArgumentParser(usage=usage,description=description)

#parser.add_argument('files', nargs='+')
parser.add_argument('--min_counts', default=3.0, type=float)
parser.add_argument('--ts_threshold', default=25.0, type=float)
parser.add_argument('--bexp_scale', default=1.0, type=float)
parser.add_argument('--nside', default=16, type=int)
parser.add_argument('--irfs', default='P8R2_SOURCE_V6')
parser.add_argument('--bexpmap', default=None,required=True)
parser.add_argument('--galdiff', default=None,required=True)
parser.add_argument('--isodiff', default=None,required=True)
parser.add_argument('--prefix', default='p8source')
parser.add_argument('--gamma', default=2.0,type=float)
parser.add_argument('--emin', default=1.5,type=float)
parser.add_argument('--emax', default=6.0,type=float)
parser.add_argument('--ethresh', default=None,type=float)
parser.add_argument('--bins_per_decade', default=4,type=int)
parser.add_argument('--event_type', default='all')

args = parser.parse_args()


filename = '{prefix:s}_n{min_counts:04.1f}_g{gamma:03.1f}_ts{ts_threshold:02.0f}_nside{nside:03d}'

filename = filename.format(**{'prefix' : args.prefix,
                              'min_counts':args.min_counts,
                              'gamma':args.gamma,
                              'ts_threshold' : args.ts_threshold,
                              'nside' : args.nside})

fn_iso = []
for isodiff in sorted(glob.glob(args.isodiff)):
    print isodiff
    iso = np.loadtxt(isodiff,unpack=True)
    fn_iso += [BSpline.fit(np.log10(iso[0]),np.log10(iso[1]),None,10,3)]

# Open Diffuse Model
irf_dir = '/Users/mdwood/fermi/custom_irfs'
if 'CUSTOM_IRF_DIR' in os.environ:
    irf_dir = os.environ['CUSTOM_IRF_DIR']


irfs = []

print 'Loading IRFs'
if args.event_type == 'all':
    types = [None]
elif args.event_type == 'fb':
    types = ['front','back']
elif args.event_type == 'psf4':
    types = ['psf0','psf1','psf2','psf3']

for t in types:

    event_types = None
    if t is not None: event_types = [t.upper()]
    
    irf = IRFManager.create(args.irfs, event_types,
                            load_from_file=True,irf_dir=irf_dir)
    irfs.append(irf)

        
nbin = np.round((args.emax-args.emin)*8)
energy_axis = Axis.create(args.emin,args.emax,nbin)
nside = args.nside
min_counts = args.min_counts
ts_threshold = args.ts_threshold

print energy_axis

rebin=8/args.bins_per_decade
energy_axis2 = Axis(energy_axis.edges[::rebin])

theta_axis = Axis.create(-1.5,1.0,50)
cth_axis = Axis.create(0.2,1.0,16)

# Setup exposure
print 'Loading exposure model'
hp_bexp = []
for t in types:

    if t is None: bexpmap = args.bexpmap
    else: bexpmap = args.bexpmap.replace('all',t)
    
    im_bexp = SkyCube.createFromFITS(bexpmap)
    bexp = im_bexp.createHEALPixMap(nside,energy_axis=energy_axis)
    bexp *= args.bexp_scale
    hp_bexp += [bexp]


    
# Setup PSF
ltfile = '/u/gl/mdwood/ki20/mdwood/fermi/data/P8_SOURCE_V6/P8_SOURCE_V6_239557414_397345414_z100_r180_gti_ft1_gtltcube_z100.fits'

ltc = LTCube(ltfile)
print 'Creating PSF'

psf_pdf = []
psf_domega = []
for irf, bexp in zip(irfs,hp_bexp):

    haxis = Axis.create(0,bexp.axis(1).nbins,bexp.axis(1).nbins)
    c = bexp.slice(0,0).center()

    print 'Filling livetime'
    lth = Histogram2D(haxis,cth_axis)
    for i in range(lth.axis(0).nbins):

        th,phi = hp.pix2ang(nside,i)
        glat = np.degrees(np.pi/2.-th)
        glon = np.degrees(phi)
        ra,dec = gal2eq(glon,glat)        
        h = ltc.get_src_lthist(ra[0],dec[0],cth_axis)
        lth._counts[i,:] = h.counts
        
#        print i, th, phi, h.counts

#    plt.figure()
#    lth.slice(0,0).plot()
#    lth.slice(0,1000).plot()
#    lth.slice(0,2000).plot()
#    lth.slice(0,3000).plot()

        
#        ltc.get_src_lthist(ra,dec,cth_axis)
#    slat = np.sin(np.radians(c[1]))
#    lon = np.radians(c[0])
    
#    # Loop over sky bins
#    for i in range(lth.axis(2).nbins):
#    lt = ltc_hist.interpolate(lon[:,np.newaxis],
#                              slat[:,np.newaxis],
#                              cth_axis.center[:,np.newaxis])
    
    psf_q68 = irf.psf_quantile(energy_axis.center,0.6)

    theta_edges = theta_axis.edges[np.newaxis,:] + np.log10(psf_q68)[:,np.newaxis]
    theta = theta_axis.center[np.newaxis,:] + np.log10(psf_q68)[:,np.newaxis]

    domega = 2*np.pi*(np.cos(np.radians(10**theta_edges[:,:-1])) - 
                      np.cos(np.radians(10**theta_edges[:,1:])))

    aeff = irf.aeff(energy_axis.center[:,np.newaxis],
                    cth_axis.center[np.newaxis,:])

    print 'Evaluating PSF'
    psf = irf.psf(10**theta[:,:,np.newaxis],
                  #10**theta_axis.center[np.newaxis,:,np.newaxis],
                  energy_axis.center[:,np.newaxis,np.newaxis],
                  cth_axis.center[np.newaxis,np.newaxis,:])

    # PSF = E, Theta, Cth
    # Aeff = E, Cth
    # LT = S, Cth

    wpsf = np.zeros((energy_axis.nbins,haxis.nbins,theta_axis.nbins))

    for i in range(0,haxis.nbins,1000):
        s = slice(i,i+1000)    
        psf2 = (psf[:,np.newaxis,:,:]*
                aeff[:,np.newaxis,np.newaxis,:]*
                lth.counts[np.newaxis,s,np.newaxis,:])

        wpsf[:,s,:] = np.sum(psf2,axis=3)/np.sum(aeff[:,np.newaxis,np.newaxis,:]*
                                                 lth.counts[np.newaxis,s,np.newaxis,:],
                                                 axis=3)

    psf = wpsf       
    psf_q68_domega = np.radians(psf_q68)**2*np.pi

    psf *= (180./np.pi)**2
    psf *= domega[:,np.newaxis,:]
    psf_pdf.append(psf)
    
#    hist = HistogramND([energy_axis,haxis,theta_axis],counts=psf)
#    hist *= (180./np.pi)**2
#    hist = hist*domega[:,np.newaxis,:]
#    psf_pdf_hist.append(hist)
    psf_domega.append(domega[:,np.newaxis,:])

# Setup galactic diffuse model
print 'Loading diffuse model'
im_gdiff = FITSImage.createFromFITS(args.galdiff)
im_gdiff._counts = np.log10(im_gdiff._counts)
hp_gdiff = im_gdiff.createHEALPixMap(nside=nside,energy_axis=energy_axis)
hp_gdiff._counts = 10**hp_gdiff._counts

hp_bdiff = []

#for fn in fn_iso:
for i, t in enumerate(types):
    bdiff = copy.deepcopy(hp_gdiff)    
    for j in range(energy_axis.nbins):
        bdiff._counts[j] += 10**fn_iso[i](energy_axis.center[j])
    hp_bdiff += [bdiff]

    
deltae = 10**energy_axis.edges[1:] - 10**energy_axis.edges[:-1]
#domega = hp.nside2pixarea(hp_gdiff.nside)

print len(hp_bdiff), len(hp_bexp)

hp_bdiff_counts = []
for i,t in enumerate(types):

    hc = hp_bdiff[i]*hp_bexp[i]
    hc *= deltae[:,np.newaxis]
    hp_bdiff_counts += [hc]
    

print 'Computing sensitivity'
    
filename = '{prefix:s}_ebin_n{min_counts:04.1f}_'
filename += 'g{gamma:04.2f}_ts{ts_threshold:02.0f}_nside{nside:03d}'

filename = filename.format(**{'prefix' : args.prefix,
                              'min_counts':args.min_counts,
                              'gamma':args.gamma,
                              'ts_threshold' : args.ts_threshold,
                              'nside' : args.nside})

if args.ethresh is None and not os.path.isfile('%s.fits'%filename):
    hp_flux = compute_flux(hp_bexp,hp_bdiff_counts,psf_pdf,psf_domega,rebin,ts_threshold,min_counts,args.gamma)
    hp_flux.save('%s.fits'%filename)

if args.ethresh is None: ethresh = args.emin
else: ethresh = args.ethresh
    
filename = '{prefix:s}_powerlaw_n{min_counts:04.1f}_'
filename += 'g{gamma:04.2f}_e{ethresh:04.2f}_ts{ts_threshold:02.0f}_nside{nside:03d}'

filename = filename.format(**{'prefix' : args.prefix,
                              'min_counts':args.min_counts,
                              'gamma': args.gamma,
                              'ts_threshold' : args.ts_threshold,
                              'nside' : args.nside,
                              'ethresh' : ethresh})

if not os.path.isfile('%s.fits'%filename):
    hp_flux = compute_flux(hp_bexp,hp_bdiff_counts,psf_pdf,psf_domega,
                           energy_axis.nbins,
                           ts_threshold,min_counts,args.gamma,ethresh,escale=3.5)

    hp_flux.save('%s.fits'%filename)
    

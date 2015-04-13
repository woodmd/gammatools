#!/usr/bin/env python

import matplotlib

#try:             os.environ['DISPLAY']
#except KeyError: matplotlib.use('Agg')

matplotlib.interactive(False)
matplotlib.use('Agg')

from gammatools.core.fits_util import *
from gammatools.fermi.irf_util import *
from gammatools.fermi.psf_model import *
from gammatools.core.stats import poisson_lnl
from gammatools.core.util import *
from gammatools.core.bspline import BSpline, PolyFn
import matplotlib.pyplot as plt
import healpy as hp
import argparse
import glob

def poisson_ts(sig,bkg):
    return 2*(poisson_lnl(sig+bkg,sig+bkg) - poisson_lnl(sig+bkg,bkg))


def compute_flux(hp_bexp,hp_bdiff_counts,psf_pdf_hist,
                 psf_domega,rebin,ts_threshold,min_counts,
                 gamma):

#    from time import sleep
#    print 'Entering flux'
#    sleep(5)

    ntype = len(hp_bexp)
    energy_axis = hp_bexp[0].axis(0)
    energy_axis_rebin = Axis(energy_axis.edges[::rebin])

    deltae = 10**energy_axis.edges[1:] - 10**energy_axis.edges[:-1]
    
    hp_flux = HealpixSkyCube([energy_axis_rebin,hp_bexp[0].axis(1)])

    scale = 10**np.linspace(-0.5,4,25)
    ts = np.zeros((energy_axis_rebin.nbins,hp_bexp[0].axis(1).nbins))
    bexps = np.zeros((energy_axis_rebin.nbins,hp_bexp[0].axis(1).nbins))
    ts_scale = np.zeros((energy_axis_rebin.nbins,hp_bexp[0].axis(1).nbins,25))

    # Loop over spatial bins
    for k in range(0,hp_flux.axis(1).nbins,1000):

        ss = slice(k,k+1000)
        print k, ss

        # Loop over energy bins
        for i in range(0,energy_axis_rebin.nbins):

            es = slice(i*rebin,(i+1)*rebin)
            ebin = np.sum(energy_axis.center[es])/float(rebin)
            ew = (10**energy_axis.center[es]/10**ebin)**-gamma
            
            # Loop over event types

            bexps = None
            for j in range(ntype):
                bexpw = (hp_bexp[j].counts[es,ss,np.newaxis]*
                         deltae[es,np.newaxis,np.newaxis]*
                         ew[:,np.newaxis,np.newaxis])

                if bexps is None:
                    bexps = np.sum(bexpw,axis=0)
                else:
                    bexps += np.sum(bexpw,axis=0)

            for j in range(ntype):

                bexpw = (hp_bexp[j].counts[es,ss,np.newaxis]*
                         deltae[es,np.newaxis,np.newaxis]*
                         ew[:,np.newaxis,np.newaxis])

                sig = (min_counts*psf_pdf_hist[j].counts[es,np.newaxis,:]*
                       bexpw/bexps[np.newaxis,:,:])
                sig_scale = sig[:,:,:,np.newaxis]*scale[np.newaxis,np.newaxis,:]

#            bkg = hp_bdiff*hp_bexp
#            hp_gdiff_counts *= deltae[:,np.newaxis]
            
                bkg = hp_bdiff_counts[j].counts[es,ss,np.newaxis]*psf_domega[j][es,np.newaxis,:]
            
                print i, j, sig.shape, bkg[es,ss,:].shape, bexps.shape

                ts[i,ss] += np.squeeze(np.apply_over_axes(np.sum,poisson_ts(sig,bkg),axes=[0,2]))
                ts_scale[i,ss,:] += np.squeeze(np.apply_over_axes(np.sum,
                                                                  poisson_ts(sig_scale,
                                                                             bkg[:,:,:,np.newaxis]),
                                                                  axes=[0,2]))
#                print ts[0,0]
#                print ts_scale[0,0,:]
#                print np.sum(sig[:,0])
#                print np.sum(psf_pdf_hist[j].counts[0,np.newaxis,:])
#                print np.sum(bexpw/bexps[np.newaxis,:,:])
                
            hp_flux._counts[i,ss] = min_counts/np.squeeze(bexps)

#    print ts.shape
#    ts_scale[ts_scale<0]=0
    
    # Loop over energy bins
    for i in range(0,energy_axis_rebin.nbins):

        # Loop over spatial bins
        for j in range(hp_flux.axis(1).nbins):

            if ts[i,j] >= ts_threshold: continue
    #            hp_flux._counts[i,j] = min_counts/bexps[j,...]
    #        else:
#            print i, j, ts[i,j]
            
            try:
                fn = PolyFn.fit(4,np.log10(scale),np.log10(ts_scale[i,j]))
                r = fn.roots(offset=np.log10(ts_threshold),imag=False)
                r = r[(r > np.log10(scale[0]))&(r < np.log10(scale[-1]))]
            except Exception, e:
                print e.message
                print 'EXCEPTION ', i, j, ts[i,j], ts_scale[i,j], r
#                plt.figure()
#                plt.plot(np.log10(scale),np.log10(ts_scale[j]))
#                plt.plot(np.log10(scale),fn(np.log10(scale)))
#                plt.axhline(np.log10(ts_threshold))
#                plt.show()
            hp_flux._counts[i,j] *= 10**r
            
    return hp_flux
    
usage = "usage: %(prog)s [options] [ft1file]"
description = "Run both gtmktime and gtselect on an FT1 file."
parser = argparse.ArgumentParser(usage=usage,description=description)

#parser.add_argument('files', nargs='+')
parser.add_argument('--min_counts', default=3.0, type=float)
parser.add_argument('--ts_threshold', default=25.0, type=float)
parser.add_argument('--bexp_scale', default=1.0, type=float)
parser.add_argument('--nside', default=16, type=int)
parser.add_argument('--irfs', default='P8_SOURCE_V5')
parser.add_argument('--bexpmap', default=None,required=True)
parser.add_argument('--galdiff', default=None,required=True)
parser.add_argument('--isodiff', default=None,required=True)
parser.add_argument('--prefix', default='p8source')
parser.add_argument('--gamma', default=2.0,type=float)
parser.add_argument('--emin', default=1.5,type=float)
parser.add_argument('--emax', default=5.75,type=float)
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

print energy_axis

rebin=8/args.bins_per_decade
energy_axis2 = Axis(energy_axis.edges[::rebin])

theta_axis = Axis.create(-1.5,1.0,50)
cth_axis = Axis.create(0.2,1.0,16)

psf_pdf_hist = []
psf_domega = []
for irf in irfs:
    psf_q68 = irf.psf_quantile(energy_axis.center,0.6)

    theta_edges = theta_axis.edges[np.newaxis,:] + np.log10(psf_q68)[:,np.newaxis]
    theta = theta_axis.center[np.newaxis,:] + np.log10(psf_q68)[:,np.newaxis]

    domega = 2*np.pi*(np.cos(np.radians(10**theta_edges[:,:-1])) - 
                      np.cos(np.radians(10**theta_edges[:,1:])))

    aeff = irf.aeff(energy_axis.center[:,np.newaxis],
                    cth_axis.center[np.newaxis,:])

    print 'Evaluating PSF'
    psf = irf.psf(10**theta[:,:,np.newaxis],  #10**theta_axis.center[np.newaxis,:,np.newaxis],
                  energy_axis.center[:,np.newaxis,np.newaxis],
                  cth_axis.center[np.newaxis,np.newaxis,:])
    
    psf = np.sum(psf*aeff[:,np.newaxis,:],axis=2)
    psf /= np.sum(aeff,axis=1)[:,np.newaxis]    
    psf_q68_domega = np.radians(psf_q68)**2*np.pi

    aeff_hist = Histogram2D(energy_axis,cth_axis,counts=aeff)
    hist = Histogram2D(energy_axis,theta_axis,counts=psf)
    hist *= (180./np.pi)**2
    hist = hist*domega

    psf_pdf_hist.append(hist)
    psf_domega.append(domega)

    
nside = args.nside
min_counts = args.min_counts
ts_threshold = args.ts_threshold

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

    
print 'Loading exposure model'
hp_bexp = []
for t in types:

#    print t, args.bexpmap.replace('all',t)
    if t is None: bexpmap = args.bexpmap
    else: bexpmap = args.bexpmap.replace('all',t)
    
    im_bexp = SkyCube.createFromFITS(bexpmap)
    bexp = im_bexp.createHEALPixMap(hp_gdiff.nside,hp_gdiff.axis(0))
    bexp *= args.bexp_scale
    hp_bexp += [bexp]


deltae = 10**energy_axis.edges[1:] - 10**energy_axis.edges[:-1]
#domega = hp.nside2pixarea(hp_gdiff.nside)

print len(hp_bdiff), len(hp_bexp)


hp_bdiff_counts = []
for i,t in enumerate(types):

    hc = hp_bdiff[i]*hp_bexp[i]
    hc *= deltae[:,np.newaxis]
    hp_bdiff_counts += [hc]
    

print 'Computing sensitivity'
    
filename = '{prefix:s}_ebin2_n{min_counts:04.1f}_'
filename += 'g{gamma:04.2f}_ts{ts_threshold:02.0f}_nside{nside:03d}'

filename = filename.format(**{'prefix' : args.prefix,
                              'min_counts':args.min_counts,
                              'gamma':args.gamma,
                              'ts_threshold' : args.ts_threshold,
                              'nside' : args.nside})

hp_flux = compute_flux(hp_bexp,hp_bdiff_counts,psf_pdf_hist,psf_domega,rebin,ts_threshold,min_counts,args.gamma)
hp_flux.save('%s.fits'%filename)



filename = '{prefix:s}_powerlaw2_n{min_counts:04.1f}_'
filename += 'g{gamma:04.2f}_ts{ts_threshold:02.0f}_nside{nside:03d}'

filename = filename.format(**{'prefix' : args.prefix,
                              'min_counts':args.min_counts,
                              'gamma': args.gamma,
                              'ts_threshold' : args.ts_threshold,
                              'nside' : args.nside})
    
hp_flux = compute_flux(hp_bexp,hp_bdiff_counts,psf_pdf_hist,psf_domega,
                       energy_axis.nbins,
                       ts_threshold,min_counts,args.gamma)

hp_flux.save('%s.fits'%filename)
    
sys.exit(0)

#sig0 = 1.*psf_pdf_hist.counts[:,np.newaxis,:]
#sig1 = 5.*np.sqrt(psf_q68_domega[:,np.newaxis]*hp_gdiff_counts.counts)
#sig1 = sig1[:,:,np.newaxis]*psf_pdf_hist.counts[:,np.newaxis,:]

#ts0 = poisson_ts(sig0,bkg)
#ts0 = np.sum(ts0,axis=2)
#ts1 = poisson_ts(sig1,bkg)
#ts1 = np.sum(ts1,axis=2)
#hp_ts0 = HealpixSkyCube(hp_bexp.axes(),counts=ts0)
#hp_ts1 = HealpixSkyCube(hp_bexp.axes(),counts=ts1)



print 'Creating flux hist'
hp_flux = HealpixSkyCube([energy_axis2,hp_bexp.axis(1)])

scale = 10**np.linspace(-0.5,5,25)
ts = np.zeros((energy_axis2.nbins,hp_bexp.axis(1).nbins))
ts_scale = np.zeros((energy_axis2.nbins,hp_bexp.axis(1).nbins,25))

for k in range(0,hp_flux.axis(1).nbins,1000):

    ss = slice(k,k+1000)
    print k, ss

    # Loop over energy bins
    for i in range(0,energy_axis2.nbins):

        es = slice(i*rebin,(i+1)*rebin)
        ebin = np.sum(energy_axis.center[es])/float(rebin)
        ew = (10**energy_axis.center[es]/10**ebin)**-args.gamma
        
        bexpw = (hp_bexp.counts[es,ss,np.newaxis]*
                 deltae[es,np.newaxis,np.newaxis]*
                 ew[:,np.newaxis,np.newaxis])
        bexps = np.sum(bexpw,axis=0)

        sig = (min_counts*psf_pdf_hist.counts[es,np.newaxis,:]*
               bexpw/bexps[np.newaxis,:,:])
        sig_scale = sig[:,:,:,np.newaxis]*scale[np.newaxis,np.newaxis,:]

        print i, sig.shape, bkg[es,ss,:].shape, bexps.shape
        
        ts[i,ss] = np.squeeze(np.apply_over_axes(np.sum,poisson_ts(sig,bkg[es,ss,:]),axes=[0,2]))
        ts_scale[i,ss,:] = np.squeeze(np.apply_over_axes(np.sum,
                                                         poisson_ts(sig_scale,bkg[es,ss,:,np.newaxis]),
                                                         axes=[0,2]))

        hp_flux._counts[i,ss] = min_counts/np.squeeze(bexps)
        
# Loop over energy bins
for i in range(0,energy_axis2.nbins):

    # Loop over spatial bins
    for j in range(hp_flux.axis(1).nbins):

        if ts[i,j] >= ts_threshold: continue
#            hp_flux._counts[i,j] = min_counts/bexps[j,...]
#        else:

        try:
            fn = PolyFn.fit(4,np.log10(scale),np.log10(ts_scale[i,j]))
            r = fn.roots(offset=np.log10(ts_threshold),imag=False)
            r = r[(r > np.log10(scale[0]))&(r < np.log10(scale[-1]))]
        except Exception, e:
            print ts_scale[j,0], r
            plt.figure()
            plt.plot(np.log10(scale),np.log10(ts_scale[j]))
            plt.plot(np.log10(scale),fn(np.log10(scale)))
            plt.axhline(np.log10(ts_threshold))
            plt.show()
        hp_flux._counts[i,j] *= 10**r
#*min_counts/bexps[j,...]

#hp_flux.slice(0,2).plot()
#plt.show()

            
hp_flux.save('%s_dflux.fits'%filename)


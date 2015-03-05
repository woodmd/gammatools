from gammatools.core.fits_util import *
from gammatools.fermi.irf_util import *
from gammatools.fermi.psf_model import *
from gammatools.core.stats import poisson_lnl
from gammatools.core.bspline import BSpline, PolyFn
import matplotlib.pyplot as plt
import healpy as hp
import argparse

def poisson_ts(sig,bkg):
    return 2*(poisson_lnl(sig+bkg,sig+bkg) - poisson_lnl(sig+bkg,bkg))

usage = "usage: %(prog)s [options] [ft1file]"
description = "Run both gtmktime and gtselect on an FT1 file."
parser = argparse.ArgumentParser(usage=usage,description=description)

#parser.add_argument('files', nargs='+')
parser.add_argument('--min_counts', default=3, type=float)
parser.add_argument('--ts_threshold', default=25.0, type=float)
parser.add_argument('--bexp_scale', default=1.0, type=float)
parser.add_argument('--nside', default=16, type=int)
parser.add_argument('--irf', default='P8_SOURCE_V5')
parser.add_argument('--bexpmap', default=None,required=True)
parser.add_argument('--prefix', default='p8source')
parser.add_argument('--emin', default=1.5,type=float)
parser.add_argument('--emax', default=5.75,type=float)
parser.add_argument('--bins_per_decade', default=4,type=int)

args = parser.parse_args()


filename = '{prefix:s}_n{min_counts:04.1f}_ts{ts_threshold:02.0f}_nside{nside:03d}'

filename = filename.format(**{'prefix' : args.prefix,
                              'min_counts':args.min_counts,
                              'ts_threshold' : args.ts_threshold,
                              'nside' : args.nside})

iso = np.loadtxt('isotropic_source_4years_P8V3.txt',unpack=True)

fn_iso = BSpline.fit(np.log10(iso[0]),np.log10(iso[1]),None,10,3)

# Open Diffuse Model
irf_dir = '/Users/mdwood/fermi/custom_irfs'
if 'CUSTOM_IRF_DIR' in os.environ:
    irf_dir = os.environ['CUSTOM_IRF_DIR']


irf = None
m = None

irf = IRFManager.create(args.irf, True,irf_dir=irf_dir)
#ltfile = '/Users/mdwood/fermi/data/p301/ltcube_5years_zmax100.fits'
m = PSFModelLT(irf,src_type='iso')

nbin = np.round((args.emax-args.emin)*8)
energy_axis = Axis.create(args.emin,args.emax,nbin)

print energy_axis

rebin=8/args.bins_per_decade
energy_axis2 = Axis(energy_axis.edges[::rebin])

theta_axis = Axis.create(-1.5,1.0,50)

psf_q68 = irf.psf_quantile(energy_axis.center,0.6)

theta_edges = theta_axis.edges[np.newaxis,:] + np.log10(psf_q68)[:,np.newaxis]
theta = theta_axis.center[np.newaxis,:] + np.log10(psf_q68)[:,np.newaxis]

psf_domega = 2*np.pi*(np.cos(np.radians(10**theta_edges[:,:-1])) - 
                      np.cos(np.radians(10**theta_edges[:,1:])))

#psf_domega = 2*np.pi*(np.cos(np.radians(10**theta_axis.edges[:-1])) - 
#                      np.cos(np.radians(10**theta_axis.edges[1:])))

cth_axis = Axis.create(0.2,1.0,16)

aeff = irf.aeff(energy_axis.center[:,np.newaxis],
                cth_axis.center[np.newaxis,:])

psf = irf.psf(10**theta[:,:,np.newaxis],  #10**theta_axis.center[np.newaxis,:,np.newaxis],
              energy_axis.center[:,np.newaxis,np.newaxis],
              cth_axis.center[np.newaxis,np.newaxis,:])
              
psf = np.sum(psf*aeff[:,np.newaxis,:],axis=2)
psf /= np.sum(aeff,axis=1)[:,np.newaxis]


psf_q68_domega = np.radians(psf_q68)**2*np.pi

aeff_hist = Histogram2D(energy_axis,cth_axis,counts=aeff)
psf_hist = Histogram2D(energy_axis,theta_axis,counts=psf)
psf_hist *= (180./np.pi)**2

psf_pdf_hist = psf_hist*psf_domega

nside = args.nside
min_counts = args.min_counts
ts_threshold = args.ts_threshold

# Setup galactic diffuse model

im_gdiff = FITSImage.createFromFITS('template_4years_P8_V2_scaled.fits')
im_gdiff._counts = np.log10(im_gdiff._counts)
hp_gdiff = im_gdiff.createHEALPixMap(nside=nside,energy_axis=energy_axis)
hp_gdiff._counts = 10**hp_gdiff._counts

for i in range(energy_axis.nbins):
    hp_gdiff._counts[i] += 10**fn_iso(energy_axis.center[i])

im_bexp = SkyCube.createFromFITS(bexpmap_file)
hp_bexp = im_bexp.createHEALPixMap(hp_gdiff.nside,hp_gdiff.axis(0))
hp_bexp *= args.bexp_scale

deltae = 10**energy_axis.edges[1:] - 10**energy_axis.edges[:-1]
domega = hp.nside2pixarea(hp_gdiff.nside)

hp_gdiff_counts = hp_gdiff*hp_bexp
hp_gdiff_counts *= deltae[:,np.newaxis]

#hp_sig_flux = HealpixSkyCube(hp_bexp.axes(),counts=min_counts)
#hp_sig_flux /= hp_bexp
#hp_sig_flux /= deltae[:,np.newaxis]

print 'Creating bkg'
bkg = hp_gdiff_counts.counts[:,:,np.newaxis]*psf_domega[:,np.newaxis,:]

print bkg.size

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


for i in range(0,energy_axis.nbins,rebin):

    ebin = np.sum(energy_axis.center[i:i+rebin])/float(rebin)
    ew = (10**energy_axis.center[i:i+rebin]/10**ebin)**-2.0    
    es = slice(i,i+rebin)

    print i, ebin, ew

    # Get Weighted Exposure
    bexpw = hp_bexp.counts[es,:,np.newaxis]*deltae[es,np.newaxis,np.newaxis]*ew[:,np.newaxis,np.newaxis]
    bexps = np.sum(bexpw,axis=0)

    scale = (10**np.linspace(-0.5,4,25))[np.newaxis,np.newaxis,:]

    sig = min_counts*psf_pdf_hist.counts[es,np.newaxis,:]*bexpw/bexps[np.newaxis,:,:] 
    sig_scale = sig[:,:,:,np.newaxis]*scale

    ts = np.apply_over_axes(np.sum,poisson_ts(sig,bkg[es,:,:]),axes=[0,2])
    ts_scale = np.apply_over_axes(np.sum,poisson_ts(sig_scale,bkg[es,:,:,np.newaxis]),axes=[0,2])

    ts = np.squeeze(ts)
    ts_scale = np.squeeze(ts_scale)

#    ipix0 = hp.ang2pix(hp_scale.nside,0.0,0.0)
#    ipix1 = hp.ang2pix(hp_scale.nside,np.pi/2.,0.0)

    scale = np.squeeze(scale)

    for j in range(hp_flux.axis(1).nbins):


        if ts[j] >= ts_threshold: 
            hp_flux._counts[i/rebin,j] = min_counts/bexps[j,...]
        else:

            try:
                fn = PolyFn.fit(4,np.log10(scale),np.log10(ts_scale[j]))
                r = fn.roots(offset=np.log10(ts_threshold),imag=False)
                r = r[(r > np.log10(scale[0]))&(r < np.log10(scale[-1]))]
            except Exception, e:
                print ts_scale[j,0], r
                plt.figure()
                plt.plot(np.log10(scale),np.log10(ts_scale[j]))
                plt.plot(np.log10(scale),fn(np.log10(scale)))
                plt.axhline(np.log10(ts_threshold))
                plt.show()

            hp_flux._counts[i/rebin,j] = 10**r*min_counts/bexps[j,...]


#        if j %1000 == 0:
#            print i, j, ts[j], r
        
#    scale = hp_scale._counts[i,:][:,np.newaxis]
#    ts = np.sum(poisson_ts(sig*scale,bkg[i,:,:]),axis=1)

#    print ts
#    print np.sum(sig*scale,axis=1)


#hp_flux = hp_scale*hp_sig_flux
hp_flux.save('%s_dflux.fits'%filename)


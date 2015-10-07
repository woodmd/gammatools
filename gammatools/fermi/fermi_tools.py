import matplotlib

#try:             os.environ['DISPLAY']
#except KeyError: matplotlib.use('Agg')

matplotlib.interactive(False)
matplotlib.use('Agg')

from gammatools.core.fits_util import *
from gammatools.fermi.irf_util import *
from gammatools.fermi.psf_model import *
from gammatools.fermi.fermi_tools import *
from gammatools.core.stats import poisson_lnl
from gammatools.core.util import *
from gammatools.core.bspline import BSpline, PolyFn
import matplotlib.pyplot as plt
import healpy as hp
import argparse
import glob

def poisson_ts(sig,bkg):
    return 2*(poisson_lnl(sig+bkg,sig+bkg) - poisson_lnl(sig+bkg,bkg))

def compute_flux(hp_bexp,hp_bdiff_counts,psf_pdf,
                 psf_domega,rebin,ts_threshold,min_counts,
                 gamma,ethresh=None,escale=None):

    ntype = len(hp_bexp)
    energy_axis = hp_bexp[0].axis(0)
    energy_axis_rebin = Axis(energy_axis.edges[::rebin])

    imin = 0
    if ethresh: imin = energy_axis.valToEdge(ethresh)[0]

    print 'imin ', imin, energy_axis.edges[imin]
    
    deltae = 10**energy_axis.edges[1:] - 10**energy_axis.edges[:-1]
    
    hp_flux = HealpixSkyCube([energy_axis_rebin,hp_bexp[0].axis(1)])

    scale = 10**np.linspace(-0.5,4,25)
    ts = np.zeros((energy_axis_rebin.nbins,hp_bexp[0].axis(1).nbins))
    bexps = np.zeros((energy_axis_rebin.nbins,hp_bexp[0].axis(1).nbins))
    ts_scale = np.zeros((energy_axis_rebin.nbins,hp_bexp[0].axis(1).nbins,
                         len(scale)))

    # Loop over spatial bins
    for k in range(0,hp_flux.axis(1).nbins,1000):

        ss = slice(k,k+1000)
        print k, ss

        # Loop over energy bins
        for i in range(0,energy_axis_rebin.nbins):

            es = slice(max(imin,i*rebin),(i+1)*rebin)

            if escale is not None: ebin=escale
            else: ebin = np.sum(energy_axis.center[es])/float(rebin)
            ew = (10**energy_axis.center[es]/10**ebin)**-gamma

            print i, ebin, es
            
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

                sig = (min_counts*psf_pdf[j][es,ss,:]*
                       bexpw/bexps[np.newaxis,:,:])
                sig_scale = sig[:,:,:,np.newaxis]*scale[np.newaxis,np.newaxis,:]

#            bkg = hp_bdiff*hp_bexp
#            hp_gdiff_counts *= deltae[:,np.newaxis]
            
                bkg = hp_bdiff_counts[j].counts[es,ss,np.newaxis]*psf_domega[j][es,:,:]
            
                print i, j, sig.shape, bkg[es,ss,:].shape, bexps.shape
                ts[i,ss] += np.squeeze(np.apply_over_axes(np.sum,poisson_ts(sig,bkg),axes=[0,2]))
                ts_scale[i,ss,:] += np.squeeze(np.apply_over_axes(np.sum,
                                                                  poisson_ts(sig_scale,
                                                                             bkg[:,:,:,np.newaxis]),
                                                                  axes=[0,2]))
                
            hp_flux._counts[i,ss] = min_counts/np.squeeze(bexps)
    
    # Loop over energy bins
    for i in range(0,energy_axis_rebin.nbins):

        # Loop over spatial bins
        for j in range(hp_flux.axis(1).nbins):

            if ts[i,j] >= ts_threshold: continue
            
            try:

                msk = (ts_scale[i,j] > ts_threshold/20.)&(ts_scale[i,j] < ts_threshold*20)
                fn = PolyFn.fit(4,np.log10(scale[msk]),np.log10(ts_scale[i,j][msk]))
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

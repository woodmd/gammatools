#!/usr/bin/env python

import matplotlib.pyplot as plt
import gammatools
from gammatools.dm.dmmodel import *
from gammatools.core.util import *
from gammatools.dm.jcalc import *
from gammatools.dm.halo_model import *
from gammatools.dm.irf_model import *
from gammatools.core.plot_util import *
import sys
import yaml
import copy

import argparse

def make_halo_plots(halo_models,cat):

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    for h in halo_models:
        dp = DensityProfile.create(cat[h])
        jp = LoSIntegralFn.create(cat[h])

        prefix = h + '_'

        x = np.linspace(-1.0,2,100)


        ax0.plot(10**x,dp.rho(10**x*Units.kpc)/Units.gev_cm3,label=h)

        psi_edge = np.radians(10**np.linspace(np.log10(0.1),np.log10(45.),100))

        psi = 0.5*(psi_edge[1:] + psi_edge[:-1])

        jval = jp(psi)

        domega = 2*np.pi*(np.cos(psi_edge[1:]) - psi_edge[:-1])
        jcum = np.cumsum(domega*jval)

        ax1.plot(np.degrees(psi),jval/Units.gev2_cm5,label=h)
        ax2.plot(np.degrees(psi),jcum/Units.gev2_cm5,label=h)


        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.axvline(8.5,label='Solar Radius')
        ax0.grid(True)
        ax0.set_xlabel('Distance [kpc]')
        ax0.set_ylabel('DM Density [GeV cm$^{-3}$]')

        ax0.legend()

        fig0.savefig(prefix + 'density_profile.png')

    #psi = np.arctan(10**x*Units.kpc/(8.5*Units.kpc))

        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.grid(True)
        ax1.set_xlabel('Angle [deg]')
        ax1.set_ylabel('J Factor [GeV$^{-2}$ cm$^{-5}$ sr$^{-1}$]')

        ax1.legend()

        fig1.savefig(prefix + 'jval.png')

        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.grid(True)
        ax2.set_xlabel('GC Angle [deg]')
        ax2.set_ylabel('Cumulative J Factor [GeV$^{-2}$ cm$^{-5}$]')

        ax2.legend()

        fig2.savefig(prefix + 'jcum.png')

usage = "usage: %(prog)s [options]"
description = """Compute the annihilation flux and yield spectrum for a 
DM halo and WIMP model."""

parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('--halo_model', default=None, required=True,
                    help = 'Set the name of the halo model.  This will be used '
                    'to look up the parameters for this halo from the halo '
                    'model file.')

parser.add_argument('--channel', default=None, required=True,
                    help = 'Set the DM annihilation/decay channel.')

parser.add_argument('--sigmav', default=3E-26, 
                    help = 'Set the annihilation cross section in cm^3 s^{-1}.')

parser.add_argument('--mass', default='1.0/4.0/25', 
                    help = 'Set the array of WIMP masses at which the DM flux '
                    'will be evaluation.')

parser.add_argument('--irf', default=None)

parser.add_argument('--psi', default='0.0/10.0/100', 
                    help = 'Set the array of WIMP masses at which the DM flux '
                    'will be evaluation.')

args = parser.parse_args()

hm = HaloModelFactory.create(args.halo_model)

massv = [float(t) for t in args.mass.split('/')]
massv = np.linspace(massv[0],massv[1],massv[2])
sigmav = args.sigmav*Units.cm3_s

if args.irf is not None:
    irf = IRFModel.createCTAIRF(args.irf)
    chm = ConvolvedHaloModel(hm,irf)
else:
    irf = None
    chm = hm

flux_header = 'Column 0 Energy Bin Center [Log10(E/MeV)]\n'
flux_header += 'Column 1 Halo Offset Bin Angle Center [deg]\n'
flux_header += 'Column 2 Halo Offset Bin Angle Low Edge [deg]\n'
flux_header += 'Column 3 Halo Offset Bin Angle High Edge [deg]\n'
flux_header += 'Column 4 E^2 dF/dE [MeV cm^{-2} s^{-1} sr^{-1}]\n'
flux_header += 'Column 5 dF/dE [MeV^{-1} cm^{-2} s^{-1} sr^{-1}]\n'

yield_header = 'Column 0 Energy [Log10(E/MeV)]\n'
yield_header += 'Column 1 Annihilation Yield [MeV^{-1}]\n'

halo_header = 'Column 0 Halo Offset Angle [radians]\n'
halo_header += 'Column 1 Halo Offset Angle [degrees]\n'
halo_header += 'Column 2 J Factor [GeV^{2} cm^{-5} sr^{-1}]\n'
halo_header += 'Column 3 Integrated J Factor [GeV^{2} cm^{-5}]\n'

#dp = DensityProfile.create(hm)
#jp = LoSIntegralFn.create(hm)

jp = hm.jp


if args.psi:
    x = [float(t) for t in args.psi.split('/')]
    psi_axis = Axis.create(np.radians(x[0]),np.radians(x[1]),x[2])
else:
    psi_edge = np.radians(10**np.linspace(np.log10(0.01),np.log10(45.),1000))    
    psi_axis = Axis(psi_edge)

#psi = 0.5*(psi_edge[1:] + psi_edge[:-1])
#psi_edge = np.radians(10**np.linspace(np.log10(0.01),np.log10(45.),200))
#psi = 0.5*(psi_edge[1:] + psi_edge[:-1])

#print psi
#print psi2

psi = psi_axis.center

print np.degrees(psi)

jval = hm.losfn(psi_axis.center)
jval_cum = hm._jp.cumsum(psi_axis.center)
domega = -2*np.pi*(np.cos(psi_axis.edges[1:])-np.cos(psi_axis.edges[:-1]))

h = copy.copy(halo_header)

halo_info  = 'Halo rs       = %10.5g [kpc]\n'%(hm.dp._rs/Units.kpc)
halo_info += 'Halo rhos     = %10.5g [GeV cm^{-3}]\n'%(hm.dp._rhos/Units.gev_cm3)
halo_info += 'Halo Distance = %10.5g [kpc]\n'%(hm.dist/Units.kpc)
halo_info += 'Halo Model    = %s'%(hm.dp.__class__.__name__)

h += halo_info

np.savetxt('jvalue_%s.txt'%(args.halo_model),
           np.array((psi_axis.center,psi_axis.center/Units.deg,
                     jval/Units.gev2_cm5,jval_cum/Units.gev2_cm5)).T,
           fmt='%12.5g',header=h)

for m in massv:

    mass = 10**m*Units.gev

    src_model = DMFluxModel.createChanModel(chm,mass,sigmav,args.channel)

    sp = DMChanSpectrum(args.channel,mass=mass)

    loge = np.linspace(-1.0,4.0,16*5+1)
    loge += np.log10(Units.gev)

    loge_axis = Axis(loge)

    yld = sp.dnde(loge_axis.center)

    flux = 1./(8.*np.pi)*sigmav*np.power(mass,-2)*sp.dnde(loge_axis.center)
    flux[flux<=0] = 0

    e2flux = flux*10**(2*loge_axis.center)

#    e2flux = np.outer(jval,e2flux)
#    flux = np.outer(jval,flux)
    e2flux = np.outer(e2flux,jval)
    flux = np.outer(flux,jval)

    x,y = np.meshgrid(loge_axis.center,psi_axis.center,indexing='ij')
    x,ylo = np.meshgrid(loge_axis.center,psi_axis.edges[:-1],indexing='ij')
    x,yhi = np.meshgrid(loge_axis.center,psi_axis.edges[1:],indexing='ij')

    flux2 = src_model.flux(np.ravel(x),np.ravel(y))
    flux2 = flux2.reshape(x.shape)

    h = copy.copy(flux_header)
    h += 'Mass                 = %10.5g [GeV]\n'%10**m
    h += 'Cross Section        = %10.5g [cm^{3} s^{-1}]\n'%(sigmav/Units.cm3_s)
    h += 'Annihilation Channel = %s\n'%(args.channel)
    h += halo_info


    np.savetxt('flux_%s_%s_m%06.3f.txt'%(args.halo_model,args.channel,m),
               np.array((np.ravel(x)-np.log10(Units.mev),
                         np.ravel(y)/Units.deg,
                         np.ravel(ylo)/Units.deg,
                         np.ravel(yhi)/Units.deg,
                         np.ravel(e2flux)/Units.mev,
                         np.ravel(flux)/(1./Units.mev))).T,
               fmt='%12.5g',
               header=h)

    h = copy.copy(yield_header)
    h += 'Mass                 = %10.5g [GeV]\n'%10**m
    h += 'Cross Section        = %10.5g [cm^{3} s^{-1}]\n'%(sigmav/Units.cm3_s)
    h += 'Annihilation Channel = %s\n'%(args.channel)

    np.savetxt('yield_%s_m%06.3f.txt'%(args.channel,m),
               np.array((loge_axis.center-np.log10(Units.mev),yld/Units._mev)).T,
               fmt='%12.5g',header=h)


sys.exit(0)

ft = FigTool()

for m in massv:

    mass = 10**m*Units.gev

    src_model = DMFluxModel.createChanModel(chm,mass,sigmav,args.channel)

    sp = DMChanSpectrum(args.channel,mass=mass)

    loge = np.linspace(1.1,4.1,5*3+1)
    loge += np.log10(Units.gev)

    loge_axis = Axis(loge)

    deltae = 10**loge_axis.edges[1:] - 10**loge_axis.edges[:-1]

    x,y = np.meshgrid(loge_axis.center,psi_axis.center,indexing='ij')
    xlo = np.ones(x.shape)*loge_axis.edges[:-1]
    xhi = np.ones(x.shape)*loge_axis.edges[1:]
    x,ylo = np.meshgrid(loge_axis.center,psi_axis.edges[:-1],indexing='ij')
    x,yhi = np.meshgrid(loge_axis.center,psi_axis.edges[1:],indexing='ij')

    flux = src_model.flux(np.ravel(x),np.ravel(y))
    flux = flux.reshape(x.shape)

    aeff = irf.aeff(loge_axis.center - Units.log10_mev)
    sig_counts_diffrate = flux*aeff[np.newaxis,:]*domega[:,np.newaxis]
    sig_counts_rate = flux*aeff[np.newaxis,:]*domega[:,np.newaxis]*deltae[np.newaxis,:]

    sig_counts_rate_density = sig_counts_rate/domega[:,np.newaxis]

    bkg_rate = irf.bkg(loge_axis.center - Units.log10_mev)
    bkg_counts_diffrate = bkg_rate*domega[:,np.newaxis]/10**loge_axis.center/np.log(10.)
    bkg_counts_rate = bkg_counts_diffrate*deltae[np.newaxis,:]
    
    bkg_counts_rate_density = bkg_counts_rate/domega[:,np.newaxis]
    e2flux = src_model.e2flux(np.ravel(x),np.ravel(y))
    e2flux = e2flux.reshape(x.shape)

    h = ''
    
    h += 'Column 0 Halo Offset Bin Angle Low Edge [deg]\n'
    h += 'Column 1 Halo Offset Bin Angle High Edge [deg]\n'
    h += 'Column 2 Energy Bin Low Edge [GeV]\n'
    h += 'Column 3 Energy Bin High Edge [GeV]\n'    
    h += 'Column 4 Convolved Flux E^2 dF/dE [MeV cm^{-2} s^{-1} sr^{-1}]\n'
    h += 'Column 5 Convolved Flux dF/dE [MeV^{-1} cm^{-2} s^{-1} sr^{-1}]\n'
    h += 'Column 6 Differential Signal Rate dN/dEdt [MeV^{-1} s^{-1}]\n'
    h += 'Column 7 Signal Rate dN/dt [s^{-1}]\n'
    h += 'Column 8 Signal Rate Density dN/dtdOmega [s^{-1} deg^{-2}]\n'
    h += 'Column 9 Differential Background Rate dN/dEdt [MeV^{-1} s^{-1}]\n'
    h += 'Column 10 Background Rate dN/dt [s^{-1}]\n'
    h += 'Column 11 Background Rate Density dN/dtdOmega [s^{-1} deg^{-2}]\n'

    h += 'Mass                 = %10.5g [GeV]\n'%10**m
    h += 'Cross Section        = %10.5g [cm^{3} s^{-1}]\n'%(sigmav/Units.cm3_s)
    h += 'Annihilation Channel = %s\n'%(args.channel)
    h += halo_info
    

    np.savetxt('counts_%s_%s_m%06.3f.txt'%(args.halo_model,args.channel,m),
               np.array((
                np.ravel(ylo)/Units.deg,
                np.ravel(yhi)/Units.deg,
                10**np.ravel(xlo)/Units.gev,
                10**np.ravel(xhi)/Units.gev,                
                np.ravel(e2flux)/Units.mev,
                np.ravel(flux)/(1./Units.mev),                
                np.ravel(sig_counts_diffrate)/(1./Units.mev),
                np.ravel(sig_counts_rate),
                np.ravel(sig_counts_rate_density)/Units._deg2,
                np.ravel(bkg_counts_diffrate)/(1./Units.mev),
                np.ravel(bkg_counts_rate),
                np.ravel(bkg_counts_rate_density)/Units._deg2,
                         )).T,
               fmt='%12.6g',
               header=h)


    fig = ft.create('counts_%s_%s_m%06.3f'%(args.halo_model,args.channel,m),nax=1,figstyle='ratio2',
                    norm_interpolation='lin',yscale='log',ylabel='Rate [s$^{-1}$ deg$^{-2}$]',
                    xlabel='Energy [log$_{10}$(E/GeV)]',legend_loc='upper right')

    fig[0].add_data(loge_axis.center-np.log10(Units.gev),bkg_counts_rate_density[0,:],marker='o',
                    label='Background',color='k')

    signal_bins = [0,1,3,9]

    for i in signal_bins:
        fig[0].add_data(loge_axis.center-np.log10(Units.gev),sig_counts_rate_density[i,:],marker='o',
                        label='Signal $\\theta$ = [%.2f, %.2f]'%(psi_axis.edges[i]/Units.deg,
                                                                 psi_axis.edges[i+1]/Units.deg))
    
    
    fig[0].ax().set_title('M = %.2f GeV bb Channel'%(10**m))
    fig[1].ax().set_yscale('log')

    fig.plot()

plt.figure()

plt.plot(loge_axis.center-np.log10(Units.gev),e2flux[0,:]/Units.erg)


plt.gca().set_yscale('log')

plt.figure()
plt.plot(loge_axis.center-np.log10(Units.gev),sp.e2dnde(loge_axis.center)/Units.gev)

plt.gca().set_yscale('log')

plt.gca().grid(True)


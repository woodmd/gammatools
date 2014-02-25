#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from gammatools.core.histogram import *
from gammatools.core.series import *
from gammatools.core.util import *
from gammatools.core.plot_util import *
from gammatools.core.model_fn import *
from gammatools.core.likelihood import *
from gammatools.dm.irf_model import *
import sys
import glob
import scipy.signal
import argparse

class AxionModel(Model):

    def __init__(self,gphist,spfn):
        Model.__init__(self,spfn.param())
        
        self._gphist = gphist
        self._spfn = spfn

    def _eval(self,x,p):
        return self._spfn(x,p)*self._gphist.interpolate(x)
        

def calc_e2flux(dh,mh,fn):

    h = dh/mh

    x = h.axis().center()
    delta = 10**h.axis().edges()[1:]-10**h.axis().edges()[:-1]
        
    h *= fn(x)*10**(2*x)*Units.mev

    return h
    
usage = "usage: %(prog)s [options] [detector file]"
description = """A description."""

parser = argparse.ArgumentParser(usage=usage,description=description)

#parser.add_argument('--model', default=None, 
#                    help = '')

parser.add_argument('--irf', default=None, 
                    help = '')

parser.add_argument('--queue', default=None, 
                    help = '')

parser.add_argument('--output', default=None, 
                    help = '')

parser.add_argument('files', nargs='+')

args = parser.parse_args()

if len(args.files) < 1:
    parser.error("At least one argument required.")

if not args.queue is None:
    dispatch_jobs(os.path.abspath(__file__),args.files,args,args.queue,
                  skip=['files'])
    sys.exit(0)


input_model = args.files[0]
    
irf = IRFModel.createCTAIRF(args.irf)
axion_data = load_object(input_model)

sd = yaml.load(open(args.irf,'r'))


fn = LogParabola.create(2.468515027e-11*Units._mev,
                        2.026562251,
                        0.09306285428,
                        1000.92)


livetime = 50*Units.hr

o = { 'chi2_sig'  : [],
      'chi2_null' : [],
      'chi2_null_fit' : [],
      'dh_null_fit' : [],
      'dh_axion0' : [],
      'dh_axion1' : [],
      'mh_null_fit' : [],
      'mh_axion0' : [],
      'mh_axion1' : [],
      'dh_excess_null_e2flux' : [],
      'dh_excess_axion1_e2flux' : [],
      'mh_axion1_e2flux' : [],
      'mh_axion1_fit_e2flux' : [],
      'pgg_hist' : [],
      'flux_ptsrc' : sd['flux_ptsrc'],
      'g' : axion_data['g'],
      'src' : axion_data['src'],
      'm' : axion_data['m'] }


np.random.seed(1)

for i in range(len(axion_data['Pgg'])):
#for i in range(2):
    print i

    pgg_hist = Histogram(Axis.createFromArray(np.log10(axion_data['EGeV'])+3.0),
                         counts=axion_data['Pgg'][i],var=0)

    axion_fn = AxionModel(pgg_hist,fn)

    cm_bkg = BkgSpectrumModel(irf,livetime)    
    cm_ps_null = CountsSpectrumModel(irf,fn,livetime,fold_edisp=True)
    cm_ps_axion0 = CountsSpectrumModel(irf,axion_fn,livetime)
    cm_ps_axion1 = CountsSpectrumModel(irf,axion_fn,livetime,fold_edisp=True)

    cm_null = CompositeSumModel([cm_bkg,cm_ps_null])
    cm_axion0 = CompositeSumModel([cm_bkg,cm_ps_axion0])
    cm_axion1 = CompositeSumModel([cm_bkg,cm_ps_axion1])
    
    axis = Axis.create(4.5,7,2.5*64)

    mh_bkg = cm_bkg.create_histogram(axis)
    mh_ps_null = cm_ps_null.create_histogram(axis)
    mh_ps_axion0 = cm_ps_axion0.create_histogram(axis)
    mh_ps_axion1 = cm_ps_axion1.create_histogram(axis)
    
    dh_bkg = mh_bkg.random()
    dh_ps_null = mh_ps_null.random()
    dh_ps_axion0 = mh_ps_axion0.random()
    dh_ps_axion1 = mh_ps_axion1.random()
    
    dh_null = dh_bkg + dh_ps_null
    dh_axion0 = dh_bkg + dh_ps_axion0 #cm_axion0.create_histogram(axis).random()
    dh_axion1 = dh_bkg + dh_ps_axion1 #cm_axion1.create_histogram(axis).random()

    dh_excess_null = dh_null - mh_bkg
    dh_excess_axion0 = dh_axion0 - mh_bkg
    dh_excess_axion1 = dh_axion1 - mh_bkg
        
    # Fit null hypothesis to axion data
    chi2_fn = BinnedChi2Fn(dh_axion1,cm_null)
    chi2_fn.param().fix(3)
    fitter = BFGSFitter(chi2_fn)
    pset_axion1 = fitter.fit()

    dh_null_fit = cm_null.create_histogram(axis,pset_axion1).random()

    # Fit null hypothesis to axion data
    chi2_fn = BinnedChi2Fn(dh_null_fit,cm_null)
    chi2_fn.param().fix(3)
    fitter = BFGSFitter(chi2_fn)
    pset_null = fitter.fit(pset_axion1)

    mh_null = cm_null.create_histogram(axis)
    mh_null_fit = cm_null.create_histogram(axis,pset_null)
    mh_axion1_fit = cm_null.create_histogram(axis,pset_axion1)
    mh_ps_axion1_fit = cm_ps_null.create_histogram(axis,pset_axion1)
    
    mh_axion0 = cm_axion0.create_histogram(axis)
    mh_axion1 = cm_axion1.create_histogram(axis)

    # Chi2 of axion spectrum with null hypothesis
    chi2_sig = dh_axion1.chi2(mh_axion1_fit,5.0)

    # Chi2 of input spectrum with null hypothesis
    chi2_null = dh_null.chi2(mh_null,5.0)

    # Chi2 of fit spectrum with null hypothesis
    chi2_null_fit = dh_null_fit.chi2(mh_null_fit,5.0)

    print chi2_sig
    print chi2_null
    print chi2_null_fit

    axis2 = mh_null.rebin(2).rebin_mincount(20).axis()
    
    
    dh_excess_null_e2flux = \
        calc_e2flux(dh_excess_null.rebin_axis(axis2),
                    mh_ps_null.rebin_axis(axis2),fn)
    
    dh_excess_axion1_e2flux = \
        calc_e2flux(dh_excess_axion1.rebin_axis(axis2),
                    mh_ps_null.rebin_axis(axis2),fn)

    mh_axion1_fit_e2flux = calc_e2flux(mh_ps_axion1_fit,
                                       mh_ps_null,fn)
    
    mh_axion1_e2flux = calc_e2flux(mh_ps_axion1,
                                   mh_ps_null,fn)


    o['chi2_sig'].append(chi2_sig)
    o['chi2_null'].append(chi2_null)
    o['chi2_null_fit'].append(chi2_null_fit)

    if i > 10: continue
    
    o['dh_null_fit'].append(dh_null_fit)
    o['dh_axion0'].append(dh_axion0)
    o['dh_axion1'].append(dh_axion1)

    o['mh_null_fit'].append(mh_null_fit)
    o['mh_axion0'].append(mh_axion0)
    o['mh_axion1'].append(mh_axion1)

    o['dh_excess_null_e2flux'].append(dh_excess_null_e2flux)
    o['dh_excess_axion1_e2flux'].append(dh_excess_axion1_e2flux)
    
    o['mh_axion1_e2flux'].append(mh_axion1_e2flux)
    o['mh_axion1_fit_e2flux'].append(mh_axion1_fit_e2flux)
    o['pgg_hist'].append(pgg_hist)


fit_data = { 'chi2_sig' : o['chi2_sig'],
             'chi2_null' : o['chi2_null'],
             'chi2_null_fit' : o['chi2_null_fit'],
             'g' : o['g'],
             'm' : o['m'],
             'src' : o['src'] }

if args.output is None:
    m = re.search('(.+)\.pickle\.gz?',input_model)
    if not m is None:
        outfile_fit = m.group(1) + '_fit.pickle'
        outfile_hist = m.group(1) + '_hist.pickle'
else:
    outfile_fit = os.path.splitext(args.output)[0] + '_fit.pickle'
    outfile_hist = os.path.splitext(args.output)[0] + '_hist.pickle'
        
save_object(o,outfile_hist,True)
save_object(fit_data,outfile_fit,True)

sys.exit(0)

ft = FigTool()

plt.figure()
x = np.linspace(4.5,7,800)
plt.plot(x,axion_fn(x)*10**(2*x)*Units.mev**2/Units.erg)
plt.plot(x,fn(x)*10**(2*x)*Units.mev**2/Units.erg)
plt.gca().set_yscale('log')


fig = ft.create(1,'axion_model_density',
                ylabel='Counts Density',
                xlabel='Energy [log$_{10}$(E/MeV)]')


fig[0].add_data(x,cm_null(x,pset_axion1),marker='None')
fig[0].add_data(x,cm_axion0(x),marker='None')
fig[0].add_data(x,cm_axion1(x),marker='None')
fig.plot(ylim_ratio=[-0.5,0.5],style='residual2')

fig = ft.create(1,'axion_model_counts',
                ylabel='Counts',
                xlabel='Energy [log$_{10}$(E/MeV)]')

fig[0].add_hist(mh_null_fit,hist_style='line')
fig[0].add_hist(mh_axion0,hist_style='line')
fig[0].add_hist(mh_axion1,hist_style='line')

fig.plot(ylim_ratio=[-0.5,0.5],style='residual2')

plt.figure()

dh_null.plot()
dh_axion0.plot()
dh_axion1.plot()
mh_axion1_fit.plot(hist_style='line',color='k')



plt.figure()

h0_e2flux.plot(linestyle='None')
h1_e2flux.plot(linestyle='None')
h2_e2flux.plot(linestyle='None')

plt.plot(x,fn(x)*10**(2*x)*Units.mev**2/Units.mev)
plt.plot(x,fn(x,pset_axion1)*10**(2*x)*Units.mev**2/Units.mev)


plt.gca().set_yscale('log')

plt.gca().grid(True)

#plt.gca().set_ylim(1E-14,1E-9)


fig = ft.create(1,'axion_counts_residual',
                ylabel='Counts',
                xlabel='Energy [log$_{10}$(E/MeV)]')


fig[0].add_hist(mh_axion1_fit,hist_style='line')
fig[0].add_hist(dh_axion1)

fig.plot(ylim_ratio=[-0.5,0.5],style='residual2')

fig = ft.create(1,'axion_flux_residual',
                yscale='log',
                ylabel='Flux',
                xlabel='Energy [log$_{10}$(E/MeV)]')


fig[0].add_data(x,fn(x,pset_axion1)*10**(2*x)*Units.mev**2/Units.mev,
                marker='None')
fig[0].add_hist(h2_e2flux)
fig[0].add_hist(mh_axion0_e2flux,hist_style='line',label='ALP')
fig[0].add_hist(mh_axion1_e2flux,hist_style='line',label='ALP Smoothed')


fig.plot(ylim_ratio=[-0.5,0.5],style='residual2')

#fig.plot(style='residual2')

plt.figure()


dh_excess_null_e2flux.plot()
dh_excess_axion1_e2flux.plot()
x = np.linspace(4.5,7,800)
plt.plot(x,axion_fn(x)*10**(2*x)*Units.mev**2/Units.mev)
plt.plot(x,fn(x)*10**(2*x)*Units.mev**2/Units.mev)

plt.gca().set_yscale('log')


plt.plot(sd['flux_ptsrc']['x']+3.0,
         sd['flux_ptsrc']['counts']*Units.gev_m2/Units.mev)


ft = FigTool()

fig = ft.create(1,'axion_model_density',
                ylabel='Counts Density',
                xlabel='Energy [log$_{10}$(E/MeV)]',
                yscale='log',linestyles=['-'])


#    fig[0].add_data(x,fn(x,pset_axion1)*10**(2*x)*Units.mev**2/Units.mev,
#                    marker='None',linestyle='-')

fig[0].add_hist(mh_axion1_fit_e2flux,hist_style='line',linestyle='-',
                label='LogParabola Fit')
    
fig[0].add_hist(mh_axion1_e2flux,hist_style='line',linestyle='-',
                label='ALP')
   
fig[0].add_hist(dh_excess_axion1_e2flux,linestyle='None',label='Data',
                color='k')


fig.plot(ylim_ratio=[-0.8,0.8],style='residual2')
    


plt.show()

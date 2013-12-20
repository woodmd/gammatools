#!/usr/bin/env python



import os
import sys
import copy
import numpy as np
import pickle
from optparse import Option
from optparse import OptionParser
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from gammatools.fermi.data import *
from gammatools.fermi.validate import *
from gammatools.core.plot_util import *
from matplotlib import font_manager
import matplotlib as mpl
import yaml

#from custom_scales import SqrtScale
#from matplotlib import scale as mscale
#mscale.register_scale(SqrtScale)

def get_next(i,l): return l[i%len(l)]


def plot_cth_quantiles(self):

    # ---------------------------------------------------------------------
    # Plot Quantiles as a function of Cos(Theta)    

    for iegy in range(len(egy_range)):

        fig_label = '%04.0f_%04.0f'%(100*egy_range[iegy][0],
                                     100*egy_range[iegy][1])
        pngfile = 'cth_' + quantile_label + '_' + fig_label + '_' + \
            opts.tag + '.png'

        fig = plt.figure()

        ax = fig.add_subplot(2,1,1)
        ax.grid(True)

        title = '%.0f < log$_10$(E/MeV) < %.0f '%(egy_range[iegy][0],
                                                  egy_range[iegy][1])

        plt.errorbar(self.cth_center,qmean[iegy],xerr=cth_width/2.,
                     yerr=qerr[iegy],fmt='o',label='vela')

        for imodel in range(len(models)):
            plt.errorbar(self.cth_center,qmodel[imodel,0,iegy],
                         xerr=0.125,yerr=0,fmt='o',
                         label=model_labels[imodel],
                         color=self.irf_colors[imodel])


        ax.set_title(title)

        ax.set_ylabel('Containment Radius [deg]')
        ax.set_xlabel('Cos$\\theta$')

        ax.set_xlim(0.25,1)
        ax.legend(prop=font)
        ax = fig.add_subplot(2,1,2)

        ax.grid(True)
        ax.set_ylabel('Fractional Residual')
        ax.set_xlabel('Cos$\\theta$')

        for imodel in range(len(models)):

            residual = (qmean[iegy]-qmodel[imodel,iegy])/qmodel[imodel,iegy]
            residual_err = qerr[iegy]/qmodel[imodel,0,iegy]

            plt.errorbar(cth,residual,xerr=0.125,
                         yerr=residual_err,fmt='o',
                         label=model_labels[imodel],
                         color=self.irf_colors[imodel])

        ax.set_xlim(0.25,1)
        ax.set_ylim(-0.6,0.6)
        ax.legend(prop=font)
        plt.savefig(pngfile)


def plot_egy_quantiles(data,model,config,output=None):

    font = font_manager.FontProperties(size=10)
#    mpl.rcParams.update({'font.size': 8})
    
    data_markers = {}


    # ---------------------------------------------------------------------
    # Plot Quantiles as a function of Energy
    pngfile = 'quantile.png'
    if output is not None: pngfile = output
    
    title = 'PSF Containment ' 
#    title += '%.2f < Cos$\\theta$ < %.2f'%(cth_lo,cth_hi)

    fig, axes = plt.subplots(1+len(config['quantiles']), 
                             sharex=True,
                             figsize=(1.2*8,
                                      1.2*8/3.*(1+len(config['quantiles']))))
    for ax in axes: ax.grid(True)
    
    idata = 0
    imodel = 0

    for i, q in enumerate(config['quantiles']):
        
        ql = 'r%2.f'%(q*100)

        mfn = None
        for md in model:
            if ql not in md.qdata.keys(): continue
            qh = md.qdata[ql].cut(1,0)

            mfn = UnivariateSpline(md.egy_center,qh._counts,s=0)
            break


        for j, d in enumerate(data):
            
            excess_hist = d.excess.cut(1,0)
            mask = excess_hist._counts > 20

            qh = d.qdata[ql].cut(1,0)
            
            logx = d.egy_center
            x = np.power(10,logx)
            xlo = (x - np.power(10,d.egy_bin_edge[:-1]))
            xhi = np.power(10,d.egy_bin_edge[1:])-x            

            mean = qh._counts
            err = np.sqrt(qh._var)

#            label = d['label'] + ' %02.f'%(q*100) + '%'
            label = config['data_labels'][j] + ' %02.f'%(q*100) + '%'
            
            x = x[mask]
            xlo = xlo[mask]
            xhi = xhi[mask]
            err = err[mask]
            mean = mean[mask]

            if not config['data_markers'] is None:
                marker = get_next(j,config['data_markers'])
            else:
                marker = get_next(i,config['quantile_markers'])

            color = get_next(j,config['data_colors'])

            axes[0].errorbar(x,mean,xerr=[xlo,xhi],yerr=err,
                             marker=marker,
                             color=color,
                             label=label,linestyle='None')

            if mfn is not None:
                residual = -(mfn(logx)-qh._counts)/mfn(logx)
                residual_err = np.sqrt(qh._var)/mfn(logx)

                residual = residual[mask]
                residual_err = residual_err[mask]

                axes[i+1].errorbar(x,residual,
                                   xerr=[xlo,xhi],
                                   yerr=residual_err,
                                   label=label,
                                   marker=marker,
                                   color=color,
                                   linestyle='None')
    


        for j, d in enumerate(model):
        
            print 'model ', j

            qh = d.qdata[ql].cut(1,0)

#            label = d['label'] + ' %02.f'%(q*100) + '%'
            label = config['model_labels'][j] + ' %02.f'%(q*100) + '%'
            x = np.power(10,d.egy_center)
            axes[0].plot(x,qh._counts,label=label,
                         color=get_next(j,config['model_colors']))
#                         linestyle=config['model_linestyles'][imodel],

            logx = d.egy_center
            residual = -(mfn(logx)-qh._counts)/mfn(logx)
            
            axes[i+1].plot(x,residual, label=label,
                           color=get_next(j,config['model_colors']))

#    if cfg.as_bool('no_title') is not True:
#        ax1.set_title(title)

    if not config['title'] is None:
        axes[0].set_title(config['title'])
            
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_xlim(np.power(10,config['xlim'][0]),
                     np.power(10,config['xlim'][1]))
    axes[0].set_ylim(0.03,30)
    axes[0].set_ylabel('Containment Radius [deg]',fontsize=12)
#    axes[0].set_xlabel('Energy [MeV]')
    axes[0].legend(prop={'size' : 8},ncol=2,numpoints=1)
    
    for ax in axes[1:]:            

        if not config['residual_ylim'] is None:
            ax.set_ylim(config['residual_ylim'][0],config['residual_ylim'][1])

        lims = ax.axis()

        ax.set_ylim(max(-1.0,lims[2]),min(1.0,lims[3]))

        ax.set_xscale('log')
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('Fractional Residual',fontsize=12)
        ax.legend(prop={'size' : 8},loc='upper left',ncol=2,numpoints=1)

    fig.subplots_adjust(hspace=0)

    for i in range(len(axes)-1):
        plt.setp([axes[i].get_xticklabels()], visible=False)
        
    print 'Printing ', pngfile
    plt.savefig(pngfile,bbox_inches='tight')

usage = "%(prog)s [options] [PSF file ...]"
description = """Plot quantiles of PSF."""
parser = argparse.ArgumentParser(usage=usage, description=description)

parser.add_argument('--config', default = None, 
                    help = '')

parser.add_argument('--show', default = False, action='store_true', 
                    help = '')

parser.add_argument('files', nargs='+')

FigTool.configure(parser)

args = parser.parse_args()

config = { 'data_labels' : ['Vela','AGN'], 'model_labels' : [] }

if not args.config is None:
    config.update(yaml.load(open(args.config,'r')))


#plt.rc('font', family='serif')
#plt.rc('font', serif='Times New Roman')

ft = FigTool(args,legend_loc='upper right')

data_colors = ['k','b']

model_colors = ['g','m','k','b']

data_fig68 = ft.create(1,'psf_quantile_r68',style='residual2',yscale='log',
                       xlabel='Energy [log$_{10}$(E/MeV)]',
                       ylabel='Containment Radius [deg]',
                       colors=data_colors)
data_fig95 = ft.create(1,'psf_quantile_r95',style='residual2',yscale='log',
                       xlabel='Energy [log$_{10}$(E/MeV)]',
                       ylabel='Containment Radius [deg]',
                       colors=data_colors)

mdl_fig68 = ft.create(1,'psf_quantile_r68',style='residual2',yscale='log',
                       xlabel='Energy [log$_{10}$(E/MeV)]',
                       ylabel='Containment Radius [deg]',
                       colors=model_colors)

mdl_fig95 = ft.create(1,'psf_quantile_r95',style='residual2',yscale='log',
                       xlabel='Energy [log$_{10}$(E/MeV)]',
                       ylabel='Containment Radius [deg]',
                       colors=model_colors)

norm_index = 0

for i, arg in enumerate(args.files):
    
    d = PSFData.load(arg)

    if d.dtype == 'data':

        msk = None

        if 'range' in config:
            xlim = config['range'][i]
            x = d.excess.xaxis().center()        
            msk = (x > xlim[0]) & (x < xlim[1])

        j = len(data_fig68[0]._data)
        
        data_fig68[0].add_hist(d.qdata[1].slice(1,0),linestyle='None',msk=msk,
                          label=config['data_labels'][j])
        data_fig95[0].add_hist(d.qdata[3].slice(1,0),linestyle='None',msk=msk,
                               label=config['data_labels'][j])
        
    else:
        norm_index = i

        j = len(mdl_fig68[0]._data)
        
        if j >= len(config['model_labels']):
            label = arg
        else:
            label = config['model_labels'][j]

        mdl_fig68[0].add_hist(d.qdata[1].slice(1,0),hist_style='line',
                              label=label)
        mdl_fig95[0].add_hist(d.qdata[3].slice(1,0),hist_style='line',
                              label=label)


data_fig68.merge(mdl_fig68)
data_fig95.merge(mdl_fig95)
        
        
data_fig68.plot(style='residual2',norm_index=norm_index)
data_fig95.plot(style='residual2',norm_index=norm_index)

if args.show: plt.show()

sys.exit(0)


config = { 'quantiles' : [0.68,0.95],
           'data_labels' : [],
           'model_labels' : [],
           'model_colors' : ['r','b','g','c','y'],
           'data_colors' : ['k','b','k','b'],
           'data_markers' : ['o','o','s','s'],
           'title' : None,
           'quantile_markers' : [], ###None, #['s','d'],
           'xlim' : [1.5,5.5],
           'residual_ylim' : None,
           'args' : []}

if opts.config is not None:
    if os.path.isfile(opts.config):
        config = yaml.load(open(opts.config,'r'))
    else:
        yaml.dump(config,open(opts.config,'w'))

if opts.title is not None:
    config['title'] = opts.title
    
if opts.data_labels is not None:
    config['data_labels'] = opts.data_labels.split(',')

if opts.model_labels is not None:
    config['model_labels'] = opts.model_labels.split(',')

data_quantiles = []
model_quantiles = []
    
args += config['args']

for i, arg in enumerate(args):

    d = PSFData.load(arg)    

    if d.dtype == 'data':
        idata = len(data_quantiles)
        data_quantiles.append(d)
        if idata >= len(config['data_labels']): 
            config['data_labels'].append(arg)
    else:
        imodel = len(model_quantiles)
        model_quantiles.append(d)
        if imodel >= len(config['model_labels']): 
            config['model_labels'].append(arg)


    
plot_egy_quantiles(data_quantiles,model_quantiles,config,opts.output)


sys.exit(0)

for i in range(data_quantiles[0].egy_nbin):
    for j in range(data_quantiles[0].cth_nbin):

        egy_range = data_quantiles[0].egy_range[i]
        cth_range = data_quantiles[0].cth_range[j]

        fig_label = 'theta_counts_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f'%(egy_range[0]*100,
                                                 egy_range[1]*100,
                                                 cth_range[0]*100,
                                                 cth_range[1]*100)

        hdata_tot = []
        hbkg = []
        hmodel_sig = []
        hmodel_bkg = []

        for d in data_quantiles:
            hdata_tot.append(d.tot_hist[i,j])
            hbkg.append(d.bkg_hist[i,j])

        for d in model_quantiles:
            hmodel_sig.append(d.sig_hist[i,j])
            hmodel_bkg.append(d.bkg_hist[i,j])


        plot_theta_cumulative(hdata_tot,hbkg,hmodel_sig,hmodel_bkg,fig_label)


#plt.show()

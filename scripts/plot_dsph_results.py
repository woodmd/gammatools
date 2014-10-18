#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
import copy

from gammatools.core.plot_util import FigTool
from gammatools.core.util import update_dict, eq2gal
from gammatools.core.histogram import Histogram, Axis
from dsphs.base.results import TargetResults, AnalysisResults
from pywcsgrid2.allsky_axes import *
from matplotlib.offsetbox import AnchoredText

def merge_targets(results):
    pass

def make_allsky_scatter(x,y,z,filename):

    fig = plt.figure()
    ax = make_allsky_axes(fig,111,"gal","AIT",lon_center=0)
    
    p = ax['fk5'].scatter(x,y,c=z,vmin=0,vmax=5,s=10)
    ax.grid(True)

    cb = plt.colorbar(p,orientation='horizontal',
                      shrink=0.9,pad=0.15,
                      fraction=0.05)
        
    plt.savefig(filename)


def make_ts_hists(hists):
    pass
    

if __name__ == "__main__":

    usage = "usage: %(prog)s [options] [results file]"
    description = "Plot results of dsph analysis."
    parser = argparse.ArgumentParser(usage=usage,description=description)

    parser.add_argument('files', nargs='+')
    parser.add_argument('--labels', default=None)
    parser.add_argument('--colors', default=None)
    parser.add_argument('--ylim_ratio', default='0.6/1.4')
    parser.add_argument('--prefix', default='')

    args = parser.parse_args()

    labels = args.labels.split(',')

    if args.colors: colors = args.colors.split(',')
    else: colors = ['b','g','r','m','c']

    ylim_ratio = [float(t) for t in args.ylim_ratio.split('/')]
    
    hist_set = {'title' : 'test',
                'ts' : Histogram(Axis.create(0.,16.,160)),                
                'fluxul' : Histogram(Axis.create(-12,-8,100)) }
    
    pwl_hists = []
    dm_hists = []

    hists = {}
    
    results = []


    limits = {}
    
    for f in args.files:
        
#        results = AnalysisResults(f)
#        print results.median('composite')        
#        print results.get_results('draco')
        ph = copy.deepcopy(hist_set)
        pwl_hists.append(ph)

        dh = []
        
        c = yaml.load(open(f,'r'),Loader=yaml.CLoader)

        o = {}
        
        for target_name, target in c.iteritems():

            limits.setdefault(target_name,{})
            limits[target_name].setdefault('masses',target['bb']['masses'])
            limits[target_name].setdefault('limits',[])

            pulimits = target['bb']['pulimits99']
            
            nancut = ~np.isnan(np.sum(pulimits,axis=1))
            
            limits[target_name]['limits'].append(np.median(pulimits[nancut],axis=0))

            if target_name == 'composite':
                print f, target_name, np.median(pulimits[nancut],axis=0)
            
            if target_name != 'composite': 
                target['bb']['masses'] = \
                    target['bb']['masses'].reshape((1,) + target['bb']['masses'].shape)
            
                update_dict(o,target,True,True)
                

        results.append(o)


#        plt.figure()
#        plt.plot(c['composite']['bb']['masses'],np.median(c['composite']['bb']['pulimits'],axis=0))       
#        plt.show()
        
        masses = o['bb']['masses'][0]
        for m in masses:
            dh.append(copy.deepcopy(hist_set))

        dm_hists.append(dh)


        if 'pwl' in o:
            ts = o['pwl']['ts']
            ts[ts<0] = 0        
            ph['ts'].fill(np.ravel(ts))

            if 'fluxes' in o['pwl']:
                ph['fluxul'].fill(np.log10(np.ravel(o['pwl']['fluxes'])))
            ph['title'] = 'Powerlaw Gamma = 2.0'
            
            hists.setdefault('pwl', [])
            hists['pwl'].append(ph)
            
        for i, (m, h) in enumerate(zip(masses,dh)):
            ts = o['bb']['ts'][:,i]
            ts[ts<0] = 0
            h['ts'].fill(ts)
            if 'fluxes' in o['bb']:
                h['fluxul'].fill(np.log10(np.ravel(o['bb']['fluxes'][:,i])))
            h['title'] = r'$b \bar b$' + ', M = %.f GeV'%m

            key = 'bb' + '_m%05.f'%m
            
            hists.setdefault(key, [])
            hists[key].append(h)
            

    ft = FigTool()
            
    for k,v in limits.iteritems():

        if k != 'composite': continue

        fig = ft.create(args.prefix + '%s_sigmav_ul'%k,
                        xscale='log',yscale='log',
                        ylim_ratio=ylim_ratio,
                        color=colors,
                        xlabel='Mass',ylabel='Sigmav',figstyle='ratio2')
        
        for j, f in enumerate(args.files):
            fig[0].add_data(v['masses'],v['limits'][j],label=labels[j])
            
        fig.plot()
            
            
    # Make All-sky hists
    for j, f in enumerate(args.files):

        continue
        
        make_allsky_scatter(results[j]['target']['ra'],
                            results[j]['target']['dec'],
                            np.sqrt(np.ravel(results[j]['pwl']['ts'])),
                            'allsky_pwl.png')

        for i, m in enumerate(masses):
            make_allsky_scatter(results[j]['target']['ra'],
                                results[j]['target']['dec'],
                                np.sqrt(np.ravel(results[j]['bb']['ts'][:,i])),
                                'allsky%02i_bb_%010.2f.png'%(j,m))
        
#        for x, y, z in zip(results[j]['target']['ra'],results[j]['target']['dec'],
#                           np.ravel(results[j]['pwl']['ts'])):
#            print x, y, z, eq2gal(x,y)
        
#        print results[j]['target']
#        print results[j]['pwl']['ts']        
        
        

        
        
    for k, hist in hists.iteritems():

        if k == 'pwl': continue
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        label1 = AnchoredText(hist[0]['title'], loc=2,
                              prop={'size':16,'color': 'k'},
                              pad=0., borderpad=0.75,
                              frameon=False)

        ax.add_artist(label1)


        handles = []
        
        for j, f in enumerate(args.files):

#            ax.set_title('M= %.2f GeV'%(m))
#            ax.set_title(hist[j]['title'])

            h = hist[j]['ts']
            h = h.normalize().cumulative(False)
            artists = h.plot(hist_style='band',linewidth=1,
                             color=colors[j],mask_neg=True,
                             label=labels[j])
            handles += [[tuple(artists),labels[j]]]

            
        from scipy.special import erfc

        label0 = AnchoredText('Preliminary', loc=3,
                              prop={'size':18,'color': 'red'},
                              pad=0., borderpad=0.75,
                              frameon=False)
        ax.add_artist(label0)
        
        x = h.axis().center        
        pl, = plt.plot(h.axis().center,0.5*erfc(np.sqrt(x)/np.sqrt(2.)),
                     color='k',linestyle='--',label='$\chi_{1}^2/2$')
        handles += [[pl,'$\chi_{1}^2/2$']]
        
        ax.grid(True)
        ax.set_yscale('log')

        print 
        
        ax.legend(zip(*handles)[0],zip(*handles)[1],
                  loc='best',prop= {'size' : 10 })
        
        ax.set_ylim(1E-4,1E1)
        ax.set_xlim(0,10)
        ax.set_xlabel('TS')
        ax.set_ylabel('Cumulative Fraction')
        
        plt.savefig(args.prefix + 'ts_%s.png'%(k))
        
        
        fig = plt.figure()

        text = ''
        
        for j, f in enumerate(args.files):

            ax = fig.add_subplot(111)
            ax.set_title(hist[j]['title'])

            h = hist[j]['fluxul'].normalize()
            h.plot(hist_style='step',linewidth=1,
                   color=colors[j],
                   label=labels[j],marker='o')

            text += '%20s = %.3f\n'%(labels[j] + ' Mean',h.mean())
            
        plt.gca().grid(True)


        plt.gca().text(0.5,0.95,text,transform=plt.gca().transAxes,
                       fontsize=10,verticalalignment='top',
                       horizontalalignment='right')
        
        plt.gca().legend(loc='upper right',prop= {'size' : 12 })
        plt.gca().set_ylim(plt.gca().axis()[2],1.25*plt.gca().axis()[3])
        
        plt.gca().set_xlabel('Flux Upper Limit [log$_{10}$(Flux/ph cm$^{-2}$ s$^{-1}$)]')
            
        plt.savefig(args.prefix + 'ul_%s.png'%(k))
            
        

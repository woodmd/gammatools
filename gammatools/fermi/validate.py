import os

os.environ['CUSTOM_IRF_DIR'] = '/u/gl/mdwood/ki10/analysis/custom_irfs/'
os.environ['CUSTOM_IRF_NAMES'] = 'P7SOURCE_V6,P7SOURCE_V6MC,P7SOURCE_V9,P7CLEAN_V6,P7CLEAN_V6MC,P7ULTRACLEAN_V6,' \
        'P7ULTRACLEAN_V6MC,P6_v11_diff,P7SOURCE_V6MCPSFC,P7CLEAN_V6MCPSFC,P7ULTRACLEAN_V6MCPSFC'

import sys
import copy
import re
import pickle
import argparse

import math
import numpy as np
import matplotlib.pyplot as plt
from gammatools.core.histogram import Histogram, Histogram2D
from matplotlib import font_manager

from gammatools.fermi.psf_model import *
import gammatools.core.stats as stats
from gammatools.fermi.catalog import Catalog
from gammatools.core.plot_util import *

from data import SkyImage
from analysis_util import *

#from psf_lnl import BinnedPulsarLnLFn

from data import PhotonData, Data
from irf_util import IRFManager

from gammatools.core.mpl_util import SqrtScale
from matplotlib import scale as mscale
mscale.register_scale(SqrtScale)


vela_phase_selection = {'on_phase' : '0.0/0.15,0.6/0.7', 'off_phase' : '0.2/0.5' }

class PSFData(Data):

    def __init__(self,egy_bin_edge,cth_bin_edge,dtype):

        egy_bin_edge = np.array(egy_bin_edge,ndmin=1)
        cth_bin_edge = np.array(cth_bin_edge,ndmin=1)

        self.dtype = dtype
        self.quantiles = [0.34,0.68,0.90,0.95]
        self.quantile_labels = ['r%2.f'%(q*100) for q in self.quantiles]

        self.egy_axis = Axis(egy_bin_edge)
        self.cth_axis = Axis(cth_bin_edge)
        self.egy_nbin = self.egy_axis.nbins()
        self.cth_nbin = self.cth_axis.nbins()

        self.chi2 = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.rchi2 = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.ndf = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.excess = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.bkg = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.bkg_density = Histogram2D(egy_bin_edge,cth_bin_edge)

        hist_shape = (self.egy_nbin,self.cth_nbin)

        self.sig_density_hist = np.empty(shape=hist_shape, dtype=object)
        self.tot_density_hist = np.empty(shape=hist_shape, dtype=object)
        self.bkg_density_hist = np.empty(shape=hist_shape, dtype=object)
        self.sig_hist = np.empty(shape=hist_shape, dtype=object)
        self.off_hist = np.empty(shape=hist_shape, dtype=object)
        self.tot_hist = np.empty(shape=hist_shape, dtype=object)
        self.bkg_hist = np.empty(shape=hist_shape, dtype=object)
        self.sky_image = np.empty(shape=hist_shape, dtype=object)

#        self.q34 = Histogram2D(egy_bin_edge,cth_bin_edge)
#        self.q68 = Histogram2D(egy_bin_edge,cth_bin_edge)
#        self.q90 = Histogram2D(egy_bin_edge,cth_bin_edge)
#        self.q95 = Histogram2D(egy_bin_edge,cth_bin_edge)

        self.qdata = []
        for i in range(len(self.quantiles)):
            self.qdata.append(Histogram2D(egy_bin_edge,cth_bin_edge))

    def init_hist(self,fn,theta_max):

        for i in range(self.egy_nbin):
            for j in range(self.cth_nbin):

                ecenter = self.egy_axis.center()[i]
                theta_max = min(theta_max,fn(ecenter))
                theta_edges = np.linspace(0,theta_max,100)

                h = Histogram(theta_edges)
                self.sig_density_hist[i,j] = copy.deepcopy(h)
                self.tot_density_hist[i,j] = copy.deepcopy(h)
                self.bkg_density_hist[i,j] = copy.deepcopy(h)
                self.sig_hist[i,j] = copy.deepcopy(h)
                self.tot_hist[i,j] = copy.deepcopy(h)
                self.bkg_hist[i,j] = copy.deepcopy(h)
                self.off_hist[i,j] = copy.deepcopy(h)


    def print_quantiles(self,prefix):

        filename = prefix + '.txt'
        f = open(filename,'w')

        for i, ql in enumerate(self.quantile_labels):

            q = self.qdata[i]

            for icth in range(self.cth_nbin):
                for iegy in range(self.egy_nbin):

                    line = '%5.3f '%(self.quantiles[i])
                    line += '%5.2f %5.2f '%(self.cth_range[icth][0],
                                           self.cth_range[icth][1])
                    line += '%5.2f %5.2f '%(self.egy_range[iegy][0],
                                            self.egy_range[iegy][1])
                    line += '%8.4f %8.4f '%(q.mean[iegy,icth],
                                            q.err[iegy,icth])

                    f.write(line + '\n')

    def print_quantiles_tex(self,prefix):

        for i, q in enumerate(self.qdata):

            filename = prefix + '_' + ql + '.tex'
            f = open(filename,'w')

            for icth in range(self.cth_nbin):
                for iegy in range(self.egy_nbin):

                    line = '%5.2f %5.2f '%(self.cth_range[icth][0],
                                           self.cth_range[icth][1])
                    line += '%5.2f %5.2f '%(self.egy_range[iegy][0],
                                            self.egy_range[iegy][1])
                    line += format_error(q.mean[iegy,icth],
                                         q.err[iegy,icth],1,True)
                    f.write(line + '\n')



class PSFValidate(object):
    def __init__(self, opts):
        """
        @type self: object
        """
        self.irf_colors = ['green', 'red', 'magenta', 'gray', 'orange']
        self.on_phases = []
        self.off_phases = []
        self.data = PhotonData()
#        self.load(opts)

        self._ft = FigTool(opts)

        self.font = font_manager.FontProperties(size=10)

        if opts.egy_bin_edge is not None:
            self.egy_bin_edge = [float(t) for t in opts.egy_bin_edge.split(',')]
        elif opts.egy_bin is not None:
            [elo, ehi, ebin] = [float(t) for t in opts.egy_bin.split('/')]
            self.egy_bin_edge = \
                np.linspace(elo, ehi, 1 + int((ehi - elo) / ebin))
        elif self.data_type == 'agn':
            self.egy_bin_edge = np.linspace(3.5, 5, 7)
        else:
            self.egy_bin_edge = np.linspace(1.5, 5.0, 15)

        self.cth_bin_edge = \
            np.array([float(t) for t in opts.cth_bin_edge.split(',')])

        self.output_prefix = opts.output_prefix
        if self.output_prefix is None:
            prefix = os.path.splitext(opts.files[0])[0]

            cth_label = '%03.f%03.f' % (self.cth_bin_edge[0] * 100,
                                        self.cth_bin_edge[1] * 100)

            if not opts.event_class_id is None:
                cth_label += '_%02i'%(opts.event_class_id)

            self.output_prefix = '%s_' % (prefix)

            if not opts.conversion_type is None:
                self.output_prefix += '%s_' % (opts.conversion_type)
            
            self.output_prefix += '%s_' % (cth_label)

        if opts.output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = opts.output_dir

        self.show = opts.show

        self.models = []
        if opts.irf is not None:
            self.models = opts.irf.split(',')

            for i in range(len(self.models)):
                if opts.conversion_type == 'front':
                    self.models[i] += '::FRONT'
                elif opts.conversion_type == 'back':
                    self.models[i] += '::BACK'

        if opts.irf_labels is not None:
            self.model_labels = opts.irf_labels.split(',')
        else:
            self.model_labels = self.models

        self.mask_srcs = opts.mask_srcs

        self.data_type = 'agn'
        if opts.phase_selection == 'vela':
            self.data_type = 'pulsar'
            opts.on_phase = vela_phase_selection['on_phase']
            opts.off_phase = vela_phase_selection['off_phase']

        self.conversion_type = opts.conversion_type
        self.opts = opts

#        self.quantiles = [float(t) for t in opts.quantiles.split(',')]
#        self.quantile_labels = ['r%2.f' % (q * 100) for q in self.quantiles]

        if self.data_type == 'pulsar':
            self.phases = parse_phases(self.opts.on_phase, self.opts.off_phase)
            self.on_phases = self.phases[0]
            self.off_phases = self.phases[1]
            self.alpha = self.phases[2]

        self.psf_data = PSFData(self.egy_bin_edge,
                                self.cth_bin_edge,
                                'data')

        theta_max_c0 = 30
        theta_max_c1 = -0.8
        theta_max_c2 = 1.0

        if self.opts.conversion_type == 'back' or \
                self.opts.conversion_type is None:
            theta_max_c0 = 35
            theta_max_c2 = 1.5

        self.thetamax_fn = \
            lambda x: np.sqrt(np.power(theta_max_c0 *
                                       np.power(np.power(10, x - 2),
                                                theta_max_c1), 2) +
                              theta_max_c2 ** 2)

        self.psf_data.init_hist(self.thetamax_fn, self.opts.theta_max)

        self.build_models()


    @staticmethod
    def configure(parser):

        IRFManager.configure(parser)
        FigTool.configure(parser)

        parser.add_argument('--ltfile', default=None,
                            help='Set the livetime cube which will be used '
                                 'to generate the exposure-weighted PSF model.')

        parser.add_argument('--irf', default=None,
                            help='Set the names of one or more IRF models.')

        parser.add_argument('--theta_max', default=25.0, type=float,
                            help='Set the names of one or more IRF models.')

        parser.add_argument('--irf_labels', default=None,
                            help='IRF labels')

        parser.add_argument('--output_dir', default=None,
                            help='Set the output directory name.')

        parser.add_argument('--output_prefix', default=None,
                            help='Set the string prefix that will be appended '
                                 'to all output files.')

        parser.add_argument('--on_phase', default=None,
                            help='Type of input data (pulsar/agn).')

        parser.add_argument('--off_phase', default=None,
                            help='Type of input data (pulsar/agn).')

        parser.add_argument('--phase_selection', default=None,
                            help='Type of input data (pulsar/agn).')
        
        parser.add_argument('--cth_bin_edge', default='0.2,1.0',
                            help='Edges of cos(theta) bins (e.g. 0.2,0.5,1.0).')

        parser.add_argument('--egy_bin_edge', default=None,
                            help='Edges of energy bins.')

        parser.add_argument('--egy_bin', default=None,
                            help='Set low/high and energy bin size.')

        parser.add_argument('--cuts', default=None,
                            help='Set min/max redshift.')

        parser.add_argument('--src', default='Vela',
                            help='Set the source model.')


        parser.add_argument('--show', default=False, action='store_true',
                            help='Draw plots to screen.')

        parser.add_argument('--make_sky_image', default=False,
                            action='store_true',
                            help='Plot distribution of photons on the sky.')

        parser.add_argument('--conversion_type', default=None,
                            help='Draw plots.')

        parser.add_argument('--event_class', default=None,
                            help='Set the event class name.')

        parser.add_argument('--event_class_id', default=None, type=int,
                            help='Set the event class ID.')

        parser.add_argument('--quantiles', default='0.34,0.68,0.90,0.95',
                            help='Draw plots.')

        parser.add_argument('--mask_srcs', default=None,
                            help='Define a list of sources to exclude.')

    def build_models(self):

        print 'Building Models'

        self.psf_models = {}

        for imodel, ml in enumerate(self.models):
            self.psf_models[ml] = []

            for icth in range(self.psf_data.cth_axis.nbins()):

                cth_range = self.psf_data.cth_axis.edges()[icth:icth+2]
                irfm = IRFManager.create(self.models[imodel], True,
                                         irf_dir=self.opts.irf_dir)

                lonlat = (0, 0)
                if self.opts.src != 'iso' and self.opts.src != 'iso2':
                    cat = Catalog()
                    src = cat.get_source_by_name(self.opts.src)
                    lonlat = (src['RAJ2000'], src['DEJ2000'])

                m = PSFModelLT(self.opts.ltfile, irfm,
                               nbin=300,cth_range=cth_range,
                               psf_type=self.opts.src,
                               lonlat=lonlat)

                #                m.set_spectrum('powerlaw_exp',(1.607,3508.6))
                #                m.set_spectrum('powerlaw',(2.0))

                self.psf_models[ml].append(m)

        self.irf_data = {}
        for ml in self.models:
            self.irf_data[ml] = PSFData(self.egy_bin_edge,
                                        self.cth_bin_edge,
                                        'model')

    def load(self, opts):


        for f in opts.files:
            print 'Loading ', f
            d = PhotonData.load(f)
            d.mask(event_class_id=opts.event_class_id,
                   conversion_type=opts.conversion_type)

            self.data.merge(d)

        self.data['dtheta'] = np.degrees(self.data['dtheta'])

    def fill(self,data):

         for iegy in range(self.psf_data.egy_axis.nbins()):
            for icth in range(self.psf_data.cth_axis.nbins()):
                if self.data_type == 'pulsar':
                    self.fill_pulsar(data, iegy, icth)
                else:
                    self.fill_agn(data, iegy, icth)
            
    def run(self):

        for f in self.opts.files:
            print 'Loading ', f
            d = PhotonData.load(f)
            d.mask(event_class_id=self.opts.event_class_id,
                   conversion_type=self.opts.conversion_type)
            d['dtheta'] = np.degrees(d['dtheta'])

            self.fill(d)
            
#        for iegy in range(self.psf_data.egy_axis.nbins()):
#            for icth in range(self.psf_data.cth_axis.nbins()):
#                if self.data_type == 'pulsar':
#                    self.fill_pulsar(self.data, iegy, icth)
#                else:
#                    self.fill_agn(self.data, iegy, icth)

        for iegy in range(self.psf_data.egy_axis.nbins()):
            for icth in range(self.psf_data.cth_axis.nbins()):
                self.fill_models(iegy,icth)

        for iegy in range(self.psf_data.egy_axis.nbins()):
            for icth in range(self.psf_data.cth_axis.nbins()):
                if self.data_type == 'pulsar':
                    self.fit_pulsar(iegy, icth)
                else:
                    self.fit_agn(iegy, icth)

        fname = os.path.join(self.output_dir,
                             self.output_prefix + 'psfdata')

        self.psf_data.save(fname + '.P')

        #        psf_data.print_quantiles()

        #        psf_data.print_quantiles_tex(os.path.join(self.output_dir,
        #                                                  self.output_prefix))

        for ml in self.models:
            fname = self.output_prefix + 'psfdata_' + ml
            self.irf_data[ml].save(fname + '.P')

    def plot(self):
        return

    def get_counts(self, data, theta_edges, mask):

        theta_mask = (data['dtheta'] >= theta_edges[0]) & \
                     (data['dtheta'] <= theta_edges[1])

        return len(data['dtheta'][mask & theta_mask])

    def plot_theta_residual(self, hsignal, hbkg, hmodel, label):


        fig = self._ft.create(1,label,xscale='sqrt')

        

        hsignal_rebin = hsignal.rebin_mincount(10)
        hbkg_rebin = Histogram(hsignal_rebin.axis().edges())
        hbkg_rebin.fill(hbkg.axis().center(),hbkg.counts(),
                        hbkg.var())

        fig[0].add_hist(hsignal_rebin,
                        marker='o', linestyle='None',label='signal')
        fig[0].add_hist(hbkg_rebin,hist_style='line',
                        linestyle='--',label='bkg',color='k')

        
        for i, h in enumerate(hmodel):
            fig[0].add_hist(h,hist_style='line',linestyle='-',
                            label=self.model_labels[i],
                            color=self.irf_colors[i],
                            linewidth=1.5)

        fig[0].set_style('ylabel','Counts Density [deg$^{-2}$]')

        fig.plot(style='residual2',norm_index=2,mask_ratio_args=[1])


        return

    #        fig = plt.figure()
        fig, axes = plt.subplots(2, sharex=True)
        axes[0].set_xscale('sqrt', exp=2.0)
        axes[1].set_xscale('sqrt', exp=2.0)

        pngfile = os.path.join(self.output_dir, label + '.png')

        #        ax = plt.gca()

        #        if hsignal.sum() > 0:
        #            ax.set_yscale('log')

        axes[0].set_ylabel('Counts Density [deg$^{-2}$]')
        axes[0].set_xlabel('$\\theta$ [deg]')

        #    ax.set_title(title)

        hsignal_rebin = hsignal.rebin_mincount(10)
        hbkg_rebin = Histogram(hsignal_rebin.axis().edges())
        hbkg_rebin.fill(hbkg.axis().center(),hbkg.counts(),
                        hbkg.var())

        hsignal_rebin.plot(ax=axes[0], marker='o', linestyle='None',
                           label='signal')
        hbkg_rebin.plot(ax=axes[0], marker='o', linestyle='None',
                        label='bkg')

        for i, h in enumerate(hmodel):

            h.plot(hist_style='line', ax=axes[0], fmt='-',
                   label=self.model_labels[i],
                   color=self.irf_colors[i],
                   linewidth=2)

            hm = Histogram(hsignal_rebin.axis().edges())
            hm.fill(h.center(),h.counts(),h.var())

            hresid = hsignal_rebin.residual(hm)
            hresid.plot(ax=axes[1],linestyle='None',
                        label=self.model_labels[i],
                        color=self.irf_colors[i],
                        linewidth=2)


        #        ax.set_ylim(1)
        axes[0].grid(True)
        axes[1].grid(True)

        axes[0].legend(prop=self.font)

        axes[1].set_ylim(-0.5, 0.5)

        fig.subplots_adjust(hspace=0)

        for i in range(len(axes) - 1):
            plt.setp([axes[i].get_xticklabels()], visible=False)

        if self.show is True:
            plt.show()

        print 'Printing ', pngfile
        plt.savefig(pngfile)

    def plot_theta_cumulative(self, hsignal, hbkg, hmodel, label,
                              theta_max=None, text=None):

        hexcess = hsignal - hbkg


        fig = self._ft.create(2,label,xscale='sqrt')

        fig[0].add_hist(hsignal,label='Data',linestyle='None')
        fig[0].add_hist(hbkg,hist_style='line',
                        label='Bkg',marker='None',linestyle='--',
                        color='k')

        for i, h in enumerate(hmodel):
            fig[0].add_hist(h,hist_style='line',
                            linestyle='-',
                            label=self.model_labels[i],
                            color=self.irf_colors[i],
                            linewidth=1.5)

        hexcess_cum = hexcess.normalize()
        hexcess_cum = hexcess_cum.cumulative()

        fig[1].add_hist(hexcess_cum,marker='None',linestyle='None',label='Data')

        for i, h in enumerate(hmodel):
            t = h - hbkg
            t = t.normalize()
            t = t.cumulative()
            fig[1].add_hist(t,hist_style='line', 
                            linestyle='-',
                            label=self.model_labels[i],
                            color=self.irf_colors[i],
                            linewidth=1.5)

        fig[0].set_style('ylabel','Counts')
        fig[0].set_style('legend_loc','upper right')
        fig[1].set_style('legend_loc','lower right')
        fig[1].set_style('ylabel','Cumulative Fraction')

        fig[1].add_hline(1.0, color='k')

#        fig[1].axhline(0.34, color='r', linestyle='--', label='34%')
        fig[1].add_hline(0.68, color='b', linestyle='--', label='68%')
#        axes[1].axhline(0.90, color='g', linestyle='--', label='90%')
        fig[1].add_hline(0.95, color='m', linestyle='--', label='95%')

        fig[1].set_style('xlabel','$\\theta$ [deg]')

        fig.plot(figure_style='twopane')
        

#        axes[1].legend(prop=self.font, loc='lower right', ncol=2)
#        if theta_max is not None:
#            axes[0].axvline(theta_max, color='k', linestyle='--')
#            axes[1].axvline(theta_max, color='k', linestyle='--')

#        if text is not None:
#            axes[0].text(0.3, 0.75, text,
#                         transform=axes[0].transAxes, fontsize=10)


    def fill_agn(self,data,iegy,icth):

        qdata = self.psf_data

        egy_range = self.psf_data.egy_axis.edges()[iegy:iegy+2]
        cth_range = self.psf_data.cth_axis.edges()[icth:icth+2]
        ecenter = self.psf_data.egy_axis.center()[iegy]
        emin = 10 ** egy_range[0]
        emax = 10 ** egy_range[1]
        theta_edges = self.psf_data.sig_hist[iegy, icth].axis().edges()

        theta_max=theta_edges[-1]

#        theta_max = min(3.0, self.thetamax_fn(ecenter))
#        theta_edges = np.linspace(0, 3.0, int(3.0 / (theta_max / 100.)))

        mask = PhotonData.get_mask(data, {'energy': egy_range,
                                          'cth': cth_range},
                                   conversion_type=self.conversion_type,
                                   event_class=self.opts.event_class,
                                   cuts=self.opts.cuts)

        hcounts = data.hist('dtheta', mask=mask, edges=theta_edges)

        domega = (theta_max ** 2) * np.pi
        bkg_edge = [min(2.5, theta_max), 3.5]

        bkg_domega = (bkg_edge[1] ** 2 - bkg_edge[0] ** 2) * np.pi
        bkg_counts = self.get_counts(data, bkg_edge, mask)
        bkg_density = bkg_counts / bkg_domega

#        print 'BKG ', bkg_counts, bkg_edge

        qdata.bkg.fill(self.psf_data.egy_axis.center()[iegy],
                       self.psf_data.cth_axis.center()[icth],
                       bkg_counts)
        qdata.bkg_density.set(iegy,icth,
                              qdata.bkg.counts()[iegy,icth]/bkg_domega,
                              qdata.bkg.counts()[iegy,icth]/bkg_domega**2)

        hbkg = copy.deepcopy(hcounts)
        hbkg.clear()
        for b in hbkg.iterbins():
            bin_area = (b.hi_edge() ** 2 - b.lo_edge() ** 2) * np.pi
            b.set_counts(bin_area * bkg_density)

        hexcess = hcounts - hbkg
        
#        htotal_density = hcounts.scale_density(lambda x: x * x * np.pi)
#        hbkg_density = copy.deepcopy(hcounts)
#        hbkg_density._counts[:] = bkg_density
#        hbkg_density._var[:] = 0

        # Fill histograms for later plotting
        
        qdata.sig_hist[iegy, icth] += hexcess
        qdata.tot_hist[iegy, icth] += hcounts
        qdata.off_hist[iegy, icth] += hbkg
        qdata.bkg_hist[iegy, icth] += hbkg
        qdata.excess.set(iegy, icth, *qdata.sig_hist[iegy, icth].sum())
        
        qdata.tot_density_hist[iegy, icth] = \
            qdata.sig_hist[iegy, icth].scale_density(lambda x: x * x * np.pi)
        qdata.bkg_density_hist[iegy, icth]._counts = bkg_density

        



    def fit_agn(self, iegy, icth):

        models = self.psf_models
        irf_data = self.irf_data
        psf_data = self.psf_data

        egy_range = psf_data.egy_axis.edges()[iegy:iegy+2]
        cth_range = psf_data.cth_axis.edges()[icth:icth+2]
        ecenter = psf_data.egy_axis.center()[iegy]
        emin = 10 ** psf_data.egy_axis.edges()[iegy]
        emax = 10 ** psf_data.egy_axis.edges()[iegy+1]

        theta_max = min(self.opts.theta_max, self.thetamax_fn(ecenter))

        bkg_hist = psf_data.bkg_hist[iegy, icth]
        sig_hist = psf_data.sig_hist[iegy, icth]
        on_hist = psf_data.tot_hist[iegy, icth]
        off_hist = psf_data.off_hist[iegy, icth]
        excess_sum = psf_data.excess._counts[iegy, icth]

        bkg_density = psf_data.bkg_density._counts[iegy,icth]
        bkg_counts = psf_data.bkg._counts[iegy,icth]
        bkg_domega = bkg_counts/bkg_density

        print 'Computing Quantiles'
        bkg_fn = lambda x: x**2 * np.pi * bkg_density
        hq = stats.HistQuantileBkgFn(on_hist, 
                                     lambda x: x**2 * np.pi / bkg_domega,
                                     bkg_counts)

        if excess_sum > 25:
            self.compute_quantiles(hq, psf_data, iegy, icth, theta_max)

        hmodel_density = []
        hmodel_counts = []
        for i, ml in enumerate(self.model_labels):
            hmodel_density.append(irf_data[ml].tot_density_hist[iegy, icth])
            hmodel_counts.append(irf_data[ml].tot_hist[iegy, icth])

        text = 'Bkg Density = %.3f deg$^{-2}$\n' % (bkg_density)
        text += 'Signal = %.3f\n' % (excess_sum)
        text += 'Background = %.3f' % (bkg_density * theta_max**2 * np.pi)

        fig_label = self.output_prefix + 'theta_density_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

#        self.plot_theta_residual(psf_data.tot_density_hist[iegy, icth],
#                                 psf_data.bkg_density_hist[iegy, icth],
#                                 hmodel_density, fig_label)

        fig_label = self.output_prefix + 'theta_counts_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        self.plot_theta_cumulative(psf_data.tot_hist[iegy, icth],
                                   psf_data.bkg_hist[iegy, icth],
                                   hmodel_counts, fig_label,None,text)
#                                   bkg_edge[0], text)


        return

        r68 = hq.quantile(0.68)
        r95 = hq.quantile(0.95)

        rs = min(r68 / 4., theta_max / 10.)

        xedge = np.linspace(-3.0, 3.0, 601)

        bin_size = 6.0 / 600.

        stacked_image = Histogram2D(xedge, xedge)

        stacked_image.fill(data['delta_ra'][mask], data['delta_dec'][mask])

        plt.figure()

        stacked_image.smooth(rs)

        stacked_image.plot()

        plt.plot(r68 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 r68 * np.sin(np.linspace(0, 2 * np.pi, 100)), color='k')

        plt.plot(r95 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 r95 * np.sin(np.linspace(0, 2 * np.pi, 100)), color='k',
                 linestyle='--')

        plt.plot(bkg_edge[0] * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 bkg_edge[0] * np.sin(np.linspace(0, 2 * np.pi, 100)), color='k',
                 linestyle='-', linewidth=2)

        plt.plot(bkg_edge[1] * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 bkg_edge[1] * np.sin(np.linspace(0, 2 * np.pi, 100)), color='k',
                 linestyle='-', linewidth=2)

        #        c68 = plt.Circle((0, 0), radius=r68, color='k',facecolor='None')
        #        c95 = plt.Circle((0, 0), radius=r95, color='k',facecolor='None')

        #        plt.gca().add_patch(c68)
        #        plt.gca().add_patch(c95)

        fig_label = self.output_prefix + 'stackedimage_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        plt.savefig(fig_label)

        for i in range(len(data._srcs)):

            if not self.opts.make_sky_image: continue

            src = data._srcs[i]

            srcra = src['RAJ2000']
            srcdec = src['DEJ2000']

            print i, srcra, srcdec

            im = SkyImage.createROI(srcra, srcdec, 3.0, 3.0 / 600.)

            src_mask = mask & (data['src_index'] == i)

            im.fill(data['ra'][src_mask], data['dec'][src_mask])
            fig = plt.figure()
            im = im.smooth(rs)
            im.plot()

            im.plot_catalog()

            im.plot_circle(r68, color='k')
            im.plot_circle(r95, color='k', linestyle='--')
            im.plot_circle(theta_max, color='k', linestyle='-', linewidth=2)

            fig_label = self.output_prefix + 'skyimage_src%03i_' % (i)
            fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                       egy_range[1] * 100,
                                                       cth_range[0] * 100,
                                                       cth_range[1] * 100)

            fig.savefig(fig_label + '.png')


    def fill_pulsar(self, data, iegy, icth):

        qdata = self.psf_data

        egy_range = qdata.egy_axis.edges()[iegy:iegy+2]
        cth_range = qdata.cth_axis.edges()[icth:icth+2]
        ecenter = qdata.egy_axis.center()[iegy]
        emin = 10 ** egy_range[0]
        emax = 10 ** egy_range[1]
        theta_edges = qdata.sig_hist[iegy, icth].axis().edges()

        theta_max = theta_edges[-1]

        mask = PhotonData.get_mask(data, {'energy': egy_range,
                                          'cth': cth_range},
                                   conversion_type=self.conversion_type,
                                   event_class=self.opts.event_class,
                                   cuts=self.opts.cuts)

        on_mask = PhotonData.get_mask(data, {'energy': egy_range,
                                             'cth': cth_range},
                                      conversion_type=self.conversion_type,
                                      event_class=self.opts.event_class,
                                      cuts=self.opts.cuts,
                                      phases=self.on_phases)



        (hon, hoff, hoffs) = getOnOffHist(data, 'dtheta', phases=self.phases,
                                          edges=theta_edges, mask=mask)

        hexcess = copy.deepcopy(hon)
        hexcess -= hoffs

        htotal_density = copy.deepcopy(hexcess)
        htotal_density += hoffs
        htotal_density = htotal_density.scale_density(lambda x: x * x * np.pi)

        hoffs_density = hoffs.scale_density(lambda x: x * x * np.pi)

        excess_sum = np.sum(hexcess._counts)
        on_sum = np.sum(hon._counts)
        off_sum = np.sum(hoff._counts)

        # Fill histograms for later plotting
        qdata.tot_density_hist[iegy, icth] += htotal_density
        qdata.bkg_density_hist[iegy, icth] += hoffs_density
        qdata.sig_hist[iegy, icth] += hexcess
        qdata.tot_hist[iegy, icth] += hon
        qdata.off_hist[iegy, icth] += hoff
        qdata.bkg_hist[iegy, icth] += hoffs
        qdata.excess.set(iegy, icth, np.sum(qdata.sig_hist[iegy, icth]._counts))

        
        src = data._srcs[0]

        if not isinstance(qdata.sky_image[iegy,icth],SkyImage):        
            im = SkyImage.createROI(src['RAJ2000'], src['DEJ2000'],
                                    theta_max, theta_max / 200.)
            qdata.sky_image[iegy,icth] = im
        else:
            im = qdata.sky_image[iegy,icth]

        
        if len(data['ra'][on_mask]) > 0:
            im.fill(data['ra'][on_mask], data['dec'][on_mask])


    def fill_models(self, iegy, icth):
        """Fill IRF model distributions."""

        models = self.psf_models
        irf_data = self.irf_data
        psf_data = self.psf_data

        egy_range = psf_data.egy_axis.edges()[iegy:iegy+2]
        cth_range = psf_data.cth_axis.edges()[icth:icth+2]
        ecenter = psf_data.egy_axis.center()[iegy]
        emin = 10 ** psf_data.egy_axis.edges()[iegy]
        emax = 10 ** psf_data.egy_axis.edges()[iegy+1]

        bkg_hist = psf_data.bkg_hist[iegy, icth]
        sig_hist = psf_data.sig_hist[iegy, icth]
        on_hist = psf_data.tot_hist[iegy, icth]
        off_hist = psf_data.off_hist[iegy, icth]
        excess_sum = psf_data.excess._counts[iegy, icth]

        for i, ml in enumerate(self.model_labels):
            m = models[ml][icth]

            print 'Fitting model ', ml
            hmodel_sig = m.histogram(emin, emax,
                                     on_hist.axis().edges()).normalize()
            model_norm = excess_sum
            hmodel_sig *= model_norm

            irf_data[ml].excess.set(iegy, icth, sig_hist.sum()[0])
            irf_data[ml].ndf.set(iegy, icth, sig_hist.nbins())

            hmd = hmodel_sig.scale_density(lambda x: x * x * np.pi)
            hmd += psf_data.bkg_density_hist[iegy, icth]

            irf_data[ml].tot_density_hist[iegy, icth] = hmd
            irf_data[ml].bkg_density_hist[iegy, icth] = \
                copy.deepcopy(psf_data.bkg_density_hist[iegy, icth])
            irf_data[ml].sig_hist[iegy, icth] = hmodel_sig
            irf_data[ml].bkg_hist[iegy, icth] = copy.deepcopy(bkg_hist)
            irf_data[ml].tot_hist[iegy, icth] = hmodel_sig + bkg_hist

            for j, q in enumerate(psf_data.quantiles):
                ql = psf_data.quantile_labels[j]
                qm = m.quantile(emin, emax, q)
                self.irf_data[ml].qdata[j].set(iegy, icth, qm)
                print ml, ql, qm

#            ndf = hexcess.nbins()

#            qmodel[ml].excess.set(iegy, icth, hexcess.sum()[0])
#            qmodel[ml].ndf.set(iegy, icth, hexcess.nbins())
#            qmodel[ml].chi2.set(iegy,icth,hexcess.chi2(hmodel))
#            qmodel[ml].rchi2.set(iegy,icth,
#            qmodel[ml].chi2[iegy,icth]/hexcess.nbins())
#           print ml, ' chi2/ndf: %f/%d rchi2: %f'%(qmodel[ml].chi2[iegy,icth],
#                                                   ndf,
#                                                   qmodel[ml].rchi2[iegy,icth])

    def fit_pulsar(self, iegy, icth):
        
        models = self.psf_models
        irf_data = self.irf_data
        psf_data = self.psf_data

        egy_range = psf_data.egy_axis.edges()[iegy:iegy+2]
        cth_range = psf_data.cth_axis.edges()[icth:icth+2]
        ecenter = psf_data.egy_axis.center()[iegy]
        emin = psf_data.egy_axis.edges()[iegy]
        emax = psf_data.egy_axis.edges()[iegy+1]

        print 'Analyzing Bin ', emin, emax
        
        theta_max = min(self.opts.theta_max, self.thetamax_fn(ecenter))

        bkg_hist = psf_data.bkg_hist[iegy, icth]
        sig_hist = psf_data.sig_hist[iegy, icth]
        on_hist = psf_data.tot_hist[iegy, icth]
        off_hist = psf_data.off_hist[iegy, icth]
        excess_sum = psf_data.excess._counts[iegy, icth]

        bkg_density = bkg_hist.sum() / (theta_max ** 2 * np.pi)
        text = 'Bkg Density = %.3f deg$^{-2}$\n' % (bkg_density[0])
        text += 'Signal = %.3f\n' % (psf_data.excess._counts[iegy, icth])
        text += 'Background = %.3f' % (bkg_hist.sum()[0])

        print 'Computing Quantiles'
        hq = stats.HistQuantileBkgHist(on_hist, off_hist, self.alpha)


        
        if excess_sum > 25:
            self.compute_quantiles(hq, psf_data, iegy, icth, theta_max)
            
        hmodel_density = []
        hmodel_counts = []

        for i, ml in enumerate(self.model_labels):
            hmodel_density.append(irf_data[ml].tot_density_hist[iegy, icth])
            hmodel_counts.append(irf_data[ml].tot_hist[iegy, icth])

        fig_label = self.output_prefix + 'theta_density_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        self.plot_theta_residual(psf_data.tot_density_hist[iegy, icth],
                                 psf_data.bkg_density_hist[iegy, icth],
                                 hmodel_density,
                                 fig_label)

        fig_label = self.output_prefix + 'theta_counts_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        self.plot_theta_cumulative(on_hist, bkg_hist, hmodel_counts,
                                   fig_label,
                                   theta_max=None, text=text)


        if excess_sum < 10: return

        
        r68 = hq.quantile(0.68)
        r95 = hq.quantile(0.95)

        rs = min(r68 / 4., theta_max / 10.)


        fig = plt.figure()
        im = psf_data.sky_image[iegy,icth].smooth(rs)
        im.plot(logz=True)

        im.plot_catalog()

        im.plot_circle(r68, color='k')
        im.plot_circle(r95, color='k', linestyle='--')
        im.plot_circle(theta_max, color='k', linestyle='-', linewidth=2)



        fig_label = self.output_prefix + 'skyimage_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        fig.savefig(fig_label + '.png')

    def compute_quantiles(self, hq, qdata, iegy, icth,
                          theta_max=None):

        emin = 10 ** qdata.egy_axis.edges()[iegy]
        emax = 10 ** qdata.egy_axis.edges()[iegy+1]

        for i, q in enumerate(qdata.quantiles):

            ql = qdata.quantile_labels[i]
            qmean = hq.quantile(fraction=q)
            qdist_mean, qdist_err = hq.bootstrap(q, niter=200, xmax=theta_max)
            qdata.qdata[i].set(iegy, icth, qmean, qdist_err ** 2)

            print ql, ' %10.4f +/- %10.4f %10.4f' % (qmean, qdist_err, 
                                                     qdist_mean)

if __name__ == '__main__':
    usage = "%(prog)s [options] [pickle file ...]"
    description = """Perform PSF validation analysis on agn or
pulsar data samples."""
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('files', nargs='+')

    PSFValidate.configure(parser)

    args = parser.parse_args()

    psfv = PSFValidate(args)
    psfv.run()

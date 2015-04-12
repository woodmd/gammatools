import os

os.environ['CUSTOM_IRF_DIR'] = '/u/gl/mdwood/ki10/analysis/custom_irfs/'
os.environ['CUSTOM_IRF_NAMES'] = 'P7SOURCE_V6,P7SOURCE_V6MC,P7SOURCE_V9,P7CLEAN_V6,P7CLEAN_V6MC,P7ULTRACLEAN_V6,' \
        'P7ULTRACLEAN_V6MC,P6_v11_diff,P7SOURCE_V6MCPSFC,P7CLEAN_V6MCPSFC,P7ULTRACLEAN_V6MCPSFC'

import sys
import copy
import re
import glob
import pickle
import argparse
import yaml

import math
import numpy as np
import matplotlib.pyplot as plt
from gammatools.core.histogram import Histogram, Histogram2D
from matplotlib import font_manager

from gammatools.fermi.exposure import *
from gammatools.fermi.psf_model import *
from gammatools.fermi.ft1_loader import *
from gammatools.core.quantile import *
from gammatools.fermi.catalog import Catalog
from gammatools.core.plot_util import *
from gammatools.core.config import Configurable

from gammatools.core.fits_util import SkyImage
from analysis_util import *

#from psf_lnl import BinnedPulsarLnLFn

from data import PhotonData, Data
from irf_util import IRFManager

from gammatools.core.mpl_util import SqrtScale, PowerNormalize
from matplotlib import scale as mscale
mscale.register_scale(SqrtScale)

vela_phase_selection = {'on_phase' : '0.0/0.15,0.6/0.7',
                        'off_phase' : '0.2/0.5' }

class PSFScalingFunction(object):

    def __init__(self,c0,c1,beta,thmax=None,thmin=None,scale=1.0):
        self._c0 = c0
        self._c1 = c1
        self._beta = beta
        self._thmax = thmax
        self._thmin = thmin
        self._scale = scale
        
    def __call__(self,x):

        x = np.array(x,ndmin=1)

        v = self._scale*np.sqrt((self._c0*np.power(np.power(10,x-2),self._beta))**2 +
                                self._c1**2)

        if self._thmax is not None:
            v[v>self._thmax] = self._thmax

        if self._thmin is not None:
            v[v<self._thmin] = self._thmin

        return v

psf_scaling_params = {
    'front' : (30.0,1.0,-0.8),
    'back' : (35.0,1.5,-0.8),
    'FRONT' : (9.873,0.295,-0.8),
    'BACK' : (16.556,0.609,-0.8),
    'PSF0' : (18.487,0.820,-0.8),
    'PSF1' : (12.637,0.269,-0.8),
    'PSF2' : (9.191,0.139,-0.8),
    'PSF3' : (6.185,0.078,-0.8),
    'EDISP0' : (16.556,0.609,-0.8),
    'EDISP1' : (16.556,0.609,-0.8),
    'EDISP2' : (16.556,0.609,-0.8),
    'EDISP3' : (16.556,0.609,-0.8),
}

psf_scaling_fn = { 
    'front' : PSFScalingFunction(30.0,1.0,-0.8),
    'back' : PSFScalingFunction(35.0,1.5,-0.8),
    'FRONT' : PSFScalingFunction(9.873,0.295,-0.8),
    'BACK' : PSFScalingFunction(16.556,0.609,-0.8),
    'PSF0' : PSFScalingFunction(18.487,0.820,-0.8),
    'PSF1' : PSFScalingFunction(12.637,0.269,-0.8),
    'PSF2' : PSFScalingFunction(9.191,0.139,-0.8),
    'PSF3' : PSFScalingFunction(6.185,0.078,-0.8) }

pass8_type_map = {
    0 : 'FRONT',
    1 : 'BACK',
    2 : 'PSF0',
    3 : 'PSF1',
    4 : 'PSF2',
    5 : 'PSF3',
    6 : 'EDISP0',
    7 : 'EDISP1',
    8 : 'EDISP2',
    9 : 'EDISP3',
    }

class AeffData(Data):

    def __init__(self,egy_axis,cth_axis,
                 class_name,class_id,**kwargs):

        self.egy_axis = egy_axis
        self.cth_axis = cth_axis
        self.class_name = class_name
        self.type_name = kwargs.get('type_name',None)
        self.class_id = class_id
        self.type_id = kwargs.get('type_id',[])
        self.fn = kwargs.get('fn',None)

        self.alpha_hist = Histogram(egy_axis)

        # Excess (S - a*B)
        self.sig_hist = Histogram(egy_axis)

        # Off (Background)
        self.off_hist = Histogram(egy_axis)

        # On (S + B)
        self.on_hist = Histogram(egy_axis)

        # Unscaled Background
        self.bkg_hist = Histogram(egy_axis)
        
        # Data Efficiency
        self.eff_hist = Histogram(egy_axis)

        # IRF Efficiency
        self.irf_eff_hist = Histogram(egy_axis)

        # Weighted Exposure
        self.exp_hist = Histogram(egy_axis)

class PSFData(Data):

    
    def __init__(self,egy_axis,cth_axis,
                 class_name,class_id,**kwargs):

        self.class_name = class_name
        self.type_name = kwargs.get('type_name',None)
        self.class_id = class_id
        self.type_id = kwargs.get('type_id',[])
        self.fn = kwargs.get('fn',None)

        self.quantiles = [0.34,0.68,0.90,0.95]
        self.quantile_labels = ['r%2.f'%(q*100) for q in self.quantiles]
        self.egy_axis = egy_axis
        self.cth_axis = cth_axis
        self.dtheta_axis = Axis.create(0.0,4.0,160)
        self.domega = (self.dtheta_axis.edges[1:]**2-
                       self.dtheta_axis.edges[:-1]**2)*np.pi

        self.sd_hist = Histogram(egy_axis)
        self.alpha_hist = Histogram(egy_axis)
        self.chi2 = Histogram(egy_axis)
        self.rchi2 = Histogram(egy_axis)
        self.ndf = Histogram(egy_axis)
        self.excess = Histogram(egy_axis)
        self.bkg = Histogram(egy_axis)
        self.bkg_density = Histogram(egy_axis)

        hist_shape = (self.egy_axis.nbins,self.cth_axis.nbins)

        self.sig_density_hist = Histogram2D(egy_axis,self.dtheta_axis)
        self.on_density_hist = Histogram2D(egy_axis,self.dtheta_axis)
        self.bkg_density_hist = Histogram2D(egy_axis,self.dtheta_axis)
        self.sig_hist = Histogram2D(egy_axis,self.dtheta_axis)
        self.off_hist = Histogram2D(egy_axis,self.dtheta_axis)
        self.on_hist = Histogram2D(egy_axis,self.dtheta_axis)
        self.bkg_hist = Histogram2D(egy_axis,self.dtheta_axis)
#        self.sky_image = 
#        self.sky_image_off = 
#        self.lat_image = np.empty(shape=hist_shape, dtype=object)
#        self.lat_image_off = np.empty(shape=hist_shape, dtype=object)

        self.irf_sig_hist = Histogram2D(egy_axis,self.dtheta_axis)

        self.qdata = []
        self.qdata_cth = []
        self.irf_qdata = []
        self.irf_qdata_cth = []
        for i in range(len(self.quantiles)):
            self.qdata.append(Histogram(egy_axis))
            self.qdata_cth.append(Histogram2D(egy_axis,cth_axis))
            self.irf_qdata.append(Histogram(egy_axis))
            self.irf_qdata_cth.append(Histogram2D(egy_axis,cth_axis))

        self.sd_hist._counts[:] = self.fn(self.sd_hist.axis(0).center)

#    def init_hist(self,fn,theta_max):

#        for i in range(self.egy_nbin):
#            for j in range(self.cth_nbin):

#                ecenter = self.egy_axis.center[i]
#                theta_max = min(theta_max,fn(ecenter))
#                theta_edges = np.linspace(0,theta_max,100)

#                h = Histogram(theta_edges)
#                self.sig_density_hist[i,j] = copy.deepcopy(h)
#                self.tot_density_hist[i,j] = copy.deepcopy(h)
#                self.bkg_density_hist[i,j] = copy.deepcopy(h)
#                self.sig_hist[i,j] = copy.deepcopy(h)
#                self.tot_hist[i,j] = copy.deepcopy(h)
#                self.bkg_hist[i,j] = copy.deepcopy(h)
#                self.off_hist[i,j] = copy.deepcopy(h)

#        parser.add_argument('--output_prefix', default=None,
#                            help='Set the string prefix that will be appended '
#                                 'to all output files.')

#        parser.add_argument('--phase_selection', default=None,
#                            help='Type of input data (pulsar/agn).')
        
#        parser.add_argument('--src', default='Vela',
#                            help='Set the source model.')

#        parser.add_argument('--mask_srcs', default=None,
#                            help='Define a list of sources to exclude.')

                    
class IRFValidate(Configurable):

    default_config = {
        'irfmanager'      : IRFManager.defaults,
        'ft1loader'       : FT1Loader.default_config,
        'output_prefix'   : (None),
        'output_dir'      : (None,'Set the output directory name.'),        
        'ltfile'          : (None,'Set the livetime cube.'),
        'evfile'          : (None),
        'egy_bin'         : (None,'Set low/high and energy bin size.','',(list,float)),
        'egy_bin_edge'    : (None,'Edges of energy bins.','',(list,float)),
        'cth_bin'         : (None,'','',(list,float)),
        'cth_bin_edge'    : (None,'Set edges of cos(theta) bins.','',(list,float)),
        'event_class'     : (None,'Set the event class ID.'),
        'event_type'      : (None,'Set the event type ID.'),
        'conversion_type' : (None,'Set conversion type.'),
        'phase_selection' : (None),
        'on_phase'        : (None),
        'off_phase'       : (None),
        'refclass'        : (None,'',''),
        'event_class_selections' : (None,'',''),
        'event_type_selections'  : (None,'',''),
        'event_class_irfs'       : (None,'',''),
        'noplots'                : (False,'','')
        }
    
    def __init__(self, config, opts,**kwargs):
        super(IRFValidate,self).__init__(config,opts,**kwargs)
#        self.update_default_config(IRFManager.defaults)  

        self._ft = FigTool(opts=opts)
        cfg = self._config

        self.data_type = 'agn'
        if cfg['phase_selection'] == 'vela':
            cfg['on_phase'] = vela_phase_selection['on_phase']
            cfg['off_phase'] = vela_phase_selection['off_phase']

        self.conversion_type = cfg['conversion_type']

        if not cfg['on_phase'] is None: self.data_type = 'pulsar'

        if self.data_type == 'pulsar':
            self.phases = parse_phases(cfg['on_phase'],cfg['off_phase'])
            self.on_phases = self.phases[0]
            self.off_phases = self.phases[1]
            self.alpha = self.phases[2]

        if self.config['output_dir'] is not None:
            mkdir(self.config['output_dir'])

    def load(self):
        """Load data from FITS."""
        
        evfile = []
        for f in self.config['evfile']:

            if os.path.splitext(f)[1] == '.txt':
                evfile += list(np.genfromtxt(f,unpack=True,dtype=None))
            else:
                evfile += glob.glob(f)

        for f in evfile: #self.config['evfile']:
            print 'Loading ', f
            basename = os.path.splitext(f)[0] 
            if self.config['output_dir']:
                basename = os.path.join(self.config['output_dir'],
                                        os.path.basename(basename))
            
            if os.path.isfile(basename + '.P.gz'):
                d = load_object(basename + '.P.gz')
            else:
#                wlif os.path.splitext(f)[1] == '.fits':
                loader = FT1Loader(self.config['ft1loader'])
                loader.load_photons(f)
                d = loader._photon_data
                loader.save(basename + '.P')

            print 'Masking'

            d.mask(event_class_id=self.config['event_class'],
                   event_type_id=self.config['event_type'],
                   conversion_type=self.config['conversion_type'])
            d['dtheta'] = np.degrees(d['dtheta'])
            self.fill(d)


class AeffValidate(IRFValidate):

    default_config = { }

    def __init__(self, config, opts,**kwargs):
        super(AeffValidate,self).__init__(config,opts,**kwargs)
        cfg = self.config
        
        import pprint
        pprint.pprint(cfg)
        pprint.pprint(config)

        if cfg['egy_bin_edge'] is not None:
            self.egy_axis = Axis(cfg['egy_bin_edge'])
        elif cfg['egy_bin'] is not None:
            self.egy_axis = Axis.create(*cfg['egy_bin'])
        
        if cfg['cth_bin_edge'] is not None:
            self.cth_axis = Axis(cfg['cth_bin_edge'])
        elif cfg['cth_bin'] is not None:
            self.cth_axis = Axis.create(*cfg['cth_bin'])
        
        self._ltc = LTCube.create(self.config['ltfile'])

        self._class_data = {}
        for c, cid in self.config['event_class_selections'].items():

            fn = PSFScalingFunction(*psf_scaling_params['BACK'],
                                     scale=1.0,thmax=30.0,thmin=1.0)

            self._class_data[c + '_ALL'] = AeffData(self.egy_axis,self.cth_axis,
                                           class_name=c,class_id=cid,fn=fn)

            for t,tid in self.config['event_type_selections'].items():

                fn = PSFScalingFunction(*psf_scaling_params[t],
                                         scale=1.0,thmax=30.0,thmin=1.0)

                self._class_data[c + '_' + t] = AeffData(self.egy_axis,self.cth_axis,
                                                         class_name=c,class_id=cid,
                                                         type_name=t,type_id=tid,fn=fn)

        self.build_models()

    def build_models(self):

        cd = self._class_data

        for k,v in self._class_data.items():
            
            class_name = '%s_%s_%s'%(self.config['event_class_irfs']['release'],
                                     v.class_name,
                                     self.config['event_class_irfs']['version'])

            
            type_names = []
            for t in v.type_id:
                type_names.append(pass8_type_map[t])
            

            print 'Creating IRF ', class_name, type_names

            irfm = IRFManager.create(class_name, type_names,True,
                                     irf_dir=self.config['irfmanager']['irf_dir'])

            exp = ExposureCalc(irfm,self._ltc)

            cth_axis = Axis.create(0.2,1.0,80)

            exph = exp.getExpByName(self.config['ft1loader']['src_list'],
                                    self.egy_axis,cth_axis)

            m = PSFModelLT(irfm,
                           src_type=self.config['ft1loader']['src_list'],
                           ltcube=self._ltc)
#                           spectrum=sp,spectrum_pars=sp_pars)

            cd[k].exp_hist += exph

            psf_corr = np.zeros(self.egy_axis.nbins)

            for i in range(self.egy_axis.nbins):
                r, cdf = m.cdf(10**self.egy_axis.edges[i],
                               10**self.egy_axis.edges[i+1],0.2,1.0)

                s = cd[k].fn(self.egy_axis.center[i])
#                print i, self.egy_axis.center[i], interpolate(r,cdf,0.1), interpolate(r,cdf,1.0), interpolate(r,cdf,s)
#                plt.figure()
#                plt.plot(r,cdf)
#                plt.axvline(0.1)
#                plt.axhline(interpolate(r,cdf,0.1))
#                plt.gca().set_xscale('log')
#                plt.show()

                psf_corr[i] = interpolate(r,cdf,s)

            cd[k].exp_hist *= psf_corr

        # Renormalize
        exp_hist0 = self._class_data[self.config['refclass']].exp_hist
        for k,v in self._class_data.items():
            self._class_data[k].irf_eff_hist = v.exp_hist/exp_hist0

    def fill(self,data):

        if self.data_type =='agn':
            self.fill_agn(data)
        else:
            self.fill_pulsar(data)

    def fill_pulsar(self,data):

        cd = self._class_data
        
        # Loop over classes
        for k,v in self._class_data.items():
            
            evclass = list_to_bitfield(v.class_id)
            evtype = list_to_bitfield(v.type_id)

            cd[k].alpha_hist._counts[:] = self.alpha

            fn = cd[k].fn

            msk = PhotonData.get_mask(data,
                                      {'cth': self.cth_axis.lims()},
                                      evclass=evclass,evtype=evtype,
                                      theta_fn=[fn])

            msk_on = PhotonData.get_mask(data,
                                         {'cth': self.cth_axis.lims()},
                                         evclass=evclass,evtype=evtype,
                                         phases=self.on_phases,
                                         theta_fn=[fn])

            msk_off = PhotonData.get_mask(data,
                                          {'cth': self.cth_axis.lims()},
                                          evclass=evclass,evtype=evtype,
                                          phases=self.off_phases,
                                          theta_fn=[fn])
            
            cd[k].on_hist.fill(data['energy'][msk_on])
            cd[k].off_hist.fill(data['energy'][msk_off])

#            emsk = (data['energy'] >3.0)&(data['energy']<3.5)
#            h0 = Histogram(Axis.create(0,1,40))
#            h1 = Histogram(Axis.create(0,1,40))
#            h2 = Histogram(Axis.create(0,1,40))
#            h0.fill(data['phase'][msk_on&emsk])
#            h1.fill(data['phase'][msk_off&emsk])
#            h2.fill(data['phase'][msk&emsk])
#            print k
#            plt.figure()

#            h0.plot()
#            h1 *= self.alpha
#            h1.plot()
#            h2.plot()

#            plt.show()

    def fill_agn(self,data):

        cd = self._class_data
        
        # Loop over classes
        for k,v in self._class_data.items():

            fn = cd[k].fn

            theta0 = fn(cd[k].alpha_hist.axis(0).center)

            print theta0

            cd[k].alpha_hist._counts[:] = theta0**2/(3.5**2-2.5**2)

#            domega_on = 1.0**2*np.pi
#            domega_off = (3.5**2-2.5**2)*np.pi
            
            evclass = list_to_bitfield(v.class_id)
            evtype = list_to_bitfield(v.type_id)

            msk_on = PhotonData.get_mask(data,
                                         {'cth': self.cth_axis.lims() },
                                         evclass=evclass,evtype=evtype)

            msk_off = PhotonData.get_mask(data,
                                          {'cth': self.cth_axis.lims(),
                                           'dtheta' : [2.5,3.5]},
                                          evclass=evclass,evtype=evtype)

            ebin = self.egy_axis.valToBinBounded(data['energy'])
            bin_energy = self.egy_axis.center[ebin]
            msk_on &= data['dtheta'] < fn(bin_energy)

            
            cd[k].on_hist.fill(data['energy'][msk_on])
            cd[k].off_hist.fill(data['energy'][msk_off])
        
    def compute_eff(self):

        cd = self._class_data

#        domega_on = 1.0**2*np.pi
#        domega_off = (3.5**2-2.5**2)*np.pi
#        alpha = domega_on/domega_off
       
        # Loop over classes
        for k,v in self._class_data.items():
            alpha = cd[k].alpha_hist.counts
            cd[k].sig_hist += cd[k].on_hist.counts - cd[k].off_hist.counts*alpha
            cd[k].bkg_hist += cd[k].off_hist.counts*alpha

#            plt.figure()
#            cd[k].on_hist.plot()
#            cd[k].off_hist.plot()
#            cd[k].bkg_hist.plot()
#            plt.show()

        ns0 = cd[self.config['refclass']].on_hist.counts
        nb0 = cd[self.config['refclass']].off_hist.counts
            
        for k,v in self._class_data.items():
            
            ns1 = cd[k].on_hist.counts
            nb1 = cd[k].off_hist.counts

            eff = (ns1-alpha*nb1)/(ns0-alpha*nb0)
            eff_var = (((ns0-ns1+alpha**2*(nb0-nb1))*eff**2 + (ns1+alpha**2*nb1)*(1-eff)**2)/
                       (ns0-alpha*nb0)**2)

            cd[k].eff_hist._counts = eff
            cd[k].eff_hist._var = eff_var
            
    def plot(self):

        for k,v in self._class_data.items():

            figname = self.config['output_prefix'] + '_' + k
            figname = os.path.join(self.config['output_dir'],figname)

            fig = self._ft.create(figname,figstyle='residual2',
                                  norm_interpolation='lin',
                                  legend_loc='best',
                                  ylim=[0.0,1.1],
                                  ylim_ratio=[-0.2,0.2])

            fig[0].ax().set_title(k)

            fig[0].add_hist(v.irf_eff_hist,label='model',hist_style='line')
            fig[0].add_hist(v.eff_hist,label='data')
            fig.plot()

    def run(self):

        self.load()
        self.compute_eff()        
        self.plot()
        self.save()
            
    def save(self):

        outfile = self.config['output_prefix'] + '_aeffdata.P'

        if self.config['output_dir']:
            outfile = os.path.join(self.config['output_dir'],
                                   os.path.basename(outfile))

        save_object(self._class_data,outfile,compress=True)

        o = {}
        for k,v in self._class_data.items():
            o[k] = self._class_data[k].to_dict()

        outfile = self.config['output_prefix'] + '_aeffdata.yaml'

        if self.config['output_dir']:
            outfile = os.path.join(self.config['output_dir'],
                                   os.path.basename(outfile))

        yaml.dump(o,open(outfile,'w'),Dumper=yaml.CDumper)
        
#        for ml in self.models:
#            fname = self.config['output_prefix'] + 'psfdata_' + ml
#            self.irf_data[ml].save(fname + '.P')
            
#        self.irf_data = {}
#        for ml in self.models:
#            self.irf_data[ml] = AeffData(self.egy_bin_edge,
#                                         self.cth_bin_edge,
#                                         'model')
        
class PSFValidate(IRFValidate):

    default_config = {
        'data_type'       : 'agn',
        'spectrum'        : (None),
        'spectrum_pars'   : (None), 
        'theta_max'       : (30.0),
#        'psf_scaling_fn'  :
#            (None,'Set the scaling function to use for determining the edge of the ROI at each energy.'),
        'irf'             : (None,'Set the names of one or more IRF models.','',list),
        'irf_labels'      : (None,'IRF Labels.'),
        'make_sky_image'  : (False,'Plot distribution of photons on the sky.'),
        'show'            : (False,'Draw plots to screen.'),
        'quantiles'       : ([0.34,0.68,0.90,0.95],'Set quantiles.'),
        'src'             : 'iso' }
    
    def __init__(self, config, opts,**kwargs):
        super(PSFValidate,self).__init__(config,opts,**kwargs)
        cfg = self.config

        self.irf_colors = ['green', 'red', 'magenta', 'gray', 'orange']
        self.font = font_manager.FontProperties(size=10)

        if cfg['egy_bin_edge'] is not None:
            self.egy_axis = Axis(cfg['egy_bin_edge'])
        elif cfg['egy_bin'] is not None:
            self.egy_axis = Axis.create(*cfg['egy_bin'])
        
        if cfg['cth_bin_edge'] is not None:
            self.cth_axis = Axis(cfg['cth_bin_edge'])
        elif cfg['cth_bin'] is not None:
            self.cth_axis = Axis.create(*cfg['cth_bin'])
        
        if cfg['output_prefix'] is None:
#            prefix = os.path.splitext(opts.files[0])[0]
            m = re.search('(.+).P.gz',self.config['evfile'][0])
            if m is None: prefix = os.path.splitext(self.config['evfile'][0])[0] 
            else: prefix = m.group(1) 
            
            cth_label = '%03.f%03.f' % (self.cth_bin_edge[0] * 100,
                                        self.cth_bin_edge[1] * 100)

            if not self.config['event_class_id'] is None:
                cth_label += '_c%02i'%(self.config['event_class_id'])

            if not self.config['event_type_id'] is None:
                cth_label += '_t%02i'%(self.config['event_type_id'])
                
            cfg['output_prefix'] = '%s_' % (prefix)

            if not cfg['conversion_type'] is None:
                self.output_prefix += '%s_' % (cfg['conversion_type'])
            
            cfg['output_prefix'] += '%s_' % (cth_label)

        if cfg['output_dir'] is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = cfg['output_dir']


#        if opts.irf_labels is not None:
#            self.model_labels = opts.irf_labels.split(',')
#        else:
#            self.model_labels = self.models

        self.conversion_type = cfg['conversion_type']
        
#        self.quantiles = [float(t) for t in opts.quantiles.split(',')]
#        self.quantile_labels = ['r%2.f' % (q * 100) for q in self.quantiles]

        self._ltc = LTCube.create(self.config['ltfile'])

        self._class_data = {}
        for c, cid in self.config['event_class_selections'].items():

            fn = PSFScalingFunction(*psf_scaling_params['BACK'],thmax=30.0/4.,
                                     thmin=0.5)
            self._class_data[c + '_ALL'] = PSFData(self.egy_axis,self.cth_axis,
                                                   class_name=c,class_id=cid,fn=fn)

            

            for t,tid in self.config['event_type_selections'].items():

                fn = PSFScalingFunction(*psf_scaling_params[t],thmax=30.0/4.,
                                         thmin=0.5)

                self._class_data[c + '_' + t] = PSFData(self.egy_axis,self.cth_axis,
                                                        class_name=c,class_id=cid,
                                                        type_name=t,type_id=tid,fn=fn)
        
        self.build_models()

    def build_models(self):

        cd = self._class_data

        for k,v in self._class_data.items():
            
            class_name = '%s_%s_%s'%(self.config['event_class_irfs']['release'],
                                     v.class_name,
                                     self.config['event_class_irfs']['version'])

            
            type_names = []
            for t in v.type_id:
                type_names.append(pass8_type_map[t])
            

            print 'Creating IRF ', class_name, type_names

            irfm = IRFManager.create(class_name, type_names,True,
                                     irf_dir=self.config['irfmanager']['irf_dir'])

            exp = ExposureCalc(irfm,self._ltc)

            cth_axis = Axis.create(0.2,1.0,80)

            exph = exp.getExpByName(self.config['ft1loader']['src_list'],
                                    self.egy_axis,cth_axis)

            m = PSFModelLT(irfm,
                           src_type=self.config['ft1loader']['src_list'],
                           ltcube=self._ltc)

            cd[k].irf_sig_hist.set_label('test')

            for i in range(self.egy_axis.nbins):

                edges = cd[k].on_hist.axis(1).edges*cd[k].sd_hist.counts[i]
                emin = self.egy_axis.edges[i]
                emax = self.egy_axis.edges[i+1]
                cth_range = self.cth_axis.lims()
                hm = m.histogram(10**emin, 10**emax,cth_range[0],cth_range[1],
                                 edges).normalize()

                cd[k].irf_sig_hist._counts[i] = hm.counts
                
                for j, q in enumerate(cd[k].quantiles):
                    ql = cd[k].quantile_labels[j]
                    qm = m.quantile(10**emin, 10**emax, cth_range[0],cth_range[1], q)
                    cd[k].irf_qdata[j].set(i, qm)
                    print ql, qm
                
#            sp = self.config['spectrum']
#            sp_pars = string_to_array(self.config['spectrum_pars'])
                #                m.set_spectrum('powerlaw_exp',(1.607,3508.6))
                #                m.set_spectrum('powerlaw',(2.0))

    def fill(self,data):

        if self.data_type == 'pulsar':
            self.fill_pulsar(data)
        else:
            self.fill_agn(data)
            
    def run(self):

        self.load()
        self.compute_quantiles()
        if not self.config['noplots']: self.plot()
        self.save()
            
    def save(self):

        outfile = self.config['output_prefix'] + '_psfdata.P'
        if self.config['output_dir']:
            outfile = os.path.join(self.config['output_dir'],
                                   os.path.basename(outfile))
            
        save_object(self._class_data,outfile,compress=True)

    def plot(self):
        
        cd = self._class_data

        for k,v in self._class_data.items():

            for i in range(cd[k].on_hist.axis(0).nbins):

                on_hist = cd[k].on_hist.slice(0,i)
                bkg_hist = cd[k].bkg_hist.slice(0,i)
                irf_hist = cd[k].irf_sig_hist.slice(0,i)

                on_hist._axes[0] = Axis(cd[k].dtheta_axis.edges * cd[k].sd_hist.counts[i])
                bkg_hist._axes[0] = Axis(cd[k].dtheta_axis.edges * cd[k].sd_hist.counts[i])
                irf_hist._axes[0] = Axis(cd[k].dtheta_axis.edges * cd[k].sd_hist.counts[i])

                erange = cd[k].on_hist.axis(0).edges[i:i+2]
                cthrange = cd[k].cth_axis.lims()

                fig_label = self.config['output_prefix'] + '_%s_psfcum_'%k
                fig_label += '%04.0f_%04.0f_%03.f%03.f' % (erange[0] * 100,
                                                           erange[1] * 100,
                                                           cthrange[0] * 100,
                                                           cthrange[1] * 100)

                fig_title = 'E = [%.3f, %.3f] '%(erange[0],erange[1])
                fig_title += 'Cos$\\theta$ = [%.3f, %.3f]'%(cthrange[0],cthrange[1])

                if self.config['output_dir']:
                    fig_label = os.path.join(self.config['output_dir'],
                                            os.path.basename(fig_label))

                self.plot_psf_cumulative(on_hist,bkg_hist, [irf_hist], fig_label,fig_title,
                                         None,None)


    def get_counts(self, data, theta_edges, mask):

        theta_mask = (data['dtheta'] >= theta_edges[0]) & \
                     (data['dtheta'] <= theta_edges[1])

        return len(data['dtheta'][mask & theta_mask])

    def plot_theta_residual(self, hsignal, hbkg, hmodel, label):


        fig = self._ft.create(label,figstyle='residual2',xscale='sqrt',
                              norm_interpolation='lin',
                              legend_loc='upper right')

        hsignal_rebin = hsignal.rebin_mincount(10)
        hbkg_rebin = Histogram(hsignal_rebin.axis().edges)
        hbkg_rebin.fill(hbkg.axis().center,hbkg.counts,
                        hbkg.var)

        hsignal_rebin = hsignal_rebin.scale_density(lambda x: x**2*np.pi)
        hbkg_rebin = hbkg_rebin.scale_density(lambda x: x**2*np.pi)
        
        for i, h in enumerate(hmodel):

            h_rebin = Histogram(hsignal_rebin.axis().edges)
            h_rebin.fill(h.axis().center,h.counts,h.var)
            h_rebin = h_rebin.scale_density(lambda x: x**2*np.pi)
            
            fig[0].add_hist(h_rebin,hist_style='line',linestyle='-',
                            label=self.model_labels[i],
                            color=self.irf_colors[i],
                            linewidth=1.5)
        
        fig[0].add_hist(hsignal_rebin,
                        marker='o', linestyle='None',label='signal')
        fig[0].add_hist(hbkg_rebin,hist_style='line',
                        linestyle='--',label='bkg',color='k')

        
        

        fig[0].set_style('ylabel','Counts Density [deg$^{-2}$]')

        fig[1].ax().set_ylim(-0.5,0.5)
        
#        fig.plot(norm_index=2,mask_ratio_args=[1])
        fig.plot()

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
        hbkg_rebin = Histogram(hsignal_rebin.axis().edges)
        hbkg_rebin.fill(hbkg.axis().center,hbkg.counts,hbkg.var)
        hsignal_rebin.plot(ax=axes[0], marker='o', linestyle='None',
                           label='signal')
        hbkg_rebin.plot(ax=axes[0], marker='o', linestyle='None',
                        label='bkg')

        for i, h in enumerate(hmodel):

            h.plot(hist_style='line', ax=axes[0], fmt='-',
                   label=self.model_labels[i],
                   color=self.irf_colors[i],
                   linewidth=2)

            hm = Histogram(hsignal_rebin.axis().edges)
            hm.fill(h.center,h.counts,h.var)

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

    def plot_psf_cumulative(self, hon, hbkg, hmodel, label,title,
                            theta_max=None, text=None):

        hsig = hon - hbkg
        fig = self._ft.create(label,figstyle='twopane',xscale='sqrt',
                              title=title,legend_loc='best')

        

        fig[0].add_hist(hon,label='Data',linestyle='None')
        fig[0].add_hist(hbkg,hist_style='line',
                        label='Bkg',marker='None',linestyle='--',
                        color='k')

        for i, hm in enumerate(hmodel):

            h = hm*np.sum(hsig.counts)+hbkg
            fig[0].add_hist(h,hist_style='line',
                            linestyle='-',label='model',
#                            label=self.model_labels[i],
#                            color=self.irf_colors[i],
                            linewidth=1.5)

        hsig_cum = hsig.normalize()
        hsig_cum = hsig_cum.cumulative()

        fig[1].add_hist(hsig_cum,marker='None',linestyle='None',label='Data')

        for i, hm in enumerate(hmodel):
            h = hm.cumulative()
            fig[1].add_hist(h,hist_style='line', 
                            linestyle='-',
#                            label=self.model_labels[i],
#                            color=self.irf_colors[i],
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

        fig.plot()
        

#        axes[1].legend(prop=self.font, loc='lower right', ncol=2)
#        if theta_max is not None:
#            axes[0].axvline(theta_max, color='k', linestyle='--')
#            axes[1].axvline(theta_max, color='k', linestyle='--')

#        if text is not None:
#            axes[0].text(0.3, 0.75, text,
#                         transform=axes[0].transAxes, fontsize=10)


    def fill_agn(self,data):

        cd = self._class_data
        
        for k,v in self._class_data.items():

            evclass = list_to_bitfield(v.class_id)
            evtype = list_to_bitfield(v.type_id)
            cd[k].alpha_hist._counts[:] = 1.0

#           theta_edges = self.psf_data.sig_hist[iegy, icth].axis().edges
#           theta_max=theta_edges[-1]

#           theta_max = min(3.0, self.thetamax_fn(ecenter))
#           theta_edges = np.linspace(0, 3.0, int(3.0 / (theta_max / 100.)))


            fn = copy.deepcopy(cd[k].fn)

#            fn._thmax=3.5
#            fnhi._thmax=3.0
#            fnlo._thmin=2.5
#            fnhi._thmin=2.5
            
            bkg_edge = [2.5, 3.5]
            bkg_domega = (bkg_edge[1] ** 2 - bkg_edge[0] ** 2) * np.pi

            msk_on = PhotonData.get_mask(data, {'cth': self.cth_axis.lims()},
                                         evclass=evclass,evtype=evtype,
                                         conversion_type=self.conversion_type)

            msk_off = PhotonData.get_mask(data, {'dtheta' : bkg_edge,
                                                 'cth': self.cth_axis.lims()},
                                          evclass=evclass,evtype=evtype,
                                          conversion_type=self.conversion_type)

#           hcounts = data.hist('dtheta', mask=mask, edges=theta_edges)
#           domega = (theta_max ** 2) * np.pi

            cd[k].bkg.fill(data['energy'][msk_off])
            bkg_counts = cd[k].bkg.counts

            ebin = self.egy_axis.valToBinBounded(data['energy'])
            bin_energy = self.egy_axis.center[ebin]
            sd = data['dtheta']/cd[k].fn(bin_energy)

            cd[k].off_hist._counts[:,:] = (bkg_counts[:,np.newaxis]*
                                           cd[k].sd_hist.counts[:,np.newaxis]**2*
                                           cd[k].domega[np.newaxis,:]/bkg_domega)

            cd[k].on_hist.fill(data['energy'][msk_on],sd[msk_on])
                                      
#           bkg_counts = self.get_counts(data, bkg_edge, mask)
#           bkg_density = bkg_counts / bkg_domega


        return
        
        qdata.tot_density_hist[iegy, icth] = \
            qdata.sig_hist[iegy, icth].scale_density(lambda x: x**2*np.pi)
        qdata.bkg_density_hist[iegy, icth]._counts = bkg_density

        xedge = np.linspace(-theta_max, theta_max, 301)

        if qdata.sky_image[iegy,icth] is None:
            qdata.sky_image[iegy,icth] = Histogram2D(xedge,xedge)
            qdata.lat_image[iegy,icth] = Histogram2D(xedge,xedge)


        if np.sum(mask):
            qdata.sky_image[iegy,icth].fill(data['delta_ra'][mask],
                                            data['delta_dec'][mask])

            qdata.lat_image[iegy,icth].fill(data['delta_phi'][mask],
                                            data['delta_theta'][mask])


    def fit_agn(self):

        models = self.psf_models
        irf_data = self.irf_data
        psf_data = self.psf_data

        egy_range = psf_data.egy_axis.edges[iegy:iegy+2]
        cth_range = psf_data.cth_axis.edges[icth:icth+2]
        ecenter = psf_data.egy_axis.center[iegy]
        emin = 10 ** psf_data.egy_axis.edges[iegy]
        emax = 10 ** psf_data.egy_axis.edges[iegy+1]

        theta_max = min(self.config['theta_max'], self.thetamax_fn(ecenter))

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
        hq = HistQuantileBkgFn(on_hist, 
                               lambda x: x**2 * np.pi / bkg_domega,
                               bkg_counts)
        
        if excess_sum > 25:
            try:
                self.compute_quantiles(hq, psf_data, iegy, icth, theta_max)
            except Exception, e:
                print e
                
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

        self.plot_theta_residual(psf_data.tot_density_hist[iegy, icth],
                                 psf_data.bkg_density_hist[iegy, icth],
                                 hmodel_density, fig_label)

        fig_label = self.output_prefix + 'theta_counts_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        fig_title = 'E = [%.3f, %.3f] '%(egy_range[0],egy_range[1])
        fig_title += 'Cos$\\theta$ = [%.3f, %.3f]'%(cth_range[0],cth_range[1])
        
        self.plot_psf_cumulative(psf_data.tot_hist[iegy, icth],
                                 psf_data.bkg_hist[iegy, icth],
                                 hmodel_counts, fig_label,fig_title,
                                 None,text)
#                                   bkg_edge[0], text)


        r68 = hq.quantile(0.68)
        r95 = hq.quantile(0.95)


        rs = min(r68 / 4., theta_max / 10.)
        bin_size = 6.0 / 600.

        stacked_image = psf_data.sky_image[iegy,icth]

#        stacked_image.fill(psf_data['delta_ra'][mask],
#                           psf_data['delta_dec'][mask])

        plt.figure()

        stacked_image = stacked_image.smooth(rs)
        

        plt.plot(r68 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 r68 * np.sin(np.linspace(0, 2 * np.pi, 100)), color='k')

        plt.plot(r95 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 r95 * np.sin(np.linspace(0, 2 * np.pi, 100)), color='k',
                 linestyle='--')

        stacked_image.plot()
        plt.gca().set_xlim(-theta_max,theta_max)
        plt.gca().set_ylim(-theta_max,theta_max)
        
        
#        plt.plot(bkg_edge[0] * np.cos(np.linspace(0, 2 * np.pi, 100)),
#                 bkg_edge[0] * np.sin(np.linspace(0, 2 * np.pi, 100)),
#                 color='k',
#                 linestyle='-', linewidth=2)

#        plt.plot(bkg_edge[1] * np.cos(np.linspace(0, 2 * np.pi, 100)),
#                 bkg_edge[1] * np.sin(np.linspace(0, 2 * np.pi, 100)),
#                 color='k',
#                 linestyle='-', linewidth=2)

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

        return
        
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


    def compute_quantiles(self):

        cd = self._class_data

        for k,v in self._class_data.items():
            alpha = cd[k].alpha_hist.counts



            cd[k].sig_hist += cd[k].on_hist
            cd[k].sig_hist -= cd[k].off_hist.counts*alpha[:,np.newaxis]
            cd[k].bkg_hist += cd[k].off_hist.counts*alpha[:,np.newaxis]
            
            qdata = cd[k].qdata

            # Loop on Energy
            for i in range(cd[k].on_hist.axis(0).nbins):

                sd = cd[k].sd_hist.counts[i]                
                on_hist = cd[k].on_hist.slice(0,i)
                off_hist = cd[k].off_hist.slice(0,i)
                sig_hist = cd[k].sig_hist.slice(0,i)

                on_hist._axes[0] = Axis(on_hist._axes[0].edges*sd)
                off_hist._axes[0] = Axis(off_hist._axes[0].edges*sd)
                sig_hist._axes[0] = Axis(sig_hist._axes[0].edges*sd)

#                alpha_hist = cd[k].alpha_hist.slice(0,i)
                
                if self.data_type == 'pulsar':
                    hq = HistQuantileOnOff(on_hist, off_hist, alpha[i])
                else:
                    bkg_counts = cd[k].bkg.counts[i]
                    bkg_domega = (3.5**2-2.5**2)*np.pi

                    hq = HistQuantileBkgFn(on_hist, 
                                           lambda x: x**2 * np.pi / bkg_domega,
                                           bkg_counts)

                print '-'*80

#                print bkg_counts, sig_hist.sum()

                for j, q in enumerate(cd[k].quantiles):
            
                    ql = cd[k].quantile_labels[j]
                    qmean = hq.quantile(fraction=q)
                    qdist_mean, qdist_err = hq.bootstrap(q, niter=200)
                    qdata[j].set(i, qmean, qdist_err ** 2)
                    print ql, ' %10.4f +/- %10.4f' % (qmean, qdist_err)

#    def compute_quantiles_bin(self,on_hist,off_hist,alpha_hist):
        
        
        


    def fill_pulsar(self, data):

        cd = self._class_data
        
        for k,v in self._class_data.items():

            evclass = list_to_bitfield(v.class_id)
            evtype = list_to_bitfield(v.type_id)
            cd[k].alpha_hist._counts[:] = self.alpha

            mask = PhotonData.get_mask(data, {'cth': self.cth_axis.lims()},
                                       evclass=evclass,evtype=evtype,
                                       conversion_type=self.conversion_type)

            msk_on = PhotonData.get_mask(data, {'cth': self.cth_axis.lims()},
                                          conversion_type=self.conversion_type,
                                          evclass=evclass,evtype=evtype,
                                          phases=self.on_phases)

            msk_off = PhotonData.get_mask(data, {'cth': self.cth_axis.lims()},
                                           conversion_type=self.conversion_type,
                                           evclass=evclass,evtype=evtype,
                                           phases=self.off_phases)
            
            ebin = self.egy_axis.valToBinBounded(data['energy'])
            bin_energy = self.egy_axis.center[ebin]
            sd = data['dtheta']/cd[k].fn(bin_energy)
            cd[k].on_hist.fill(data['energy'][msk_on],sd[msk_on])
            cd[k].off_hist.fill(data['energy'][msk_off],sd[msk_off])
            

#            plt.figure()
#            cd[k].on_hist.slice(0,4).plot()
#            cd[k].off_hist.slice(0,4).plot()
#            plt.show()


#            (hon, hoff, hoffs) = getOnOffHist(data, 'dtheta', phases=self.phases,
#                                              edges=theta_edges, mask=mask)

            


        return
        hexcess = copy.deepcopy(hon)
        hexcess -= hoffs

        htotal_density = copy.deepcopy(hexcess)
        htotal_density += hoffs
        htotal_density = htotal_density.scale_density(lambda x: x * x * np.pi)

        hoffs_density = hoffs.scale_density(lambda x: x * x * np.pi)

        excess_sum = np.sum(hexcess._counts)
        on_sum = np.sum(hon._counts)
        off_sum = np.sum(hoff._counts)

        
        src = data._srcs[0]
        xedge = np.linspace(-theta_max, theta_max, 301)

        if qdata.sky_image[iegy,icth] is None:
            qdata.sky_image[iegy,icth] = Histogram2D(xedge,xedge)
            qdata.sky_image_off[iegy,icth] = Histogram2D(xedge,xedge)

            qdata.lat_image[iegy,icth] = Histogram2D(xedge,xedge)
            qdata.lat_image_off[iegy,icth] = Histogram2D(xedge,xedge)
            
        qdata.sky_image[iegy,icth].fill(data['delta_ra'][on_mask],
                                        data['delta_dec'][on_mask])

        qdata.sky_image_off[iegy,icth].fill(data['delta_ra'][off_mask],
                                            data['delta_dec'][off_mask])

        qdata.lat_image[iegy,icth].fill(data['delta_phi'][on_mask],
                                        data['delta_theta'][on_mask])

        qdata.lat_image_off[iegy,icth].fill(data['delta_phi'][off_mask],
                                            data['delta_theta'][off_mask])

        
#        if not isinstance(qdata.sky_image[iegy,icth],SkyImage):        
#            im = SkyImage.createROI(src['RAJ2000'], src['DEJ2000'],
#                                    theta_max, theta_max / 200.)
#            qdata.sky_image[iegy,icth] = im
#        else:
#            im = qdata.sky_image[iegy,icth]

        
#        if len(data['ra'][on_mask]) > 0:
#            im.fill(data['ra'][on_mask], data['dec'][on_mask])

    def fit_pulsar(self, iegy, icth):
        
        models = self.psf_models
        irf_data = self.irf_data
        psf_data = self.psf_data

        egy_range = psf_data.egy_axis.edges[iegy:iegy+2]
        cth_range = psf_data.cth_axis.edges[icth:icth+2]
        ecenter = psf_data.egy_axis.center[iegy]
        emin = psf_data.egy_axis.edges[iegy]
        emax = psf_data.egy_axis.edges[iegy+1]

        print 'Analyzing Bin ', emin, emax
        
        theta_max = min(self.config['theta_max'], self.thetamax_fn(ecenter))

        bkg_hist = psf_data.bkg_hist[iegy, icth]
        sig_hist = psf_data.sig_hist[iegy, icth]
        on_hist = psf_data.tot_hist[iegy, icth]
        off_hist = psf_data.off_hist[iegy, icth]
        excess_sum = psf_data.excess.counts[iegy, icth]

        bkg_density = bkg_hist.sum() / (theta_max ** 2 * np.pi)
        text = 'Bkg Density = %.3f deg$^{-2}$\n' % (bkg_density[0])
        text += 'Signal = %.3f\n' % (psf_data.excess._counts[iegy, icth])
        text += 'Background = %.3f' % (bkg_hist.sum()[0])

        print 'Computing Quantiles'
        hq = HistQuantileOnOff(on_hist, off_hist, self.alpha)
        
        if excess_sum > 25:

            try:
                self.compute_quantiles(hq, psf_data, iegy, icth, theta_max)
            except Exception, e:
                print e
                
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

        self.plot_theta_residual(on_hist, bkg_hist, hmodel_counts,
            #psf_data.tot_density_hist[iegy, icth],
                                 #psf_data.bkg_density_hist[iegy, icth],
                                 #hmodel_density,
                                 fig_label)

        fig_label = self.output_prefix + 'theta_counts_'
        fig_label += '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)

        fig_title = 'E = [%.3f, %.3f] '%(egy_range[0],egy_range[1])
        fig_title += 'Cos$\\theta$ = [%.3f, %.3f]'%(cth_range[0],cth_range[1])
        
        self.plot_psf_cumulative(on_hist, bkg_hist, hmodel_counts,
                                 fig_label,fig_title,
                                 theta_max=None, text=text)



        
        r68 = hq.quantile(0.68)
        r95 = hq.quantile(0.95)

        imin = psf_data.sky_image[iegy,icth].axis(0).valToBin(-r68)
        imax = psf_data.sky_image[iegy,icth].axis(0).valToBin(r68)+1
        

        model_hists = []

        for k,m in models.iteritems():

            xy = psf_data.sky_image[iegy,icth].center()
            r = np.sqrt(xy[0]**2 + xy[1]**2)
            psf = m.pdf(10**emin,10**emax,cth_range[0],cth_range[1],r)
            psf = psf.reshape(psf_data.sky_image[iegy,icth].shape())
            
            h = Histogram2D(psf_data.sky_image[iegy,icth].xaxis(),
                            psf_data.sky_image[iegy,icth].yaxis(),
                            counts=psf)

            h = h.normalize()
            h *= excess_sum                        
            model_hists.append(h)
            
#            im._counts = psf


        fig_suffix = '%04.0f_%04.0f_%03.f%03.f' % (egy_range[0] * 100,
                                                   egy_range[1] * 100,
                                                   cth_range[0] * 100,
                                                   cth_range[1] * 100)
            
        # 2D Sky Image
        self.make_onoff_image(psf_data.sky_image[iegy,icth],
                              psf_data.sky_image_off[iegy,icth],
                              self.alpha,model_hists,
                              'RA Offset [deg]','DEC Offset [deg]',
                              fig_title,r68,r95,
                              'skyimage_' + fig_suffix)
        
        # 2D LAT Image
        self.make_onoff_image(psf_data.lat_image[iegy,icth],
                              psf_data.lat_image_off[iegy,icth],
                              self.alpha,model_hists,
                              'Phi Offset [deg]','Theta Offset [deg]',
                              fig_title,r68,r95,
                              'latimage_' + fig_suffix)

        
        return

        
        # X Projection
        fig = self._ft.create(self.output_prefix + 'xproj_' + fig_suffix,
                              xlabel='Delta RA [deg]')

        imx = im.project(0,[[imin,imax]])
        
        fig[0].add_hist(imx,label='Data')
        for i, h in enumerate(model_hists):
            fig[0].add_hist(h.project(0,[[imin,imax]]),hist_style='line',linestyle='-',
                            label=self.model_labels[i])

        imx2 = imx.slice(0,[[imin,imax]])
            
        mean_err = imx.stddev()/np.sqrt(excess_sum)            
        data_stats = 'Mean = %.3f\nMedian = %.2f'%(imx2.mean(),imx2.quantile(0.5))

        fig[0].ax().set_title(fig_title)
        fig[0].ax().text(0.05,0.95,
                         '%s'%(data_stats),
                         verticalalignment='top',
                         transform=fig[0].ax().transAxes,fontsize=10)


        fig.plot()

        # Y Projection
        fig = self._ft.create(self.output_prefix + 'yproj_' + fig_suffix,
                              xlabel='Delta DEC [deg]')

        imy = im.project(1,[[imin,imax]])

        fig[0].add_hist(imy,label='Data')
        for i, h in enumerate(model_hists):
            fig[0].add_hist(h.project(1,[[imin,imax]]),hist_style='line',linestyle='-',
                            label=self.model_labels[i])

        mean_err = imx.stddev()/np.sqrt(excess_sum)            
        data_stats = 'Mean = %.3f\nRMS = %.2f'%(imx.mean(),imx.stddev())
            
        fig[0].ax().set_title(fig_title)
        fig[0].ax().text(0.05,0.95,
                         '%s'%(data_stats),
                         verticalalignment='top',
                         transform=fig[0].ax().transAxes,fontsize=10)
            
        fig.plot()
            
        

        
        return
        
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

    def make_onoff_image(self,im,imoff,alpha,model_hists,
                         xlabel,ylabel,fig_title,r68,r95,
                         fig_suffix):

        imin = im.axis(0).valToBin(-r68)
        imax = im.axis(0).valToBin(r68)+1
    
        fig = plt.figure(figsize=(8,8))
        plt.gca().set_title(fig_title)
        
        im = copy.deepcopy(im)
        im -= imoff*alpha
        im.smooth(r68/4.).plot(norm=PowerNormalize(2.))
        
        plt.gca().grid(True)
        plt.gca().set_xlabel(xlabel)
        plt.gca().set_ylabel(ylabel)
        
        plt.plot(r68 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 r68 * np.sin(np.linspace(0, 2 * np.pi, 100)), color='w')
    
        plt.plot(r95 * np.cos(np.linspace(0, 2 * np.pi, 100)),
                 r95 * np.sin(np.linspace(0, 2 * np.pi, 100)), color='w',
                 linestyle='--')
    
        plt.gca().set_xlim(*im.xaxis().lims())
        plt.gca().set_ylim(*im.yaxis().lims())
        fig_label = self.output_prefix + fig_suffix
        fig.savefig(fig_label + '.png')

        # X Projection
        fig = self._ft.create(self.output_prefix + fig_suffix + '_xproj',
                              xlabel=xlabel)

        imx = im.project(0,[[imin,imax]])
        
        fig[0].add_hist(imx.rebin(2),label='Data',linestyle='None')
        for i, h in enumerate(model_hists):
            fig[0].add_hist(h.project(0,[[imin,imax]]).rebin(2),
                            hist_style='line',linestyle='-',
                            label=self.model_labels[i])

        imx2 = imx.slice(0,[[imin,imax]]).rebin(2)
        med = imx2.quantile(0.5)        
        data_stats = 'Mean = %.3f\nMedian = %.3f $\pm$ %.3f'%(imx2.mean(),
                                                              med[0],med[1])

        fig[0].ax().set_title(fig_title)
        fig[0].ax().text(0.05,0.95,
                         '%s'%(data_stats),
                         verticalalignment='top',
                         transform=fig[0].ax().transAxes,fontsize=10)


        fig.plot()

        # Y Projection
        fig = self._ft.create(self.output_prefix + fig_suffix + '_yproj',
                              xlabel=ylabel)

        imy = im.project(1,[[imin,imax]])
        
        fig[0].add_hist(imy.rebin(2),label='Data',linestyle='None')
        for i, h in enumerate(model_hists):
            fig[0].add_hist(h.project(0,[[imin,imax]]).rebin(2),
                            hist_style='line',linestyle='-',
                            label=self.model_labels[i])

        imy2 = imy.slice(0,[[imin,imax]])
        med = imy2.quantile(0.5)        
        data_stats = 'Mean = %.3f\nMedian = %.3f $\pm$ %.3f'%(imy2.mean(),med[0],med[1])

        fig[0].ax().set_title(fig_title)
        fig[0].ax().text(0.05,0.95,
                         '%s'%(data_stats),
                         verticalalignment='top',
                         transform=fig[0].ax().transAxes,fontsize=10)


        fig.plot()
        
    def compute_quantiles_bin(self, hq, qdata, iegy, icth,
                              theta_max=None):

        emin = 10 ** qdata.egy_axis.edges[iegy]
        emax = 10 ** qdata.egy_axis.edges[iegy+1]

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

    PSFValidate.add_arguments(parser)

    args = parser.parse_args()

    psfv = PSFValidate(args)
    psfv.run()

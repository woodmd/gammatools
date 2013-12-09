import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from analysis_util import *
from data import *
import stats
import yaml

from psf_likelihood import *
from exposure import ExposureCalc

from optparse import Option
from optparse import OptionParser

class HaloFit(Data):
    
    def __init__(self,lnl,pset_null=None,pset=None):
        self['lnl'] = lnl
        self['pset_null'] = pset_null
        self['pset'] = pset
        self['ts'] = 0.0
        
    def print_fit(self):
        print 'Null Fit Parameters'
        print self['pset_null']

        print 'Signal Fit Parameters'
        print self['pset']

        print 'Signal TS: ', self['ts']

    def plot(self):
        pass

class Config(object):

    def __init__(self):
        pass

    def __getitem__(self,key):
        return self._data[key]

    def __setitem__(self,key,val):
        self._data[key] = val

    def save(self,outfile):

        import cPickle as pickle
        fp = open(outfile,'w')
        pickle.dump(self,fp,protocol = pickle.HIGHEST_PROTOCOL)
        fp.close()

class BinnedHaloAnalysis(Data,Configurable):

    default_config = { 'ebin_edges'  : [],
                       'cth_edges'   : [],
                       'phases'      : [],
                       'halo_sigma'  : 0.1,
                       'event_types' : [] }
    
    def __init__(self,exp,pulsar_data,agn_data,config=None):
        super(BinnedHaloAnalysis,self).__init__()    
        self.configure(BinnedHaloAnalysis.default_config,config)
        self.cfg = self.config()
        
        self._ha_bin = []

        for i in range(len(self.cfg['ebin_edges'][:-1])):
            
            print 'Setting up energy bin ', i, self.cfg['ebin_edges'][i:i+2]

            ebin_cfg = copy.deepcopy(cfg)
            ebin_cfg['ebin_edges'] = self.cfg['ebin_edges'][i:i+2]

            ha = HaloAnalysis(exp,pulsar_data,agn_data,ebin_cfg)
            self._ha_bin.append(ha)

    def __iter__(self):
        return iter(self._ha_bin)

    def setup_lnl(self):

        for ha in self._ha_bin: ha.setup_lnl()

    def fit(self):        

        for ha in self._ha_bin: ha.fit()

    def print_fit(self):

        for ha in self._ha_bin: ha.print_fit()

    def plot_images(self):

        for ha in self._ha_bin: ha.plot_images()

    def plot(self):
        
        for ha in self._ha_bin: ha.plot()

    def get_ts_halo(self):

        ts_halo = []
        for ha in self._ha_bin: ts_halo.append(ha._ts_halo)

        return ts_halo

    def plot_ul(self,fig=None):

        flux = []
        loge = []

#        for i in range(len(self.cfg['ebin_edges'][:-1])):
        for i, ha in enumerate(self._ha_bin):

            ecenter = 0.5*np.sum(self.cfg['ebin_edges'][i:i+2])

            loge.append(ecenter)

            fn = copy.deepcopy(ha._halo_model)
            fn.setParamByName('halo_flux',ha._flux_ul_halo)

            flux.append((fn.eval(ecenter)*10**(2*ecenter)).flat[0])
            print ecenter, fn.eval(ecenter)*10**(2*ecenter)

#            self._flux_ul_halo

        if fig is None: fig = plt.figure()
        from plot_util import plot_ul

        plot_ul(loge,flux)

#        plt.arrow(loge[0],flux[0],0,-flux[0]*0.5,arrowstyle='simple')
        plt.gca().set_yscale('log')
        plt.gca().set_ylim(1E-7,2E-5)
        plt.gca().grid(True)

        plt.gca().set_xlabel('Energy [log$_{10}$(E/MeV)]')
        plt.gca().set_ylabel('E$^{2}$dF/dE [MeV cm$^{-2}$ s$^{-1}$]')

#        fig.savefig('flux_ul.png')
        
#        plt.show()
#            flux.append()


class HaloAnalysis(Data,Configurable):

    default_config = { 'ebin_edges'  : [],
                       'theta_edges' : np.linspace(0,3.0,31),
                       'cth_edges'   : [],
                       'phases'      : [],
                       'halo_sigma'  : 0.1,
                       'cuts_file'   : None,
                       'event_types' : [] }

    def __init__(self,exp,pulsar_data,agn_data,config=None):
        super(HaloAnalysis,self).__init__()    
        self.configure(HaloAnalysis.default_config,config)        
        self.cfg = self.config()

        self._pset = ParameterSet()
        self._pset_null = None
        self._pset_halo = None
        self._ts_halo = None
        self._plnl_halo_flux = None
        self._plnl_halo_dlnl = None
        self._srcs = []
        
        self.load_data(exp,pulsar_data,agn_data)

    def set_seed_pset(self,pset):

        for p in pset:
            if p.name() in self._pset.names():
                self._pset.getParByName(p.name()).set(p.value())

    def fix_param(self,regex):
        self._pset.fixAll(True,regex)
        
    def load_data(self,exp,pulsar_data,agn_data):

        nebin = len(self.cfg['ebin_edges'])-1
        ntype = len(self.cfg['event_types'])

        self.data = np.empty(shape=(nebin,ntype), dtype=object)
        self._srcs = copy.deepcopy(agn_data._srcs)
        
        for i in range(len(self.data)):

            erange = self.cfg['ebin_edges'][i:i+2]
            print erange
            for j, c in enumerate(self.cfg['event_types']):
                self.data[i,j] = self.load_bin_data(c,erange,exp[i],
                                                    pulsar_data,agn_data)
                                                    
            

    def load_bin_data(self,c,erange,exp,pulsar_data,agn_data):

        data = Data()

        
        data['agn_exp'] = exp
        data['energy'] = erange
        data['cth'] = [0.4,1.0]
        data['conversion_type'] = c['conversion_type']
        data['cuts'] = None
        data['par_id'] = []
        if 'cuts' in c: data['cuts'] = c['cuts']
        if 'label' in c: data['label'] = c['label']
        else: data['label'] = c['conversion_type']
        
        ctype = c['conversion_type']

        mask = PhotonData.get_mask(pulsar_data,
                                   {'energy' : erange,
                                    'cth' : self.cfg['cth_edges']},
                                   conversion_type=ctype,
                                   cuts=data['cuts'],
                                   cuts_file=self.cfg['cuts_file'])

#        data['pulsar_mask'] = mask

        (hon,hoff,hoffs) = getOnOffHist(pulsar_data,'dtheta',
                                        self.cfg['phases'],
                                        mask=mask,
                                        edges=self.cfg['theta_edges'])

        data['pulsar_theta_hist_on'] = hon
        data['pulsar_theta_hist_off'] = hoff
        data['pulsar_theta_hist_bkg'] = hoffs
        data['pulsar_theta_hist_excess'] = hon - hoffs

        hq = stats.HistQuantileBkgHist(hon,hoff,self.cfg['phases'][2])

        data['pulsar_q68'] = hq.quantile(0.68)

        mask = PhotonData.get_mask(agn_data,
                                   {'energy' : erange,
                                    'cth' : self.cfg['cth_edges']},
                                   conversion_type=ctype,
                                   cuts=data['cuts'],
                                   cuts_file=self.cfg['cuts_file'])

#        data['agn_mask'] = mask

        data['agn_theta_hist'] = getHist(agn_data,'dtheta',mask=mask,
                                         edges=self.cfg['theta_edges'])


        
        
        stacked_image = Histogram2D(np.linspace(-3.0,3.0,301),
                                    np.linspace(-3.0,3.0,301))

        stacked_image.fill(agn_data['delta_ra'][mask],
                           agn_data['delta_dec'][mask])

        
        src = agn_data._srcs[0]

        srcra = src['RAJ2000']
        srcdec = src['DEJ2000']

        im = SkyImage.createROI(srcra,srcdec,3.0,3.0/300.)
        im.fill(agn_data['ra'][mask],agn_data['dec'][mask])

        data['agn_image'] = im
#        data['agn_image_smoothed'] = im.smooth(data['pulsar_q68']/4.)
        data['agn_stacked_image'] = stacked_image
#        data['agn_stacked_image_smoothed'] = \
#            stacked_image.smooth(data['pulsar_q68']/4.)

        return data

    def setup_lnl(self):
        
        self._joint_lnl = JointLnL()

        self._pset.clear()
        
        halo_flux = self._pset.createParameter(0.0,'halo_flux',True,
                                               [0.0,1E-6])
        halo_gamma = self._pset.createParameter(2.0,'halo_gamma',True)
        halo_sigma = self._pset.createParameter(0.3,'halo_sigma',True)
        halo_norm = self._pset.createParameter(1.0,'halo_norm',True)

        self._halo_model = PowerlawFn(halo_flux,halo_gamma)

        self._pset.setParByName('halo_sigma',self.cfg['halo_sigma'])

        for i in range(len(self.data)):
            self.setup_lnl_bin(self.data[i],self._halo_model)
            for j in range(len(self.data[i].flat)):
                self._joint_lnl.add(self.data[i,j]['lnl'])

    def setup_lnl_bin(self,data,halo_model):
        """Construct the likelihood function for a single
        energy/inclination angle bin."""

        agn_excess = []
        pulsar_excess = []
        pulsar_fraction = []
        agn_bkg = []

        
        
        for d in data:

            excess, bkg_density, r68 = analyze_agn(d['agn_theta_hist'],2.5)
            agn_excess.append(excess)
            agn_bkg.append(bkg_density)
            pulsar_excess.append(d['pulsar_theta_hist_excess'].sum())


#        agn_tot_hist = getHist(agn_data,'dtheta',mask=agn_mask,
#                               edges=theta_edges)
#        excess, bkg_density, r68 = analyze_agn(agn_tot_hist,2.5)

        agn_excess_tot = np.sum(np.array(agn_excess))
        pulsar_excess_tot = np.sum(np.array(pulsar_excess))

        for x in pulsar_excess:
            pulsar_fraction.append(x/pulsar_excess_tot)

        pset = self._pset

        elabel = 'e%04.f_%04.f'%(data[0]['energy'][0]*100,
                               data[0]['energy'][1]*100)
        
        # Total Counts in Vela and AGN
        p0 = pset.createParameter(pulsar_excess_tot,'vela_norm_%s'%(elabel),
                                  False,[0,max(10,10*pulsar_excess_tot)])
        p1 = pset.createParameter(agn_excess_tot,'agn_norm_%s'%(elabel),
                                  False,[0,max(10,10*agn_excess_tot)])

        pfname = []
        ndata = len(data)

        for i in range(1,ndata):            
            p = pset.createParameter(pulsar_fraction[i],'acc_f%i_%s'%(i,elabel))
            data[i]['par_id'].append(p.pid())
            pfname.append(p.name())

        if ndata > 1:
            expr0 = '(1.0 - (' + '+'.join(pfname) + '))'
        else:
            expr0 = '(1.0)'

        for i, d in enumerate(data):

 #           excess, bkg_density, r68 = analyze_agn(d['agn_hist'],2.5)
            r68 = d['pulsar_q68']

            pulsar_model = CompositeModel()
            agn_model = CompositeModel()

            p2 = pset.createParameter(0.5*r68,'psf_sigma_%s_%02i'%(elabel,i))
            p3 = pset.createParameter(2.0,'psf_gamma_%s_%02i'%(elabel,i),
                                      False,[1.1,6.0])
            p4 = pset.createParameter(agn_bkg[i],
                                      'agn_iso_%s_%02i'%(elabel,i),False,
                                      [0.0,10.*agn_bkg[i]])

            d['par_id'].append(p2.pid())
            d['par_id'].append(p3.pid())
            d['par_id'].append(p4.pid())
            
            expr = None
            if len(data) > 1 and i == 0: expr = expr0
            elif len(data) > 1: expr = pfname[i-1]

            pulsar_src_model = ScaledModel(KingFn(p2,p3,p0),pset,expr,
                                           name='pulsar')
            agn_src_model = ScaledModel(KingFn(p2,p3,p1),pset,expr,name='agn')

            halo_norm = pset.getParByName("halo_norm")
            halo_sigma = pset.getParByName("halo_sigma")

            halo_spatial_model = ConvolvedGaussFn(halo_norm,
                                                  halo_sigma,KingFn(p2,p3))

            halo_spectral_model = BinnedPLFluxModel(halo_model,
                                                    halo_spatial_model,
                                                    data[i]['energy'],
                                                    data[i]['agn_exp'])
                                                    
            halo_src_model = ScaledModel(halo_spectral_model,
                                         pset,expr,name='halo')

            pulsar_model.addModel(pulsar_src_model)
            agn_model.addModel(agn_src_model)
            agn_model.addModel(halo_src_model)

            agn_model.addModel(PolarPolyFn(ParameterSet([p4]),name='iso'))

            d['pulsar_model'] = pulsar_model
            d['agn_model'] = agn_model

            agn_norm_expr = p1.name()
            if not expr is None: agn_norm_expr += '*' + expr

            d['par_agn_norm'] = CompositeParameter(agn_norm_expr,pset)
            d['par_agn_iso'] = p4
            d['par_psf_sigma'] = p2
            d['par_psf_gamma'] = p3
            d['pulsar_lnl'] = \
                OnOffBinnedLnL.createFromHist(d['pulsar_theta_hist_on'],
                                              d['pulsar_theta_hist_off'],
                                              self.cfg['phases'][2],
                                              pulsar_model)

            d['agn_lnl'] = BinnedLnL.createFromHist(d['agn_theta_hist'],
                                                    agn_model)

            joint_lnl = JointLnL([d['pulsar_lnl'],d['agn_lnl']])
            d['lnl'] = joint_lnl

    def print_fit(self):

        print 'Null Fit Parameters'
        print self._pset_null

        print 'Signal Fit Parameters'
        print self._pset_halo

        print 'Signal TS: ', self._ts_halo
        print 'Halo UL: ', self._flux_ul_halo

    def set_halo_prop(self,halo_sigma, halo_gamma=2.0):
        self.cfg['halo_sigma'] = halo_sigma
        self.cfg['halo_gamma'] = halo_gamma

    def fit(self):
        
        print 'Fitting'

        fixed = self._pset.fixed()

        print fixed
        
        self._joint_lnl.setParam(self._pset)
        print self._joint_lnl.param()
        
        fitter = Fitter(self._joint_lnl)

        print 'Null fit'
        pset = copy.deepcopy(self._pset)
        
        nbin = len(self.data.flat)
        for i in range(nbin):
            print i, self.data.flat[i]['par_id']

            pset.fixAll(True)        
            for pid in self.data.flat[i]['par_id']:

                if not self._pset.getParByIndex(pid).fixed():
                    pset.getParByIndex(pid).fix(False)

            print pset                
            pset = fitter.fit(pset)
            print pset

        for pid in pset.pids():
            pset.getParByIndex(pid).fix(self._pset.getParByIndex(pid).fixed())
            
        pset_null = fitter.fit(pset)
        print pset_null

#        fitter.plot_lnl_scan(pset_null)
#        plt.show()
#        sys.exit(0)

        self._pset_null = pset_null
        self._pset_halo = None
        self._ts_halo = None
        self._plnl_halo_flux = None
        self._plnl_halo_dlnl = None        
        self._flux_ul_halo = 0.0

        pset = copy.deepcopy(pset_null)

        pset.getParByName('halo_flux').fix(False)

        print 'Halo fit'
        pset_halo = fitter.fit(pset)
        print pset_halo

        print 'Computing UL'        
        self._pset_halo = pset_halo
        self._ts_halo = -2*(pset_halo.fval() - pset_null.fval())
        self.compute_flux_ul()

    def compute_flux_ul(self):

        delta_lnl = 2.72/2.

        print 'Computing Profile Likelihood'
        
        self._plnl_halo_flux, self._plnl_halo_dlnl = \
            self.compute_flux_plnl()

        print 'Flux: ', self._plnl_halo_flux
        print 'lnL:  ', self._plnl_halo_dlnl
        
        self._plnl_fn = UnivariateSpline(self._plnl_halo_flux,
                                         self._plnl_halo_dlnl,s=0,
                                         k=2)
        
        i = np.argmin(self._plnl_halo_dlnl)

        if self._plnl_halo_flux[i] < 0: offset = self._plnl_fn(0)
        else: offset = self._plnl_halo_dlnl[i]

        x0 = brentq(lambda t: self._plnl_fn(t) - delta_lnl - offset,
                    self._plnl_halo_flux[i],
                    self._plnl_halo_flux[-1],xtol=1E-16)

        self._flux_ul_halo = x0

        return

        x = np.linspace(self._plnl_halo_flux[0],self._plnl_halo_flux[-1],100)
        plt.plot(x,fn(x),marker='o')
        plt.gca().grid(True)
        plt.axhline(2.72/2.)
        plt.axvline(x0)
        plt.show()


    def compute_flux_plnl(self):

        pset = self._pset_halo

        fitter = Fitter(self._joint_lnl)
        fmin = pset.fval()
        pset = copy.deepcopy(pset)

        pset.fixAll()
        pset.fixAll(False,'agn_norm')

        halo_flux = np.ravel(pset.getParByName('halo_flux').value())

        if halo_flux > 0: xmin = max(-14,np.log10(halo_flux))
        else: xmin = -14

        print 'xmin ', xmin
        
        x = np.linspace(xmin,-8,100)
        p = pset.makeParameterArray(0,10**x)
        fv = fitter._objfn.eval(p)
        fn = UnivariateSpline(x,fv,s=0,k=2)

#        plt.figure()
#        plt.plot(x,fv-fmin)
#        plt.show()

        
        x0 = brentq(lambda t: fn(t) - fmin - 100.,x[0],x[-1],xtol=1E-16)
        
        err = pset.getParError('halo_flux')
        err = max(err,1E-13)
        
        v = np.ravel(pset.getParByName('halo_flux').value())

        pmin = v - 5*err
        pmin[pmin<0] = 0

        pmax = 10**x0
        pval = np.linspace(v,pmax,20)
        
        fval = fitter.profile(pset,'halo_flux',pval,True)-fmin

        return pval,fval

    def plot_images(self):

        nbin = len(self.data.flat)

        bin_per_fig = 4

        nfig = int(np.ceil(float(nbin)/float(bin_per_fig)))
        
        nx = 2
        ny = 2
        figsize = (8*1.5,6*1.5)    

        for i in range(nfig):
            
            fig = plt.figure(figsize=figsize)
            for j in range(i*bin_per_fig,(i+1)*bin_per_fig):

                if j >= len(self.data.flat): continue

                d = self.data.flat[j]

                elabel = 'e%04.f_%04.f'%(d['energy'][0]*100,
                                         d['energy'][1]*100)

                
                subplot = '%i%i%i'%(nx,ny,j%bin_per_fig+1)
                self.plot_images_bin(self.data.flat[j],subplot)

            fig.savefig('skyimage_%s_%02i.png'%(elabel,i))

    def plot_images_bin(self,data,subplot=111):

        im = data['agn_image'].smooth(data['pulsar_q68']/4.)

        title = '%s E = [%.3f, %.3f]'%(data['label'],
                                       data['energy'][0],
                                       data['energy'][1])
            

        ax = im.plot(subplot=subplot)
        ax.set_title(title)

        ax.add_beam_size(data['pulsar_q68'],data['pulsar_q68'],0.0,loc=2,
                         patch_props={'ec' : 'white', 'fc' : 'None'})
        im.plot_catalog()
        im.plot_circle(3.0,linewidth=2,linestyle='--',color='k')
        


    def plot(self,pset=None):
                
        pset = [self._pset_null,self._pset_halo]
        pset_labels = ['Null','Halo']

        nbin = len(self.data.flat)
        bin_per_fig = 4
        nfig = int(np.ceil(float(nbin)/float(bin_per_fig)))
        
        nx = 2
        ny = 2
        figsize = (8*1.5,6*1.5)            


        for i in range(nfig):
        
            fig0 = plt.figure(figsize=figsize)
            fig1 = plt.figure(figsize=figsize)
            for j in range(i*bin_per_fig,(i+1)*bin_per_fig):

                if j >= len(self.data.flat): continue

                print i, j
                
                d = self.data.flat[j]

                elabel = 'e%04.f_%04.f'%(d['energy'][0]*100,
                                         d['energy'][1]*100)


                title = '%s E = [%.3f, %.3f]'%(d['label'],
                                               d['energy'][0],
                                               d['energy'][1])
            
                ax1 = fig0.add_subplot(ny,nx,j%bin_per_fig+1)
                ax1.set_title(title)
            
                plt.sca(ax1)
                self.plot_pulsar_bin(self.data.flat[j],ax1,pset,pset_labels)
                ax1.grid(True)
                ax1.legend()

                ax2 = fig1.add_subplot(ny,nx,j%bin_per_fig+1)
                ax2.set_title(title)
            
                plt.sca(ax2)
                self.plot_agn_bin(self.data.flat[j],ax2,pset,pset_labels)
                ax2.grid(True)
                ax2.legend()

            fig0.savefig('vela_dtheta_%s_%02i.png'%(elabel,i))
            fig1.savefig('src_dtheta_%s_%02i.png'%(elabel,i))
                

    def plot_pulsar_bin(self,data,ax,pset,pset_labels):
        edges = data['pulsar_theta_hist_on'].edges()

        data['pulsar_theta_hist_on'].plot(ax=ax,label='on')
        data['pulsar_theta_hist_bkg'].plot(ax=ax,label='off')

        for i, p in enumerate(pset):
            hm = data['pulsar_model'].histogram(edges,p=p)
            hm += data['pulsar_theta_hist_bkg']
            hm.plot(ax=ax,style='line',label=pset_labels[i])
        
    
    def plot_agn_bin(self,data,ax,pset,pset_labels):
        edges = data['agn_theta_hist'].edges()
        data['agn_theta_hist'].plot(ax=ax,label='Data')


        psf_sigma_pid = data['par_psf_sigma'].pid()
        psf_gamma_pid = data['par_psf_gamma'].pid()

        text = '$\sigma$ = %6.3f '%(self._pset_null.
                                    getParByID(psf_sigma_pid).value())
        text += '$\pm$ %6.3f deg\n'%(self._pset_null.
                                 getParError(psf_sigma_pid))

        text += '$\gamma$ = %6.3f '%(self._pset_null.
                                    getParByID(psf_gamma_pid).value())
        text += '$\pm$ %6.3f\n'%(self._pset_null.
                                 getParError(psf_gamma_pid))

        text += '$\Sigma_{b}$ = %6.3f deg$^{-2}$\n'%(data['par_agn_iso'].value())
        text += 'N$_{src}$ = %6.3f\n'%(data['par_agn_norm'].
                                       eval(self._pset_halo))
        text += 'TS = %6.3f'%(self._ts_halo)

        for i, p in enumerate(pset):
            
            hm = data['agn_model'].histogram(edges,p=pset[i])
            hm.plot(ax=ax,style='line',label=pset_labels[i])
            ax.text(0.1,0.7,text,transform=ax.transAxes,fontsize=10)

    def plot_bin(self,data,pset=None):

        if pset is None:
            pset = [self._pset_null,self._pset_halo]

        


        for i in range(len(data.flat)):

            d = self.data.flat[i]

            edges = d['pulsar_theta_hist_on'].edges()
            ax0 = fig0.add_subplot(2,2,i+1)
            plt.sca(ax0)

            d['pulsar_theta_hist_on'].plot(label='on')
            d['pulsar_theta_hist_bkg'].plot(label='off')

            for p in pset:
                hm = d['pulsar_model'].histogram(edges,p=p)
                hm += d['pulsar_theta_hist_bkg']
                hm.plot(style='line')

            plt.gca().grid(True)

            ax1 = fig1.add_subplot(2,2,i+1)
            plt.sca(ax1)

#            plt.figure()
            d['agn_theta_hist'].plot()
            for j in range(len(pset)):
                hm1 = d['agn_model'].histogram(edges,p=pset[j])
                hm1.plot(style='line')
#            hm1c = d['agn_model'].histogramComponents(edges,p=pset)
#            for h in hm1c: h.plot(style='line')

            plt.gca().grid(True)
            plt.gca().legend()

#        plt.show()
        

class HaloAnalysisManager(Configurable):
    """Halo analysis object.  Responsible for parsing configuration
    file and passing data to HaloAnalysisData."""

    default_config = {'pulsar_data' : None,
           'agn_data'    : None,
           'on_phase'    : '0.0/0.15,0.6/0.7',
           'off_phase'   : '0.2/0.5',
           'energy_bins' : None,
           'theta_bins'  : '0.0/3.0/60',
           'halo_sigma'  : None,
           'halo_gamma'  : None,
           'ltfile'      : None,
           'irf'         : None,
           'irf_dir'     : None,
           'cuts_file'   : None,
           'output_file' : None,
           'event_types' : None }

    def __init__(self,config=None):
        super(HaloAnalysisManager,self).__init__()    
        self.configure(HaloAnalysisManager.default_config,config)

        cfg = self.config()
        
        (bmin,bmax,nbin) = cfg['energy_bins'].split('/')
        self._ebin_edges = np.linspace(float(bmin),float(bmax),int(nbin)+1)
                                      
        (bmin,bmax,nbin) = cfg['theta_bins'].split('/')
        self._theta_edges = np.linspace(float(bmin),float(bmax),int(nbin)+1)

        self._cth_edges = np.linspace(0.4,1.0,2)
        

    def load(self):

        cfg = self.config()
        
        self._phases = parse_phases(cfg['on_phase'],
                                    cfg['off_phase'])

        self._pulsar_data = PhotonData.load(cfg['pulsar_data']['file'])
        self._agn_data = PhotonData.load(cfg['agn_data']['file'])

        self._exp_calc = ExposureCalc.create(cfg['irf'],
                                             cfg['ltfile'],
                                             cfg['irf_dir'])

        if 'srcs' in cfg['agn_data'] and not \
                cfg['agn_data']['srcs'] is None:
            self._agn_data.get_srcs(cfg['agn_data']['srcs'])        

        src_names = []
        for s in self._agn_data._srcs:
            src_names.append(s['Source_Name'])
            
        self._exp = self._exp_calc.getExpByName(src_names, self._ebin_edges)
        # Convert exposure to cm^2 s
        self._exp *= 1E4

        self._pulsar_data['dtheta'] = np.degrees(self._pulsar_data['dtheta'])
        self._agn_data['dtheta'] = np.degrees(self._agn_data['dtheta'])

        
        mask = PhotonData.get_mask(self._pulsar_data,
                                   {'energy' : [self._ebin_edges[0],
                                                self._ebin_edges[-1]] })
        self._pulsar_data.apply_mask(mask)

    def run(self):
        """Run both the binned and joint halo analysis and write
        analysis objects to an output file."""

        fit_data = Data()

        cfg = copy.deepcopy(self.config())

        cfg['ebin_edges'] = self._ebin_edges
        cfg['theta_edges'] = self._theta_edges
        cfg['cth_edges'] = self._cth_edges
        cfg['phases'] = self._phases

        # Setup Joint Fit
        ha_binned = BinnedHaloAnalysis(self._exp,self._pulsar_data,
                                       self._agn_data,cfg)        
        ha_binned.setup_lnl()
        ha_binned.fit()

        fit_data['binned_fit'] = ha_binned

        ha_joint = HaloAnalysis(self._exp,self._pulsar_data,self._agn_data,
                                cfg)        
        ha_joint.setup_lnl()

        for ha in ha_binned:
            ha_joint.set_seed_pset(ha._pset_null)

        ha_joint.fix_param('vela')
        ha_joint.fix_param('psf')
        ha_joint.fix_param('acc\_f')
        ha_joint.fit()

        fit_data['joint_fit'] = ha_joint


        fit_data.save(cfg['output_file'])


def analyze_agn(hon,theta_cut):

    i = hon.getBinByValue(theta_cut)


    xlo = hon._xedges[i]
    xhi = hon._xedges[-1]


    sig_domega = xlo**2*np.pi
    bkg_domega = (xhi**2-xlo**2)*np.pi
    bkg_counts = np.sum(hon._counts[i:])
    
    bkg_density = bkg_counts/bkg_domega

    excess = np.sum(hon._counts[:i]) - bkg_density*sig_domega

    if excess <= 0: return 0, bkg_density, 0.0
    
    hq = stats.HistQuantileBkgFn(hon,lambda x: x*x*np.pi/bkg_domega,
                                 bkg_counts)

    return excess, bkg_density, hq.quantile(0.68)


if __name__ == '__main__':

    gfn = ConvolvedGaussFn.create(3.0,0.1,KingFn.create(0.1,3.0),4)

    pset = gfn.param()
    
    p0 = pset.createParameter(1.0,'norm')
    p1 = pset.createParameter(2.0,'gamma')

    pfn = PowerlawFn(p0,p1)

    cm = CompProdModel()
    cm.addModel(gfn)
    cm.addModel(pfn)

    x = np.linspace(1,2,3)
    y = np.linspace(0,1,3)

    print pfn.eval(2.0)
    print pfn.eval(3.0)
    print pfn.integrate(2.0,3.0)
    print pfn.eval({'energy' : 3.0 })

    print pset

    v0 = pfn.eval(x)
    v1 = gfn.eval(y)

    print cm.eval(2.0)
    print cm.eval({'energy': x, 'dtheta' : y})
    print v0*v1


    h = Histogram2D([0,1,2],[0,1,2])


    print h._xedges
    print h._yedges

    h.fill(0.5,0.5,1.0)

    lnl = Binned2DLnL.createFromHist(h,cm)


    print lnl.eval(pset.makeParameterArray(0,np.linspace(0.1,2,3)))

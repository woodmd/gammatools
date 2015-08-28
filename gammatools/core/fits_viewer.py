
import numpy as np
import os
import copy
from astropy_helper import pyfits
from gammatools.core.plot_util import *
from gammatools.core.stats import poisson_lnl
from gammatools.core.histogram import Histogram, Axis
from gammatools.core.fits_util import SkyImage, SkyCube
from gammatools.fermi.catalog import Catalog
import matplotlib.pyplot as plt
from itertools import cycle

class FITSPlotter(object):

    fignum = 0
    
    def __init__(self,im,im_mdl,irf=None,prefix=None,outdir='plots',
                 title=None,format='png',
                 rsmooth=0.2,nosrclabels=False,srclabels_thresh=5.0):

        self._ft = FigTool(fig_dir=outdir)
        self._im = im
        self._im_mdl = im_mdl
        self._irf = irf
        self._prefix = prefix
        self._rsmooth = rsmooth
        self._title = title
        self._format = format
        self._nosrclabels = nosrclabels
        if nosrclabels:
            self._srclabels_thresh = None
        else:
            self._srclabels_thresh = srclabels_thresh
        
        if self._prefix is None: self._prefix = 'fig'
        if outdir: self._prefix_path = os.path.join(outdir,self._prefix)
        
    def make_mdl_plots_skycube(self,**kwargs):

        self.make_plots_skycube(model=True,**kwargs)

    def make_title(self,**kwargs):

        emin = kwargs.get('emin',None)
        emax = kwargs.get('emax',None)
        
        if self._title: return self._title
        elif emin and emax:
            return 'log$_{10}$(E/MeV) = [%.3f, %.3f]'%(emin,emax)
        else:
            return ''
        
    def make_energy_residual(self,suffix=''):

        h = self._im.project(2)
        hm = self._im_mdl.project(2)

        bins = np.linspace(0,h.axis(0).nbins,h.axis(0).nbins+1)
        bins = np.concatenate((bins[:10],bins[10:16:2],bins[16::4]))
        
        h = h.rebin(bins)
        hm = hm.rebin(bins)

        title = self.make_title()
        
        fig = self._ft.create(self._prefix + '_' + suffix,
                              format=self._format,
                              figstyle='residual2',
                              yscale='log',
                              #title=title,
                              ylabel='Counts',
                              xlabel='Energy [log$_{10}$(E/MeV)]')

        fig[0].add_hist(hm,hist_style='line',label='Model')
        fig[0].add_hist(h,linestyle='None',label='Data',color='k')
        fig[1].set_style('ylim',[-0.4,0.4])
        fig[0].ax().set_title(title)
        fig.plot()


        fig = self._ft.create(self._prefix + '_' + suffix + 'e2',
                              format=self._format,
                              figstyle='residual2',
                              yscale='log',
                              #title=title,
                              ylabel='E$^{2}$dN/dE [MeV]',
                              xlabel='Energy [log$_{10}$(E/MeV)]')

        e2 = 10**(2*h.axis(0).center)/(10**h.axis(0).edges[1:]-
                                       10**h.axis(0).edges[:-1])
        
        fig[0].add_hist(hm*e2,hist_style='line',label='Model')
        fig[0].add_hist(h*e2,linestyle='None',label='Data',color='k')
        fig[1].set_style('ylim',[-0.2,0.2])
        fig[0].ax().set_title(title)
        fig.plot()
        
#        fig.savefig('%s_%s.png'%(self._prefix,suffix))

    def make_plots_skycube(self,delta_bin=None,paxis=None,plots_per_fig=4,
                           smooth=False, resid_type=None,make_projection=False,
                           suffix='',model=False, projection='psf68',**kwargs):

        if model: im = self._im_mdl
        else: im = self._im
        
        nbins = im.axis(2).nbins
        if delta_bin is None:
            bins = np.array([0,nbins])
            nplots = 1
        else:
            bins = np.cumsum(np.concatenate(([0],delta_bin)))
            nplots = len(bins)-1

        nfig = int(np.ceil(float(nplots)/float(plots_per_fig)))
        
        if plots_per_fig > 4 and plots_per_fig <= 8:
            nx, ny = 4, 2
        elif plots_per_fig <= 4 and plots_per_fig > 1:
            nx, ny = 2, 2
        else:
            nx, ny = 1, 1

        fig_sx = 5.0*nx
        fig_sy = 5.0*ny


        figs = []
        for i in range(nfig):

            if nfig > 1: fig_suffix = '_%02i'%i
            else: fig_suffix = ''
                
            fig_name = '%s_%s%s.%s'%(self._prefix_path,suffix,fig_suffix,self._format)
            fig2_name = '%s_%s_zproj%s.%s'%(self._prefix_path,suffix,fig_suffix,self._format)
            fig3_name = '%s_%s_xproj%s.%s'%(self._prefix_path,suffix,fig_suffix,self._format)
            fig4_name = '%s_%s_yproj%s.%s'%(self._prefix_path,suffix,fig_suffix,self._format)
            
            figs.append({'fig'  : self.create_figure(figsize=(fig_sx,fig_sy)),
                         'fig2' : self.create_figure(figsize=(fig_sx,fig_sy)),
                         'fig3' : self.create_figure(figsize=(fig_sx,fig_sy)),
                         'fig4' : self.create_figure(figsize=(fig_sx,fig_sy)),
                         'fig_name' : fig_name,
                         'fig2_name' : fig2_name,
                         'fig3_name' : fig3_name,
                         'fig4_name' : fig4_name,
                         })
            
        plots = []
        for i in range(nplots):
                           
            ifig = i/plots_per_fig
            plots.append({'fig'  : figs[ifig]['fig'],
                          'fig2' : figs[ifig]['fig2'],
                          'fig3' : figs[ifig]['fig3'],
                          'fig4' : figs[ifig]['fig4'],
                          'ibin' : [bins[i], bins[i+1]],
                          'subplot' : i%plots_per_fig,
                         })
                          
        for i, p in enumerate(plots):
            
            ibin0, ibin1 = p['ibin']
            emin = im.axis(2).pix_to_coord(ibin0)
            emax = im.axis(2).pix_to_coord(ibin1)
            
            rpsf68 = self._irf.quantile(10**emin,10**emax,0.2,1.0,0.68)
            rpsf95 = self._irf.quantile(10**emin,10**emax,0.2,1.0,0.95)

            if smooth:
                delta_proj = 0.0
            elif projection== 'psf68':
                delta_proj = rpsf68
            elif isinstance(projection,float):
                delta_proj = projection
                
            x0 = im.axis(0).coord_to_pix(delta_proj,False)
            x1 = im.axis(0).coord_to_pix(-delta_proj,False)
            y0 = im.axis(1).coord_to_pix(-delta_proj,False)
            y1 = im.axis(1).coord_to_pix(delta_proj,False)

            if smooth:
                x1 = x0+1
                y1 = y0+1

            title = self.make_title(emin=emin,emax=emax)
            subplot = '%i%i%i'%(ny,nx,p['subplot']+1)
            
            fig = p['fig']
            fig2 = p['fig2']
            fig3 = p['fig3']
            fig4 = p['fig4']

            h = im.marginalize(2,[[ibin0,ibin1]])
            hm = None
            if self._im_mdl:
                hm = self._im_mdl.marginalize(2,[[ibin0,ibin1]])

            mc_resid = []

            if resid_type:
                for k in range(10):
                    mc_resid.append(self.make_residual_map(h,hm,
                                                           smooth,mc=True,
                                                           resid_type=resid_type))
                h = self.make_residual_map(h,hm,smooth,resid_type=resid_type)
            elif smooth:
                h = h.smooth(self._rsmooth,compute_var=True)
                hm = hm.smooth(self._rsmooth,compute_var=True)
                
#                h = self.make_counts_map(im,ibin,ibin+delta_bin,
#                                         residual,smooth)
                

            self.make_image_plot(subplot,h,fig,fig2,
                                 title,rpsf68,rpsf95,
                                 smooth=smooth,
                                 resid_type=resid_type,
                                 mc_resid=mc_resid,**kwargs)

            if make_projection:
                ax = fig3.add_subplot(subplot)
                self.make_projection_plot(ax,h,0,[x0,x1],title=title,
                                          hm=hm,**kwargs)                
                ax = fig4.add_subplot(subplot)
                self.make_projection_plot(ax,h,1,[y0,y1],title=title,
                                          hm=hm,**kwargs) 

        for f in figs:
            f['fig'].savefig(f['fig_name'])

            if not resid_type is None:
                f['fig2'].savefig(f['fig2_name'])

            if make_projection:
                f['fig3'].savefig(f['fig3_name'])
                f['fig4'].savefig(f['fig4_name'])

    def create_figure(self,**kwargs):
        fig = plt.figure('Figure %i'%FITSPlotter.fignum,**kwargs)
        FITSPlotter.fignum += 1
        return fig

    def make_projection_plot(self,ax,h,pindex,plims,title='',hm=None,**kwargs):
        plt.sca(ax)
                
        hp = h.project(pindex,[plims],offset_coord=True)
        hp.plot(ax=ax,linestyle='None',label='Data',**kwargs)
                
        if hm:
            hmp = hm.project(pindex,[plims],offset_coord=True)
            hmp.plot(ax=ax,label='Model',hist_style='line',linestyle='-',**kwargs)

        ax.grid(True)

        if pindex == 0:
            ax.set_xlabel('GLON Offset')
        elif pindex == 1:
            ax.set_xlabel('GLAT Offset')
            
        ax.set_xlim(*hp.axis().lims())
        ax.legend(loc='upper right')
        ax.set_ylim(0)
        ax.set_title(title)
    
    def make_residual_map(self,h,hm,smooth,mc=False,resid_type='fractional'):
        
        if mc:
            h = copy.deepcopy(h)
            h._counts = np.array(np.random.poisson(hm.counts),
                                 dtype='float')
            
        if smooth:
            hm = hm.smooth(self._rsmooth,compute_var=True,summed=True)
            h = h.smooth(self._rsmooth,compute_var=True,summed=True)

        ts = 2.0*(poisson_lnl(h.counts,h.counts) -
                  poisson_lnl(h.counts,hm.counts))

        s = h.counts - hm.counts

        if resid_type == 'fractional':
            h._counts = s/hm.counts
            h._var = np.zeros(s.shape)
        else:
            sigma = np.sqrt(ts)
            sigma[s<0] *= -1
            h._counts = sigma
            h._var = np.zeros(sigma.shape)
        
#        h._counts -= hm._counts
#        h._counts /= np.sqrt(hm._var)

        return h
        

    def make_image_plot(self,subplot,h,fig,fig2,title,rpsf68,rpsf95,
                        smooth=False,
                        resid_type=None,mc_resid=None,**kwargs):

        plt.figure(fig.get_label())

        cb_label='Counts'

        if resid_type == 'significance':
            kwargs['vmin'] = -5
            kwargs['vmax'] = 5
            kwargs['levels'] = [-5.0,-3.0,3.0,5.0]
            cb_label = 'Significance [$\sigma$]'
        elif resid_type == 'fractional':
            kwargs['vmin'] = -1.0
            kwargs['vmax'] = 1.0
            kwargs['levels'] = [-1.0,-0.5,0.5,1.0]
            cb_label = 'Fractional Residual'

        if smooth:
            kwargs['beam_size'] = [self._rsmooth,self._rsmooth,0.0,4]
            
        axim = h.plot(subplot=subplot,cmap='ds9_b',**kwargs)
        h.plot_circle(rpsf68,color='w',lw=1.5)
        h.plot_circle(rpsf95,color='w',linestyle='--',lw=1.5)
        h.plot_marker(marker='x',color='w',linestyle='--')
        ax = h.ax()
        ax.set_title(title)
        cb = plt.colorbar(axim,orientation='horizontal',
                          shrink=0.9,pad=0.15,
                          fraction=0.05)

        if kwargs.get('zscale',None) is not None:
            import matplotlib.ticker        
            cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
            cb.update_ticks()
        
        cb.set_label(cb_label)

        cat = Catalog.get('3fgl')
        cat.plot(h,ax=ax,src_color='w',
                 label_threshold=self._srclabels_thresh)

        if resid_type is None: return
        
        plt.figure(fig2.get_label())        
        ax2 = fig2.add_subplot(subplot)

        z = h.counts[10:-10,10:-10]

        if resid_type == 'significance':
            zproj_axis = Axis.create(-6,6,120)
        elif resid_type == 'fractional':
            zproj_axis = Axis.create(-1.0,1.0,120)
        else:
            zproj_axis = Axis.create(-10,10,120)


        hz = Histogram(zproj_axis)
        hz.fill(np.ravel(z))

        nbin = np.prod(z.shape)
        
        hz_mc = Histogram(zproj_axis)    
        
        if mc_resid:
            for mch in mc_resid:
                z = mch.counts[10:-10,10:-10]
                hz_mc.fill(np.ravel(z))

            hz_mc /= float(len(mc_resid))

            
        fn = lambda t : 1./np.sqrt(2*np.pi)*np.exp(-t**2/2.)
        
        hz.plot(label='Data',linestyle='None')

        if resid_type == 'significance':
            plt.plot(hz.axis().center,
                     fn(hz.axis().center)*hz.axis().width*nbin,
                     color='k',label='Gaussian ($\sigma = 1$)')
        
        hz_mc.plot(label='MC',hist_style='line')
        plt.gca().grid(True)
        plt.gca().set_yscale('log')
        plt.gca().set_ylim(0.5)

        ax2.legend(loc='upper right',prop= {'size' : 10 })

        data_stats = 'Mean = %.2f\nRMS = %.2f'%(hz.mean(),hz.stddev())
        mc_stats = 'MC Mean = %.2f\nMC RMS = %.2f'%(hz_mc.mean(),
                                                    hz_mc.stddev())
        
        ax2.set_xlabel(cb_label)
        ax2.set_title(title)
        ax2.text(0.05,0.95,
                 '%s\n%s'%(data_stats,mc_stats),
                 verticalalignment='top',
                 transform=ax2.transAxes,fontsize=10)

                 
def make_projection_plots_skycube(im,paxis,delta_bin=2):

    nbins = im.axis(2).nbins
    nfig = nbins/(8*delta_bin)
    
    for i in range(nfig):

        fig, axes = plt.subplots(2,4,figsize=(1.5*10,1.5*5))
        for j in range(8):

            ibin = i*nfig*delta_bin + j*delta_bin

            if ibin >= nbins: break
            
            print i, j, ibin
        
            h = im.marginalize(2,[ibin,ibin+1])
            emin = im.axis(2).pix_to_coord(ibin)
            emax = im.axis(2).pix_to_coord(ibin+delta_bin)

    
            axes.flat[j].set_title('E = [%.3f %.3f]'%(emin,emax))

            hp = h.project(paxis)
            hp.plot(ax=axes.flat[j])
            axes.flat[j].set_xlim(*hp.axis().lims())
            axes.flat[j].set_ylim(0)
        


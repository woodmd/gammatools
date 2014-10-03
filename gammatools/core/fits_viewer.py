import wx
import numpy as np
import os
import copy
from astropy_helper import pyfits

from gammatools.core.plot_util import *
from gammatools.core.stats import poisson_lnl
from gammatools.core.histogram import Histogram, Axis
from gammatools.core.fits_util import SkyImage, SkyCube
from gammatools.fermi.catalog import Catalog
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wxagg import Toolbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.colors import NoNorm, LogNorm, Normalize

from itertools import cycle

def make_projection_plots_skyimage(im):

    plt.figure()

    im.project(0).plot()
    
    plt.figure()

    im.project(1).plot()




    
class FITSPlotter(object):

    fignum = 0
    
    def __init__(self,im,im_mdl,irf=None,prefix=None,outdir='plots',
                 rsmooth=0.2):

        self._ft = FigTool(fig_dir=outdir)
        self._im = im
        self._im_mdl = im_mdl
        self._irf = irf
        self._prefix = prefix
        self._rsmooth = rsmooth
        if self._prefix is None: self._prefix = 'fig'
        if outdir: self._prefix_path = os.path.join(outdir,self._prefix)
        
    def make_mdl_plots_skycube(self,**kwargs):

        self.make_plots_skycube(model=True,**kwargs)
        
    def make_energy_residual(self,suffix=''):

        h = self._im.project(2)
        hm = self._im_mdl.project(2)

        fig = self._ft.create(self._prefix + suffix,
                              figstyle='residual2',
                              yscale='log',
                              ylabel='Counts',
                              xlabel='Energy [log$_{10}$(E/MeV)]')

        fig[0].add_hist(hm,hist_style='line',label='Model')
        fig[0].add_hist(h,linestyle='None',label='Data')

        fig[1].set_style('ylim',[-0.3,0.3])


        fig.plot()
#        fig.savefig('%s_%s.png'%(self._prefix,suffix))

    def make_plots_skycube(self,delta_bin=None,paxis=None,plots_per_fig=4,
                           smooth=False, resid_type=None,
                           suffix='',model=False, **kwargs):

        if model: im = self._im_mdl
        else: im = self._im
        
        nbins = im.axis(2).nbins
        if delta_bin is None:
            delta_bin = np.array([0,nbins])
            nplots = 1
        else:
            delta_bin = np.array([0,2,4,4,12])
            nplots = 4

        bins = np.cumsum(delta_bin)
        nfig = int(np.ceil(float(nplots)/float(plots_per_fig)))
        
        if plots_per_fig > 4 and plots_per_fig <= 8:
            nx, ny = 4, 2
        elif plots_per_fig <= 4 and plots_per_fig > 1:
            nx, ny = 2, 2
        else:
            nx, ny = 1, 1

        fig_sx = 5.0*nx
        fig_sy = 5.0*ny

#        figs = []
#        for i in range(nfig): figs.append(plt.figure(figsize=(fig_sx,fig_sy)))
        
        for i in range(nfig):

#        fig, axes = plt.subplots(2,4,figsize=(1.5*10,1.5*5))

            fig = self.create_figure(figsize=(fig_sx,fig_sy))
            fig2 = self.create_figure(figsize=(fig_sx,fig_sy))
            ##plt.figure(FITSPlotter.fignum,figsize=(fig_sx,fig_sy))

            imin, imax = i*nfig, min(nplots,i*nfig+plots_per_fig)            
            ibin0, ibin1 = bins[imin], bins[imax]
            
            fig_emin = im.axis(2).pix_to_coord(ibin0)
            fig_emax = im.axis(2).pix_to_coord(ibin1)

            #            fig2 = plt.figure(100+i,figsize=(fig_sx,fig_sy))
            for j in range(plots_per_fig):

                iplot = i*nfig+j

                if iplot >= nplots: break
                
                ibin0 = bins[iplot]
                ibin1 = bins[iplot+1]
                print i, j, ibin0, ibin1
                
                
                if ibin0 >= nbins: break
#                print i, j, ibin

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
                elif smooth: h = h.smooth(self._rsmooth,compute_var=True)
                
#                h = self.make_counts_map(im,ibin,ibin+delta_bin,
#                                         residual,smooth)

                emin = im.axis(2).pix_to_coord(ibin0)
                emax = im.axis(2).pix_to_coord(ibin1)
                
                rpsf68 = self._irf.quantile(10**emin,10**emax,0.2,1.0,0.68)
                rpsf95 = self._irf.quantile(10**emin,10**emax,0.2,1.0,0.95)
                
                title = 'log$_{10}$(E/MeV) = [%.3f %.3f]'%(emin,emax)
                
                subplot = '%i%i%i'%(ny,nx,j+1)
                
                if paxis is None:
                    self.make_image_plot(subplot,h,fig,fig2,
                                         title,rpsf68,rpsf95,
                                         resid_type=resid_type,
                                         mc_resid=mc_resid,**kwargs)
                else:
                    ax = fig.add_subplot(subplot)
                    hp = h.project(paxis)
                    hp.plot(ax=ax,**kwargs)
                    ax.set_xlim(*hp.axis().lims())
                
#                ax.set_ylim(0)

        fig_label = '%04.f_%04.f'%(fig_emin*1000,fig_emax*1000)
                    
        fig.savefig('%s_%s%s.png'%(self._prefix_path,fig_label,suffix))
        fig2.savefig('%s_%s%s_zproj.png'%(self._prefix_path,fig_label,suffix))

    def create_figure(self,**kwargs):
        fig = plt.figure('Figure %i'%FITSPlotter.fignum,**kwargs)
        FITSPlotter.fignum += 1
        return fig
        
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

        axim = h.plot(subplot=subplot,cmap='ds9_b',**kwargs)
        h.plot_circle(rpsf68,color='w')
        h.plot_circle(rpsf95,color='w',linestyle='--')
        h.plot_marker(marker='+',color='w',linestyle='--')
        ax = h.ax()
        ax.set_title(title)
        cb = plt.colorbar(axim,orientation='horizontal',
                          shrink=0.9,pad=0.15,
                          fraction=0.05)

        cb.set_label(cb_label)

        cat = Catalog.get('3fgl')
        cat.plot(h,ax=ax,src_color='w',label_threshold=5.0)

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
        

class Knob:
    """
    Knob - simple class with a "setKnob" method.  
    A Knob instance is attached to a Param instance, e.g., param.attach(knob)
    Base class is for documentation purposes.
    """
    def setKnob(self, value):
        pass


class Param(object):
    """
    The idea of the "Param" class is that some parameter in the GUI may have
    several knobs that both control it and reflect the parameter's state, e.g.
    a slider, text, and dragging can all change the value of the frequency in
    the waveform of this example.  
    The class allows a cleaner way to update/"feedback" to the other knobs when 
    one is being changed.  Also, this class handles min/max constraints for all
    the knobs.
    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """
    def __init__(self, initialValue=None, minimum=0., maximum=1.):
        self.minimum = minimum
        self.maximum = maximum
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []
        
    def attach(self, knob):
        self.knobs += [knob]

    def setMax(self, maximum):

        self.maximum = maximum
        
    def set(self, value, knob=None):
        self.value = value
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    def value(self):
        return self.value

    def constrain(self, value):
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value

class CtrlGroup(object):

    def __init__(self, parent, label, pmin, pmax, fn, pdefault=None):
        self.label = wx.StaticText(parent, label=label)
    
class SpinCtrlGroup(CtrlGroup):
    def __init__(self, parent, label, pmin, pmax, fn, pdefault=None):
        CtrlGroup.__init__(self,parent,label,pmin,pmax,fn,pdefault=None)
        
        self.spinCtrl = wx.SpinCtrl(parent,style=wx.SP_ARROW_KEYS)#, pos=(150, 75), size=(60, -1))
        self.spinCtrl.SetRange(pmin,pmax) 
        self.spinCtrl.SetValue(pmin)
#        self.spinCtrl.Bind(wx.EVT_SPINCTRL, self.spinCtrlHandler)
        parent.Bind(wx.EVT_SPINCTRL, self.spinCtrlHandler,self.spinCtrl)

        self.sizer = wx.GridBagSizer(1,2)#wx.BoxSizer(wx.HORIZONTAL)

        self.sizer.Add(self.label, pos=(0,0),
                       flag = wx.EXPAND | wx.ALIGN_CENTER,
                       border=5)

        self.sizer.Add(self.spinCtrl, pos=(0,1),
                       flag = wx.EXPAND | wx.ALIGN_CENTER | wx.ALIGN_RIGHT,
                       border=5)

#        self.sizer.Add(self.label, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
#                       border=2)

#        self.sizer.Add(self.spinCtrl, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
#                       border=2)

        self.fn = fn

    def init(self,pmin,pmax,pdefault=None):

        print 'spinCtrlGroup init ', pmin, pmax, pdefault
        
        self.spinCtrl.SetRange(pmin,pmax)
        if pdefault is None: self.spinCtrl.SetValue(pmin)
        else: self.spinCtrl.SetValue(pdefault)

    def spinCtrlHandler(self, evt):
        v = evt.GetPosition() 
        print 'spinCtrlHandler ', v
        self.fn(v)

class TwinSliderGroup(object):
    def __init__(self, parent, label, pmin, pmax, fnlo, fnhi, pdefault=None):
        
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.slider_lo = SliderGroup(parent,label,pmin,pmax,fn=self.loSliderHandler,float_arg=True)
        self.slider_hi = SliderGroup(parent,label,pmin,pmax,fn=self.hiSliderHandler,float_arg=True)
                
        sizer.Add(self.slider_lo.sizer, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
                  border=2)
        sizer.Add(self.slider_hi.sizer, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
                  border=2)
        
        self.sizer = sizer

        self.fnlo = fnlo
        self.fnhi = fnhi

    def init(self,pmin,pmax,pdefault=None):

        self.slider_lo.SetMin(pmin)
        self.slider_lo.SetMax(pmax)
        self.slider_hi.SetMin(pmin)
        self.slider_hi.SetMax(pmax)
#        if pdefault is None: self.set(pmin)
#        else: self.set(pdefault)
                
    def loSliderHandler(self, v):
#        self.slider_hi.set(v)

        if self.slider_lo.value() > self.slider_hi.value():
            self.slider_hi.set(self.slider_lo.value())
        
        self.fnlo(v)
        
    def hiSliderHandler(self, v):

        if self.slider_hi.value() < self.slider_lo.value():
            self.slider_lo.set(self.slider_hi.value())
        
        self.fnhi(v)
        


        
class SliderGroup(object):
    def __init__(self, parent, label, pmin, pmax, fn,
                 pdefault=None, float_arg=False):
        self.sliderLabel = wx.StaticText(parent, label=label)
        self.sliderText = wx.TextCtrl(parent, -1, style=wx.TE_PROCESS_ENTER)

#        self.spinCtrl = wx.SpinCtrl(parent, value='0')#, pos=(150, 75), size=(60, -1))
#        self.spinCtrl.SetRange(pmin,pax)       
#        self.spinCtrl.Bind(wx.EVT_SPINCTRL, self.sliderSpinCtrlHandler)

        self.float_arg = float_arg
        
        self.slider_min = 0
        self.slider_max = 1000
        
        self.slider = wx.Slider(parent, -1,style=wx.SL_MIN_MAX_LABELS)#,style=wx.SL_AUTOTICKS | wx.SL_LABELS)

        self.init(pmin,pmax,pdefault)
        
#        sizer = wx.GridBagSizer(1,3)
#        sizer.Add(self.sliderLabel, pos=(0,0), 
#                  border=5,flag =  wx.ALIGN_CENTER)#,flag=wx.EXPAND)
#        sizer.Add(self.sliderText, pos=(0,1), 
#                  border=5,flag =  wx.ALIGN_CENTER)#,flag=wx.EXPAND)
#        sizer.Add(self.slider, pos=(0,2),flag=wx.EXPAND | wx.ALIGN_CENTER,border=5)


        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderLabel, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
                  border=2)
        sizer.Add(self.sliderText, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
                  border=2)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.sizer = sizer

        self.slider.Bind(wx.EVT_SLIDER, self.sliderHandler)
        self.sliderText.Bind(wx.EVT_TEXT_ENTER, self.sliderTextHandler)

        self.fn = fn

    def getValue(self):

        if self.float_arg:        
            return float(self.sliderText.GetValue())
        else:
            return int(self.sliderText.GetValue())
            
    def init(self,pmin,pmax,pdefault=None):

        print 'sliderGroup init ', pmin, pmax, pdefault
        
        self.pmin = pmin
        self.pmax = pmax
        
        if not self.float_arg:
            self.slider_min = pmin
            self.slider_max = pmax

        self.slider.SetMin(self.slider_min)
        self.slider.SetMax(self.slider_max)

        if pdefault is None: self.set(pmin)
        else: self.set(pdefault)
        
#        if pdefault is None: self.set(pmin)            
#        self.slider.SetMin(pmin)
#        self.slider.SetMax(pmax)
#        if pdefault is None: self.set(pmin)
#        else: self.set(pdefault)
        
    def disable(self):
        self.slider.Enable(False)
        self.sliderText.Enable(False)
        
    def sliderHandler(self, evt):
        v = evt.GetInt()

        if self.float_arg:
            v = self.pmin + float(v)/1000.*(self.pmax-self.pmin)
            
        self.set(v)
        self.fn(v)
        
    def sliderTextHandler(self, evt):
        v = self.sliderText.GetValue()
        self.set(v)
        self.fn(v)

    def value(self):

        v = self.slider.GetValue()
        if self.float_arg:
            return self.pmin + float(v)/1000.*(self.pmax-self.pmin)
        else: return v
        
    def set(self, value):

        print 'sliderGroup value ', value
        
        self.sliderText.SetValue('%s'%value)

        if self.float_arg:
            v = 1000*((value-self.pmin)/(self.pmax-self.pmin))
            #v = min(max(v,0),1000)

            print 'set ', v, value
            
            self.slider.SetValue(int(v))
        else:
            self.slider.SetValue(int(value))

class FITSViewerApp(wx.App):

    def __init__(self,im):

        self._im = im
        wx.App.__init__(self)


    def OnInit(self):

        print 'im: ', self._im
        
        self.frame = FITSViewerFrame(self._im,parent=None,
                                     title="FITS Viewer",
                                     size=(1.5*640, 1.5*480))
        
#        self.frame1 = FITSViewerFrame(parent=None,
#                                      title="FITS Viewer",
#                                      size=(640, 480))
        self.frame.Show()
        return True
        
class FITSViewerFrame(wx.Frame):
    def __init__(self, files, hdu=0,*args, **kwargs):
        wx.Frame.__init__(self, *args,**kwargs)

        self.files = files
        self.hdulist = []
        self.show_image = []
        self.image_window = []

        for i, f in enumerate(files):
            self.hdulist.append(pyfits.open(f))
            self.show_image.append(True)
            self.image_window.append(ImagePanel(self,i))
            
        self.hdu = hdu
        self.slice = 0
        self.nbin = 1
        self._projx_width = 10.
        self._projx_center = 50.
        self._projy_width = 10.
        self._projy_center = 50.

        self.projx_window = PlotPanel(self,12,0,'LAT Projection','','')
        self.projy_window = PlotPanel(self,13,1,'LON Projection','','')

#        self.ctrl_slice = SliderGroup(self,'Slice',0,6,fn=self.update_slice)
        
        self.ctrl_slice = SpinCtrlGroup(self,'Slice',0,6,fn=self.update_slice)
        self.ctrl_nbins = SpinCtrlGroup(self,'NBins',0,6,fn=self.update_nbin)
        self.ctrl_hdu = SpinCtrlGroup(self,'HDU',0,6,fn=self.update_hdu)

#        self.spinctrl0 = wx.SpinCtrl(self, value='0')#, pos=(150, 75), size=(60, -1))
#        self.spinctrl0.SetRange(0,6)
        
 #       self.spinctrl0.Bind(wx.EVT_SPINCTRL, lambda evt: self.update_slice(evt.GetPosition()))


        
        self.ctrl_projx_center = SliderGroup(self,'X Center',0,100,
                                             self.update_projx_center,
                                             float_arg=True)

        self.ctrl_projx_width = SliderGroup(self,'X Width',0,100,
                                            self.update_projx_width,
                                            float_arg=True)

        
        self.ctrl_projy_center = SliderGroup(self,'Y Center',0,100,
                                             self.update_projy_center,
                                             float_arg=True)

        self.ctrl_projy_width = SliderGroup(self,'Y Width',0,100,
                                            self.update_projy_width,
                                            float_arg=True)
        
        self.ctrl_rebin = SpinCtrlGroup(self,'Rebin',1,4,
                                          fn=self.update_rebin)

        self.ctrl_hdu.init(0,len(self.hdulist[0])-1)

        sb0 = wx.StaticBox(self, label="Optional Attributes")
        sb0sizer = wx.StaticBoxSizer(sb0, wx.VERTICAL)

        sb1 = wx.StaticBox(self, label="Projection")
        sb1sizer = wx.StaticBoxSizer(sb1, wx.VERTICAL)

        sb2 = wx.StaticBox(self, label="Transform")
        sb2sizer = wx.StaticBoxSizer(sb2, wx.VERTICAL)

        sizer_main = wx.BoxSizer(wx.HORIZONTAL)

        sizer_plots = wx.BoxSizer(wx.HORIZONTAL)

        sizer_proj = wx.BoxSizer(wx.VERTICAL)
        self.sizer_image = wx.BoxSizer(wx.VERTICAL)
        
        sizer_ctrls = wx.BoxSizer(wx.VERTICAL)
#        sizer.Add(self.window, 1, wx.EXPAND)
        sizer_ctrls.Add(sb0sizer, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer_ctrls.Add(sb1sizer, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer_ctrls.Add(sb2sizer, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        sb0sizer.Add(self.ctrl_slice.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sb0sizer.Add(self.ctrl_nbins.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sb0sizer.Add(self.ctrl_hdu.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        fn = []

        for i, w in enumerate(self.image_window):
            cb = wx.CheckBox(self, label="Image %i"%i)
            cb.Bind(wx.EVT_CHECKBOX, lambda t,i=i: self.toggle_image(t,i))
            cb.SetValue(True)

            sb0sizer.Add(cb, 0,
                         wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        # Projection Controls

        cb_proj_norm = wx.CheckBox(self, label="Normalize")
        cb_proj_norm.Bind(wx.EVT_CHECKBOX, self.toggle_proj_norm)
            
        sb1sizer.Add(self.ctrl_rebin.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        sb1sizer.Add(self.ctrl_projx_center.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        sb1sizer.Add(self.ctrl_projx_width.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        sb1sizer.Add(self.ctrl_projy_center.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        sb1sizer.Add(self.ctrl_projy_width.sizer, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        
        sb1sizer.Add(cb_proj_norm, 0,
                     wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        
        tc0 = wx.TextCtrl(self, -1, style=wx.TE_PROCESS_ENTER)

        bt0 = wx.Button(self, label="Update")
        bt0.Bind(wx.EVT_BUTTON, self.update)

        sizer_ctrls.Add(bt0, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        cb1 = wx.CheckBox(self, label="Log Scale")
        cb1.Bind(wx.EVT_CHECKBOX, self.toggle_yscale)

        cb0 = wx.CheckBox(self, label="Smooth")
        cb0.Bind(wx.EVT_CHECKBOX, self.update_smoothing)

        sb2sizer.Add(cb0,flag=wx.LEFT|wx.TOP, border=5)
        sb2sizer.Add(cb1,flag=wx.LEFT|wx.TOP, border=5)
        sb2sizer.Add(bt0,flag=wx.LEFT|wx.TOP, border=5)
        sb2sizer.Add(tc0,flag=wx.LEFT|wx.TOP, border=5)

        sizer_main.Add(sizer_ctrls,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)
        sizer_main.Add(sizer_plots,3, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)

        sizer_plots.Add(self.sizer_image,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)
        sizer_plots.Add(sizer_proj,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)

        for w in self.image_window:
            self.sizer_image.Add(w,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)

        sizer_proj.Add(self.projx_window,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)
        sizer_proj.Add(self.projy_window,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)

        self.SetSizer(sizer_main)

        self.update_hdu(self.hdu)
        self.update_slice(self.slice)
        
    def update_hdu(self,value):

        self.hdu = int(value)

        for w in self.image_window: w.clear()
        self.projx_window.clear()
        self.projy_window.clear()

        for i, hl in enumerate(self.hdulist):
            hdu = hl[self.hdu]        
            self.load_hdu(hdu,self.image_window[i])

        self.update()
        
    def load_hdu(self,hdu,image_window):

        print 'Loading HDU'

        style = {}

        if 'CREATOR' in hdu.header and hdu.header['CREATOR'] == 'gtsrcmaps':
            style['hist_style'] = 'line'
            style['linestyle'] = '-'

        if hdu.header['NAXIS'] == 3:
            im = SkyCube.createFromHDU(hdu)

            if image_window: image_window.add(im,style)
            self.projx_window.add(im,style)
            self.projy_window.add(im,style)
            self.ctrl_slice.init(0,im.axis(2).nbins-1)
            self.ctrl_nbins.init(1,im.axis(2).nbins)

            self.nbinx = im.axis(0).nbins
            self.nbiny = im.axis(1).nbins

            self._projx_center = self.nbinx/2.
            self._projx_width = 10.0

            self._projy_center = self.nbiny/2.
            self._projy_width = 10.0
            
        else:
            im = SkyImage.createFromHDU(hdu)
            self.ctrl_slice.init(0,0)
            self.ctrl_slice.disable()
            self.ctrl_nbins.init(0,0)
            self.ctrl_nbins.disable()

        self.ctrl_projx_center.init(0,self.nbinx,self._projx_center)
        self.ctrl_projx_width.init(0,self.nbinx,self._projx_width)

        self.ctrl_projy_center.init(0,self.nbiny,self._projy_center)
        self.ctrl_projy_width.init(0,self.nbiny,self._projy_width)
        
    def update_slice(self,value):

        self.slice = int(value)

        for w in self.image_window: w.set_slice(value)
        self.projx_window.set_slice(value)
        self.projy_window.set_slice(value)
#        self.update()

    def update_nbin(self,value):

        self.nbin = int(value)

        for w in self.image_window: w.set_nbin(value)
        self.projx_window.set_nbin(value)
        self.projy_window.set_nbin(value)
#        self.update()

    def update_rebin(self,value):
        
        self.projx_window.set_rebin(value)
        self.projy_window.set_rebin(value)
#        self.update()

    def update_projx_lo(self,value):
        
        for w in self.image_window: w.set_projx_range_lo(value)
        self.projx_window.set_proj_range_lo(value)
        for w in self.image_window: w.update_lines()

    def update_projx_hi(self,value):
        
        for w in self.image_window: w.set_projx_range_hi(value)
        self.projx_window.set_proj_range_hi(value)
        for w in self.image_window: w.update_lines()

    def update_projx_center(self,value):

        self._projx_center = value
        self.update_projx()

    def update_projx_width(self,value):

        self._projx_width = value
        self.update_projx()

    def update_projx(self):

        projx_lo = self._projx_center - self._projx_width*0.5
        projx_hi = self._projx_center + self._projx_width*0.5
        
        for w in self.image_window: w.set_projx_range(projx_lo,projx_hi)
        self.projx_window.set_proj_range(projx_lo,projx_hi)
        for w in self.image_window: w.update_lines()
        self.projx_window.update()
        
    def update_projy_lo(self,value):
        
        for w in self.image_window: w.set_projy_range_lo(value)
        self.projy_window.set_proj_range_lo(value)
        for w in self.image_window: w.update_lines()

    def update_projy_hi(self,value):
        
        for w in self.image_window: w.set_projy_range_hi(value)
        self.projy_window.set_proj_range_hi(value)
        for w in self.image_window: w.update_lines()

    def update_projy_center(self,value):

        self._projy_center = value
        self.update_projy()

    def update_projy_width(self,value):

        self._projy_width = value
        self.update_projy()

    def update_projy(self):

        projy_lo = self._projy_center - self._projy_width*0.5
        projy_hi = self._projy_center + self._projy_width*0.5
        
        for w in self.image_window: w.set_projy_range(projy_lo,projy_hi)
        self.projy_window.set_proj_range(projy_lo,projy_hi)
        for w in self.image_window: w.update_lines()
        self.projy_window.update()
        
    def update_smoothing(self, e):
        
        sender = e.GetEventObject()
        isChecked = sender.GetValue()

        if isChecked: 
            self.image_window[0].smooth = True
        else: 
            self.image_window[0].smooth = False

#        self.update()

    def toggle_image(self, e, i):

        w = self.image_window[i]

        sender = e.GetEventObject()
        if sender.GetValue(): w.Show()
        else: w.Hide()

        self.sizer_image.Layout()
#        self.update()

    def toggle_proj_norm(self, e):

        sender = e.GetEventObject()
        if sender.GetValue():
            self.projx_window.set_norm(True)
            self.projy_window.set_norm(True)
        else:
            self.projx_window.set_norm(False)
            self.projy_window.set_norm(False)
        self.projx_window.update()
        self.projy_window.update()

    def toggle_smoothing(self, e):
        
        sender = e.GetEventObject()
        if sender.GetValue(): 
            self.image_window[0].smooth = True
        else: 
            self.image_window[0].smooth = False

        self.update()

    def toggle_yscale(self, evt):
        
        sender = evt.GetEventObject()
        if sender.GetValue():
            for w in self.image_window:
                w.update_style('logz',True)
        else: 
            for w in self.image_window:
                w.update_style('logz',False)

        self.update()

    def update(self,evt=None):

        self.update_projx()
        self.update_projy()
        
        for w in self.image_window: w.update()
        self.projx_window.update()
        self.projy_window.update()
        
#    def OnPaint(self, event):
#        print 'OnPaint'
#        self.window.canvas.draw()
        

class BasePanel(wx.Panel):
    
    def __init__(self, parent,fignum):
        wx.Panel.__init__(self, parent, -1)

        self.slice = 0
        self.nbin = 1
        self.rebin = 0
        
    def update_style(self,k,v):
        for i, s in enumerate(self._style):
            self._style[i][k] = v


            
class ImagePanel(BasePanel):

    def __init__(self, parent,fignum):
        BasePanel.__init__(self, parent, fignum)

        self._fignum = fignum
        self._fig = plt.figure(fignum,figsize=(5,4), dpi=75)
        self.canvas = FigureCanvasWxAgg(self, -1, self._fig)
        self.toolbar = NavigationToolbar2Wx(self._fig.canvas)
        #Toolbar(self.canvas) #matplotlib toolbar
        self.toolbar.Realize()

        #self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        self._ax = None
        self._projx_lines = [None,None]
        self._projy_lines = [None,None]
        self._im = []
        self._cm = []
        self._style = []
        self._axim = []
        self.smooth = False

        self.projx_range = [None, None]
        self.projy_range = [None, None]
        
#        self.toolbar.update() # Not sure why this is needed - ADS

        
    def set_projx_range_lo(self,value):
        self.projx_range[0] = value

    def set_projx_range_hi(self,value):
        self.projx_range[1] = value

    def set_projx_range(self,lovalue,hivalue):
        self.projx_range[0] = lovalue
        self.projx_range[1] = hivalue

    def set_projy_range_lo(self,value):
        self.projy_range[0] = value

    def set_projy_range_hi(self,value):
        self.projy_range[1] = value

    def set_projy_range(self,lovalue,hivalue):
        self.projy_range[0] = lovalue
        self.projy_range[1] = hivalue

    def draw(self):

        bin_range = [self.slice,self.slice+self.nbin]

        if isinstance(self._im,SkyCube):
            im = self._im.marginalize(2,bin_range=bin_range)
        else:
            im = self._im

        self.scf()
        self._axim = im.plot()

        self.scf()
        self._axim.set_data(im.counts.T)
        self._axim.autoscale()
        self.canvas.draw()
        self._fig.canvas.draw()

    def clear(self):

        self._im = []
        self._style = []
        self._axim = []

    def add(self,im,style):

        style.setdefault('logz',False)

        self._im.append(im)
        self._style.append(style)

    def scf(self):
        plt.figure(self._fignum)

    def set_slice(self,value):
        self.slice = int(value)

    def set_nbin(self,value):
        self.nbin = int(value)


    def update(self):

        self.update_image()
        self.update_lines()
        
    def update_image(self):

        self.scf()

        bin_range = [self.slice,self.slice+self.nbin]

        self._cm = []

        for im in self._im:
            if isinstance(im,SkyCube):
                self._cm.append(im.marginalize(2,bin_range=bin_range))
            else:
                self._cm.append(im)

        for i in range(len(self._cm)):
            if self.smooth: self._cm[i] = self._cm[i].smooth(0.1)

        if len(self._cm) == 0: return

        cm = self._cm
        
        cat = Catalog.get('3fgl')
        
        if len(self._axim) == 0:

            axim = cm[0].plot(**self._style[0])

            cat.plot(cm[0],src_color='w')
            
            self._axim.append(axim)
            self._ax = plt.gca()

            
            
            
            
            plt.colorbar(axim,orientation='horizontal',shrink=0.7,pad=0.15,
                         fraction=0.05)

            return

        self._axim[0].set_data(cm[0].counts.T)
        self._axim[0].autoscale()

        if self._style[0]['logz']:
            self._axim[0].set_norm(LogNorm())
        else:
            self._axim[0].set_norm(Normalize())

        self.canvas.draw()
#        self._fig.canvas.draw()
#        self.toolbar.update()
#        self._fig.canvas.draw_idle()

    def update_lines(self):

        if len(self._cm) == 0: return
        
        print 'update lines ', self.projx_range, self.projy_range
        
        cm = self._cm

        if self._projx_lines[0]:            
            self._ax.lines.remove(self._projx_lines[0])

        if self._projx_lines[1]:
            self._ax.lines.remove(self._projx_lines[1])

        if self._projy_lines[0]:            
            self._ax.lines.remove(self._projy_lines[0])

        if self._projy_lines[1]:
            self._ax.lines.remove(self._projy_lines[1])
            
        ixlo = max(0,self.projx_range[0])
        ixhi = max(0,self.projx_range[1])

        iylo = max(0,self.projy_range[0])
        iyhi = max(0,self.projy_range[1])


        self._projx_lines[0] = self._ax.axhline(ixlo,color='w')
        self._projx_lines[1] = self._ax.axhline(ixhi,color='w')

        self._projy_lines[0] = self._ax.axvline(iylo,color='w')
        self._projy_lines[1] = self._ax.axvline(iyhi,color='w')
            
        self.canvas.draw()
        
class PlotPanel(BasePanel):

    def __init__(self, parent,fignum,pindex,title,xlabel,ylabel):
        BasePanel.__init__(self, parent, fignum)
#        wx.Panel.__init__(self, parent, -1)

        self._fignum = fignum
        self._pindex = pindex
        self._fig = plt.figure(fignum,figsize=(5,4), dpi=75)
        self.canvas = FigureCanvasWxAgg(self, -1, self._fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        #Toolbar(self.canvas) #matplotlib toolbar
        self.toolbar.Realize()

        #self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        self._ax = None
        self._im = []
        self._style = []
        self._lines = []
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._proj_range = None
        self._norm = False
#        self.toolbar.update() # Not sure why this is needed - ADS
        
    def clear(self):        
        self._im = []
        self._style = []
        self._lines = []

    def add(self,im,style):

        style.setdefault('linestyle','None')

        if self._proj_range is None:
            self._proj_range = [0,im.axis(self._pindex).nbins]
        
        self._im.append(im)
        self._style.append(style)

    def scf(self):
        plt.figure(self._fignum)

    def set_slice(self,value):
        self.slice = int(value)

    def set_nbin(self,value):
        self.nbin = int(value)

    def set_rebin(self,value):
        self.rebin = int(value)

    def set_proj_range(self,lo,hi):
        self._proj_range = [lo,hi]

    def set_norm(self,v):
        self._norm = v
        
    def update(self):

        self.scf()

        bin_range = [self.slice,self.slice+self.nbin]

        print 'proj_range ', self._proj_range

        proj_bin_range = []
        proj_bin_range += [max(0,np.round(self._proj_range[0]))]
        proj_bin_range += [np.round(self._proj_range[1])]

        print 'proj_bin_range ', proj_bin_range
        
#        proj_bin_range = [self._im[0].axis(self._pindex).binToVal(self._proj_range[0])[0],
#                          self._im[0].axis(self._pindex).binToVal(self._proj_range[1])[0]]
        
        pcm = []

        for im in self._im:
            if isinstance(im,SkyCube):
                cm = im.marginalize(2,bin_range=bin_range)
            else:
                cm = im

            h = cm.project(self._pindex,
                           bin_range=proj_bin_range,
                           offset_coord=True).rebin(self.rebin)

            if self._norm: h = h.normalize()
            
            pcm.append(h)

        if len(self._lines) == 0:

            for i, p in enumerate(pcm): self._lines.append(p.plot(**self._style[i]))
            self._ax = plt.gca()
            self._ax.grid(True)
            self._ax.set_title(self._title)
            return

        for i, p in enumerate(pcm):
            p.update_artists(self._lines[i])

        self._ax.relim()
        self._ax.autoscale(axis='y')

        self.canvas.draw()
#        self._fig.canvas.draw()
#        self.toolbar.update()
#        self._fig.canvas.draw_idle()

class FourierDemoWindow(wx.Window, Knob):
    def __init__(self, *args, **kwargs):
        wx.Window.__init__(self, *args, **kwargs)
        self.lines = []
        self.figure = plt.Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.canvas.callbacks.connect('button_press_event', self.mouseDown)
        self.canvas.callbacks.connect('motion_notify_event', self.mouseMotion)
        self.canvas.callbacks.connect('button_release_event', self.mouseUp)
        self.state = ''
        self.mouseInfo = (None, None, None, None)
        self.f0 = Param(2., minimum=0., maximum=6.)
        self.A = Param(1., minimum=0.01, maximum=2.)


        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)

        self.draw()
        
        # Not sure I like having two params attached to the same Knob,
        # but that is what we have here... it works but feels kludgy -
        # although maybe it's not too bad since the knob changes both params
        # at the same time (both f0 and A are affected during a drag)
        self.f0.attach(self)
        self.A.attach(self)
        self.Bind(wx.EVT_SIZE, self.sizeHandler)

        self.add_toolbar()
       
    def sizeHandler(self, *args, **kwargs):
        self.canvas.SetSize(self.GetSize())
        
    def mouseDown(self, evt):
        if self.lines[0] in self.figure.hitlist(evt):
            self.state = 'frequency'
        elif self.lines[1] in self.figure.hitlist(evt):
            self.state = 'time'
        else:
            self.state = ''
        self.mouseInfo = (evt.xdata, evt.ydata, max(self.f0.value, .1), self.A.value)

    def mouseMotion(self, evt):
        if self.state == '':
            return
        x, y = evt.xdata, evt.ydata
        if x is None:  # outside the axes
            return
        x0, y0, f0Init, AInit = self.mouseInfo
        self.A.set(AInit+(AInit*(y-y0)/y0), self)
        if self.state == 'frequency':
            self.f0.set(f0Init+(f0Init*(x-x0)/x0))
        elif self.state == 'time':
            if (x-x0)/x0 != -1.:
                self.f0.set(1./(1./f0Init+(1./f0Init*(x-x0)/x0)))
                    
    def mouseUp(self, evt):
        self.state = ''

    def draw(self):
        if not hasattr(self, 'subplot1'):
            self.subplot1 = self.figure.add_subplot(211)
            self.subplot2 = self.figure.add_subplot(212)
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        color = (1., 0., 0.)
        self.lines += self.subplot1.plot(x1, y1, color=color, linewidth=2)
        self.lines += self.subplot2.plot(x2, y2, color=color, linewidth=2)
        #Set some plot attributes
        self.subplot1.set_title("Click and drag waveforms to change frequency and amplitude", fontsize=12)
        self.subplot1.set_ylabel("Frequency Domain Waveform X(f)", fontsize = 8)
        self.subplot1.set_xlabel("frequency f", fontsize = 8)
        self.subplot2.set_ylabel("Time Domain Waveform x(t)", fontsize = 8)
        self.subplot2.set_xlabel("time t", fontsize = 8)
        self.subplot1.set_xlim([-6, 6])
        self.subplot1.set_ylim([0, 1])
        self.subplot2.set_xlim([-2, 2])
        self.subplot2.set_ylim([-2, 2])
        self.subplot1.text(0.05, .95, r'$X(f) = \mathcal{F}\{x(t)\}$', \
            verticalalignment='top', transform = self.subplot1.transAxes)
        self.subplot2.text(0.05, .95, r'$x(t) = a \cdot \cos(2\pi f_0 t) e^{-\pi t^2}$', \
            verticalalignment='top', transform = self.subplot2.transAxes)

    def compute(self, f0, A):
        f = np.arange(-6., 6., 0.02)
        t = np.arange(-2., 2., 0.01)
        x = A*np.cos(2*np.pi*f0*t)*np.exp(-np.pi*t**2)
        X = A/2*(np.exp(-np.pi*(f-f0)**2) + np.exp(-np.pi*(f+f0)**2))
        return f, X, t, x

    def repaint(self):
        self.canvas.draw()

    def setKnob(self, value):
        # Note, we ignore value arg here and just go by state of the params
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        plt.setp(self.lines[0], xdata=x1, ydata=y1)
        plt.setp(self.lines[1], xdata=x2, ydata=y2)
        self.repaint()

    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()

        print 'Adding toolbar'
        
        if wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()

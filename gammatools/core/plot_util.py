import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from scipy.interpolate import UnivariateSpline
import itertools
import copy
import numpy as np
from histogram import *
from series import *
from util import update_dict
from config import Configurable

__author__   = "Matthew Wood (mdwood@slac.stanford.edu)"
__abstract__ = ""

def set_font_size(ax,size):
    """Update the font size for all elements in a matplotlib axis
    object."""

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def get_cycle_element(cycle,index):

    return cycle[index%len(cycle)]

    
class FigureSubplot(object):
    """Class implementing a single pane in a matplotlib figure. """

    style = { 'xlabel' : None,
              'ylabel' : None,
              'zlabel' : None,
              'fontsize' : None,
              'xlim'   : None,
              'ylim'   : None,
              'ylim_ratio' : None,
              'show_args' : None,
              'mask_args' : None,
              'title'  : None,
              'yscale' : 'lin',
              'xscale' : 'lin',
              'logz'   : False,
              'marker' : None,
              'color' : None,
              'linestyle' : None,
              'linewidth' : None,
              'markersize' : None,
              'hist_style' : None,
              'hist_xerr' : True,
              'nolegend'  : False,
              'grid'      : True,
              'legend' : {'loc' : 'best',
                          'ncol' : 1,
                          'frameon' : True,
                          'size' : 10},
              'norm_style' : 'ratio',
              'norm_index' : None,
              'norm_interpolation' : 'log' }

    def __init__(self,ax,**kwargs):
        
        self._style = copy.deepcopy(FigureSubplot.style)
        update_dict(self._style,kwargs)

        self._ax = ax
        self._cb = None
        self._data = []        

        self._style_counter = {
            'color' : 0,
            'marker' : 0,
            'markersize' : 0,
            'linestyle' : 0,
            'linewidth' : 0,
            'hist_style' : 0,
            'hist_xerr' : 0
            }
        
        self._hline = []
        self._hline_style = []
        self._text = []
        self._text_style = []

    def ax(self):
        return self._ax

    def set_style(self,k,v):
        self._style[k] = v

    def set_title(self,title):
        self.set_style('title',title)
        
    def get_style(self,h,**kwargs):
        """Generate style dictionary for a subplot element."""
        
        style = copy.deepcopy(h.style())       
        style.update(kwargs)
        
        for k in self._style_counter.keys():  

            if not k in style: continue
            if not style[k] is None: continue

            if isinstance(self._style[k],list):
                style[k] = get_cycle_element(self._style[k],
                                             self._style_counter[k])
                self._style_counter[k] += 1
            else:
                style[k] = self._style[k]
                
        return copy.deepcopy(style)
        
    def add_text(self,x,y,s,**kwargs):

        style = { 'color' : 'k', 'fontsize' : 10 }
        update_dict(style,kwargs,False)

        self._text.append([x,y,s])
        self._text_style.append(style)

    def add_data(self,x,y,yerr=None,**kwargs):

        s = Series(x,y,yerr)        
        style = self.get_style(s,**kwargs)
        s.update_style(style)

        self._data.append(s)

    def add_series(self,s,**kwargs):

        s = copy.deepcopy(s)
        style = self.get_style(s,**kwargs)
        s.update_style(style)
        self._data.append(s)

    def add_hist(self,h,**kwargs):
        
        h = copy.deepcopy(h)
        style = self.get_style(h,**kwargs)
        h.update_style(style)
        self._data.append(h)

    def add(self,h,**kwargs):
        
        h = copy.deepcopy(h)
        style = self.get_style(h,**kwargs)
        h.update_style(style)
        self._data.append(h)

    def add_hline(self,x,**kwargs):

        style = { 'color' : None, 'linestyle' : None, 'label' : None }

        update_dict(style,kwargs,False)

        self._hline.append(x)
        self._hline_style.append(style)

    def merge(self,sp):

        for d in sp._data: self._data.append(d)
 
    def cumulative(self,**kwargs):

        for i, d in enumerate(self._data):            
            self._data[i] = self._data[i].normalize()
            self._data[i] = self._data[i].cumulative()

    def normalize(self,residual=False,**kwargs):

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)
        
        norm_index = 0
        if not style['norm_index'] is None:
            norm_index = style['norm_index']
            
        if isinstance(self._data[norm_index],Histogram):
            x = copy.deepcopy(self._data[norm_index].axis(0).center)
            y = copy.deepcopy(self._data[norm_index].counts)
        elif isinstance(self._data[norm_index],Series):
            x = copy.deepcopy(self._data[norm_index].x())
            y = copy.deepcopy(self._data[norm_index].y())


        if style['norm_interpolation'] == 'log':
            fn = UnivariateSpline(x,np.log10(y),k=1,s=0)
        else:
            fn = UnivariateSpline(x,y,k=1,s=0)
            
#        msk = y>0
        for i, d in enumerate(self._data):

            if isinstance(d,Series):
                msk = (d.x() >= x[0]*0.95) & (d.x() <= x[-1]*1.05)
                if style['norm_interpolation'] == 'log':
                    ynorm = 10**fn(d.x())
                else: ynorm = fn(d.x())
                self._data[i]._msk &= msk
                self._data[i] /= ynorm

                if style['norm_style'] == 'residual':
                    self._data[i] -= 1.0
                                    
            elif isinstance(d,Histogram):                    
                if style['norm_interpolation'] == 'log':
                    ynorm = 10**fn(d.axis().center)
                else: ynorm = fn(d.axis().center)
                self._data[i] /= ynorm
                if style['norm_style'] == 'residual':
                    self._data[i]._counts -= 1.0

    def plot(self,**kwargs):
        
        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)
        
        ax = self._ax

        yscale = style['yscale']
        if 'yscale' in kwargs: yscale = kwargs.pop('yscale')

        xscale = style['xscale']
        if 'xscale' in kwargs: xscale = kwargs.pop('xscale')

        logz = style['logz']
        if 'logz' in kwargs: logz = kwargs.pop('logz')
        
        if not style['title'] is None:
            ax.set_title(style['title'])

        labels = []

        iargs = range(len(self._data))
        if not style['show_args'] is None:
            iargs = style['show_args']
         
        if not style['mask_args'] is None:
            iargs = [x for x in iargs if x not in style['mask_args']]

        for i in iargs:
            s = self._data[i]
            labels.append(s.label())
            p = s.plot(ax=ax,logz=logz)

            if isinstance(p,QuadMesh):
                self._cb = plt.colorbar(p,ax=ax)
                if not style['zlabel'] is None:
                    self._cb.set_label(style['zlabel'])
            
        for i, h in enumerate(self._hline):
            ax.axhline(self._hline[i],**self._hline_style[i])

        for i, t in enumerate(self._text):
            ax.text(*t,transform=ax.transAxes, **self._text_style[i])

        ax.grid(style['grid'])
        if len(labels) > 0 and not style['nolegend']:
            
            legkwargs = copy.deepcopy(style['legend'])
            legkwargs['prop'] = {'size' : legkwargs.pop('size',12)}

            ax.legend(numpoints=1,**legkwargs)

        if not style['ylabel'] is None:
            ax.set_ylabel(style['ylabel'])
        if not style['xlabel'] is None:
            ax.set_xlabel(style['xlabel'])

        if not style['xlim'] is None: ax.set_xlim(style['xlim'])
        if not style['ylim'] is None: ax.set_ylim(style['ylim'])
        
#            if ratio: ax.set_ylim(0.0,2.0)            
#        if not ylim is None: ax.set_ylim(ylim)

        if yscale == 'log': ax.set_yscale('log')
        elif yscale == 'sqrt': ax.set_yscale('sqrt',exp=2.0)

        if xscale == 'log': ax.set_xscale('log')
        elif xscale == 'sqrt': ax.set_xscale('sqrt',exp=2.0)

        if style['fontsize'] is not None:
            set_font_size(ax,style['fontsize'])
        
        
class RatioSubplot(FigureSubplot):

    def __init__(self,ax,src_subplot=None,**kwargs):
        super(RatioSubplot,self).__init__(ax,**kwargs)
        self._src_subplot = src_subplot

    def plot(self,**kwargs):
        
        if not self._src_subplot is None:
            self._data = copy.deepcopy(self._src_subplot._data)
            self.normalize()            
            super(RatioSubplot,self).plot(**kwargs)
        else:
            data = copy.deepcopy(self._data)
            self.normalize()
            super(RatioSubplot,self).plot(**kwargs)
            self._data = data
        

class Figure(object):

    fignum = 100
    
    style = { 'show_ratio_args' : None,
              'mask_ratio_args' : None,
              'figstyle' : None,
              'fontsize' : None,
              'format' : 'png',
              'fig_dir' : './',
              'figscale' : 1.0,
              'subplot_margins' : {'left' : 0.12, 'bottom' : 0.12,
                                   'right' : 0.9, 'top': 0.9 },
              'figsize' : [8.0,6.0],
              'panes_per_fig' : 1 }

    def __init__(self,figlabel,nsubplot=0,**kwargs):
        
        self._style = copy.deepcopy(Figure.style)
        
        update_dict(self._style,FigureSubplot.style,True)
        update_dict(self._style,kwargs)

        figsize = self._style['figsize']
        
        figsize[0] *= self._style['figscale']
        figsize[1] *= self._style['figscale']
        
        self._fig = plt.figure('Figure %i'%Figure.fignum,figsize=figsize)
        Figure.fignum += 1
        
        self._figlabel = figlabel
        self._subplots = []        
        self.add_subplot(nsubplot)        

    def __getitem__(self,key):

        return self._subplots[key]

    def add_subplot(self,n=1,**kwargs):

        if n == 0: return
        
        if isinstance(n,tuple): nx, ny = n        
        elif n == 1: nx, ny = 1,1
        elif n == 2: nx, ny = 2,1
        elif n > 2 and n <= 4: nx, ny = 2,2
        
        for i in range(nx*ny):        
            style = copy.deepcopy(self._style)
            update_dict(style,kwargs)

            ax = self._fig.add_subplot(ny,nx,i+1)

            if self._style['figstyle'] == 'ratio':
                self._subplots.append(RatioSubplot(ax,**style))
            else:
                self._subplots.append(FigureSubplot(ax,**style))

    def normalize(self,**kwargs):

        for s in self._subplots: s.normalize(**kwargs)

    def merge(self,fig):

        for i, s in enumerate(self._subplots): 
            s.merge(fig._subplots[i])
            
    def _plot_twopane_shared_axis(self,sp0,sp1,height_ratio=1.6,**kwargs):
        """Generate a figure with two panes the share a common x-axis.
        Tick labels will be suppressed in the x-axis of the upper pane."""
        fig = plt.figure()

        gs1 = gridspec.GridSpec(2, 1, height_ratios = [height_ratio,1])
        ax0 = fig.add_subplot(gs1[0,0])
        ax1 = fig.add_subplot(gs1[1,0],sharex=ax0)

        fig.subplots_adjust(hspace=0.1)
        plt.setp([ax0.get_xticklabels()],visible=False)

        sp0.plot(ax0,**kwargs)
        sp1.plot(ax1,**kwargs)

        fig.canvas.draw()
        plt.subplots_adjust(left=0.12, bottom=0.12,right=0.95, top=0.95)

        return fig

    def plot(self,**kwargs):

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        fig_name = '%s'%(self._figlabel)
        fig_name = os.path.join(style['fig_dir'],fig_name)
        
        for p in self._subplots: p.plot(**kwargs)

        if not style['fontsize'] is None:
            for p in self._subplots:
                set_font_size(p.ax(),style['fontsize'])
        self._fig.subplots_adjust(**style['subplot_margins'])

        if isinstance(style['format'],list):
            formats = style['format']
        else:
            formats = [style['format']]

        for fmt in formats:
            self._fig.savefig(fig_name + '.%s'%fmt)
        
class TwoPaneFigure(Figure):

    def __init__(self,figlabel,**kwargs):

        super(TwoPaneFigure,self).__init__(figlabel,**kwargs)

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        height_ratio=1.6

        gs1 = gridspec.GridSpec(2, 1, height_ratios = [height_ratio,1])
        ax0 = self._fig.add_subplot(gs1[0,0])
        ax1 = self._fig.add_subplot(gs1[1,0],sharex=ax0)

        style0 = copy.deepcopy(style)
        style1 = copy.deepcopy(style)

        style0['xlabel'] = None
        style1['nolegend'] = True
        style1['title'] = None

        fp0 = FigureSubplot(ax0,**style0)
        fp1 = FigureSubplot(ax1,**style1)

        self._fig.subplots_adjust(hspace=0.1)
        plt.setp([ax0.get_xticklabels()],visible=False)

        self._subplots.append(fp0)
        self._subplots.append(fp1)
            
class TwoPaneRatioFigure(Figure):

    def __init__(self,figlabel,**kwargs):

        super(TwoPaneRatioFigure,self).__init__(figlabel,**kwargs)

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        height_ratio=1.6

        gs1 = gridspec.GridSpec(2, 1, height_ratios = [height_ratio,1])
        ax0 = self._fig.add_subplot(gs1[0,0])
        ax1 = self._fig.add_subplot(gs1[1,0],sharex=ax0)

        style0 = copy.deepcopy(style)
        style1 = copy.deepcopy(style)

        style0['xlabel'] = None

        if style['norm_style'] == 'ratio':        
            style1['ylabel'] = 'Ratio'
        elif style['norm_style'] == 'residual':        
            style1['ylabel'] = 'Fractional Residual'
            
        style1['yscale'] = 'lin'
        style1['ylim'] = style['ylim_ratio']
        style1['nolegend'] = True
        style1['title'] = ''

#        ratio_subp.set_style('show_args',style['show_ratio_args'])
#        ratio_subp.set_style('mask_args',style['mask_ratio_args'])

        fp0 = FigureSubplot(ax0,**style0)
        fp1 = RatioSubplot(ax1,fp0,**style1)

        self._fig.subplots_adjust(hspace=0.1)
        plt.setp([ax0.get_xticklabels()],visible=False)

        self._subplots.append(fp0)
        self._subplots.append(fp1)


class FigTool(Configurable):

    default_config = {
        'format' :( 'png', 'Set the output image format.' ),
        'marker' : (['s','o','d','^','v','<','>'],'Set the marker style sequence.'),
        'color'  : ['b','g','r','m','c','grey','brown'],
        'linestyle' : ['-','--','-.','-','--','-.','-'],
        'linewidth' : [1.0],
        'markersize' : [6.0],
        'figsize' : [8.0,6.0],
        'norm_index'  : None,
        'legend_kwargs' : {'loc' : 'best'}, 
        'fig_dir' :
            ( './', 'Set the output directory.' ),
        'fig_prefix'   :
            ( None, 'Set the common prefix for image files.') }

    
    def __init__(self,config=None,opts=None,**kwargs):
        super(FigTool,self).__init__()
        self.configure(config,opts=opts,**kwargs)
         
    def create(self,figlabel,figstyle=None,nax=1,**kwargs):

        if self.config['fig_prefix']:
            figlabel = self.config['fig_prefix'] + '_' + figlabel

        style = copy.deepcopy(self.config)
        style.update(kwargs)
        
        if figstyle == 'twopane':
            return TwoPaneFigure(figlabel,**style)
        elif figstyle == 'ratio2':
            return TwoPaneRatioFigure(figlabel,**style)
        elif figstyle == 'residual2':
            return TwoPaneRatioFigure(figlabel,
                                      norm_style='residual',**style)
        elif figstyle == 'ratio':
            return Figure(figlabel,nax,figstyle='ratio',
                          **style)
        else:
            return Figure(figlabel,nax,**style)

if __name__ == '__main__':

    from optparse import Option
    from optparse import OptionParser

    usage = "usage: %prog [options] <h5file>"
    description = """A description."""

    parser = OptionParser(usage=usage,description=description)
    FigTool.configure(parser)

    (opts, args) = parser.parse_args()
    
    x0 = np.linspace(0,2*np.pi,100.)
    y0 = 2. + np.sin(x0)
    y1 = 2. + 0.5*np.sin(x0+np.pi/4.)


    ft = FigTool(opts)

    fig = ft.create(1,'theta_cut',
                    xlabel='X Label [X Unit]',
                    ylabel='Y Label [Y Unit]',
                    markers=['d','x','+'])

    fig[0].add_data(x0,y0,label='label1')
    fig[0].add_data(x0,y1,label='label2')

    fig.plot(style='ratio2')

    fig1 = ft.create(1,'theta_cut',
                    xlabel='Energy [log${10}$(E/GeV)]',
                    ylabel='Cut Value [deg]',colors=['r','grey','maroon'])

    h0 = Histogram([-3,3],100)
    h1 = Histogram([-3,3],100)
    h2 = Histogram([-4,4],100)

    h0.fill(np.random.normal(0,1.0,size=10000))
    h1.fill(np.random.normal(0,0.5,size=10000))
    h2.fill(np.random.normal(0,3.0,size=10000))

    fig1[0].add_hist(h0,label='label1',hist_style='filled')
    fig1[0].add_hist(h1,label='label2',hist_style='errorbar')
    fig1[0].add_hist(h2,label='label3',hist_style='step',linestyle='-')

    fig1.plot(xlim=[-5,5],legend_loc='upper right')

    fig2 = ft.create(1,'theta_cut',
                    xlabel='Energy [log${10}$(E/GeV)]',
                    ylabel='Cut Value [deg]',colors=['r','grey','maroon'])

    h3 = Histogram2D([-3,3],[-3,3],100,100)

    x = np.random.normal(0,0.5,size=100000)
    y = np.random.normal(0,1.0,size=100000)

    h3.fill(x,y)

    fig2[0].add_hist(h3,logz=True)

    fig2.plot()

    plt.show()

import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import itertools
import copy
import numpy as np
from histogram import *
from series import *
from util import update_dict

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
              'markersize' : None,
              'hist_style' : None,
              'hist_xerr' : True,
              'legend_loc' : 'upper right',
              'legend_fontsize' : 10,
              'legend'   : True,
              'norm_index' : None }

    def __init__(self,**kwargs):
        
        self._style = copy.deepcopy(FigureSubplot.style)
        update_dict(self._style,kwargs)

        self._ax = None
        self._data = []
        

        self._style_counter = {
            'color' : 0,
            'marker' : 0,
            'markersize' : 0,
            'linestyle' : 0,
            'hist_style' : 0,
            'hist_xerr' : 0
            }
        
        self._hline = []
        self._hline_style = []
        self._text = []
        self._text_style = []

    def set_style(self,k,v):
        self._style[k] = v

    def set_title(self,title):
        self.set_style('title',title)
        
    def get_style(self,**kwargs):
            
        style = { 'color'     : None,
                  'marker'    : None,
                  'markersize'    : None,
                  'linestyle' : None,
                  'hist_style' : None,
                  'hist_xerr' : None,
                  'msk'       : None }
        
        style.update(kwargs)
        
        for k in self._style_counter.keys():        
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
        
        style = self.get_style(**kwargs)
        s = Series(x,y,yerr,style)
        s.update_style(style)
        self._data.append(s)

    def add_series(self,s,**kwargs):
        s = copy.deepcopy(s)
        style = self.get_style(**kwargs)
        s.update_style(style)
        self._data.append(s)

    def add_hist(self,h,**kwargs):
        
        h = copy.deepcopy(h)
        style = self.get_style(**kwargs)
        h.update_style(style)
        self._data.append(h)

    def add_hline(self,x,**kwargs):

        style = { 'color' : None, 'linestyle' : None, 'label' : None }

        update_dict(style,kwargs,False)

        self._hline.append(x)
        self._hline_style.append(style)

    def merge(self,sp):

        for d in sp._data: self._data.append(d)
        
    def create_ratio_subplot(self,residual=False,**kwargs):

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)
        
        subp = copy.deepcopy(self)
        subp.set_style('yscale','lin')
        subp.set_style('ylim',style['ylim_ratio'])

        if residual: subp.set_style('ylabel','Fractional Residual')
        else: subp.set_style('ylabel','Ratio')

        subp.normalize(residual,**kwargs)

        return subp

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
            x = copy.deepcopy(self._data[norm_index].center())
            y = copy.deepcopy(self._data[norm_index].counts())
        elif isinstance(self._data[norm_index],Series):
            x = copy.deepcopy(self._data[norm_index].x())
            y = copy.deepcopy(self._data[norm_index].y())

        fn = UnivariateSpline(x,np.log10(y),k=1,s=0)
        
#        msk = y>0
        for i, d in enumerate(self._data):

            if isinstance(d,Series):
                msk = (d.x() >= x[0]*0.95) & (d.x() <= x[-1]*1.05)
                ynorm = 10**fn(d.x())
                self._data[i]._msk &= msk
                self._data[i] /= ynorm

                if residual:
                    self._data[i] -= 1.0
                                    
            elif isinstance(d,Histogram):                    
                ynorm = 10**fn(d.center())            
                self._data[i]._counts /= ynorm
                self._data[i]._var /= ynorm**2
                
                if residual:
                    self._data[i]._counts -= 1.0

    def plot(self,ax,**kwargs):
        
        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        self._ax = ax

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
            s.plot(ax=ax,logz=logz)

        for i, h in enumerate(self._hline):
            ax.axhline(self._hline[i],**self._hline_style[i])

        for i, t in enumerate(self._text):
            ax.text(*t,transform=ax.transAxes, **self._text_style[i])

        ax.grid(True)
        if len(labels) > 0 and style['legend']:
            ax.legend(prop={'size' : style['legend_fontsize']},
                      loc=style['legend_loc'],ncol=1)

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
        
        
class RatioSubplot(FigureSubplot):

    def __init__(self):

        pass

class Figure(object):
    
    style = { 'show_ratio_args' : None,
              'mask_ratio_args' : None,
              'style' : 'normal',
              'figure_style' : 'onepane',
              'format' : 'png',
              'fig_dir' : './',
              'figscale' : 1.0,
              'subplots_per_fig' : 1 }

    def __init__(self,figlabel,nx=1,ny=1,**kwargs):
        
        self._style = copy.deepcopy(Figure.style) 
        update_dict(self._style,FigureSubplot.style,True)
        update_dict(self._style,kwargs)

        nsub = nx*ny

        self._figlabel = figlabel
        self._subplots = []        
        self.add_subplot(nsub)        

    def __getitem__(self,key):

        return self._subplots[key]

    def add_subplot(self,n=1,**kwargs):

        for i in range(n):        
            style = copy.deepcopy(self._style)

            update_dict(style,kwargs)

#            for k, v in style.iteritems():
#                if k in kwargs: style[k] = kwargs[k]
        
            self._subplots.append(FigureSubplot(**style))

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

    def _plot_twopane(self,**kwargs):
        
#        nfig = int(len(self._subplots))
#        if nfig > 1:
#            fig_name = '%s_%02i.%s'%(self._figlabel,
#                                     i,self._style['format'])
#        else:
        fig_name = '%s.%s'%(self._figlabel,self._style['format'])
            

#        subp_kwargs['legend'] = False
#        subp_kwargs['yscale'] = 'lin'
#        subp_kwargs['ylim'] = kwargs['ylim_ratio']
#        subp_kwargs['show_args'] = kwargs['show_ratio_args']        
#        kwargs['xlabel'] = None
        
        fig = self._plot_twopane_shared_axis(self._subplots[0],
                                             self._subplots[1],
                                             **kwargs)

        fig.savefig(fig_name)

#                                wspace=None, hspace=None)
            
#            ax1.set_yticklabels([ lbl.get_text()
#                                  for lbl in ax1.get_yticklabels()[:-1] ]+ 
#                                ['']) 
            
#                if not fontsize is None: set_font_size(ax,fontsize)
            #,bbox_inches='tight')
            

    def plot(self,**kwargs):

        style = copy.deepcopy(self._style)
        style.update(kwargs)
        
        if style['style'] == 'ratio2' or style['style'] == 'residual2': 
            style['figure_style'] = 'twopane'
            

        residual = False
        if style['style'] == 'residual2' or style['style'] == 'residual':
            residual = True

        if style['style'] == 'ratio2' or style['style'] == 'residual2':
            ratio_subp = self._subplots[0].create_ratio_subplot(residual,
                                                                **kwargs)
            ratio_subp.set_style('legend',False)
            ratio_subp.set_style('yscale','lin')
            ratio_subp.set_style('show_args',style['show_ratio_args'])
            ratio_subp.set_style('mask_args',style['mask_ratio_args'])
            self._subplots[0].set_style('xlabel',None)
            self._subplots.append(ratio_subp)
            
        if style['figure_style'] == 'twopane':
            self._plot_twopane(**kwargs)
        else:
            self._plot(**kwargs)        
        
    def _plot(self,**kwargs):

        style = copy.deepcopy(self._style)
        style.update(kwargs)
        
        nsub = self._style['subplots_per_fig']        
        nfig = int(np.ceil(float(len(self._subplots))/float(nsub)))

        for i in range(nfig):

            if nfig > 1:
                fig_name = '%s_%02i.%s'%(self._figlabel,i,style['format'])
            else:
                fig_name = '%s.%s'%(self._figlabel,style['format'])

            fig_name = os.path.join(style['fig_dir'],fig_name)

            if nsub == 1:
                (nx,ny) = (1,1)
                figsize = (8,6)
                fontsize=None
            elif nsub == 2:
                (nx,ny) = (1,2)
                figsize = (8*1.5,6)
                fontsize=10
            elif nsub > 2 and nsub <= 4:
                (nx,ny) = (2,2)
                figsize = (8*1.4,6*1.4)
                fontsize=10
            else:
                (nx,ny) = (2,4)
                figsize = (8*1.4*2,6*1.4)
                fontsize=10

            fig = plt.figure(figsize=figsize)
            for j in range(nsub):
                isub = j+i*nsub
                if isub + 1 > len(self._subplots): continue
                
                ax = fig.add_subplot(nx,ny,j+1)
                self._subplots[isub].plot(ax,**kwargs)
#                if not fontsize is None: set_font_size(ax,fontsize)

            plt.subplots_adjust(left=0.12, bottom=0.12,
                                right=0.95, top=0.95)
                
            fig.savefig(fig_name)#,bbox_inches='tight')

class FigTool(object):

    opts = {
        'format' :
            ( 'png', str,
              'Set the output image format.' ),
        'fig_dir' :
            ( './', str,
              'Set the output directory.' ),
        'fig_prefix'   :
            ( None,  str,
              'Set the common prefix for output image files.') }

    
    def __init__(self,opts=None,**kwargs):

        style = { 'marker' : ['s','o','d','^','v','<','>'],
                  'color' : ['b','g','r','m','c','grey','brown'],
                  'linestyle' : ['-','--','-.','-','--','-.','-'],
                  'markersize' : [6.0],
                  'hist_style' : 'errorbar',
                  'norm_index'  : None,
                  'legend_loc' : 'lower right',
                  'format' : 'png',
                  'fig_prefix' : None,
                  'fig_dir' : './' }

        if not opts is None:
            update_dict(style,opts.__dict__)        
        update_dict(style,kwargs)
        
        self._fig_dir = style['fig_dir']
        self._fig_prefix = style['fig_prefix']
        self._style = style

    @staticmethod
    def configure(parser):
        
        for k, v in FigTool.opts.iteritems():

            if isinstance(v[0],bool):
                parser.add_argument('--' + k,default=v[0],
                                    action='store_true',
                                    help=v[2] + ' [default: %s]'%v[0])
            else:
                parser.add_argument('--' + k,default=v[0],type=v[1],
                                    help=v[2] + ' [default: %s]'%v[0])
        
        
    def create(self,nax,figlabel,**kwargs):

        if not self._fig_prefix is None:
            figlabel = self._fig_prefix + '_' + figlabel

        for k, v in self._style.iteritems():
            if not k in kwargs: kwargs[k] = v
            
        return Figure(figlabel,nax,**kwargs)

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

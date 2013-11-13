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

    def __init__(self,**kwargs):

        style = { 'xlabel' : None,
                  'ylabel' : None,
                  'xlim'   : None,
                  'ylim'   : None,
                  'ylim_ratio' : None,
                  'title'  : None,
                  'logy'   : False,
                  'logx'   : False,
                  'logz'   : False,
                  'markers' : None,
                  'colors' : None,
                  'linestyles' : None,
                  'legend_loc' : 'upper right',
                  'legend_fontsize' : 8,
                  'legend'   : True }

        update_dict(style,kwargs)
            
        self._style = style
        self._data = []
        self._hist = []
        self._hist2d = []
        self._color_index = 0
        self._linestyle_index = 0
        self._marker_index = 0

    def set_style(self,k,v):
        self._style[k] = v

    def set_title(self,title):
        self.set_style('title',title)
        
    def get_style(self,**kwargs):
            
        style = { 'color'     : None,
                  'marker'    : None,
                  'linestyle' : None  }
        
        style.update(kwargs)

        if style['color'] is None:
            style['color'] = get_cycle_element(self._style['colors'],
                                               self._color_index)
            self._color_index += 1

        if style['marker'] is None:
            style['marker'] = get_cycle_element(self._style['markers'],
                                                self._marker_index)
            self._marker_index += 1

        if style['linestyle'] is None:
            style['linestyle'] =  get_cycle_element(self._style['linestyles'],
                                                    self._linestyle_index)
            self._linestyle_index += 1

        return copy.deepcopy(style)
        
    def add_data(self,x,y,yerr=None,**kwargs):
        
        style = self.get_style(**kwargs)
        s = Series(x,y,yerr,style)
        s.update_style(style)
        self._data.append(s)

    def add_hist(self,h,**kwargs):
        
        h = copy.deepcopy(h)
        style = self.get_style(**kwargs)
        h.update_style(style)

        print style

        if isinstance(h,Histogram2D):  
            self._hist2d.append(h)
        else:
            self._hist.append(h)


    def normalize(self,**kwargs):

        norm_index = 0
        if 'norm_index' in kwargs: norm_index = kwargs['norm_index']

        if len(self._hist) > 0:
            x = copy.deepcopy(self._hist[norm_index].center())
            y = copy.deepcopy(self._hist[norm_index].counts())
        else:
            x = copy.deepcopy(self._data[norm_index].x())
            y = copy.deepcopy(self._data[norm_index].y())

        fn = UnivariateSpline(x,np.log10(y),k=1,s=0)
        
#        msk = y>0
        for i in range(len(self._data)):

            msk = (self._data[i].x() >= x[0]*0.95) & \
                (self._data[i].x() <= x[-1]*1.05)
            ynorm = 10**fn(self._data[i].x()[msk])

            self._data[i]._x = self._data[i].x()[msk]
            self._data[i]._y = self._data[i].y()[msk]/ynorm
            
            if not self._data[i]._yerr is None:
                self._data[i]._yerr = self._data[i].yerr()[msk]
                self._data[i]._yerr /= ynorm
            
        for i in range(len(self._hist)):

            ynorm = 10**fn(self._hist[i].center())
            
            self._hist[i]._counts /= ynorm
            self._hist[i]._var /= ynorm**2
            
    def plot(self,ax,**kwargs):
        
        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)
            
        logy = style['logy']
        if 'logy' in kwargs: logy = kwargs.pop('logy')

        logx = style['logx']
        if 'logx' in kwargs: logx = kwargs.pop('logx')

        logz = style['logz']
        if 'logz' in kwargs: logz = kwargs.pop('logz')
        
        if not style['title'] is None:
            ax.set_title(style['title'])

        labels = []
            
        for i, s in enumerate(self._data):
            labels.append(s.label())
            s.plot(ax=ax)

        for i, h in enumerate(self._hist):
            labels.append(h.label())            
            h.plot(ax=ax)

        for i in range(len(self._hist2d)):
            self._hist2d[i].plot(ax=ax,logz=logz)
            
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

        if logy: ax.set_yscale('log')
        if logx: ax.set_xscale('log')
        
        
class Figure(object):
    
    def __init__(self,figlabel,nsub=1,**kwargs):
        
        style = { 'xlabel' : None,
                  'ylabel' : None,
                  'zlabel' : None,
                  'xlim'   : None,
                  'ylim'   : None,
                  'ylim_ratio' : None,
                  'title'  : None,
                  'logy'   : False,
                  'markers' : None,
                  'colors' : None,
                  'linestyles' : None,
                  'style' : 'normal',
                  'legend_loc' : 'upper right',
                  'legend_fontsize' : 12,
                  'format' : 'png',
                  'fig_dir' : './',
                  'figscale' : 1.0,
                  'subplots_per_fig' : 1 }
 
        update_dict(style,kwargs)

        self._style = style
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

    def _plot_ratio_onepane(self,**kwargs):

        self.normalize(**kwargs)

        kwargs['logy'] = False
        kwargs['ylim'] = self._style['ylim_ratio']
        
#        if 'ylim' in kwargs: kwargs.pop('ylim')
        
        self.plot(**kwargs)
        
    def _plot_ratio_twopane(self,**kwargs):
        
        nfig = int(len(self._subplots))
        for i in range(nfig):

            if nfig > 1:
                fig_name = '%s_%02i.%s'%(self._figlabel,
                                         i,self._style['format'])
            else:
                fig_name = '%s.%s'%(self._figlabel,
                                    self._style['format'])
            
            fig = plt.figure()

            gs1 = gridspec.GridSpec(2, 1, height_ratios = [1.6,1])
            ax0 = fig.add_subplot(gs1[0,0])
            ax1 = fig.add_subplot(gs1[1,0],sharex=ax0)

            subp = copy.deepcopy(self._subplots[i])
            subp.normalize(**kwargs)
            
            fig.subplots_adjust(hspace=0.1)
            plt.setp([ax0.get_xticklabels()],visible=False)

            subp_kwargs = copy.deepcopy(kwargs)
            subp_kwargs['ylabel'] = 'Ratio'
            subp_kwargs['legend'] = False
            subp_kwargs['logy'] = False
            subp_kwargs['ylim'] = kwargs['ylim_ratio']        
            kwargs['xlabel'] = None
            
            self._subplots[i].plot(ax0,**kwargs)
            subp.plot(ax1,**subp_kwargs)

            fig.canvas.draw()
            plt.subplots_adjust(left=0.12, bottom=0.12,
                                right=0.95, top=0.95)
#                                wspace=None, hspace=None)
            
#            ax1.set_yticklabels([ lbl.get_text()
#                                  for lbl in ax1.get_yticklabels()[:-1] ]+ 
#                                ['']) 
            
#                if not fontsize is None: set_font_size(ax,fontsize)
            fig.savefig(fig_name)#,bbox_inches='tight')
            

    def plot(self,**kwargs):

        style = copy.deepcopy(self._style)
        style.update(kwargs)
        
        if style['style'] is None or style['style'] == 'normal':
            self._plot(**style)
        elif style['style'] == 'ratio': self._plot_ratio_onepane(**style)
        elif style['style'] == 'ratio2': self._plot_ratio_twopane(**style)
        
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
            ( 'png', 'string',
              'Set the output image format.' ),
        'fig_dir' :
            ( './', 'string',
              'Set the output directory.' ),
        'fig_prefix'   :
            ( None,  'string',
              'Set the common prefix for output image files.') }
    
    markers = ['s','o','d','^','v','<','>']
    colors = ['b','g','r','m','c','grey','brown']
    linestyles = ['-','--','-.','-','--','-.','-']
    
    def __init__(self,opts=None,**kwargs):

        style = { 'markers' : FigTool.markers,
                  'colors' : FigTool.colors,
                  'linestyles' : FigTool.linestyles,
                  'legend_loc' : 'lower right',
                  'format' : 'png',
                  'fig_prefix' : None,
                  'fig_dir' : './' }

        for k, v in style.iteritems():
            if k in kwargs: style[k] = kwargs[k]
            elif not opts is None and k in opts.__dict__:
                style[k] = opts.__dict__[k]
        
        self._fig_dir = style['fig_dir']
        self._fig_prefix = style['fig_prefix']
        self._style = style

    @staticmethod
    def configure(parser):
        
        for k, v in FigTool.opts.iteritems():

            if isinstance(v[0],bool):
                parser.add_option('--' + k,default=v[0],
                                  action='store_true',
                                  help=v[2] + ' [default: %s]'%v[0])
            else:
                parser.add_option('--' + k,default=v[0],type=v[1],
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

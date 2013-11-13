import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import itertools
import copy
import numpy as np
from histogram import *

__source__   = "$Source: /nfs/slac/g/glast/ground/cvs/users/kadrlica/eventSelect/python/pPlotUtils.py,v $"
__author__   = "Matthew Wood (mdwood@slac.stanford.edu)"
__abstract__ = ""
__date__     = "$Date: 2013/09/09 07:18:06 $"
__revision__ = "$Revision: 1.3 $, $Author: mdwood $"

def set_font_size(ax,size):

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def get_cycle_element(cycle,index):

    return cycle[index%len(cycle)]

    
class FigureSubplot(object):

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

        for k, v in style.iteritems():
            if k in kwargs: style[k] = kwargs[k]
            
        self._style = style
        self._data = []
        self._hist = []
        self._hist2d = []
        self._hist_styles = []
        self._hist2d_styles = []
        self._data_styles = []

        self._color_index = 0
        self._linestyle_index = 0
        self._marker_index = 0

    def set_style(self,k,v):
        self._style[k] = v

    def set_title(self,title):
        self.set_style('title',title)
        
    def get_style(self,**kwargs):

        style = { 'marker' : None,
                  'color' : None,
                  'linestyle' : None,
                  'linewidth' : 1,
                  'label' : None,
                  'max_frac_error' : None,
                  'msk' : None }

        for k, v in style.iteritems():
            if k in kwargs: style[k] = kwargs[k]
            
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

        if not yerr is None: yerr = np.array(yerr)
        
        style = self.get_style(**kwargs)
        self._data.append([np.array(x),np.array(y),yerr])
        self._data_styles.append(style)

    def add_hist(self,h,**kwargs):

        if isinstance(h,Histogram):        
            style = self.get_style(**kwargs)
            self._hist.append(h)
            self._hist_styles.append(style)
        elif isinstance(h,Histogram2D):  
            self._hist2d.append(h)
            self._hist2d_styles.append({})


    def normalize(self,**kwargs):

        norm_index = 0
        if 'norm_index' in kwargs: norm_index = kwargs['norm_index']

        if len(self._hist) > 0:
            x = copy.deepcopy(self._hist[norm_index].center())
            y = copy.deepcopy(self._hist[norm_index].counts())
        else:
            x = copy.deepcopy(self._data[norm_index][0])
            y = copy.deepcopy(self._data[norm_index][1])

        fn = UnivariateSpline(x,np.log10(y),k=1,s=0)
        
#        msk = y>0
        for i in range(len(self._data)):

            msk = (self._data[i][0] >= x[0]*0.95) & \
                (self._data[i][0] <= x[-1]*1.05)
            ynorm = 10**fn(self._data[i][0][msk])

            self._data[i][0] = self._data[i][0][msk]
            self._data[i][1] = self._data[i][1][msk]
            self._data[i][1] /= ynorm
            
            if not self._data[i][2] is None:
                self._data[i][2] = self._data[i][2][msk]
                self._data[i][2] /= ynorm

            
        for i in range(len(self._hist)):

            ynorm = 10**fn(self._hist[i].center())
            
            self._hist[i]._counts /= ynorm
            self._hist[i]._var /= ynorm**2
            
    def plot(self,ax,**kwargs):
        
        style = copy.deepcopy(self._style)
        for k, v in kwargs.iteritems():
            if k in style: style[k] = v
            
        logy = style['logy']
        if 'logy' in kwargs: logy = kwargs.pop('logy')

        logx = style['logx']
        if 'logx' in kwargs: logx = kwargs.pop('logx')

        logz = style['logz']
        if 'logz' in kwargs: logz = kwargs.pop('logz')
        
        if not style['title'] is None:
            ax.set_title(style['title'])

        labels = []
            
        for i in range(len(self._data)):

            if not self._data_styles[i]['label'] is None:
                labels.append(self._data_styles[i]['label'])

            self._data_styles[i].pop('max_frac_error')
            self._data_styles[i].pop('msk')
            ax.errorbar(self._data[i][0],self._data[i][1],self._data[i][2],
                        **self._data_styles[i])

        for i in range(len(self._hist)):

            if not self._hist_styles[i]['label'] is None:
                labels.append(self._hist_styles[i]['label'])
            
            self._hist[i].plot(ax=ax,**self._hist_styles[i])

        for i in range(len(self._hist2d)):
            self._hist2d[i].plot(ax=ax,logz=logz)
            
        ax.grid(True)
        if len(labels) > 0 and style['legend']:
            ax.legend(prop={'size' : style['legend_fontsize']},
                      loc=style['legend_loc'],ncol=1)       
#        ax.legend(prop={'size':8},loc=style['loc'],ncol=2)
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
        
        for k,v in style.iteritems():            
            if k in kwargs: style[k] = kwargs[k]

        self._style = style
        self._figlabel = figlabel
        self._subplots = []        
        self.add_subplot(nsub)        

    def __getitem__(self,key):

        return self._subplots[key]

    def add_subplot(self,n=1,**kwargs):

        for i in range(n):        
            style = copy.deepcopy(self._style)

            for k, v in style.iteritems():
                if k in kwargs: style[k] = kwargs[k]
        
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
                  'format' : format,
                  'fig_prefix' : None,
                  'fig_dir' : './' }

        for k, v in style.iteritems():
            if k in kwargs: style[k] = kwargs[k]
            elif k in opts.__dict__:
#                if isinstance(v[0],list)
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
                    xlabel='Energy [log${10}$(E/GeV)]',
                    ylabel='Cut Value [deg]')

    fig[0].add_data(x0,y0,label='label1')
    fig[0].add_data(x0,y1,label='label2')


    fig.plot(style='ratio2')

    plt.show()

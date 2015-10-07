#!/usr/bin/env python

"""
@file  histogram.py

@brief Python classes for creating, manipulating, and plotting
histograms."

@author Matthew Wood <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood <mdwood@slac.stanford.edu>"
__date__     = "$Date: 2013/10/20 23:59:49 $"
__revision__ = "$Revision: 1.30 $, $Author: mdwood $"

import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
import scipy.stats as stats
from gammatools.core.util import *
from gammatools.core.mpl_util import *
from matplotlib.colors import NoNorm, LogNorm, Normalize

def get_quantile(x,y,f):
    y = np.cumsum(y)    
    fn = UnivariateSpline(x,y,s=0,k=1)
    return brentq(lambda t: fn(t)-y[-1]*f,x[0],x[-1])

def get_quantile_error(x,y,yvar,f):
    y = np.cumsum(y)    
    yvar = np.cumsum(yvar)    

    if y[-1] == 0: return [np.inf,np.inf]
    
    yerr = np.sqrt(yvar)
    
    fn = UnivariateSpline(x,y,s=0,k=1)
    xq = brentq(lambda t: fn(t)-y[-1]*f,x[0],x[-1])

    i = np.where(y > y[-1]*f)[0][0]

    yhi = y[-1]*f+yerr[i]
    ylo = y[-1]*f-yerr[i]

    if yhi < y[-1]: xq_hi = brentq(lambda t: fn(t)-yhi,x[0],x[-1])
    else: xq_hi = x[-1]

    if ylo > y[0]: xq_lo = brentq(lambda t: fn(t)-ylo,x[0],x[-1])
    else: xq_lo = x[0]
    
    xq_err = 0.5*(xq_hi-xq_lo)
    xq_var = xq_err**2

#    print xq, xq_hi, xq_lo, xq_err

    return [xq,xq_var]

class HistogramND(object):
    """
    N-dimensional histogram class.  This class serves as the base
    class for lower dimensionality histograms (Histogram and
    Histogram2D) and provides a common functionality for filling the
    counts and error arrays as well as various transformation
    operations (slicing, projecting, marginalizing).  Bin occupation
    values and squared errors are stored in two numpy arrays.  The
    binning in each dimension is defined by an array of Axis objects
    which are provided to the constructor.  

    Histograms support most operations that can be applied to numpy
    arrays (addition, multiplication, division, etc.).  All arithmetic
    operations will recompute the bin errors using standard error
    propagation rules.

    Several transformation methods are provided which can be used to
    project, slice, and marginalize the histogram.  All transformation
    methods return a new histogram instance.  If the output histogram
    has a lower dimensionality it will be automatically instantiated
    as a 1D or 2D histogram when appropriate.

    Note that the recommended method for instantiating a histogram
    object is with the HistogramND.create() method.
    """

    def __init__(self, axes, counts=None, var=None, 
                 style=None, label = '__nolabel__'):

        self._axes = []
        self._style = {}
        if not style is None: update_dict(self._style,style)
        self._style['label'] = label

        for ax in axes:
            if isinstance(ax,Axis): self._axes.append(copy.deepcopy(ax))
            else: self._axes.append(Axis(ax))

        shape = []
        for ax in self._axes: shape.append(ax.nbins)

        if counts is None: self._counts = np.zeros(shape=shape)
        else: 
            self._counts = np.array(counts,copy=True)
            if len(self._counts.shape) == 0: 
                self._counts = np.ones(shape=shape)*self._counts

        if var is None: self._var = np.zeros(shape=self._counts.shape)
        else: 
            self._var = np.array(var,copy=True)
            if len(self._var.shape) == 0: 
                self._var = np.ones(shape=shape)*self._var

        self._ndim = self._counts.ndim

        if self._ndim != len(self._axes):
            raise Exception('Mismatch of data and axis dimensions: %i %i'
                            %(len(self._axes),self._ndim))
        
        self._dims = np.array(range(self._ndim),dtype=int)

    @property
    def shape(self):
        """Return the shape of this histogram (length along each dimension)."""
        return self._counts.shape

    @property
    def counts(self):
        """Return the counts array."""
        return self._counts
    
    @property
    def var(self):
        return self._var

    @property
    def err(self):
        return np.sqrt(self._var)

    def ndim(self):
        return len(self._axes)

    def style(self):
        return self._style
    
    def axes(self):
        """Return the axes vector."""
        return self._axes

    def axis(self,idim=0):
        return self._axes[idim]

    def center(self):
        """Return the center coordinate of each bin in this histogram
        as an NxM array."""
        
        if self._ndim == 1:
            return np.array(self._axes[0].center,ndmin=2)

        else:
            c = []
            for i in self._dims:
                c.append(self._axes[i].center)

            c = np.meshgrid(*c,indexing='ij')

            cv = []
            for i in range(len(c)): cv.append(np.ravel(c[i]))

            return np.array(cv)

 

    @staticmethod
    def createFromTH3(hist,label = '__nolabel__'):
        nx = hist.GetNbinsX()
        ny = hist.GetNbinsY()
        nz = hist.GetNbinsZ()

        xmin = hist.GetXaxis().GetBinLowEdge(1)
        xmax = hist.GetXaxis().GetBinLowEdge(nx+1)

        ymin = hist.GetYaxis().GetBinLowEdge(1)
        ymax = hist.GetYaxis().GetBinLowEdge(ny+1)

        zmin = hist.GetZaxis().GetBinLowEdge(1)
        zmax = hist.GetZaxis().GetBinLowEdge(nz+1)
        
        xaxis = Axis.create(xmin,xmax,nx)
        yaxis = Axis.create(ymin,ymax,ny)
        yaxis = Axis.create(zmin,zmax,nz)

        counts = np.zeros(shape=(nx,ny,nz))
        var = np.zeros(shape=(nx,ny,nz))

        for ix in range(1,nx+1):
            for iy in range(1,ny+1):
                for iz in range(1,nz+1):
                    counts[ix-1][iy-1][iz-1] = hist.GetBinContent(ix,iy,iz)
                    var[ix-1][iy-1][iz-1] = hist.GetBinError(ix,iy,iz)**2

        style = {}
        style['label'] = label
        style['title'] = hist.GetTitle()

        h = HistogramND([xaxis,yaxis,zaxis],
                        counts=counts,var=var,style=style)

        return h

    def to_dict(self):

        axes = []
        for ax in self.axes():
            axes.append(ax.to_dict())

        o = {'axes' : axes,
             'counts' : copy.copy(self.counts),
             'var' : copy.copy(self.var),
             }

        return o

    def to_root(self,name,title=None):

        if title is None: title=name
        
        import ROOT
        h = ROOT.TH3F(name,title,self.xaxis().nbins,
                      self.xaxis().lo_edge(),self.xaxis().hi_edge(),
                      self.yaxis().nbins,
                      self.yaxis().lo_edge(),self.yaxis().hi_edge(),
                      self.zaxis().nbins,
                      self.zaxis().lo_edge(),self.zaxis().hi_edge())

        for i in range(self.xaxis().nbins):
            for j in range(self.yaxis().nbins):
                for k in range(self.yaxis().nbins):
                    h.SetBinContent(i+1,j+1,k+1,self._counts[i,j,k])
                    h.SetBinError(i+1,j+1,k+1,np.sqrt(self._var[i,j,k]))

        return h

    def set_counts(self,counts):
        self._counts = counts

    def set_label(self,label):
        self._style['label'] = label

    def inverse(self):
        c = 1./self._counts
        var = c**2*self._var/self._counts**2
        return HistogramND.create(self._axes,c,var,self._style)

    def abs(self):
        c = np.abs(self._counts)
        return HistogramND.create(self._axes,c,self._var,self._style)
            
    def power10(self):

        err = np.sqrt(self._var)

        msk = (self._counts != np.inf)

        c = np.zeros(shape=self._counts.shape)
        clo = np.zeros(shape=self._counts.shape)
        chi = np.zeros(shape=self._counts.shape)
        
        c[msk] = np.power(10,self._counts[msk])
        clo[msk] = np.power(10,self._counts[msk]-err[msk])
        chi[msk] = np.power(10,self._counts[msk]+err[msk])

        counts = c 
        var = (0.5*(chi-clo))**2
        return HistogramND.create(self._axes,counts,var,self._style)

    def fill_from_fn(self,fn):
        
        v = fn(*self.center()).reshape(self.shape())
        self._counts = fn(*self.center()).reshape(self.shape())        
        
    def fill(self,z,w=1.0,v=None):
        """
        Fill the histogram from a set of points arranged in an NxM
        matrix where N is the histogram dimension and M is a variable
        number of points.

        @param z: Array of NxM points.
        @param w: Array of M bin weights (optional).
        @param v: Array of M bin variances (optional).
        @return:
        """

        z = np.array(z,ndmin=2)
        w = np.array(w,ndmin=1)
        if v is None: v = w
        else: v = np.array(v,ndmin=1)

        if z.shape[0] != self._ndim:
            print(z.shape, self._ndim)
            raise Exception('Coordinate dimension of input array must be '
                            'equal to histogram dimension.')

        if w.shape[0] < z.shape[1]: w = np.ones(z.shape[1])*w
        if v.shape[0] < z.shape[1]: v = np.ones(z.shape[1])*v

        edges = []
        for i in self._dims: edges.append(self._axes[i].edges)

        counts = np.histogramdd(z.T,bins=edges,weights=w)[0]
        var = np.histogramdd(z.T,bins=edges,weights=v)[0]

        self._counts += counts
        self._var += var

    def random(self,method='poisson',scale=1.0):
        """Generate a randomized histogram."""

        c = np.array(np.random.poisson(scale*self._counts),dtype='float')
        v = copy.copy(c)

        return HistogramND.create(self._axes,c,v,self._style)

    def project(self,pdims,bin_range=None):
        """Project the contents of this histogram into a histogram
        spanning the subspace defined by the list pdims.  The optional
        bin_range argument defines the range of bins over which to
        integrate each of the non-projected dimensions.  This
        operation is the complement to marginalize."""

        pdims = np.array(pdims,ndmin=1,copy=True)

        mdims = np.setdiff1d(self._dims,pdims)
        axes = []
        new_shape = []
        for i in pdims:
            axes.append(self._axes[i])
            new_shape.append(self._axes[i].nbins)
            
        if not bin_range is None:

            bin_range = np.array(bin_range,ndmin=2,copy=True)

            if len(bin_range) != len(mdims):
                raise Exception('Length of bin range list must be equal to '
                                'the number of marginalized dimensions.')

            slices = len(self._dims)*[None]

            for i, idim in enumerate(mdims):
                slices[idim] = slice(bin_range[i][0],bin_range[i][1])

            for idim in self._dims:
                if idim in mdims: continue 
                slices[idim] = slice(self._axes[idim].nbins)
            
            c = np.apply_over_axes(np.sum,self._counts[slices],
                                   mdims).reshape(new_shape)
            v = np.apply_over_axes(np.sum,self._var[slices],
                                   mdims).reshape(new_shape)
        else:
            c = np.apply_over_axes(np.sum,self._counts,mdims).reshape(new_shape)
            v = np.apply_over_axes(np.sum,self._var,mdims).reshape(new_shape)
            
        return HistogramND.create(axes,c,v,self._style)

    def quantile(self,dim,fraction=0.5,method='var',niter=100):

        sdims = np.setdiff1d(self._dims,[dim])
        
        axes = []
        for i in self._dims:

            if i == dim: continue
            else: axes.append(self._axes[i])

        h = HistogramND.create(axes,style=self._style)

#        h._counts = np.percentile(self._counts,fraction*100.,axis=dim)
                
        for index, x in np.ndenumerate(h._counts):

            hs = self.slice(sdims,index)
            
            if method == 'var':

                cs = np.concatenate(([0],hs.counts))
                vs = np.concatenate(([0],hs.var))
            
                q, qvar = get_quantile_error(hs.axis().edges,cs,vs,fraction)
#            h._counts[index] = HistQuantile(hs).eval(fraction)
                h._counts[index] = q
                h._var[index] = qvar

            elif method == 'bootstrap_poisson':

                c = np.random.poisson(np.concatenate(([0],hs.counts)),
                                      (niter,hs.axis().nbins+1))

                xq = []
                
                for i in range(niter):
                    xq.append(get_quantile(hs.axis().edges,c[i],fraction))

                h._counts[index] = \
                    get_quantile(hs.axis().edges,
                                 np.concatenate(([0],hs.counts)),fraction)
                h._var[index] = np.std(np.array(xq))**2
                    
        return h
        
    
    def normalize(self,dims=None):

        if dims is None: dims = self._dims
        else: dims = np.array(dims,ndmin=1,copy=True)
        
        norm =  np.apply_over_axes(np.sum,self._counts,dims)
        inorm = np.zeros(norm.shape)

        inorm[norm!=0] = 1./norm[norm!=0]
        
        c = self._counts*inorm
        v = self._var*inorm**2

        return HistogramND.create(self._axes,c,v,self._style)

    def marginalize(self,mdims,bin_range=None):
        mdims = np.array(mdims,ndmin=1,copy=True)
        pdims = np.setdiff1d(self._dims,mdims)
        return self.project(pdims,bin_range)

    def cumulative(self,dims=None,reverse=False):

        if dims is None: dims = self._dims
        else: dims = np.array(dims,ndmin=1,copy=True)

        slices = []
        for i in self._dims:
            if i in dims and reverse: slices.append(slice(None,None,-1))
            else: slices.append(slice(None))
            
        c = np.apply_over_axes(np.cumsum,self._counts[slices],dims)[slices]
        v = np.apply_over_axes(np.cumsum,self._var[slices],dims)[slices]

        return HistogramND.create(self._axes,c,v,self._style)

    def sum(self):

        return np.array([np.sum(self._counts),np.sqrt(np.sum(self._var))])
    
    def slice(self,sdims,dim_index):
        """Generate a new histogram from a subspace of this histogram.
        The subspace of the new histogram is defined by providing a
        list of slice dimensions and list of bin indices for the
        dimensions along which the slice will be performed.  The size
        of any dimensions not in this list will be preserved in the
        output histogram."""

        sdims = np.array(sdims,ndmin=1,copy=True)
        dim_index = np.array(dim_index,ndmin=1,copy=True)
        dims = np.setdiff1d(self._dims,sdims)

        axes= len(self._dims)*[None]
        new_shape = len(self._dims)*[None]
        slices = len(self._dims)*[None]

        for i in dims:
            axes[i] = self._axes[i]
            new_shape[i] = self._axes[i].nbins
            slices[i] = slice(self._axes[i].nbins)
            
        for i, idim in enumerate(sdims):

            if dim_index[i] is None:
                index_range = np.array([0,self.axis(idim).nbins])
            else:
                index_range = np.array(dim_index[i],ndmin=1)
            
            if idim >= self._ndim or index_range[0] >= self._axes[idim].nbins:
                raise ValueError('Dimension or Index out of range')
            
            if len(index_range) == 2:
                new_axis = self._axes[idim].slice(index_range[0],
                                                  index_range[1])

                axes[idim] = new_axis
                slices[idim] = slice(index_range[0],index_range[1])
                new_shape[idim] = new_axis.nbins                
            elif index_range[0] < 0: 
                slices[idim] = slice(index_range[0],index_range[0]-1,-1)
            else:
                slices[idim] = slice(index_range[0],index_range[0]+1)

        axes = list(filter(None,axes))
        new_shape = list(filter(None,new_shape))
        
        c = self._counts[slices].reshape(new_shape)
        v = self._var[slices].reshape(new_shape)

        return HistogramND.create(axes,c,v,self._style)

    def sliceByValue(self,sdims,dim_coord):

        sdims = np.array(sdims,ndmin=1,copy=True)
        dim_coord = np.array(dim_coord,ndmin=1,copy=True)

        dim_index = []

        for i, idim in enumerate(sdims):
            dim_index.append(self._axes[idim].valToBinBounded(dim_coord[i]))

        return self.slice(sdims,dim_index)

    def interpolate(self,*x):
        """Note: All input arrays must have the same dimension."""

        center = []
        for i in range(self._ndim): center.append(self._axes[i].center)
        
        if len(x) == 1:
            xv = x[0]
            shape = x[0].shape[1]
        else:            
            xv, shape = expand_array(*x)
            
        v = interpolatend(center,self._counts,xv)
        return v.reshape(shape)

    def interpolateSlice(self,sdims,dim_coord):

        sdims = np.array(sdims,ndmin=1,copy=True)
        dims = np.setdiff1d(self._dims,sdims)
        dim_coord = np.array(dim_coord,ndmin=1,copy=True)

        h = self.sliceByValue(sdims,dim_coord)

        x = np.zeros(shape=(self._ndim,h._counts.size))
        c = h.center

        for i, idim in enumerate(dims): x[idim] = c[i]
        for i, idim in enumerate(sdims): x[idim,:] = dim_coord[i]

        center = []
        for i in range(self._ndim): center.append(self._axes[i].center)
        h._counts = interpolatend(center,self._counts,x).reshape(h._counts.shape)
        h._var = interpolatend(center,self._var,x).reshape(h._counts.shape)

        return h

    def clear(self,fill_value=0.0):
        """Clear the contents of this histogram."""
        self._counts.fill(fill_value)
        self._var.fill(fill_value)
    
    def __add__(self,x):

        o = copy.deepcopy(self)

        if x is None:
            return o
        elif isinstance(x, HistogramND):
            o._counts += x._counts
            o._var += x._var
        else:
            o._counts += x

        return o

    def __sub__(self,x):

        o = copy.deepcopy(self)

        if isinstance(x, HistogramND):
            o._counts -= x._counts
            o._var += x._var
        else:
            o._counts -= x

        return o

    def __mul__(self,x):
        
        o = copy.deepcopy(self)

        if isinstance(x, HistogramND):

            y1 = self._counts
            y2 = x._counts
            y1v = self._var
            y2v = x._var
            
            f1 = np.zeros(self.shape)
            f2 = np.zeros(self.shape)

            msk1 = (y1 != 0)
            msk2 = (y2 != 0)

            f1[msk1] = y1v[msk1]/y1[msk1]**2
            f2[msk2] = y2v[msk2]/y2[msk2]**2

            o._counts = y1*y2
            o._var = (y1*y2)**2*(f1+f2)
        else:
            o._counts *= x
            o._var *= x*x

        return o

    def __div__(self,x):
        return self.divide(x)

    def __truediv__(self,x):
        return self.divide(x)

    def divide(self,x):

        if isinstance(x, HistogramND):
            o = copy.deepcopy(self)

            y1 = self._counts
            y2 = x._counts
            y1_var = self._var
            y2_var = x._var

            msk = ((y1!=0) & (y2!=0))

            o._counts[~msk] = 0.
            o._var[~msk] = 0.
            
            o._counts[msk] = y1[msk] / y2[msk]
            o._var[msk] = (y1[msk] / y2[msk])**2
            o._var[msk] *= (y1_var[msk]/y1[msk]**2 + y2_var[msk]/y2[msk]**2)

            return o
        else:
            x = np.array(x,ndmin=1)
            msk = x != 0
            x[msk] = 1./x[msk]
            x[~msk] = 0.0
            return self.__mul__(x)

    @staticmethod
    def createFromFn(axes,fn,style=None,label='__nolabel__'):

        if not isinstance(axes,list): axes = [axes]

        h = HistogramND.create(axes,style=style,label=label)
        h.fill_from_fn(fn)
        return h
        
    @staticmethod
    def create(axes,c=None,v=None,style=None,label='__nolabel__'):
        """Factory method for instantiating an empty histogram object.
        Will automatically produce a 1D or 2D histogram if appropriate
        based on the length of the input axes array."""
        ndim = len(axes)
        if ndim == 1:
            return Histogram(axes[0],counts=c,var=v,style=style,label=label)
        elif ndim == 2: 
            return Histogram2D(axes[0],axes[1],counts=c,var=v,
                               style=style,label=label)
        else: return HistogramND(axes,counts=c,var=v,style=style,label=label)

    @staticmethod
    def createFromTree(t,vars,axes,cut=None,fraction=1.0,
                       label = '__nolabel__'):
        """Factory method for instantiating a histogram from a ROOT
        tree."""
        nentries=t.GetEntries()
        first_entry = min(int((1.0-fraction)*nentries),nentries)
        nentries = nentries - first_entry

        x = []
        for v in vars: x.append(get_vector(t,v,cut,nentries,first_entry))
        z = np.vstack(x)

        for i, a in enumerate(axes):
            if isinstance(a,dict):
                vmin = min(z[i])
                vmax = max(z[i])

                vmin -= 1E-8*(vmax-vmin)
                vmax += 1E-8*(vmax-vmin)

                axis_label = None
                if 'label' in a: axis_label = a['label']
                axes[i] = Axis.create(vmin,vmax,a['nbin'],label=axis_label)
        
        h = HistogramND(axes)
        h.fill(z)

        return HistogramND.create(h.axes(),h.counts,h.var,h.style())

class HistogramIterator(object):
    """Iterator class that can be used to loop over the bins of a
    one-dimensional histogram."""

    def __init__(self,ibin,h):
        self._ibin = ibin
        self._h = h

    def set_counts(self,counts):
        self._h._counts[self._ibin] = counts

    def bin(self):
        return self._ibin

    def counts(self):
        return self._h.counts(self._ibin)

    def center(self):
        return self._h.axis().center[self._ibin]

    def lo_edge(self):
        return self._h.axis().edges[self._ibin]

    def hi_edge(self):
        return self._h.axis().edges[self._ibin+1]

    def next(self):

        self._ibin += 1

        if self._ibin == self._h.nbins:
            raise StopIteration
        else:
            return self

    def __iter__(self):
        return self


class Axis(object):
    """Axis object representing a sequence of coordinate bins in one
    dimension.  Axis objects are used to define the N-dimensional
    space of the histogram class."""

    def __init__(self,edges,nbins=None,center=None,name=None,label=None,units=None):
        """Construct an axis object from a sequence of bins edges."""

        self._name = name
        self._label = label
        self._units = units

        edges = np.array(edges,copy=True)

        if len(edges) < 2:
            raise ValueError("Axis must be initialized with at least two "
                             "bin edges.")

        if not nbins is None:
            edges = np.linspace(edges[0],edges[-1],nbins+1)

        self._edges = edges
        self._nbins = len(edges)-1

        if center is None:
            self._center = 0.5*(self._edges[1:] + self._edges[:-1])
        else:
            self._center = copy.copy(center)

        self._err = 0.5*(self._edges[1:] - self._edges[:-1])
        self._width = 2*self._err

    @staticmethod
    def createFromDict(d):

        kw = {'name' : None, 'label' : None}
        kw = merge_dict(kw,d)
        
        if 'edges' in d: return Axis(d['edges'],**kw)
        elif 'lo' in d: return Axis.create(d['lo'],d['hi'],d['nbin'],**kw)

        
    @staticmethod
    def create(lo,hi,nbin,**kwargs):
        """Create an axis object given lower and upper bounds for the
        coordinate and a number of bins."""
        return Axis(np.linspace(lo,hi,nbin+1),**kwargs)

    @staticmethod
    def createFromArray(x,**kwargs):

        x = np.array(x,ndmin=1)
        if len(x) == 1: 
            delta = np.array(1.0,ndmin=1)
        else: 
            delta = x[1:]-x[:-1]

        center = x
        edges = 0.5*(center[1:]+center[:-1])
        edges = np.insert(edges,0,center[0]-0.5*delta[0])
        edges = np.append(edges,center[-1]+0.5*delta[-1])
        return Axis(edges,center=center,**kwargs)

    def slice(self,lobin,hibin):
        if hibin is None: hibin = self._nbins
        if lobin < 0: lobin -= 1
        
        edges = self._edges[lobin:hibin+1]        
        return Axis(edges,label=self._label,name=self._label)

    @property
    def nbins(self):
        return self._nbins

    @property
    def shape(self):
        return (self._nbins,)
    
    def label(self):
        return self._label

    def lims(self):
        return (self._edges[0],self._edges[-1])
    
    def lo_edge(self):
        return self._edges[0]

    def hi_edge(self):
        return self._edges[-1]

    def bins(self):
        return np.array(range(self._nbins))

    @property
    def edges(self):
        """Return array of bin edges."""
        return self._edges

    @property
    def width(self):
        """Return array of bin widths."""
        return self._width

    @property
    def center(self):
        """Return array of bin centers."""
        return self._center

    def binToVal(self,ibin):
        """Convert bin index to axis coordinate."""

        return self._center[ibin]

    def binCoordToVal(self,ibin):
        """Convert bin coordinate to axis coordinate."""

        return interpolate(np.linspace(0,self._nbins,self.nbins+1),self.edges,
                           ibin)

    def valToEdge(self,x):
        """Convert axis coordinate to edge index."""
        w = np.insert(self._width,0,self._width[0])
        ibin = np.digitize(np.array(x,ndmin=1),
                           self._edges-0.5*w)-1
        ibin[ibin < 0] = 0
        return ibin
        
    def valToBin(self,x):
        """Convert axis coordinate to bin index."""
        ibin = np.digitize(np.array(x,ndmin=1),self._edges)-1
        return ibin

    def valToBinCoord(self,x):
        return interpolate(self.edges,np.linspace(0,self._nbins,self.nbins+1),x)
        
    def valToBinBounded(self,x):
        ibin = self.valToBin(x)
        ibin[ibin < 0] = 0
        ibin[ibin > self.nbins-1] = self.nbins-1
        return ibin

    def to_dict(self):
        o = {'edges' : copy.copy(self.edges),
             'center' : copy.copy(self.center),
             'label' : self._label,
             'name'  : self._name}
        return o

    def __str__(self):
        return '%i %s'%(self._nbins,self._edges)

class Histogram(HistogramND):
    """One-dimensional histogram class.  Each bin is assigned both a
    content value and an error.  Non-equidistant bin edges can be
    defined with an input bin edge array.  Supports multiplication,
    addition, and division by a scalar or another histogram object."""

    default_style = { 'marker' : None,
                      'markersize' : None,
                      'color' : None,
                      'drawstyle' : 'default',
                      'markerfacecolor' : None,
                      'markeredgecolor' : None,
                      'linestyle' : None,
                      'linewidth' : 1,
                      'label' : None,
                      'hist_style' : 'errorbar',
                      'hist_xerr' : None,
                      'hist_yerr' : None,
                      'msk'   : None,
                      'max_frac_error' : None }

    def __init__(self,axis,counts=None,var=None,
                 style=None,label = '__nolabel__'):
        """
        Create a 1D histogram object.
                
        @param axis:  Axis object or array of bin edges.
        @param label: Label for this histogram.
        @param counts: Vector of bin values.  The histogram will be
        initialized with this vector when this argument is defined.
        If this is a scalar its value will be used to initialize all
        bins in the histogram.
        @param var: Vector of bin variances.
        @param style: Style dictionary.
        @return:
        """

        super(Histogram, self).__init__([axis],counts,var,label=label)

        self._underflow = 0
        self._overflow = 0

        self._style = copy.deepcopy(Histogram.default_style)
        update_dict(self._style,style)
        self._style['label'] = label
        

    def iterbins(self):
        """Return an iterator object that steps over the bins in this
        histogram."""
        return HistogramIterator(-1,self)

    @staticmethod
    def createEfficiencyHistogram(htot,hcut,label = '__nolabel__'):
        """Create a histogram of cut efficiency."""

        h = Histogram(htot.axis(),label=label)
        eff = hcut._counts/htot._counts

        h._counts = eff
        h._var = eff*(1-eff)/htot._counts

        return h

    @staticmethod
    def createFromTree(t,varname,cut,hdef=None,fraction=0.0,
                       label = '__nolabel__'):

        from ROOT import gDirectory

        draw = '%s>>hist'%(varname)

        if not hdef is None:
            draw += '(%i,%f,%f)'%(hdef[0],hdef[1],hdef[2])

        nevent = t.GetEntries()
        first_entry = int(nevent*fraction)

        t.Draw(draw,cut,'goff',nevent,first_entry)
        h = gDirectory.Get('hist')
        h.SetDirectory(0)
        return Histogram.createFromTH1(h,label=label)

    @staticmethod
    def createFromTH1(hist,label = '__nolabel__'):
        n = hist.GetNbinsX()
        xmin = hist.GetBinLowEdge(1)
        xmax = hist.GetBinLowEdge(n+1)

        h = Histogram(np.linspace(xmin,xmax,n+1),label=label)
        h._counts = np.array([hist.GetBinContent(i) for i in range(1, n + 1)])
        h._var = np.array([hist.GetBinError(i)**2 for i in range(1, n + 1)])
        h._underflow = hist.GetBinContent(0)
        h._overflow = hist.GetBinContent(n+1)

        return h
    
    @staticmethod
    def createFromCDF(xedge,ncount,min_count=0):

        if np.sum(ncount) == 0: return Histogram(xedge)

        h = Histogram(xedge)
        h._counts = copy.deepcopy(ncount)
        h._var = copy.deepcopy(ncount)
        h = h.rebin_mincount(min_count)

        ncum = np.concatenate(([0],np.cumsum(h._counts)))
        fn = UnivariateSpline(h.axis().edges,ncum,s=0,k=1)
        mu_count = fn(xedge[1:])-fn(xedge[:-1])
        mu_count[mu_count<0] = 0

        return Histogram(xedge,counts=mu_count,var=copy.deepcopy(mu_count))
    
    def to_root(self,name,title=None):

        if title is None: title=name
        
        import ROOT
        h = ROOT.TH1F(name,title,self.axis().nbins,
                      self.axis().lo_edge(),self.axis().hi_edge())

        for i in range(self.axis().nbins):
            h.SetBinContent(i+1,self._counts[i])
            h.SetBinError(i+1,np.sqrt(self._var[i]))

        return h
    
    def update_artists(self,artists):
        
        artists[0].set_ydata(self.counts)
        artists[0].set_xdata(self.axis().center)

        if len(artists[1]) == 0: return

        x = self.axis().center
        y = self.counts
        xerr = np.abs(0.5*self.axis().width)
        yerr = self.err

        error_positions = (x-xerr,y), (x+xerr,y), (x,y-yerr), (x,y+yerr) 

        # Update the caplines 
        for i,pos in enumerate(error_positions): 
            artists[1][i].set_data(pos) 

        sx = np.array([[x - xerr, y], [x + xerr, y]]).transpose(2,0,1)
        sy = np.array([[x, y-yerr], [x, y+yerr]]).transpose(2,0,1)

        artists[2][0].set_segments(sx)
        artists[2][1].set_segments(sy)

    def update_style(self,style):
        update_dict(self._style,style)

    def label(self):
        return self._style['label']

    def _band(self,ax=None,style='step',alpha=0.2,mask_neg=False,**kwargs):
        if ax is None: ax = plt.gca()

        kw = extract_dict_by_keys(kwargs,MPLUtil.errorbar_kwargs)
        clear_dict_by_vals(kw,None)

        kw_fill = extract_dict_by_keys(kwargs,MPLUtil.fill_kwargs)
        clear_dict_by_vals(kw_fill,None)
        kw_fill['alpha'] = alpha

        artists = []
        
        if style=='step':
            xedge = self._axes[0].edges
            x = np.vstack((xedge[:-1],xedge[1:],xedge[1:])).ravel([-1])
            y = np.vstack((self.counts,
                           self.counts,
                           self.counts)).ravel([-1])

            yerr = np.vstack((self.err,
                              self.err,
                              self.err)).ravel([-1])
        
            ebar = ax.errorbar(x,y,**kw)
            kw_fill['color'] = ebar[0].get_color()
            if mask_neg: kw_fill['where']=y-yerr>0
            ax.fill_between(x,y-yerr,y+yerr,**kw_fill)
        elif style=='center':

            ebar = ax.errorbar(self._axes[0].center,self.counts,**kw)
            kw_fill['color'] = ebar[0].get_color()
            if mask_neg: kw_fill['where']=self.counts-self.err>0
            ax.fill_between(self._axes[0].center,
                            self.counts+self.err,
                            self.counts-self.err,**kw_fill)
        else:
            raise Exception('Unrecognized band style: ', style)

        if 'where' in kw_fill: kw_fill.pop('where')
        
        patch, = [ax.add_patch(plt.Rectangle((0,0),0,0,**kw_fill))]
        
        artists += [ebar]
        artists += [patch]

        return artists


    def _errorbar(self, label_rotation=0,
                 label_alignment='center', ax=None, 
                 counts=None, x=None,**kwargs):
        """
        Draw this histogram in the 'errorbar' style.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.errorbar`.
        """
        style = copy.deepcopy(kwargs)

        if ax is None: ax = plt.gca()
        if counts is None: counts = self._counts
        if x is None: x = self._axes[0].center

        if style['msk'] is None:
            msk = np.empty(len(counts),dtype='bool'); msk.fill(True)
        else:
            msk = style['msk']
            
        xerr = None
        yerr = None

        if style['hist_xerr'] or style['hist_xerr'] is None:
            xerr = self._axes[0].width[msk]/2.
        if style['hist_yerr'] or style['hist_yerr'] is None:
            yerr = np.sqrt(self._var[msk])
        if not style.has_key('fmt'): style['fmt'] = '.'

        kw = extract_dict_by_keys(style,MPLUtil.errorbar_kwargs)
        clear_dict_by_vals(kw,None)

        ebar = ax.errorbar(x[msk], counts[msk],xerr=xerr,yerr=yerr,**kw)
        self._prepare_xaxis(label_rotation, label_alignment)
        return [ebar]

    def hist(self,ax=None,counts=None,**kwargs):
        """Plot this histogram using the 'hist' matplotlib method."""
        if ax is None: ax = plt.gca()
        if counts is None: counts = self._counts

        style = kwargs
        
        kw = extract_dict_by_keys(style,MPLUtil.hist_kwargs)
        clear_dict_by_vals(kw,None)
        
        hist = ax.hist(self._axes[0].center, self._axes[0].nbins,
                       range=[self._axes[0].lo_edge(),self._axes[0].hi_edge()],
                       weights=counts,**kw)
        return [hist]


    def plot(self,ax=None,overflow=False,**kwargs):
        """
        Draw this histogram in the 'plot' style.

        All additional keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Returns an array of matplotlib artists.
        """

        style = copy.deepcopy(self._style)
        style.update(kwargs)

        if ax is None: ax = plt.gca()
        if style['msk'] is None:
            style['msk'] = np.empty(self._axes[0].nbins,dtype='bool')
            style['msk'].fill(True)
            
        if overflow:
            c = copy.deepcopy(self._counts)
            c[0] += self._underflow
            c[-1] += self._overflow
        else:
            c = self._counts

        if not style['max_frac_error'] is None:
            frac_err = np.sqrt(self._var)/self._counts
            style['msk'] = frac_err <= style['max_frac_error']

        hs = style['hist_style']

        if hs == 'errorbar':
            return self._errorbar(ax=ax,counts=c,**style)
        elif hs == 'line':

            style['marker'] = 'None'
            style['hist_xerr'] = False
            style['hist_yerr'] = False
            style['fmt'] = '-'
            return self._errorbar(ax=ax,counts=c,**style)
        elif hs == 'stepfilled' or hs == 'bar':

            style['histtype'] = hs
            
#            draw_style['linewidth'] = 0
#            del draw_style['linestyle']
 #           del draw_style['marker']
 #           del draw_style['drawstyle']

            return self.hist(ax=ax,counts=c,**style)
        elif hs == 'step':
            
            c = np.concatenate(([0],c,[0]))
            edges = np.concatenate((self._axes[0].edges,
                                    [self._axes[0].edges[-1]]))
#            msk = np.concatenate((msk,[True]))
            style['hist_xerr'] = False
            style['hist_yerr'] = False
            style['fmt'] = '-'
            style['drawstyle'] = 'steps-pre'
            style['marker'] = 'None'
            return self._errorbar(ax=ax,counts=c,
                                 x=edges,**style)

        elif hs == 'band' or hs == 'band_step' :
            return self._band(style='step',**style)  
        elif hs == 'band_center' :
            return self._band(style='center',**style)  
        else:
            raise Exception('Unrecognized style: ' + hs)

    def scale_density(self,fn):

        h = copy.deepcopy(self)

        for i in range(self.axis().nbins):
            xhi = fn(self.axis().edges[i+1])
            xlo = fn(self.axis().edges[i])

            area = xhi - xlo

            h._counts[i] /= area
            h._var[i] /= area**2

        return h

    def nbins(self):
        """Return the number of bins in this histogram."""
        return self._axes[0].nbins

    def underflow(self):
        return self._underflow

    def overflow(self):
        return self._overflow

    def stddev(self):
        """Comput the standard deviation of this histogram."""
        m = self.mean()
        n = np.sum(self.counts)
        dx = self.axis().center - m        
        return np.sqrt(np.sum(self.counts*dx**2)/n)
        
    def mean(self):
        x = self.axis().center
        return np.sum(self.counts*x)/np.sum(self.counts)
        
    def sum(self,overflow=False):
        """Return the sum of counts in this histogram."""

        s = 0
        if overflow:
            s = np.sum(self._counts) + self._overflow + self._underflow
        else:
            s = np.sum(self._counts)

        return np.array([s,np.sqrt(np.sum(self._var))])

    def cumulative(self,lhs=True):
        """Convert this histogram to its cumulative distribution."""

        if lhs:
            counts = np.cumsum(self._counts)
            var = np.cumsum(self._var)
        else:
            counts = np.cumsum(self._counts[::-1])[::-1]
            var = np.cumsum(self._var[::-1])[::-1]

        return Histogram(self._axes[0].edges,
                         counts=counts,var=var,label=self.label())

    def getBinByValue(self,x):
        return np.argmin(np.abs(self._x-x))

    def interpolate(self,x,noerror=True):
        
        x = np.array(x,ndmin=1)
        c = interpolate(self._axes[0].center,self._counts,x)
        v = interpolate(self._axes[0].center,self._var,x)

        if noerror:
            return c
        else:
            if len(x)==1: return np.array([c[0],np.sqrt(v)[0]])
            else: return np.vstack((c,np.sqrt(v)))

    def merge(self,h):
        edges = np.concatenate((h.axis().edges,
                                self.axis().edges))
        edges = np.unique(edges)
        o = Histogram(edges)
        o.fill(h.axis().center,h.counts)
        o.fill(self.axis().center,self.counts)
        return o

    def normalize(self):

        s = np.sum(self._counts)

        counts = self._counts/s
        var = self._var/s**2

        return Histogram(self._axes[0],
                         counts=counts,var=var,label=self.label())

    def quantile(self,fraction=0.68,method='var',niter=100,**kwargs):

        if method == 'var':

            cs = np.concatenate(([0],self.counts))
            vs = np.concatenate(([0],self.var))
            
            return get_quantile_error(self.axis().edges,cs,vs,fraction)
        elif method == 'bootstrap_poisson':

            c = np.random.poisson(np.concatenate(([0],self.counts)),
                                  (niter,self.axis().nbins+1))

            xq = []                
            for i in range(niter):
                xq.append(get_quantile(self.axis().edges,c[i],fraction))

            q = get_quantile(self.axis().edges,
                             np.concatenate(([0],self.counts)),fraction)
            qvar = np.std(np.array(xq))**2

            return q, qvar
            
#        import stats
#        return stats.HistQuantile(self).eval(fraction,**kwargs)

    def central_quantile(self,fraction=0.68,unbias_method='median',
                         method='bootstrap_poisson'):

        if  unbias_method == 'median': loc = self.quantile(fraction=0.5)[0]
        elif unbias_method == 'mean': loc = self.mean()
        else:
            raise('Exception')

        delta_max = max(loc-self.axis().center)
        
        habs = Histogram(Axis.create(0,delta_max,self._axes[0].nbins*2))
        habs.fill(np.abs(self.axis().center-loc),
                  self._counts,var=self._var)
        
        return habs.quantile(fraction,method=method,niter=100)
    
    def chi2(self,model,min_counts=None):

        msk = (self._var > 0)
        if not min_counts is None:
            msk &= (self._counts>5)

        if isinstance(model,Histogram):
            diff = self._counts[msk] - model._counts[msk]
        else:
            diff = self._counts[msk] - model(self._axes[0].center)[msk]

        chi2 = np.sum(np.power(diff,2)/self._var[msk])
        ndf = len(msk[msk==True])

        pval = 1 - stats.chi2.cdf(chi2, ndf)
        return [chi2,ndf,pval]

    def set(self,i,w,var=None):
        self._counts[i] = w
        if not var is None: self._var[i] = var
    
    def fill(self,x,w=1,var=None):
        """
        Add counts to the histogram at the coordinates given in the
        array x.  By default each point is given unit weight.  The
        weight assigned to each element of x can be specified by
        providing a scalar or vector for the w argument.
        """
        x = np.array(x,ndmin=1,copy=True)
        w = np.array(w,ndmin=1,copy=True)
        if len(x) == 0: return
        if len(w) < len(x): w = np.ones(len(x))*w
        
        if var is None: var = w
        else:
            var = np.array(var,ndmin=1,copy=True)
            if len(var) < len(w): var = np.ones(len(w))*var
        
        if w.ndim == 1:

            if self._axes[0].lo_edge() > self._axes[0].hi_edge(): 
                c1 = np.histogram(x,bins=self._axes[0].edges[::-1],
                                  weights=w)[0][::-1]
                c2 = np.histogram(x,bins=self._axes[0].edges[::-1],
                                  weights=var)[0][::-1]
            else:
                c1 = np.histogram(x,bins=self._axes[0].edges,weights=w)[0]
                c2 = np.histogram(x,bins=self._axes[0].edges,weights=var)[0]
                
            self._counts += c1
            self._var += c2

            if np.any(x>=self._axes[0].hi_edge()):
                self._overflow += np.sum(w[x>=self._axes[0].hi_edge()])
            if np.any(x<self._axes[0].lo_edge()):
                self._underflow += np.sum(w[x<self._axes[0].lo_edge()])

        else:
            c = np.histogram(x,bins=self._axes[0].edges)[0]

            self._counts += w*c
            self._var += var*c

            if np.any(x>=self._axes[0].hi_edge()):
                self._overflow += np.sum(x>=self._axes[0].hi_edge())
            if np.any(x<self._axes[0].lo_edge()):
                self._underflow += np.sum(x<self._axes[0].lo_edge())


    def rebin_mincount(self,min_count,max_bins=None):
        """Return a rebinned copy of this histogram such that all bins
        have an occupation of at least min_count.
        
        Parameters
        ----------

        min_count : int
            Minimum occupation of each bin.

        max_bins : int
            Maximum number of bins that can be combined.
        
        """

        bins = [0]
        c = np.concatenate(([0],np.cumsum(self._counts)))

        for ibin in range(self._axes[0].nbins+1):

            nbin = ibin-bins[-1]            
            if not max_bins is None and nbin > max_bins:
                bins.append(ibin)
            elif c[ibin] - c[bins[-1]] >= min_count or \
                    ibin == self._axes[0].nbins:
                bins.append(ibin)

        return self.rebin(bins)

    def rebin_range(self,n=2,xmin=0,xmax=0):

        bins = []

        i = 0
        while i < self._axes[0].nbins:

            if xmin != xmax and self._axes[0].center[i] >= xmin and \
                    self._axes[0].center[i] <= xmax:

                m = n
                if i+n > self._axes[0].nbins:
                    m = self._axes[0].nbins-i

                i+= m
                bins.append(m)
            else:
                i+= 1
                bins.append(1)

        return self.rebin(bins)


    def rebin(self,bins=2):

        if isinstance(bins,int) and bins <= 1: return copy.deepcopy(self)

        bins = np.array(bins,dtype=int)

        if bins.ndim == 0:
            bin_index = list(range(0,self._axes[0].nbins,bins))
            bin_index.append(self._axes[0].nbins)
            bin_index = np.array(bin_index)
        else:
#            if np.sum(bins) != self._axes[0].nbins:
#                raise ValueError("Sum of bins is not equal to histogram bins.")
            bin_index = bins

#np.concatenate((np.array([0],dtype='int'),
#                                        np.cumsum(bins,dtype='int')))

        xedges = self._axes[0].edges[bin_index]

        h = Histogram(xedges,label=self.label())
        h.fill(self.axis().center,self._counts,self._var)
        return h

    def rebin_axis(self,axis):

        h = Histogram(axis,style=self._style)
        h.fill(self.axis().center,self._counts,self._var)
        return h

    def residual(self,h):
        """
        Generate a residual histogram.

        @param h: Input histogram
        @return: Residual histogram
        """
        o = copy.deepcopy(self)
        o -= h
        return o/h

    def dump(self,outfile=None):

        if not outfile is None:
            f = open(outfile,'w')
        else:
            f = sys.stdout

        for i in range(len(self._x)):
            s = '%5i %10.5g %10.5g '%(i,self._axes[0].edges[i],
                                      self._axes[0].edges[i+1])
            s += '%10.5g %10.5g\n'%(self._counts[i],self._var[i])
            f.write(s)


    def fit(self,expr,p0):

        import ROOT
        import re

        g = ROOT.TGraphErrors()
        f1 = ROOT.TF1("f1",expr)
#        f1.SetDirectory(0)

        for i in range(len(p0)): f1.SetParameter(i,p0[i])

        npar = len(tuple(re.finditer('\[([\d]+)\]',expr)))
        for i in range(self._axes[0].nbins):
            g.SetPoint(i,self._axes[0].center[i],self._counts[i])
            g.SetPointError(i,0.0,self._counts[i]*0.1)

        g.Fit("f1","Q")

        p = np.zeros(npar)
        for i in range(npar): p[i] = f1.GetParameter(i)
        return p

    def find_root(self,y,x0=None,x1=None):
        """Solve for the x coordinate at which f(x)-y=0 where f(x) is
        a smooth interpolation of the histogram contents."""

        fn = UnivariateSpline(self._axes[0].center,self._counts,k=2,s=0)

        if x0 is None: x0 = self._axes[0].lo_edge()
        if x1 is None: x1 = self._axes[0].hi_edge()

        return brentq(lambda t: fn(t) - y,x0,x1)

    def find_max(self,msk=None):

        if msk is None:
            msk = np.empty(self._axes[0].nbins,dtype='bool'); msk.fill(True)

        return np.argmax(self._counts[msk])

    def _prepare_xaxis(self, rotation=0, alignment='center'):
        """Apply bounds and text labels on x axis."""
#        if self.binlabels is not None:
#            plt.xticks(self._x, self.binlabels,
#                       rotation=rotation, ha=alignment)
        plt.xlim(self._axes[0].edges[0], self._axes[0].edges[-1])

class Histogram2D(HistogramND):

    default_style = { 'keep_aspect' : False, 
                      'logz' : False,
                      'levels' : None, 
                      'shading' : 'flat',
                      'interpolation' : 'nearest' }

    def __init__(self, xaxis, yaxis,  
                 counts=None, var=None, style=None,
                 label = '__nolabel__'):

        super(Histogram2D, self).__init__([xaxis,yaxis],counts,var,label=label)

        self._xaxis = self._axes[0]
        self._yaxis = self._axes[1]

        self._nbins = self._xaxis.nbins*self._yaxis.nbins

        self._style = copy.deepcopy(Histogram2D.default_style)
        update_dict(self._style,style)
        self._style['label'] = label

    @staticmethod
    def createFromTree(t,vars,axes,cut,fraction=0.0,
                       label = '__nolabel__'):

        from ROOT import gDirectory

        draw = '%s:%s>>hist'%(vars[1],vars[0])

        if not axes is None:
            draw += '(%i,%f,%f,%i,%f,%f)'%(axes[0].nbins,
                                           axes[0].lo_edge(),axes[0].hi_edge(),
                                           axes[1].nbins,
                                           axes[1].lo_edge(),axes[1].hi_edge())

        nevent = t.GetEntries()
        first_entry = int(nevent*fraction)

        ncut = t.Draw(draw,cut,'goff',nevent,first_entry)

        h = gDirectory.Get('hist')
        h.SetDirectory(0)
        return Histogram2D.createFromTH2(h)

    @staticmethod
    def createFromTH2(hist,label = '__nolabel__'):
        nx = hist.GetNbinsX()
        ny = hist.GetNbinsY()

        xmin = hist.GetXaxis().GetBinLowEdge(1)
        xmax = hist.GetXaxis().GetBinLowEdge(nx+1)

        ymin = hist.GetYaxis().GetBinLowEdge(1)
        ymax = hist.GetYaxis().GetBinLowEdge(ny+1)

        xaxis = Axis.create(xmin,xmax,nx)
        yaxis = Axis.create(ymin,ymax,ny)

        counts = np.zeros(shape=(nx,ny))
        var = np.zeros(shape=(nx,ny))

        for ix in range(1,nx+1):
            for iy in range(1,ny+1):
                counts[ix-1][iy-1] = hist.GetBinContent(ix,iy)
                var[ix-1][iy-1] = hist.GetBinError(ix,iy)**2

        style = {}
        style['label'] = label

        h = Histogram2D(xaxis,yaxis,counts=counts,var=var,style=style)

        return h

    def to_root(self,name,title=None):

        if title is None: title=name
        
        import ROOT
        h = ROOT.TH2F(name,title,self.xaxis().nbins,
                      self.xaxis().lo_edge(),self.xaxis().hi_edge(),
                      self.yaxis().nbins,
                      self.yaxis().lo_edge(),self.yaxis().hi_edge())

        for i in range(self.xaxis().nbins):
            for j in range(self.yaxis().nbins):
                h.SetBinContent(i+1,j+1,self._counts[i,j])
                h.SetBinError(i+1,j+1,np.sqrt(self._var[i,j]))

        return h
    
    def label(self):
        return self._style['label']
    
    def update_style(self,style):
        update_dict(self._style,style)

    def nbins(self,idim=None):
        """Return the number of bins in this histogram."""
        if idim is None: return self._nbins
        elif idim == 0: return self.xaxis().nbins
        elif idim == 1: return self.yaxis().nbins
        else: return 0

    def xaxis(self):
        return self._axes[0]

    def yaxis(self):
        return self._axes[1]

    def lo_edge(self,ix,iy):
        return (self._xaxis.edges[ix],self._yaxis.edges[iy])

    def hi_edge(self,ix,iy):
        return (self._xaxis.edges[ix+1],self._yaxis.edges[iy+1])

    def maxIndex(self,ix_range=None,iy_range=None):
        """Return x,y indices of maximum histogram element."""

        if ix_range is None: ix_range = [0,self._xaxis.nbins+1]
        elif len(ix_range) == 1: ix_range = [ix_range[0],self._xaxis.nbins+1]

        if iy_range is None: iy_range = [0,self._yaxis.nbins+1]
        elif len(iy_range) == 1: iy_range = [iy_range[0],self._yaxis.nbins+1]

        a = np.argmax(self._counts[ix_range[0]:ix_range[1],
                                   iy_range[0]:iy_range[1]])

        cv = self._counts[ix_range[0]:ix_range[1],iy_range[0]:iy_range[1]]
        ixy = np.unravel_index(a,cv.shape)
        return (ixy[0]+ix_range[0],ixy[1]+iy_range[0])

    def interpolate(self,x,y):
        from util import interpolate2d

        x = np.array(x,ndmin=1)
        y = np.array(y,ndmin=1)
        
        xv, shape = expand_array(x,y)
        v = interpolate2d(self._xaxis.center,
                          self._yaxis.center,self._counts,xv[0],xv[1])

        return v.reshape(shape)
        
    def integrate(self,iaxis=1,bin_range=None):
        if iaxis == 1:

            if bin_range is None: bin_range = [0,self._yaxis.nbins]

            h = Histogram(self._xaxis.edges)
            for iy in range(bin_range[0],bin_range[1]):
                h._counts[:] += self._counts[:,iy]*self._ywidth[iy]
                h._var[:] += self._var[:,iy]*self._ywidth[iy]**2

            return h


    def marginalize(self,iaxis,bin_range=None):
        """Return 1D histogram marginalized over x or y dimension.

        @param: iaxis Dimension over which to marginalize.
        """

        h = Histogram(self._axes[(iaxis+1)%2],style=self._style)        

        if iaxis == 1:

            if bin_range is None: 
                h._counts = np.apply_over_axes(np.sum,self._counts,[1]).reshape(h._counts.shape)
                h._var = np.apply_over_axes(np.sum,self._var,[1]).reshape(h._counts.shape)
            else:
                c = self._counts[:,bin_range[0]:bin_range[1]]
                v = self._var[:,bin_range[0]:bin_range[1]]

                h._counts = np.apply_over_axes(np.sum,c,[1]).reshape(h._counts.shape)
                h._var = np.apply_over_axes(np.sum,v,[1]).reshape(h._counts.shape)
        else:

            if bin_range is None: 
                h._counts = np.apply_over_axes(np.sum,self._counts,[0]).reshape(h._counts.shape)
                h._var = np.apply_over_axes(np.sum,self._var,[0]).reshape(h._counts.shape)
            else:
                c = self._counts[bin_range[0]:bin_range[1],:]
                v = self._var[bin_range[0]:bin_range[1],:]

                h._counts = np.apply_over_axes(np.sum,c,[0]).reshape(h._counts.shape)
                h._var = np.apply_over_axes(np.sum,v,[0]).reshape(h._counts.shape)

        return h

    def mean(self,iaxis,**kwargs):

        hq = Histogram(self._xaxis.edges)

        for i in range(self.nbins(0)):
            h = self.slice(iaxis,i)

            x = h.axis().center

            mean = np.sum(h.counts*x)/np.sum(h.counts)
            mean_var = mean**2*(np.sum(h.var*x**2)/np.sum(h.counts*x)**2 +
                                np.sum(h.var)/np.sum(h.counts)**2)

            hq._counts[i] = mean
            hq._var[i] = mean_var

        return hq

#    def quantile(self,iaxis,fraction=0.68,**kwargs):

#        import stats

#        iaxis = (iaxis+1)%2

#        hq = Histogram(self._axes[iaxis].edges)

#        for i in range(self.nbins(iaxis)):
#            h = self.slice(iaxis,i)

#            q,qerr = stats.HistQuantile(h).eval(fraction,**kwargs)

#            hq._counts[i] = q
#            hq._var[i] = qerr**2

#        return hq

    def central_quantile(self,fraction=0.68,unbias_method=None):
        hmed = self.quantile(fraction=0.5)
        hmean = self.mean()

        lo_index = np.argmin(np.abs(h.yedges()))
        
        habs = Histogram2D(h.xedges(),h.yedges()[lo_index:])
        
        for i in range(h.nbins(0)):

            yc = h.yaxis().center
            
            if unbias_method is None: y = np.abs(yc)
            elif unbias_method == 'median': y = np.abs(yc-hmed.counts(i))
            elif unbias_method == 'mean': y = np.abs(yc-hmean.counts(i))

            
            habs.fill(np.ones(h.nbins(1))*h.xaxis().center[i],y,h._counts[i],
                      var=h._counts[i])
            

        return habs.quantile(fraction=0.68,method='mc',niter=100)

    def set(self,ix,iy,w,var=None):
        self._counts[ix,iy] = w
        if not var is None: self._var[ix,iy] = var

    def fill(self,x,y,w=1,var=None):
    
        x = np.array(x,copy=True,ndmin=1)
        y = np.array(y,copy=True,ndmin=1)

        if len(x) < len(y): x = np.ones(len(y))*x[0]
        if len(y) < len(x): y = np.ones(len(x))*y[0]

        HistogramND.fill(self,np.vstack((x,y)),w,var)

    def fill2(self,x,y,w=1,var=None):

        x = np.array(x,ndmin=1)
        y = np.array(y,ndmin=1)
        w = np.array(w,ndmin=0)

        if var is None: var = w

        bins = [self._xaxis.edges,self._yaxis.edges]

        if w.ndim == 1:
            c1 = np.histogram2d(x,y,bins=bins,weights=w)[0]
            c2 = np.histogram2d(x,y,bins=bins,weights=var)[0]

            self._counts += c1
            self._var += c2
        else:
            c = np.histogram2d(x,y,bins=bins)[0]

            self._counts += w*c
            self._var += var*c

    def smooth(self,sigma):

        from scipy import ndimage

        sigma_bins = sigma/(self._xaxis.edges[1]-self._xaxis.edges[0])

        counts = ndimage.gaussian_filter(self._counts, sigma=sigma_bins)
        var = ndimage.gaussian_filter(self._var, sigma=sigma_bins)

        return Histogram2D(self._xaxis,self._yaxis,counts=counts,var=var)

    def update_axis_labels(self,ax):
        if not self._axes[0].label() is None:
            ax.set_xlabel(self._axes[0].label())
        if not self._axes[1].label() is None:
            ax.set_ylabel(self._axes[1].label())

    def plot(self,ax=None,**kwargs):
        return self.pcolor(ax,**kwargs)
    
    def pcolor(self,ax=None,**kwargs):
        """Render this histogram with the matplotlib pcolormesh
        method.  This method is generally be preferred over the imshow
        method since it allows non-equidistant bins (which is
        permitted by the Histogram2D class."""

        style = copy.deepcopy(self._style)
        style.update(kwargs)

        if ax is None: ax = plt.gca()
        
        if style['logz']: norm = LogNorm()
        else: norm = Normalize()

        xedge, yedge = np.meshgrid(self.axis(0).edges,self.axis(1).edges)

        kw = extract_dict_by_keys(style,MPLUtil.pcolormesh_kwargs)

        p = ax.pcolormesh(xedge,yedge,self._counts.T,norm=norm,
                           **kw)

        self.update_axis_labels(ax)

        return p
        
    
    def contour(self,ax=None,clabel=False,**kwargs):
        """Render this histogram with the matplotlib imshow method."""
#        levels = [2.,4.,6.,8.,10.]

        style = copy.deepcopy(self._style)
        style.update(kwargs)

        if ax is None: ax = plt.gca()

        if style['logz']: norm = LogNorm()
        else: norm = Normalize()

        kw = extract_dict_by_keys(style,MPLUtil.contour_kwargs)
        kw['origin'] = 'lower'

        cs = plt.contour(self._xaxis.center,self._yaxis.center,
                         self._counts.T,**kw)
        if clabel: plt.clabel(cs, fontsize=9, inline=1)

        self.update_axis_labels(ax)

        return cs

    def imshow(self,ax=None,**kwargs):
        """Render this histogram with the matplotlib imshow method."""
        style = copy.deepcopy(self._style)
        style.update(kwargs)

        if ax is None: ax = plt.gca()

        dx = self._xaxis.hi_edge() - self._xaxis.lo_edge()
        dy = self._yaxis.hi_edge() - self._yaxis.lo_edge()

        aspect_ratio = 1
        if not style['keep_aspect']: aspect_ratio=dx/dy

        if style['logz']: norm = LogNorm()
        else: norm = Normalize()

        kw = extract_dict_by_keys(style,MPLUtil.imshow_kwargs)
        kw['origin'] = 'lower'
        
        im = ax.imshow(self._counts.transpose(),
                       aspect=aspect_ratio,norm=norm,
                       extent=[self._xaxis.lo_edge(), self._xaxis.hi_edge(),
                               self._yaxis.lo_edge(), self._yaxis.hi_edge()],
                       **kw)

        self.update_axis_labels(ax)

        return im

def get_vector(chain,var,cut=None,nentries=None,first_entry=0):

    if cut is None: cut = ''
    
    chain.SetEstimate(chain.GetEntries())
    if nentries is None: nentries = chain.GetEntries()
    ncut = chain.Draw('%s'%(var),cut,'goff',nentries,first_entry)
    return copy.deepcopy(np.frombuffer(chain.GetV1(),
                                       count=ncut,dtype='double'))


if __name__ == '__main__':

    fig = plt.figure()

    hnd = HistogramND([np.linspace(0,1,10),
                       np.linspace(0,1,10)])




    h1d = Histogram([0,10],10)


    h1d.fill(3.5,5)
    h1d.fill(4.5,3)
    h1d.fill(1.5,5)
    h1d.fill(8.5,1)
    h1d.fill(9.5,1)

    for x in h1d.iterbins():
        print(x.center, x.counts)

    h1d.rebin([4,4,2])

    print (h1d.axis().edges)
        

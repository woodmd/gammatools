#!/usr/bin/env python

"""
@file  series.py

@brief Python class representing series data (arrays of x/y pairs)."

@author Matthew Wood <mdwood@slac.stanford.edu>
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from util import *

class Band(object):

    default_draw_style = { 'marker' : None,
                           'facecolor' : None,                           
                           'linestyle' : None,
                           'linewidth' : None,
                           'alpha' : 0.4,
                           'label' : None }

    default_style = { }

    def __init__(self,x,ylo,yhi,style=None):

        self._x = np.array(x,copy=True)
        self._ylo = np.array(ylo,copy=True)
        self._yhi = np.array(yhi,copy=True)

        self._style = copy.deepcopy(dict(Series.default_style.items() +
                                         Series.default_draw_style.items()))
        if not style is None: update_dict(self._style,style)

    def plot(self,ax=None,**kwargs):

        if ax is None: ax = plt.gca()

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        clear_dict_by_keys(style,Series.default_draw_style.keys(),False)
        clear_dict_by_vals(style,None)

        ax.fill_between(self._x,self._ylo,self._yhi,**style)

class Series(object):

    default_errorbar_style = { 'marker' : None,
                               'color' : None,
                               'markersize' : None,
                               'markerfacecolor' : None,
                               'markeredgecolor' : None,
                               'linestyle' : None,
                               'linewidth' : None,
                               'label' : None }

    default_scatter_style = { 'marker' : None,
                               'color' : None,
                              'edgecolor' : None,
#                               'markersize' : None,
#                               'markerfacecolor' : None,
#                               'markeredgecolor' : None,
#                               'linestyle' : None,
#                               'linewidth' : 1,
                               'label' : None }

    default_style = { 'msk' : None, 'draw_style' : 'errorbar' }
    

    def __init__(self,x,y,yerr=None,style=None):
        self._x = np.array(x,copy=True)
        self._y = np.array(y,copy=True)
        if not yerr is None: self._yerr = np.array(yerr,copy=True)
        else: self._yerr = yerr

        self._msk = np.empty(len(self._x),dtype='bool')
        self._msk.fill(True)
        
        self._style = copy.deepcopy(dict(Series.default_style.items() +
                                         Series.default_scatter_style.items() +
                                         Series.default_errorbar_style.items()))
        if not style is None: update_dict(self._style,style)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def yerr(self):
        return self._yerr

    def label(self):
        return self._style['label']

    def style(self):
        return self._style

    def update_style(self,style):
        update_dict(self._style,style)

    def plot(self,ax=None,**kwargs):

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        if style['draw_style'] == 'errorbar':
            self._errorbar(ax,**style)
        else:
            self._scatter(ax,**style)

        
    def _errorbar(self,ax=None,**kwargs):

        if ax is None: ax = plt.gca()

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        if style['msk'] is None: msk = self._msk
        else: msk = style['msk']
        
        clear_dict_by_keys(style,Series.default_errorbar_style.keys(),False)
        clear_dict_by_vals(style,None)

        if not self._yerr is None: yerr = self._yerr[msk]
        else: yerr = self._yerr

        ax.errorbar(self._x[msk],self._y[msk],yerr,**style)

    def _scatter(self,ax=None,**kwargs):

        if ax is None: ax = plt.gca()

        style = copy.deepcopy(self._style)
        update_dict(style,kwargs)

        if style['msk'] is None: msk = self._msk
        else: msk = style['msk']
        
        clear_dict_by_keys(style,Series.default_scatter_style.keys(),False)
        clear_dict_by_vals(style,None)

        if not self._yerr is None: yerr = self._yerr[msk]
        else: yerr = self._yerr

        print style

        ax.scatter(self._x[msk],self._y[msk],**style)


    @staticmethod
    def createFromDict(d):

        o  = {'x' : None, 'y' : None, 'yerr' : None, 'style' : None }
        o.update(d)
        return Series(**d)

    @staticmethod
    def createFromFile(filename):

        d = np.loadtxt(filename,unpack=True)

        if len(d) == 2: return Series(d[0],d[1])
        elif len(d) == 3: return Series(d[0],d[1],d[2])

    def interpolate(self,x):
        return interpolate(self._x,self._y,x)

    def mask(self,msk):

        o = copy.deepcopy(self)
        o._x = self._x[msk]
        o._y = self._y[msk]
        o._msk = self._msk[msk]
        if not o._yerr is None:
            o._yerr = self._yerr[msk]

        return o
        
    def __sub__(self,x):

        o = copy.deepcopy(self)
        o._y -= x
        return o

    def __div__(self,x):

        o = copy.deepcopy(self)
        o._y /= x
        if not o._yerr is None: o._yerr /= x
        return o

    def __mul__(self,x):

        o = copy.deepcopy(self)
        o._y *= x
        if not o._yerr is None: o._yerr *= x
        return o
        

if __name__ == '__main__':

    fig = plt.figure()


    x0 = np.linspace(0,2*np.pi,100.)
    y0 = 2. + np.sin(x0)
    y1 = 2. + 0.5*np.sin(x0+np.pi/4.)

    
    s = Series(x0,y0)

    s.plot(marker='o',color='b',markerfacecolor='w',markeredgecolor='b')

    plt.show()

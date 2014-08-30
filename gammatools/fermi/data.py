"""
@file  data.py

@brief Python classes for storing and manipulating photon data.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood"
__date__     = "01/01/2013"

import numpy as np
import re
import copy
import pyfits
from gammatools.core.algebra import Vector3D
import matplotlib.pyplot as plt
from catalog import Catalog, CatalogSource

from gammatools.core.histogram import *
import yaml
from gammatools.core.util import expand_aliases, eq2gal, interpolate2d


class Data(object):

    def __init__(self):
        self._data = {}

    def __getitem__(self,key):
        return self._data[key]

    def __setitem__(self,key,val):
        self._data[key] = val

    def save(self,outfile):

        import cPickle as pickle
        fp = open(outfile,'w')
        pickle.dump(self,fp,protocol = pickle.HIGHEST_PROTOCOL)
        fp.close()

    @staticmethod
    def load(infile):

        import cPickle as pickle
        return pickle.load(open(infile,'rb'))





class PhotonData(object):

    def __init__(self):
        self._data = { 'ra'            : np.array([]),
                       'dec'           : np.array([]),
                       'delta_ra'      : np.array([]),
                       'delta_dec'     : np.array([]),
                       'delta_phi'     : np.array([]),
                       'delta_theta'   : np.array([]),
                       'energy'        : np.array([]),
                       'time'          : np.array([]),
                       'psfcore'       : np.array([]),
                       'event_class'   : np.array([],dtype='int'),
                       'event_type'    : np.array([],dtype='int'),
                       'conversion_type'    : np.array([],dtype='int'),
                       'src_index'     : np.array([],dtype='int'),
                       'dtheta'        : np.array([]),
                       'phase'         : np.array([]),
                       'cth'           : np.array([]) }

        self._srcs = []

    def get_srcs(self,names):

        src_index = []
        srcs = []

        for i, s in enumerate(self._srcs):

            src = CatalogSource(s)
            for n in names:
                if src.match_name(n): 
                    src_index.append(i)
                    srcs.append(s)

        self._srcs = srcs

        mask = PhotonData.get_mask(self,src_index=src_index)
        self.apply_mask(mask)
    

    def merge(self,d):

        self._srcs = d._srcs
        
        for k, v in self._data.iteritems():
            self._data[k] = np.append(self._data[k],d._data[k])
        
    def append(self,col,d):
        self._data[col] = np.append(self._data[col],d)

    def __getitem__(self,col):
        return self._data[col]

    def __setitem__(self,col,val):
        self._data[col] = val

    def apply_mask(self,mask):

        for k in self._data.keys():
            self._data[k] = self._data[k][mask]
    
    def save(self,outfile):

        import cPickle as pickle
        fp = open(outfile,'w')
        pickle.dump(self,fp,protocol = pickle.HIGHEST_PROTOCOL)
        fp.close()

    def hist(self,var_name,mask=None,edges=None):
    
        h = Histogram(edges)    
        if not mask is None: h.fill(self._data[var_name][mask])
        else: h.fill(self._data[var_name])
        return h

    def mask(self,selections=None,conversion_type=None,
             event_class=None,
             event_class_id=None,
             event_type_id=None,
             phases=None,cuts=None,
             src_index=None,cuts_file=None):

        msk = PhotonData.get_mask(self,selections,conversion_type,event_class,
                                  event_class_id,event_type_id,phases,
                                  cuts,src_index,
                                  cuts_file)
        
        self.apply_mask(msk)
    
    @staticmethod
    def get_mask(data,selections=None,conversion_type=None,
                 event_class=None,
                 event_class_id=None,
                 event_type_id=None,
                 phases=None,cuts=None,
                 src_index=None,cuts_file=None):
        
        mask = data['energy'] > 0

        if not selections is None:
            for k, v in selections.iteritems():
                mask &= (data[k] >= v[0]) & (data[k] <= v[1])

#    mask = (data['energy'] >= egy_range[0]) & \
#        (data['energy'] <= egy_range[1]) & \
#        (data['cth'] >= cth_range[0]) & (data['cth'] <= cth_range[1]) \
        
        if not conversion_type is None:
            if conversion_type == 'front':
                mask &= (data['conversion_type'] == 0)
            else:
                mask &= (data['conversion_type'] == 1)
        
        if not cuts is None and not cuts_file is None:
            cut_defs = yaml.load(open(cuts_file,'r'))
            cut_defs['CTBBestLogEnergy'] = 'data[\'energy\']'
            cut_defs['CTBCORE'] = 'data[\'psfcore\']'
            cut_defs['pow'] = 'np.power'

            for c in cuts.split(','):
            
                cv = c.split('/')

                if len(cv) == 1:
                    cut_expr = expand_aliases(cut_defs,cv[0])
                    mask &= eval(cut_expr)
                else:
                    clo = float(cv[1])
                    chi = float(cv[2])

                if len(cv) == 3 and cv[0] in data._data:
                    mask &= (data[cv[0]] >= clo)&(data[cv[0]] <= chi)

        if not event_class_id is None:
            mask &= (data['event_class'].astype('int')&
                     ((0x1)<<event_class_id)>0)
        elif event_class == 'source':
            mask &= (data['event_class'].astype('int')&((0x1)<<2)>0)
        elif event_class == 'clean':
            mask &= (data['event_class'].astype('int')&((0x1)<<3)>0)
        elif event_class == 'ultraclean':
            mask &= (data['event_class'].astype('int')&((0x1)<<4)>0)

        if not event_type_id is None:

            print np.sum(mask)
            
            mask &= (data['event_type'].astype('int')&
                     ((0x1)<<event_type_id)>0)

            print np.sum(mask)
            
        if src_index is not None:

            src_mask = data['src_index'].astype('int') < 0
            for isrc in src_index: 
                src_mask |= (data['src_index'].astype('int') == int(isrc))

            mask &= src_mask

        if phases is not None:
            
            phase_mask = data['phase'] < 0
            for p in phases:            
                phase_mask |= ((data['phase'] > p[0]) & (data['phase'] < p[1]))
            mask &= phase_mask

        return mask

    @staticmethod
    def load(infile):

        import cPickle as pickle
        return pickle.load(open(infile,'rb'))
        

class QuantileData(object):

    def __init__(self,quantile,egy_nbin,cth_nbin):

        self.quantile = quantile
        self.label = 'r%2.f'%(quantile*100)
        self.egy_nbin = egy_nbin
        self.cth_nbin = cth_nbin
        self.mean = np.zeros(shape=(egy_nbin,cth_nbin))
        self.err = np.zeros(shape=(egy_nbin,cth_nbin))

    
        

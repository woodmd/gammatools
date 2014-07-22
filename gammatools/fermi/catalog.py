#!/usr/bin/env python

"""
@file  catalog.py

@brief Python classes for manipulating source catalogs.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood"
__date__     = "01/01/2013"
__date__     = "$Date: 2013/10/08 01:03:01 $"
__revision__ = "$Revision: 1.13 $, $Author: mdwood $"

import numpy as np
import sys
import pyfits
import os
import yaml
import copy
import re

import matplotlib.pyplot as plt

import gammatools
from gammatools.core.util import save_object, load_object, gal2eq, eq2gal
from gammatools.core.algebra import Vector3D


def latlon_to_xyz(lat,lon):
    phi = lon
    theta = np.pi/2.-lat
    return np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)]).T


class CatalogSource(object):

    def __init__(self,data):

        self.__dict__.update(data)

        self._cel_vec = Vector3D.createLatLon(np.radians(self.DEJ2000),
                                              np.radians(self.RAJ2000))

        self._gal_vec = Vector3D.createLatLon(np.radians(self.GLAT),
                                              np.radians(self.GLON))
        
        #[np.sin(theta)*np.cos(phi),
        #                     np.sin(theta)*np.sin(phi),
        #                     np.cos(theta)]
        
        self._names = []
        for k in Catalog.src_name_cols:

            name = self.__dict__[k].lower().replace(' ','')
            if name != '': 
                self._names.append(name)

    def names(self):
        return self._names

    def ra(self):
        return self.RAJ2000

    def dec(self):
        return self.DEJ2000
    
    def match_name(self,name):

        match_string = name.lower().replace(' ','')
        if name in self._names: return True
        else: return False

    def get_roi_cut(self,radius):

        dec = np.radians(self.DEJ2000)
        ra = np.radians(self.RAJ2000)

        cut = '(acos(sin(%.8f)*sin(FT1Dec*%.8f)'%(dec,np.pi/180.)
        cut += '+ cos(%.8f)'%(dec)
        cut += '*cos(FT1Dec*%.8f)'%(np.pi/180.)
        cut += '*cos(%.8f-FT1Ra*%.8f))'%(ra,np.pi/180.)
        cut += ' < %.8f)'%(np.radians(radius))
        return cut

    def __str__(self):

        s = 'Name: %s\n'%(self._names[0])
        s += 'GLON/GLAT: %f %f\n'%(self.GLON,self.GLAT)
        s += 'RA/DEC:    %f %f'%(self.RAJ2000,self.DEJ2000) 
        return s
        
    def __getitem__(self,k):
        return self.__dict__[k]


#        for k in Catalog.src_name_cols:

#            k = k.lower().replace(' ','')
#            if k == match_string: return True

#        return False


class Catalog(object):


    cache = {}
    
    catalog_files = { '2fgl' : os.path.join(gammatools.PACKAGE_ROOT,
                                            'data/gll_psc_v08.P.gz'),
                      '3fgl' : os.path.join(gammatools.PACKAGE_ROOT,
                                            'data/gll_psc4yearsource_v12r3_assoc_v6r6p0_flags.P.gz'),
                      }

    src_name_cols = ['Source_Name',
                     'ASSOC1','ASSOC2','ASSOC_GAM1','ASSOC_GAM2','ASSOC_TEV']

    def __init__(self):

        self._src_data = []
        self._src_index = {}
        self._src_radec = np.zeros(shape=(0,3))

    def get_source_by_name(self,name):

        if name in self._src_index:
            return self._src_data[self._src_index[name]]
        else:
            print 'No Source with name found: ', name
            sys.exit(1)

    def get_source_by_position(self,ra,dec,radius):
        
        x = latlon_to_xyz(np.radians(dec),np.radians(ra))
        costh = np.sum(x*self._src_radec,axis=1)        
        costh[costh>1.0] = 1.0
        msk = np.where(np.arccos(costh) < np.radians(radius))[0]

        srcs = [ self._src_data[i] for i in msk]
        return srcs

    def sources(self):
        return self._src_data

    @staticmethod
    def get(name='2fgl'):

        if not name in Catalog.cache:
            Catalog.cache[name] = Catalog.create(Catalog.catalog_files[name])

        return Catalog.cache[name]
    
    @staticmethod
    def create(filename):

        if re.search('\.fits$',filename):        
            return Catalog.create_from_fits(filename)
        elif re.search('(\.P|\.P\.gz)',filename):
            return load_object(filename)
        else:
            sys.exit(1)

    @staticmethod
    def create_from_fits(fitsfile=None):

        if fitsfile is None:
            fitsfile = os.path.join(gammatools.PACKAGE_ROOT,
                                    'data/gll_psc_v08.fit')

        cat = Catalog()
        hdulist = pyfits.open(fitsfile)
        cols = hdulist[1].columns.names

        nsrc = len(hdulist[1].data)

        cat._src_radec = np.zeros(shape=(nsrc,3))

        for i in range(nsrc):

            src = {}
            for icol, col in enumerate(cols):
                v = hdulist[1].data[i][icol]
                if type(v) == np.float32: src[col] = float(v)
                elif type(v) == str: src[col] = v
                elif type(v) == np.int16: src[col] = int(v)

            cat.load_source(src,i)

        return cat

    def load_source(self,src,i):
        src_name = src['Source_Name']

        self._src_data.append(src)
        phi = np.radians(src['RAJ2000'])
        theta = np.pi/2.-np.radians(src['DEJ2000'])

        self._src_radec[i] = [np.sin(theta)*np.cos(phi),
                             np.sin(theta)*np.sin(phi),
                             np.cos(theta)]

        for s in Catalog.src_name_cols:
            if s in src and src[s] != '':
                self._src_index[src[s]] = i
                self._src_index[src[s].replace(' ','')] = i

    def save(self,outfile,format='pickle'):

        if format == 'pickle': 
            save_object(self,outfile,compress=True)
        elif format == 'yaml': self.save_to_yaml(outfile)
        else:
            print 'Unrecognized output format: ', format
            sys.exit(1)

    def save_to_pickle(self,outfile):

        import cPickle as pickle
        fp = open(outfile,'w')
        pickle.dump(self,fp,protocol = pickle.HIGHEST_PROTOCOL)
        fp.close()

    def plot(self,im,src_color='k',marker_threshold=0,
             label_threshold=20., ax=None,**kwargs):

        if ax is None: ax = plt.gca()
        
        if im.axis(0)._coordsys == 'gal':
            ra, dec = gal2eq(im.lon,im.lat)
        else:
            ra, dec = im.lon, im.lat

        #srcs = cat.get_source_by_position(ra,dec,self._roi_radius_deg)
        srcs = self.get_source_by_position(ra,dec,10.0)

        src_lon = []
        src_lat = []

        labels = []
        signif_avg = []
        
        for s in srcs:
            
#            print s['RAJ2000'], s['DEJ2000'], s['GLON'], s['GLAT']
            src_lon.append(s['RAJ2000'])
            src_lat.append(s['DEJ2000'])
            labels.append(s['Source_Name'])
            signif_avg.append(s['Signif_Avg'])
        
        if im.axis(0)._coordsys == 'gal':
            src_lon, src_lat = eq2gal(src_lon,src_lat)
            
            
        pixcrd = im.wcs.wcs_sky2pix(src_lon,src_lat, 0)

        for i in range(len(labels)):

            if signif_avg[i] > label_threshold:             
                plt.gca().text(pixcrd[0][i]+2.0,pixcrd[1][i]+2.0,labels[i],
                               color=src_color,size=8,clip_on=True)

            if signif_avg[i] > marker_threshold:      
#                print i, pixcrd[0][i],pixcrd[1][i]      
                plt.gca().plot(pixcrd[0][i],pixcrd[1][i],
                               linestyle='None',marker='+',
                               color='g', markerfacecolor = 'None',
                               markeredgecolor=src_color,clip_on=True)
        
        plt.gca().set_xlim(im.axis(0).lo_edge(),im.axis(0).hi_edge())
        plt.gca().set_ylim(im.axis(1).lo_edge(),im.axis(1).hi_edge())
        
    def save_to_yaml(self,outfile):

        print 'Saving catalog ', outfile

        yaml.dump({ 'src_data' : self._src_data, 
                    'src_name_index' : self._src_index,
                    'src_radec' : self._src_radec },
                  file(outfile,'w'))

    def load_from_yaml(self,infile):

        print 'Loading catalog', infile

        d = yaml.load(file(infile,'r'))
        
        self._src_data = d['src_data']
        self._src_index = d['src_name_index']
        self._src_radec = d['src_radec']

        
            


SourceCatalog = { 'vela' :     (128.83606354, -45.17643181),
                  'vela2' :    (-45.17643181, 128.83606354),
                  'geminga' :  (98.475638,    17.770253),
                  'crab' :     (83.63313,     22.01447),
                  'draco' :    (260.05163,     57.91536),
                  'slat+100' : (0.0, 90.00),
                  'slat+090' : (0.0, 64.16),
                  'slat+080' : (0.0, 53.13),
                  'slat+060' : (0.0, 36.87),
                  'slat+040' : (0.0, 23.58),
                  'slat+020' : (0.0, 11.54),
                  'slat+000' : (0.0,  0.00),
                  'slat-020' : (0.0,-11.54),
                  'slat-040' : (0.0,-23.58),
                  'slat-060' : (0.0,-36.87),
                  'slat-080' : (0.0,-53.13),
                  'slat-090' : (0.0,-64.16),
                  'slat-100' : (0.0,-90.00) }


if __name__ == '__main__':

    import argparse
    import re

    usage = "usage: %(prog)s [options] [catalog FITS file]"
    description = "Load a FITS catalog and write to an output file."
    parser = argparse.ArgumentParser(usage=usage,description=description)

    parser.add_argument('files', nargs='+')
    
    parser.add_argument('--output', default = None, 
                        help = 'Output file')
    
    parser.add_argument('--source', default = None, 
                        help = 'Output file')
    
    parser.add_argument('--roi_radius', default = 10., type=float,
                        help = 'Output file')

    args = parser.parse_args()


    if len(args.files) == 1:           
        cat = Catalog.create_from_fits(args.files[0])
    else:
        cat = Catalog()

#    if not opts.source is None:
#        src = CatalogSource(cat.get_source_by_name(opts.source))

    if not args.output is None:

        print 
        
        if re.search('\.P$',args.output):
            save_object(cat,args.output,compress=True)

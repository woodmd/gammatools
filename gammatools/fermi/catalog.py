#!/usr/bin/env python

"""
@file  catalog.py

@brief Python classes for manipulating source catalogs.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""
__source__   = "$Source: /nfs/slac/g/glast/ground/cvs/users/mdwood/python/catalog.py,v $"
__author__   = "Matthew Wood"
__date__     = "01/01/2013"
__date__     = "$Date: 2013/10/08 01:03:01 $"
__revision__ = "$Revision: 1.13 $, $Author: mdwood $"

import numpy as np
import sys
import pyfits
import os
#from algebra import Vector3D
import yaml
import copy

def latlon_to_xyz(lat,lon):
    phi = lon
    theta = np.pi/2.-lat
    return [np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)]


class CatalogSource(object):

    def __init__(self,data):

        self.__dict__.update(data)

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

        cut = 'acos(sin(%.8f)*sin(FT1Dec*%.8f)'%(dec,np.pi/180.)
        cut += '+ cos(%.8f)'%(dec)
        cut += '*cos(FT1Dec*%.8f)*'%(np.pi/180.)
        cut += '*cos(%.8f-FT1Ra*%.8f))'%(ra,np.pi/180.)
        cut += ' < %.8f'%(np.radians(radius))
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

    src_name_cols = ['Source_Name',
                     'ASSOC1','ASSOC2','ASSOC_GAM1','ASSOC_GAM2','ASSOC_TEV']

    def __init__(self,infile=None):

        if infile is None:
            dirname = os.path.dirname(os.path.realpath(__file__))
            infile = os.path.join(dirname,'2fgl_catalog_v08.P')
            
        self._src_data = []
        self._src_index = {}
        self._src_radec = []

        if not infile is None: self.load(infile)

    def get_source_by_name(self,name):

        if name in self._src_index:
            return self._src_data[self._src_index[name]]
        else:
            print 'No Source with name found: ', name
            sys.exit(1)

    def get_source_by_radec(self,ra,dec,radius):

        x = latlon_to_xyz(np.radians(dec),np.radians(ra))

        costh = np.sum(x*self._src_radec,axis=1)        
        costh[costh>1.0] = 1.0
        msk = np.where(np.arccos(costh) < np.radians(radius))[0]

        srcs = [ self._src_data[i] for i in msk]
        return srcs


    @staticmethod
    def load_from_fits(fitsfile):

        cat = Catalog()

        print 'Loading from FITS catalog'

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

        if format == 'pickle': self.save_to_pickle(outfile)
        elif format == 'yaml': self.save_to_yaml(outfile)
        else:
            print 'Unrecognized output format: ', format
            sys.exit(1)

    def save_to_pickle(self,outfile):

        import cPickle as pickle
        fp = open(outfile,'w')
        pickle.dump(self,fp,protocol = pickle.HIGHEST_PROTOCOL)
        fp.close()

    def load(self,infile):

        import cPickle as pickle
        c = pickle.load(open(infile,'rb'))
        
        self._src_data = c._src_data
        self._src_index = c._src_index
        self._src_radec = c._src_radec

        for i, s in enumerate(self._src_data):
            src = CatalogSource(self._src_data[i])            
            for n in src.names(): self._src_index[n] = i


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

    from optparse import Option
    from optparse import OptionParser

    usage = "usage: %prog [options] [catalog file]"
    description = "Load a FITS catalog and write to an output file."
    parser = OptionParser(usage=usage,description=description)

    parser.add_option('--output', default = None, type='string',
                      help = 'Output file')

    parser.add_option('--source', default = None, type='string',
                      help = 'Output file')

    parser.add_option('--roi_radius', default = 10., type='float',
                      help = 'Output file')
    
    parser.add_option('--format', default = 'pickle', 
                      choices=['pickle','yaml'],
                      help = 'Set output file format.')

    (opts, args) = parser.parse_args()


    if len(args) == 1:           
        cat = Catalog.load_from_fits(sys.argv[-1])
    else:
        cat = Catalog()


    if not opts.source is None:
        src = CatalogSource(cat.get_source_by_name(opts.source))

        print src
        print src.get_roi_cut(opts.roi_radius)

    if not opts.output is None:
        cat.save(opts.output,opts.format)

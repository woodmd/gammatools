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
import os
import yaml
import copy
import re

import xml.etree.cElementTree as et

from gammatools.core.astropy_helper import pyfits
import matplotlib.pyplot as plt

import gammatools
from gammatools.core.util import prettify_xml
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

        self._names_dict = {}
        self._cel_vec = Vector3D.createLatLon(np.radians(self.DEJ2000),
                                              np.radians(self.RAJ2000))

        self._gal_vec = Vector3D.createLatLon(np.radians(self.GLAT),
                                              np.radians(self.GLON))
        
        #[np.sin(theta)*np.cos(phi),
        #                     np.sin(theta)*np.sin(phi),
        #                     np.cos(theta)]
        
        self._names = []
        for k in Catalog.src_name_cols:

            if not k in self.__dict__: continue

            name = self.__dict__[k].strip()
            if name != '':  self._names.append(name)

            self._names_dict[k] = name
            
#            name = self.__dict__[k].lower().replace(' ','')
#            if name != '': 


    def names(self):
        return self._names

    def get_name(self,key=None):

        if key is None:
            return self._names[0]
        else: 
            return self._names_dict[key]

    @property
    def name(self):
        return self._names[0]

    @property
    def ra(self):
        return self.RAJ2000

    @property
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

        if not k in self.__dict__: return None
        else: return self.__dict__[k]


#        for k in Catalog.src_name_cols:

#            k = k.lower().replace(' ','')
#            if k == match_string: return True

#        return False

def create_xml_element(root,name,attrib):
    el = et.SubElement(root,name)
    for k, v in attrib.iteritems(): el.set(k,v)
    return el

class Catalog(object):

    cache = {}
    
    catalog_files = { '2fgl' : os.path.join(gammatools.PACKAGE_ROOT,
                                            'data/gll_psc_v08.fit'),
                      '1fhl' : os.path.join(gammatools.PACKAGE_ROOT,
                                            'data/gll_psch_v07.fit'),
                      '3fgl' : os.path.join(gammatools.PACKAGE_ROOT,
                                            'data/gll_psc_v11.fit'),
                      '3fglp' : os.path.join(gammatools.PACKAGE_ROOT,
                                            'data/gll_psc4yearsource_v12r3_assoc_v6r6p0_flags.fit'),
                      }

    src_name_cols = ['Source_Name',
                     'ASSOC1','ASSOC2','ASSOC_GAM','1FHL_Name','2FGL_Name',
                     'ASSOC_GAM1','ASSOC_GAM2','ASSOC_TEV']

    def __init__(self):

        self._src_data = []
        self._src_index = {}
        self._src_radec = np.zeros(shape=(0,3))

    def get_source_by_name(self,name):

        if name in self._src_index:
            return self._src_data[self._src_index[name]]
        else:
            return None

    def get_source_by_position(self,ra,dec,radius,min_radius=None):
        
        x = latlon_to_xyz(np.radians(dec),np.radians(ra))
        costh = np.sum(x*self._src_radec,axis=1)        
        costh[costh>1.0] = 1.0

        if min_radius is not None:
            msk = np.where((np.arccos(costh) < np.radians(radius)) &
                           (np.arccos(costh) > np.radians(min_radius)))[0]
        else:
            msk = np.where(np.arccos(costh) < np.radians(radius))[0]

        srcs = [ self._src_data[i] for i in msk]
        return srcs

    def sources(self):
        return self._src_data

    def create_roi(self,ra,dec,isodiff,galdiff,xmlfile,radius=180.0):

        root = et.Element('source_library')
        root.set('title','source_library')

        srcs = self.get_source_by_position(ra,dec,radius)

        for s in srcs:
            
            source_element = create_xml_element(root,'source',
                                                dict(name=s['Source_Name'],
                                                     type='PointSource'))

            spec_element = et.SubElement(source_element,'spectrum')

            stype = s['SpectrumType'].strip()            
            spec_element.set('type',stype)

            if stype == 'PowerLaw':
                Catalog.create_powerlaw(s,spec_element)
            elif stype == 'LogParabola':
                Catalog.create_logparabola(s,spec_element)
            elif stype == 'PLSuperExpCutoff':
                Catalog.create_plsuperexpcutoff(s,spec_element)
                
            spat_el = et.SubElement(source_element,'spatialModel')
            spat_el.set('type','SkyDirFunction')

            create_xml_element(spat_el,'parameter',
                               dict(name = 'RA',
                                    value = str(s['RAJ2000']),
                                    free='0',
                                    min='-360.0',
                                    max='360.0',
                                    scale='1.0'))

            create_xml_element(spat_el,'parameter',
                               dict(name = 'DEC',
                                    value = str(s['DEJ2000']),
                                    free='0',
                                    min='-90.0',
                                    max='90.0',
                                    scale='1.0'))
            
        isodiff_el = Catalog.create_isotropic(root,isodiff)
        galdiff_el = Catalog.create_galactic(root,galdiff)
        
        output_file = open(xmlfile,'w')
        output_file.write(prettify_xml(root))

    @staticmethod
    def create_isotropic(root,filefunction,name='isodiff'):

        el = create_xml_element(root,'source',
                                dict(name=name,
                                     type='DiffuseSource'))
        
        spec_el = create_xml_element(el,'spectrum',
                                     dict(file=filefunction,
                                          type='FileFunction',
                                          ctype='-1'))

        create_xml_element(spec_el,'parameter',
                           dict(name='Normalization',
                                value='1.0',
                                free='1',
                                max='10000.0',
                                min='0.0001',
                                scale='1.0'))
                        
        spat_el = create_xml_element(el,'spatialModel',
                                     dict(type='ConstantValue'))

        create_xml_element(spat_el,'parameter',
                           dict(name='Value',
                                value='1.0',
                                free='0',
                                max='10.0',
                                min='0.0',
                                scale='1.0'))

        return el

    @staticmethod
    def create_galactic(root,mapcube,name='galdiff'):

        el = create_xml_element(root,'source',
                                dict(name=name,
                                     type='DiffuseSource'))

        spec_el = create_xml_element(el,'spectrum',
                                     dict(type='PowerLaw'))
        
                
        create_xml_element(spec_el,'parameter',
                           dict(name='Prefactor',
                                value='1.0',
                                free='1',
                                max='10.0',
                                min='0.1',
                                scale='1.0'))
        
        create_xml_element(spec_el,'parameter',
                           dict(name='Index',
                                value='0.0',
                                free='0',
                                max='1.0',
                                min='-1.0',
                                scale='-1.0'))

        create_xml_element(spec_el,'parameter',
                           dict(name='Scale',
                                value='1000.0',
                                free='0',
                                max='1000.0',
                                min='1000.0',
                                scale='1.0'))

        spat_el = create_xml_element(el,'spatialModel',
                                     dict(type='MapCubeFunction',
                                          file=mapcube))
                
        create_xml_element(spat_el,'parameter',
                           dict(name='Normalization',
                                value='1.0',
                                free='0',
                                max='1E3',
                                min='1E-3',
                                scale='1.0'))

        return el
    
        
    @staticmethod
    def create_powerlaw(src,root):

        if src['Flux_Density'] > 0:        
            scale = np.round(np.log10(1./src['Flux_Density']))
        else:
            scale = 0.0
            
        value = src['Flux_Density']*10**scale
                
        create_xml_element(root,'parameter',
                           dict(name='Prefactor',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(value),
                                scale=str(10**-scale)))

        create_xml_element(root,'parameter',
                           dict(name='Index',
                                free='0',
                                min='-5.0',
                                max='5.0',
                                value=str(src['Spectral_Index']),
                                scale=str(-1.0)))
        
        create_xml_element(root,'parameter',
                           dict(name='Scale',
                                free='0',
                                min=str(src['Pivot_Energy']),
                                max=str(src['Pivot_Energy']),
                                value=str(src['Pivot_Energy']),
                                scale=str(1.0)))

    @staticmethod
    def create_logparabola(src,root):

        norm_scale = np.round(np.log10(1./src['Flux_Density']))
        norm_value = src['Flux_Density']*10**norm_scale

        eb_scale = np.round(np.log10(1./src['Pivot_Energy']))
        eb_value = src['Pivot_Energy']*10**eb_scale
        
        create_xml_element(root,'parameter',
                           dict(name='norm',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(norm_value),
                                scale=str(10**-norm_scale)))

        create_xml_element(root,'parameter',
                           dict(name='alpha',
                                free='0',
                                min='-5.0',
                                max='5.0',
                                value=str(src['Spectral_Index']),
                                scale=str(1.0)))

        create_xml_element(root,'parameter',
                           dict(name='beta',
                                free='0',
                                min='0.0',
                                max='5.0',
                                value=str(src['beta']),
                                scale=str(1.0)))

        
        create_xml_element(root,'parameter',
                           dict(name='Eb',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(eb_value),
                                scale=str(10**-eb_scale)))
        
    @staticmethod
    def create_plsuperexpcutoff(src,root):

        norm_scale = np.round(np.log10(1./src['Flux_Density']))
        norm_value = src['Flux_Density']*10**norm_scale

        eb_scale = np.round(np.log10(1./src['Pivot_Energy']))
        eb_value = src['Pivot_Energy']*10**eb_scale
        
        create_xml_element(root,'parameter',
                           dict(name='norm',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(norm_value),
                                scale=str(10**-norm_scale)))

        create_xml_element(root,'parameter',
                           dict(name='alpha',
                                free='0',
                                min='-5.0',
                                max='5.0',
                                value=str(src['Spectral_Index']),
                                scale=str(1.0)))

        create_xml_element(root,'parameter',
                           dict(name='beta',
                                free='0',
                                min='0.0',
                                max='5.0',
                                value=str(src['beta']),
                                scale=str(1.0)))
        
        create_xml_element(root,'parameter',
                           dict(name='Eb',
                                free='0',
                                min='0.01',
                                max='100.0',
                                value=str(eb_value),
                                scale=str(10**-eb_scale)))
        
    @staticmethod
    def get(name='2fgl'):

        if not name in Catalog.cache:

            filename = Catalog.catalog_files[name]

            try:
                Catalog.cache[name] = Catalog.create(filename)
            except Exception, message:

                print 'Exception ', message
                # Retry loading fits
                m = re.search('(.+)(\.P|\.P\.gz)',filename)
                if m:
                    fits_path = m.group(1) + '.fit'
                    Catalog.cache[name] = Catalog.create(fits_path)

        return Catalog.cache[name]
    
    @staticmethod
    def create(filename):

        if re.search('\.fits$',filename) or re.search('\.fit$',filename):
            return Catalog.create_from_fits(filename)
        elif re.search('(\.P|\.P\.gz)',filename):
            return load_object(filename)
        else:
            raise Exception("Unrecognized suffix in catalog file: %s"%(filename))

    @staticmethod
    def create_from_fits(fitsfile):

        cat = Catalog()
        hdulist = pyfits.open(fitsfile)
        table = hdulist[1]

        cols = {}
        for icol, col in enumerate(table.columns.names):

            col_data = hdulist[1].data[col]
            if type(col_data[0]) == np.float32: 
                cols[col] = np.array(col_data,dtype=float)
            elif type(col_data[0]) == str: 
                cols[col] = np.array(col_data,dtype=str)
            elif type(col_data[0]) == np.int16: 
                cols[col] = np.array(col_data,dtype=int)

        nsrc = len(hdulist[1].data)

        cat._src_radec = np.zeros(shape=(nsrc,3))

        for i in range(nsrc):

            src = {}
            for icol, col in enumerate(cols):
                if not col in cols: continue
                src[col] = cols[col][i]

            src['Source_Name'] = src['Source_Name'].strip()                
            cat.load_source(CatalogSource(src))

        return cat
    
    def load_source(self,src):
        src_name = src['Source_Name']
        src_index = len(self._src_data)

        self._src_data.append(src)
        phi = np.radians(src['RAJ2000'])
        theta = np.pi/2.-np.radians(src['DEJ2000'])

        self._src_radec[src_index] = [np.sin(theta)*np.cos(phi),
                                      np.sin(theta)*np.sin(phi),
                                      np.cos(theta)]

        for s in Catalog.src_name_cols:
            if s in src.__dict__ and src[s] != '':

                name = src[s].strip()
                self._src_index[name] = src_index
                self._src_index[name.replace(' ','')] = src_index
                self._src_index[name.replace(' ','').lower()] = src_index

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
             label_threshold=20., ax=None,radius_deg=10.0,
             **kwargs):

        if ax is None: ax = plt.gca()
        min_radius_deg = kwargs.get('min_radius_deg',None)
        fontweight = kwargs.get('fontweight','normal')
        
        if im.axis(0)._coordsys == 'gal':
            ra, dec = gal2eq(im.lon,im.lat)
        else:
            ra, dec = im.lon, im.lat

        #srcs = cat.get_source_by_position(ra,dec,self._roi_radius_deg)
        # Try to determine the search radius from the input file
        srcs = self.get_source_by_position(ra,dec,radius_deg,
                                           min_radius=min_radius_deg)

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
            
            
#        pixcrd = im.wcs.wcs_sky2pix(src_lon,src_lat, 0)
        pixcrd = im.wcs.wcs_world2pix(src_lon,src_lat, 0)

#        ax.autoscale(enable=False, axis='both')
#        ax.set_autoscale_on(False)

        for i in range(len(labels)):

            scale = (min(max(signif_avg[i],5.0),50.0)-5.0)/45.
            
            mew = 1.0 + 1.0*scale
            ms = 5.0 + 3.0*scale
            
            if label_threshold is not None and signif_avg[i] > label_threshold:

                ax.text(pixcrd[0][i]+2.0,pixcrd[1][i]+2.0,labels[i],
                        color=src_color,size=8,clip_on=True,
                        fontweight=fontweight)

            if marker_threshold is not None and \
                    signif_avg[i] > marker_threshold:      
                ax.plot(pixcrd[0][i],pixcrd[1][i],
                        linestyle='None',marker='+',
                        color='g', markerfacecolor = 'None',mew=mew,ms=ms,
                        markeredgecolor=src_color,clip_on=True)
        
        plt.gca().set_xlim(im.axis(0).lims())
        plt.gca().set_ylim(im.axis(1).lims())
        
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
        
        if re.search('\.P$',args.output):
            save_object(cat,args.output,compress=True)

import os
import sys
import copy
import argparse

from gammatools.core.astropy_helper import pyfits

import numpy as np
from gammatools.core.algebra import Vector3D
from gammatools.fermi.catalog import Catalog
from gammatools.core.util import separation_angle, dispatch_jobs
from gammatools.core.util import bitarray_to_int
from gammatools.core.util import save_object
from gammatools.fermi.data import PhotonData
from gammatools.core.config import *

class FT1Loader(Configurable):

    default_config = {
        'zmax'            : (None),
        'conversion_type' : (None),
        'event_class_id'  : (None),
        'event_type_id'   : (None),
        'max_dist'        : (None),
        'max_events'      : (None),
        'phase_selection' : (None),
        'erange'          : (None),
        'ft2file'         : (None),
        'src_list'        : (None)
        }
    
    def __init__(self,config,opts=None,**kwargs):
        super(FT1Loader,self).__init__()

        self.configure(config,opts=opts,**kwargs)

        if self.config['ft2file'] is not None:
            self.setFT2File(self.config['ft2file'])
        
        if self.config['src_list'] is not None:
            self.loadsrclist(self.config['src_list'])

        self._phist = None
        self._photon_data = PhotonData()

    def setFT2File(self,ft2file):

        import pointlike
        if ft2file is None:
            return        
        elif not os.path.isfile(ft2file):
            print 'Error invalid FT2 file: ', ft2file
            sys.exit(1)
        
        self._phist = pointlike.PointingHistory(ft2file)
        
    def load_photons(self,ft1file):

        hdulist = pyfits.open(ft1file)

        evhdu = hdulist['EVENTS']
        
        if self.config['max_events'] is not None:
            table = evhdu.data[0:self.config['max_events']]
        else:
            table = evhdu.data
        
        msk = table.field('ZENITH_ANGLE')<self.config['zmax']

        if not self.config['event_class_id'] is None:
            event_class = bitarray_to_int(table.field('EVENT_CLASS'),True)
            msk &= (event_class&((0x1)<<int(self.config['event_class_id']))>0)

        if not self.config['event_type_id'] is None:
            event_type = bitarray_to_int(table.field('EVENT_TYPE'),True)
            msk &= (event_type&((0x1)<<int(self.config['event_type_id']))>0)
            
        table = table[msk]
        
        if self.config['erange'] is not None:

            erange = [float(t) for t in self.config['erange'].split('/')]
            
            msk = ((np.log10(table.field('ENERGY')) > erange[0]) &
                   (np.log10(table.field('ENERGY')) < erange[1]))
            table = table[msk]
                            
        if self.config['conversion_type'] is not None:

            if self.config['conversion_type'] == 'front':
                msk = table.field('CONVERSION_TYPE') == 0
            else:
                msk = table.field('CONVERSION_TYPE') == 1

            table = table[msk]
                
        if self.config['phase_selection'] is not None:
            msk = table.field('PULSE_PHASE')<0
            phases = self.config['phase_selection'].split(',')
            for p in phases:
                (plo,phi) = p.split('/')
                msk |= ((table.field('PULSE_PHASE')>float(plo)) & 
                        (table.field('PULSE_PHASE')<float(phi)))

            table = table[msk]
            
            
        nevent = len(table)
        
        print 'Loading ', ft1file, ' nevent: ', nevent

        pd = self._photon_data
        
        for isrc, (src_ra,src_dec) in enumerate(zip(self.src_ra_deg,self.src_dec_deg)):

            vsrc = Vector3D.createLatLon(np.radians(src_dec),
                                         np.radians(src_ra))
            
            msk = ((table.field('DEC')>(src_dec-self.config['max_dist'])) & 
                   (table.field('DEC')<(src_dec+self.config['max_dist'])))
            table_src = table[msk]

            vra = np.array(table_src.field('RA'),dtype=float)
            vdec = np.array(table_src.field('DEC'),dtype=float)
            
            vptz_ra = np.array(table_src.field('PtRaz'),dtype=float)
            vptz_dec = np.array(table_src.field('PtDecz'),dtype=float)
            
            dth = separation_angle(self.src_radec[isrc][0],
                                   self.src_radec[isrc][1],
                                   np.radians(vra),
                                   np.radians(vdec))

            msk = dth < np.radians(self.config['max_dist'])
            table_src = table_src[msk]
            vra = vra[msk]
            vdec = vdec[msk]
            vptz_ra = vptz_ra[msk]
            vptz_dec = vptz_dec[msk]
            
            
            veq = Vector3D.createLatLon(np.radians(vdec),
                                        np.radians(vra))

            eptz = Vector3D.createLatLon(np.radians(vptz_dec),
                                         np.radians(vptz_ra))
            

            vp = veq.project2d(vsrc)
            vx = np.degrees(vp.theta()*np.sin(vp.phi()))
            vy = -np.degrees(vp.theta()*np.cos(vp.phi()))            

            vptz = eptz.project2d(vsrc)

#            print vptz.phi()

            vp2 = copy.deepcopy(vp)
            vp2.rotatez(-vptz.phi())

            vx2 = np.degrees(vp2.theta()*np.sin(vp2.phi()))
            vy2 = -np.degrees(vp2.theta()*np.cos(vp2.phi()))  

#            import matplotlib.pyplot as plt

#            print vp.theta()[:10]
#            print vp2.theta()[:10]
            
#            print np.sqrt(vx**2+vy**2)[:10]
#            print np.sqrt(vx2**2+vy2**2)[:10]
            
            
#            plt.figure()
#            plt.plot(vx2,vy2,marker='o',linestyle='None')

#            plt.gca().set_xlim(-80,80)
#            plt.gca().set_ylim(-80,80)
            
#            plt.figure()
#            plt.plot(vx,vy,marker='o',linestyle='None')
            
#            plt.show()

            
#            vx2 = np.degrees(vp.theta()*np.sin(vp.phi()))
#            vy2 = -np.degrees(vp.theta()*np.cos(vp.phi()))   
                        
            src_index = np.zeros(len(table_src),dtype=int)
            src_index[:] = isrc

            src_phase = np.zeros(len(table_src))
            if 'PULSE_PHASE' in evhdu.columns.names:
                src_phase = list(table_src.field('PULSE_PHASE'))

            psf_core = np.zeros(len(table_src))
            if 'CTBCORE' in evhdu.columns.names:
                psf_core = list(table_src.field('CTBCORE'))

            event_type = np.zeros(len(table_src),dtype='int')
            if 'EVENT_TYPE' in evhdu.columns.names:
                event_type = bitarray_to_int(table_src.field('EVENT_TYPE'),True)
            
            event_class = bitarray_to_int(table_src.field('EVENT_CLASS'),True)
                
            pd.append('psfcore',psf_core)
            pd.append('time',list(table_src.field('TIME')))
            pd.append('ra',list(table_src.field('RA')))
            pd.append('dec',list(table_src.field('DEC')))
            pd.append('delta_ra',list(vx))
            pd.append('delta_dec',list(vy))
            pd.append('delta_phi',list(vx2))
            pd.append('delta_theta',list(vy2))            
            pd.append('energy',list(np.log10(table_src.field('ENERGY'))))
            pd.append('dtheta',list(dth[msk]))
            pd.append('event_class',list(event_class))
            pd.append('event_type',list(event_type))
            pd.append('conversion_type',
                      list(table_src.field('CONVERSION_TYPE').astype('int')))
            pd.append('src_index',list(src_index))            
            pd.append('phase',list(src_phase))

            cthv = []
            
            for k in range(len(table_src)):

#                event = table_src[k]
#                ra = float(event.field('RA'))
#                dec = float(event.field('DEC'))
#                sd = skymaps.SkyDir(ra,dec)  
#                event = table_src[k]
#                theta = float(event.field('THETA'))*deg2rad
#                time = event.field('TIME')

                if self._phist is not None: 
                    import skymaps
                    sd = skymaps.SkyDir(src_ra,src_dec)
                    pi = self._phist(table_src.field('TIME')[k])            
                    cth = np.cos(pi.zAxis().difference(src))
                    cthv.append(cth)
                else:
                    cthv.append(np.cos(np.radians(table_src.field('THETA'))))

            pd.append('cth',cthv)


#        print 'Loaded ', len(self.dtheta), ' events'
                    
        hdulist.close()

    def loadsrclist(self,fname):
        self.srcs = []
        self.src_names = []
        self.src_redshifts = []
        self.src_ra_deg = []
        self.src_dec_deg = []
        self.src_radec = []

#        if not fname is None:
#            src_names = np.genfromtxt(fname,unpack=True,dtype=None)
#        else:
#            src_names = np.array(srcs.split(','))

        if isinstance(fname,list):
            src_names = np.array(fname)
            
        if src_names.ndim == 0: src_names = src_names.reshape(1)

        cat = Catalog.get()
        
        for name in src_names:
            src = cat.get_source_by_name(name)
            name = src['Source_Name']
            ra = src['RAJ2000']
            dec = src['DEJ2000']
#            self._photon_data._srcs.append(src)
#            self.src_skydirs.append(sd)
            self.src_names.append(name)
            self.src_ra_deg.append(ra)
            self.src_dec_deg.append(dec)
            self.src_radec.append((np.radians(ra),np.radians(dec)))
            
    def save(self,fname):
        save_object(self._photon_data,fname,compress=True)
        #        self._photon_data.save(fname)

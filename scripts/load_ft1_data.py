#!/usr/bin/env python


import os
import sys
import copy
from optparse import Option
from optparse import OptionParser


import pyfits
import skymaps
import pointlike
import numpy as np
from algebra import Vector3D
from catalog import Catalog
from util import separation_angle, dispatch_jobs
from data import PhotonData

class FT1Loader(object):

    def __init__(self, zenith_cut, conversion_type, event_class_id,
                 max_events, max_dist_deg, phase_selection,erange):
        self.zenith_cut = zenith_cut
        self.max_events = max_events
        self.max_dist = np.radians(max_dist_deg)
        self._phist = None
        self.phase_selection = phase_selection
        self.conversion_type = conversion_type
        self.event_class_id = event_class_id
        self.erange = erange

        self._photon_data = PhotonData()


    def setFT2File(self,ft2file):

        if ft2file is None:
            return        
        elif not os.path.isfile(ft2file):
            print 'Error invalid FT2 file: ', ft2file
            sys.exit(1)
        
        self._phist = pointlike.PointingHistory(ft2file)
        
    def load_photons(self,fname,ft2file=None):

        if ft2file is not None: setFT2File(ft2file)

        hdulist = pyfits.open(fname)
#        hdulist.info()
#        print hdulist[1].columns.names
        
        if self.max_events is not None:
            table = hdulist[1].data[0:self.max_events]
        else:
            table = hdulist[1].data
        
        msk = table.field('ZENITH_ANGLE')<self.zenith_cut

        if not self.event_class_id is None:
            msk &= (table.field('EVENT_CLASS').astype('int')&
                    ((0x1)<<int(self.event_class_id))>0)

        table = table[msk]


        
        if self.erange is not None:

            erange = [float(t) for t in self.erange.split('/')]
            
            msk = ((np.log10(table.field('ENERGY')) > erange[0]) &
                   (np.log10(table.field('ENERGY')) < erange[1]))
            table = table[msk]
                            
        if self.conversion_type is not None:

            if self.conversion_type == 'front':
                msk = table.field('CONVERSION_TYPE') == 0
            else:
                msk = table.field('CONVERSION_TYPE') == 1

            table = table[msk]
                
#        if 'PULSE_PHASE' in hdulist[1].columns.names and \
#                self.phase_selection == 'on':
#            msk = ((table.field('PULSE_PHASE')>0.1) & 
#                   (table.field('PULSE_PHASE')<0.15)) | \
#                   ((table.field('PULSE_PHASE')>0.52) & 
#                    (table.field('PULSE_PHASE')<0.57)) 
#            table = table[msk]
#        elif 'PULSE_PHASE' in hdulist[1].columns.names and \
#                self.phase_selection == 'off':
#            msk = (table.field('PULSE_PHASE')>0.7)
#            table = table[msk]
        if self.phase_selection is not None:
            msk = table.field('PULSE_PHASE')<0
            phases = self.phase_selection.split(',')
            for p in phases:
                (plo,phi) = p.split('/')
                msk |= ((table.field('PULSE_PHASE')>float(plo)) & 
                        (table.field('PULSE_PHASE')<float(phi)))

            table = table[msk]
            
            
        nevent = len(table)

        
        
        print 'Loading ', fname, ' nevent: ', nevent

        pd = self._photon_data
        
        for isrc, src in enumerate(self.srcs):

            vsrc = Vector3D.createLatLon(np.radians(src.dec()),
                                         np.radians(src.ra()))
            
#            print 'Source ', isrc
            msk = (table.field('DEC')>(src.dec()-np.degrees(self.max_dist))) & \
                (table.field('DEC')<(src.dec()+np.degrees(self.max_dist)))
            table_src = table[msk]

            vra = np.array(table_src.field('RA'),dtype=float)
            vdec = np.array(table_src.field('DEC'),dtype=float)

            dth = separation_angle(self.src_radec[isrc][0],
                                   self.src_radec[isrc][1],
                                   np.radians(vra),
                                   np.radians(vdec))

            msk = dth < self.max_dist
            table_src = table_src[msk]
            vra = vra[msk]
            vdec = vdec[msk]

            veq = Vector3D.createLatLon(np.radians(vdec),
                                        np.radians(vra))

            vp = veq.project(vsrc)

            vx = np.degrees(vp.theta()*np.sin(vp.phi()))
            vy = -np.degrees(vp.theta()*np.cos(vp.phi()))            
            
            src_index = np.zeros(len(table_src),dtype=int)
            src_index[:] = isrc

#            src_redshift = np.zeros(len(table_src))
#            src_redshift[:] = self.src_redshifts[isrc]
#            self.redshift += list(src_redshift)

            src_phase = np.zeros(len(table_src))
            if 'PULSE_PHASE' in hdulist[1].columns.names:
                src_phase = list(table_src.field('PULSE_PHASE'))

            psf_core = np.zeros(len(table_src))
            if 'CTBCORE' in hdulist[1].columns.names:
                psf_core = list(table_src.field('CTBCORE'))

            pd.append('psfcore',psf_core)
            pd.append('time',list(table_src.field('TIME')))
            pd.append('ra',list(table_src.field('RA')))
            pd.append('dec',list(table_src.field('DEC')))
            pd.append('delta_ra',list(vx))
            pd.append('delta_dec',list(vy))
            pd.append('energy',list(np.log10(table_src.field('ENERGY'))))
            pd.append('dtheta',list(dth[msk]))
            pd.append('event_class',
                      list(table_src.field('EVENT_CLASS').astype('int')))
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
                    pi = self._phist(table_src.field('TIME')[k])            
                    cth = np.cos(pi.zAxis().difference(src))
                    cthv.append(cth)

            pd.append('cth',cthv)


#        print 'Loaded ', len(self.dtheta), ' events'
                    
        hdulist.close()

    def loadsrclist(self,fname,srcs):
        self.srcs = []
        self.src_names = []
        self.src_redshifts = []
        self.src_ra_deg = []
        self.src_dec_deg = []
        self.src_radec = []

        if not fname is None:
            src_names = np.genfromtxt(fname,unpack=True,dtype=None)
        else:
            src_names = np.array(opts.srcs.split(','))
            
        if src_names.ndim == 0: src_names = src_names.reshape(1)


        cat = Catalog()
        
        for name in src_names:

            src = cat.get_source_by_name(name)

            name = src['Source_Name']
            ra = src['RAJ2000']
            dec = src['DEJ2000']

            print 'Loading ', name
            
            self._photon_data._srcs.append(src)

            sd = skymaps.SkyDir(ra,dec)
            self.srcs.append(sd)
            self.src_names.append(name)

            self.src_ra_deg.append(ra)
            self.src_dec_deg.append(dec)
            self.src_radec.append((np.radians(ra),np.radians(dec)))
            
            self.src_redshifts.append(0.)
            
    def save(self,fname):

        self._photon_data.save(fname)

        return
        
        np.savez(fname,
                 data=np.array([self.dtheta,self.energy,self.cth,
                                self.ra,self.dec,self.phase,
                                self.dx,self.dy]),
                 ct=np.array(self.ct),
                 event_class=np.array(self.event_class),
                 time=np.array(self.time),
                 redshift=np.array(self.redshift),
                 src_names=np.array(self.src_names),
                 isrc=np.array(self.isrc),
                 src_radec=np.array([self.src_ra_deg,self.src_dec_deg]))
                

usage = "usage: %prog [options] [FT1 file ...]"
description = """Generate a numpy file containing a list of all photons within
max_dist_deg of a source defined in src_list.  The script accepts as input
a list of FT1 files."""

parser = OptionParser(usage=usage,description=description)

parser.add_option('--zenith_cut', default = 105, type='float',
                  help = 'Set the zenith angle cut.')

parser.add_option('--conversion_type', default = None, type='string',
                  help = 'Set the conversion type.')

parser.add_option('--event_class_id', default = None, type='string',
                  help = 'Set the event class bit.')

parser.add_option('--output', default = None, type='string',
                  help = 'Set the output filename.')

parser.add_option('--src_list',
                  default = None,
                  type='string',
                  help = 'Set the list of sources.')

parser.add_option('--srcs',
                  default = None,
                  type='string',
                  help = 'Set a comma-delimited list of sources.')

parser.add_option('--sc_file', default = None, type='string',
                  help = 'Set the spacecraft (FT2) file.')

parser.add_option('--max_events', default = None, type='int',
                  help = 'Set the maximum number of events that will be '
                  'read from each file.')

parser.add_option('--erange', default = None, type='string',
                  help = 'Set the energy range in log10(E/MeV).')

parser.add_option('--max_dist_deg', default = 25.0, type='float',
                  help = 'Set the maximum distance.')

parser.add_option('--phase', default = None, type='string',
                  help = 'Select the pulsar phase selection (on/off).')

parser.add_option("--batch",action="store_true",
                  help="Split this job into several batch jobs.")

(opts, args) = parser.parse_args()

if opts.batch:
    dispatch_jobs(os.path.abspath(__file__),args,opts)
    sys.exit(0)

if opts.output is None:
    opts.output = os.path.basename(os.path.splitext(args[0])[0] + '.P')
    
ft1_files = args
ft2_file = opts.sc_file

pl = FT1Loader(opts.zenith_cut,
               opts.conversion_type,
               opts.event_class_id,
               opts.max_events,opts.max_dist_deg,
               opts.phase,opts.erange)

pl.loadsrclist(opts.src_list,opts.srcs)

pl.setFT2File(opts.sc_file)

for f in ft1_files:
    pl.load_photons(f)

pl.save(opts.output)

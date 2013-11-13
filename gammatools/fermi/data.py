"""
@file  data.py

@brief Python classes for storing and manipulating photon data.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""
__source__   = "$Source: /nfs/slac/g/glast/ground/cvs/users/mdwood/python/data.py,v $"
__author__   = "Matthew Wood"
__date__     = "01/01/2013"
__date__     = "$Date: 2013/10/20 23:59:49 $"
__revision__ = "$Revision: 1.13 $, $Author: mdwood $"

import numpy as np
import re
import copy
import pywcs
import pyfits
from algebra import Vector3D
import matplotlib.pyplot as plt
from catalog import Catalog, CatalogSource
from matplotlib.colors import LogNorm
import pywcsgrid2
import pywcsgrid2.allsky_axes
from histogram import *
import yaml
from util import expand_aliases, eq2gal, interpolate2d

from pywcsgrid2.allsky_axes import make_allsky_axes_from_header

def get_circle(ra,dec,rad_deg,n=100):
    th = np.linspace(np.radians(rad_deg),
                     np.radians(rad_deg),n)
    phi = np.linspace(0,2*np.pi,n)

    v = Vector3D.createThetaPhi(th,phi)

    v.rotatey(np.pi/2.-np.radians(dec))
    v.rotatez(np.radians(ra))

    return np.degrees(v.lon()), np.degrees(v.lat())

class FITSAxis(object):

    def __init__(self,header,iaxis,logaxis=False,offset=0.5):

        self._type = header.get('CTYPE'+str(iaxis+1))
        self._refpix = header.get('CRPIX'+str(iaxis+1))-1
        self._refval = header.get('CRVAL'+str(iaxis+1))
        self._delta = header.get('CDELT'+str(iaxis+1)) 
        self._naxis = header.get('NAXIS'+str(iaxis+1))
        if logaxis:
            self._delta = np.log10((self._refval+self._delta)/self._refval)
            self._refval = np.log10(self._refval)

        xmin = self._refval+(self._refpix-offset)*self._delta
        xmax = self._refval+(self._refpix+self._naxis-offset)*self._delta
            
        self._edges = np.linspace(xmin,xmax,self._naxis+1)
        
        if re.search('GLON',self._type) is None and \
                re.search('GLAT',self._type) is None:
            self._galcoord = False
        else:
            self._galcoord = True

    def edges(self):
        return self._edges

    def naxis(self):
        return self._naxis
    
    def pix_to_coord(self,p):
        return self._refval + (p-self._refpix)*self._delta

    def coord_to_pix(self,x):
        return self._refpix + (x-self._refval)/self._delta

    def coord_to_index(self,x):
        pix = self.coord_to_pix(x)
        index = np.round(pix)
        return index

    @staticmethod
    def create_axes(header):

        if 'NAXIS' in header: naxis = header.get('NAXIS')
        elif 'WCSAXES' in header: naxis = header.get('WCSAXES')

        axes = []
        for i in range(naxis):
            
            ctype = header.get('CTYPE'+str(i+1))

            if ctype == 'Energy' or ctype == 'photon energy':
                axes.append(FITSAxis(header,i,True,0.0))
            else:
                axes.append(FITSAxis(header,i))

        return axes


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


class SkyCube(object):

    def __init__(self,wcs,axes,counts):
        self._counts = counts
        self._axes = axes
        self._wcs = wcs

    def get_spectrum(self,lon,lat):

        xy = self._wcs.wcs_sky2pix(lon,lat, 0)
        ilon = np.round(xy[0][0])
        ilat = np.round(xy[1][0])

        ilon = min(max(0,ilon),self._axes[0]._naxis-1)
        ilat = min(max(0,ilat),self._axes[1]._naxis-1)

        c = self._counts.T[ilon,ilat,:]
        edges = self._axes[2].edges()
        return Histogram.createFromArray(edges,c)


    def plot_energy_slices(self,rebin=4,logz=False):

        frame_per_fig = 1
        nx = 1
        ny = 1

        plt.figure()
        
        images = self.get_energy_slices(rebin)
        for i, im in enumerate(images):
            subplot = '%i%i%i'%(nx,ny,i%frame_per_fig+1)

#            im = im.smooth(0.1)
            
            im.plot(subplot=subplot,logz=logz)

            break
#            im.plot_catalog()
#            im.plot_circle(2.0)
    
    def get_energy_slices(self,rebin=2):

        nslice = int(np.ceil(self._axes[2].naxis()/float(rebin)))

        print self._axes[2].naxis(), rebin, nslice
        
        images = []
        
        for i in range(nslice):

            ilo = i*rebin
            ihi = min((i+1)*rebin,self._axes[2].naxis())

            print i, ilo, ihi
            
            counts = np.sum(self._counts[ilo:ihi],axis=0)
            image = SkyImage(self._wcs,self._axes[:2],counts)
            images.append(image)

        return images
    
    def energy_slice(self,ibin):

        counts = np.sum(self._counts[ibin:ibin+1],axis=0)
        return SkyImage(self._wcs,self._axes[:2],counts)
        
    def marginalize(self,emin,emax):

        ebins = self._axes[2].edges()
        imin = np.argmin(np.abs(emin-ebins))
        imax = np.argmin(np.abs(emax-ebins))

        print imin, imax

        print self._counts.shape
        
        counts = np.sum(self._counts[imin:imax+1],axis=0)

        print counts.shape
        
        return SkyImage(self._wcs,self._axes[:2],counts)
        
    def get_integrated_map(self,emin,emax):
        
        ebins = self._axes[2].edges()

        loge = 0.5*(ebins[1:] + ebins[:-1])
        dloge = ebins[1:] - ebins[:-1]

        imin = np.argmin(np.abs(emin-ebins))
        imax = np.argmin(np.abs(emax-ebins))



        edloge = 10**loge[imin:imax+1]*dloge[imin:imax+1]

        counts = np.sum(self._counts[imin:imax+1].T*edloge*np.log(10.),
                        axis=2)

#        print imin, imax
#        print self._counts.shape
#        print counts.shape
#        print self._wcs.wcs.crpix 
#        print self._wcs.wcs.cdelt         
#        print self._wcs.wcs.crval
        return SkyImage(self._wcs,self._axes[:2],counts)
        
    @staticmethod
    def createFromFITS(fitsfile,ihdu=0):
        
        hdu = pyfits.open(fitsfile)
        
        header = hdu[ihdu].header
        wcs = pywcs.WCS(header,naxis=[1,2],relax=True)
#        wcs1 = pywcs.WCS(header,naxis=[3])
        
        axes = copy.deepcopy(FITSAxis.create_axes(header))
        
        return SkyCube(wcs,axes,copy.deepcopy(hdu[ihdu].data.astype(float)))

class SkyImage(object):

    def __init__(self,wcs,axes,counts,ra=0.0,dec=0.0,roi_radius_deg=180.):
        self._ra = ra
        self._dec = dec
        self._roi_radius = roi_radius_deg

        self._wcs = wcs
        self._header = self._wcs.to_header(True)
        self._counts = counts
        self._axes = axes #FITSAxis.create_axes(self._wcs.to_header(True))

        xmin = 0.0
        ymin = 0.0

        self._xedge = np.linspace(xmin,xmin+counts.shape[0],counts.shape[0]+1)
        self._yedge = np.linspace(ymin,ymin+counts.shape[1],counts.shape[1]+1)

    def __getnewargs__(self):

        self._wcs = pywcs.WCS(self._header)
        return ()
#        return (self._wcs,self._counts,self._ra,self._dec,self._roi_radius)


    
    @staticmethod
    def createROI(ra,dec,roi_radius_deg,bin_size_deg=0.2):
        nbin = np.ceil(2.0*roi_radius_deg/bin_size_deg)
        deg_to_pix = bin_size_deg
        
        wcs = pywcs.WCS(naxis=2)

        wcs.wcs.crpix = [0.0, 0.0]
        wcs.wcs.cdelt = np.array([-deg_to_pix,
                                   deg_to_pix])
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.ctype = ["RA---AIT", "DEC--AIT"]
        wcs.wcs.equinox = 2000.0

        lon, lat = get_circle(ra,dec,roi_radius_deg)

        xy =  wcs.wcs_sky2pix(lon, lat, 0)

        xmin = np.min(xy[0])
        xmax = np.max(xy[0])

        if roi_radius_deg >= 90.:
            xypole0 = wcs.wcs_sky2pix(0.0, -90.0, 0)
            xypole1 = wcs.wcs_sky2pix(0.0, 90.0, 0)
            ymin = xypole0[1]
            ymax = xypole1[1]
        else:
            ymin = np.min(xy[1])
            ymax = np.max(xy[1])

#        self._pix_center = self._wcs.wcs_sky2pix(self._ra, self._dec, 0)
#        self._xedge = np.linspace(np.min(xy[0]),np.max(xy[0]),nbin+1)
#        self._yedge = np.linspace(np.min(xy[1]),np.max(xy[1]),nbin+1)

        counts = np.zeros(shape=(nbin,nbin))

        header = wcs.to_header(True)
        header['NAXIS1'] = nbin
        header['NAXIS2'] = nbin
        
        axes = FITSAxis.create_axes(header)
        im = SkyImage(wcs,axes,counts,ra,dec,roi_radius_deg)

        im._xedge = np.linspace(xmin,xmax,nbin+1)
        im._yedge = np.linspace(ymin,ymax,nbin+1)

#        print wcs.wcs_pix2sky(im._xedge[nbin/2],im._yedge[0],0)
#        xy =  wcs.wcs_pix2sky([im._xedge[nbin/2.]]*len(im._yedge),im._yedge,0)
        return im

    def fill(self,ra,dec):

        pixcrd = self._wcs.wcs_sky2pix(ra,dec, 0)
        counts = np.histogramdd(pixcrd,bins=[self._xedge,self._yedge])[0]

        self._counts += counts

    def interpolate(self,lon,lat):
        
        pixcrd = self._wcs.wcs_sky2pix(lon, lat, 0)
        return interpolate2d(self._xedge,self._yedge,self._counts,
                             *pixcrd)

    def smooth(self,sigma):
        sigma /= np.abs(self._axes[0]._delta)
        
        from scipy import ndimage

        im = copy.deepcopy(self)
        im._counts = ndimage.gaussian_filter(self._counts, sigma=sigma,
                                             mode='nearest')

        return im

    def plot_catalog(self):

        labelcolor = 'k'
        
        cat = Catalog()
        srcs = cat.get_source_by_radec(self._ra,self._dec,self._roi_radius)

        src_lon = []
        src_lat = []

        labels = []
        signif_avg = []
        
        for s in srcs:

#            print s['Source_Name'], s['RAJ2000'], s['DEJ2000']

            src_lon.append(s['RAJ2000'])
            src_lat.append(s['DEJ2000'])
            labels.append(s['Source_Name'])
            signif_avg.append(s['Signif_Avg'])
        
        if self._axes[0]._galcoord:
            src_lon, src_lat = eq2gal(src_lon,src_lat)
            
            
        pixcrd = self._wcs.wcs_sky2pix(src_lon,src_lat, 0)

        for i in range(len(labels)):

            if signif_avg[i] < 10: continue
            
            plt.gca().text(pixcrd[0][i],pixcrd[1][i],labels[i],
                           color=labelcolor,size=8,clip_on=True)
        
        plt.gca().plot(pixcrd[0],pixcrd[1],linestyle='None',marker='+',
                       color='g', markerfacecolor = 'None',
                       markeredgecolor=labelcolor,clip_on=True)


        
        plt.gca().set_xlim(self._xedge[0],self._xedge[-1])
        plt.gca().set_ylim(self._yedge[0],self._yedge[-1])
        
            
    def plot_circle(self,rad_deg,radec=None,**kwargs):

        if radec is None: radec = (self._ra,self._dec)

        lon,lat = get_circle(radec[0],radec[1],rad_deg)
        xy =  self._wcs.wcs_sky2pix(lon,lat, 0)
        plt.gca().plot(xy[0],xy[1],**kwargs)

        plt.gca().set_xlim(self._xedge[0],self._xedge[-1])
        plt.gca().set_ylim(self._yedge[0],self._yedge[-1])
    

    def plot(self,subplot=111,logz=False):


        from matplotlib.colors import NoNorm, LogNorm, Normalize

        if logz: norm = LogNorm()
        else: norm = Normalize()
        
        
#        print self._wcs.wcs

        ax = pywcsgrid2.subplot(subplot, header=self._wcs.to_header())
#        ax = pywcsgrid2.axes(header=self._wcs.to_header())
#        ax = make_allsky_axes_from_header(plt.gcf(), rect=111,
#                                          header=self._wcs.to_header(True),
#                                          lon_center=0.)


        im = ax.imshow(self._counts.T,#np.power(self._counts.T,1./3.),
                       interpolation='nearest',origin='lower',norm=norm,
                       extent=[self._xedge[0],self._xedge[-1],
                               self._yedge[0],self._yedge[-1]])

#        norm=LogNorm()
        
        ax.set_ticklabel_type("d", "d")
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')

        plt.colorbar(im)
        ax.grid()
#        ax.add_compass(loc=1)
#       
#        ax.set_display_coord_system("gal")       
 #       ax.locator_params(axis="x", nbins=12)

        return ax


class PhotonData(object):

    def __init__(self):
        self._data = { 'ra'           : np.array([]),
                       'dec'          : np.array([]),
                       'delta_ra'     : np.array([]),
                       'delta_dec'    : np.array([]),
                       'energy'       : np.array([]),
                       'time'         : np.array([]),
                       'psfcore'         : np.array([]),
                       'event_class'    : np.array([],dtype='int'),
                       'conversion_type'    : np.array([],dtype='int'),
                       'src_index'    : np.array([],dtype='int'),
                       'dtheta' : np.array([]),
                       'phase'  : np.array([]),
                       'cth'    : np.array([]) }

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
             event_class=None,event_class_id=None,
             phases=None,cuts=None,
             src_index=None,cuts_file=None):

        msk = PhotonData.get_mask(self,selections,conversion_type,event_class,
                                  event_class_id,phases,cuts,src_index,
                                  cuts_file)

        self.apply_mask(msk)
    
    @staticmethod
    def get_mask(data,selections=None,conversion_type=None,
                 event_class=None,event_class_id=None,
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

    
        
class PSFData(Data):

    def __init__(self,egy_bin_edge,cth_bin_edge,quantiles,dtype):

        egy_bin_edge = np.asarray(egy_bin_edge)
        cth_bin_edge = np.asarray(cth_bin_edge)

        self.dtype = dtype
        self.quantiles = quantiles
        self.quantile_labels = ['r%2.f'%(q*100) for q in self.quantiles]

        self.egy_bin_edge = egy_bin_edge
        self.cth_bin_edge = cth_bin_edge
        self.egy_range = [[self.egy_bin_edge[i],self.egy_bin_edge[i+1]]
                          for i in range(len(self.egy_bin_edge)-1)]
        self.egy_center = 0.5*(self.egy_bin_edge[:-1] + self.egy_bin_edge[1:])
        self.egy_width = (self.egy_bin_edge[1:] - self.egy_bin_edge[:-1])
        
        self.cth_range = [[self.cth_bin_edge[i],self.cth_bin_edge[i+1]]
                          for i in range(len(self.cth_bin_edge)-1)]        
        self.cth_center = 0.5*(self.cth_bin_edge[:-1] + self.cth_bin_edge[1:])

        self.cth_width = (self.cth_bin_edge[1:] - self.cth_bin_edge[:-1])
        self.egy_nbin = len(self.egy_range)
        self.cth_nbin = len(self.cth_range)

        self.chi2 = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.rchi2 = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.ndf = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.excess = Histogram2D(egy_bin_edge,cth_bin_edge)
        self.qdata = {}
        
        self.sig_density_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                         dtype=object)
        self.tot_density_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                         dtype=object)
        self.bkg_density_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                         dtype=object)
        self.sig_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                 dtype=object)
        self.off_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                 dtype=object)        
        self.tot_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                 dtype=object)
        self.bkg_hist = np.empty(shape=(self.egy_nbin,self.cth_nbin), 
                                 dtype=object)

        for i in range(len(self.quantile_labels)):
            l = self.quantile_labels[i]
            self.qdata[l] = Histogram2D(egy_bin_edge,cth_bin_edge)


    def init_hist(self,fn,theta_max):
        
        for i in range(self.egy_nbin):
            for j in range(self.cth_nbin):

                ecenter = 0.5*(self.egy_bin_edge[i] + self.egy_bin_edge[i+1])
                theta_max = min(theta_max,fn(ecenter))
                theta_edges = np.linspace(0,theta_max,100)
                
                h = Histogram(theta_edges)
                self.sig_density_hist[i,j] = copy.deepcopy(h)
                self.tot_density_hist[i,j] = copy.deepcopy(h)
                self.bkg_density_hist[i,j] = copy.deepcopy(h)
                self.sig_hist[i,j] = copy.deepcopy(h)
                self.tot_hist[i,j] = copy.deepcopy(h)
                self.bkg_hist[i,j] = copy.deepcopy(h)
                self.off_hist[i,j] = copy.deepcopy(h)
                
            
    def print_quantiles(self,prefix):

        filename = prefix + '.txt'
        f = open(filename,'w')     
        
        for i, ql in enumerate(self.quantile_labels):

            q = self.qdata[ql]
            
            for icth in range(self.cth_nbin):
                for iegy in range(self.egy_nbin):

                    line = '%5.3f '%(self.quantiles[i])
                    line += '%5.2f %5.2f '%(self.cth_range[icth][0],
                                           self.cth_range[icth][1])
                    line += '%5.2f %5.2f '%(self.egy_range[iegy][0],
                                            self.egy_range[iegy][1])
                    line += '%8.4f %8.4f '%(q.mean[iegy,icth],
                                            q.err[iegy,icth])
                    
                    f.write(line + '\n')

    def print_quantiles_tex(self,prefix):

        for ql, q in self.qdata.iteritems():

            filename = prefix + '_' + ql + '.tex'
            f = open(filename,'w')  
            
            for icth in range(self.cth_nbin):
                for iegy in range(self.egy_nbin):

                    line = '%5.2f %5.2f '%(self.cth_range[icth][0],
                                           self.cth_range[icth][1])
                    line += '%5.2f %5.2f '%(self.egy_range[iegy][0],
                                            self.egy_range[iegy][1])
                    line += format_error(q.mean[iegy,icth],
                                         q.err[iegy,icth],1,True)
                    f.write(line + '\n')
                    
            

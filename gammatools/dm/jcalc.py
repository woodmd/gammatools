#!/usr/bin/env python

"""
@file   jcalc.py

@brief Python modules that are used to compute the line-of-sight
integral over a spherically symmetric DM distribution.

@author Matthew Wood       <mdwood@slac.stanford.edu>
@author Alex Drlica-Wagner <kadrlica@stanford.edu>
"""

__author__   = "Matthew Wood"
__date__     = "12/01/2011"

import copy
import numpy as np

from scipy.integrate import quad
from scipy.interpolate import bisplrep
from scipy.interpolate import bisplev
from scipy.interpolate import interp1d, UnivariateSpline
import scipy.special as spfn
import scipy.optimize as opt
from gammatools.core.util import *

class LoSFn(object):
    """Integrand function for LoS parameter (J).  The parameter alpha
    introduces a change of coordinates x' = x^(1/alpha).  The change
    of variables means that we need make the substitution:

    dx = alpha * (x')^(alpha-1) dx'

    A value of alpha > 1 weights the points at which we sample the
    integrand closer to x = 0 (distance of closest approach).

    Parameters
    ----------
    d: Distance to halo center.
    xi: Offset angle in radians.
    dp: Density profile.
    alpha: Rescaling exponent for line-of-sight coordinate.
    """

    def __init__(self,d,xi,dp,alpha=4.0):
        self._d = d
        self._d2 = d*d
        self._xi = xi
        self._sinxi = np.sin(xi)
        self._sinxi2 = np.power(self._sinxi,2)
        self._dp = dp
        self._alpha = alpha

    def __call__(self,xp):
        #xp = np.asarray(xp)
        #if xp.ndim == 0: xp = np.array([xp])

        x = np.power(xp,self._alpha)
        r = np.sqrt(x*x+self._d2*self._sinxi2)
        rho2 = np.power(self._dp.rho(r),2)
        return rho2*self._alpha*np.power(xp,self._alpha-1.0)
    
class LoSFnDecay(LoSFn):
    def __init__(self,d,xi,dp,alpha=1.0):
        super(LoSFnDecay,self).__init__(d,xi,dp,alpha)
        
    def __call__(self,xp):
        #xp = np.asarray(xp)
        #if xp.ndim == 0: xp = np.array([xp])

        x = np.power(xp,self._alpha)
        r = np.sqrt(x*x+self._d2*self._sinxi2)
        rho = self._dp.rho(r)
        return rho*self._alpha*np.power(xp,self._alpha-1.0)

class LoSIntegralFn(object):
    """Object that computes integral over DM density squared along a
    line-of-sight offset by an angle psi from the center of the DM
    halo.  We introduce a change of coordinates so that the integrand
    is more densely sampled near the distance of closest of approach
    to the halo center.

    Parameters
    ----------
    dist: Distance to halo center.
    dp: Density profile.
    alpha: Parameter determining the integration variable: x' = x^(1/alpha)
    rmax: Radius from center of halo at which LoS integral is truncated.
    """
    def __init__(self, dp, dist, rmax=None, alpha=3.0,ann=True):
        if rmax is None: rmax = np.inf

        self._dp = dp
        self._dist = dist
        self._rmax = rmax
        self._alpha = alpha
        self._ann = ann

    @classmethod
    def create(cls,config,method='fast'):
        
        dp = DensityProfile.create(config)        
        return LoSIntegralFnFast(dp,config['dist']*Units.kpc,
                                 config['rmax']*Units.kpc)


    def __call__(self,psi,dhalo=None):
        """Evaluate the LoS integral at the offset angle psi for a halo
        located at the distance dhalo.

        Parameters
        ----------
        psi : array_like 
        Array of offset angles (in radians)

        dhalo : array_like
        Array of halo distances.
        """

        if dhalo is None: dhalo = np.array(self._dist,ndmin=1)
        else: dhalo = np.array(dhalo,ndmin=1)

        psi = np.array(psi,ndmin=1)

        if dhalo.shape != psi.shape:
            dhalo = dhalo*np.ones(shape=psi.shape)

        v = np.zeros(shape=psi.shape)

        for i, t in np.ndenumerate(psi):

            s0 = 0
            s1 = 0

            if self._ann:
                losfn = LoSFn(dhalo[i],t,self._dp,self._alpha)
            else:
                losfn = LoSFnDecay(dhalo[i],t,self._dp,self._alpha)

            # Closest approach to halo center
            rmin = dhalo[i]*np.sin(psi[i])

            # If observer inside the halo...
            if self._rmax > dhalo[i]:

                if psi[i] < np.pi/2.:

                    x0 = np.power(dhalo[i]*np.cos(psi[i]),1./self._alpha)
                    s0 = 2*quad(losfn,0.0,x0)[0]

                    x1 = np.power(np.sqrt(self._rmax**2 -
                                          rmin**2),1./self._alpha)
                
                    s1 = quad(losfn,x0,x1)[0]
                else:
                    x0 = np.power(np.abs(dhalo[i]*np.cos(psi[i])),
                                  1./self._alpha)

                    x1 = np.power(np.sqrt(self._rmax**2 -
                                          rmin**2),1./self._alpha)
                    s1 = quad(losfn,x0,x1)[0]

            # If observer outside the halo...
            elif self._rmax > rmin:
                x0 = np.power(np.sqrt(self._rmax**2 -
                                      rmin**2),1./self._alpha)
                s0 = 2*quad(losfn,0.0,x0)[0]
                
            v[i] = s0+s1

        return v

class LoSIntegralFnFast(LoSIntegralFn):
    """Vectorized version of LoSIntegralFn that performs midpoint
    integration with a fixed number of steps.

    Parameters
    ----------
    dist: Distance to halo center.
    dp:   Density profile.
    alpha: Parameter determining the integration variable: x' = x^(1/alpha)
    rmax: Radius from center of halo at which LoS integral is truncated.
    nstep: Number of integration steps.  Increase this parameter to
    improve the accuracy of the LoS integral.
    """
    def __init__(self, dp, dist, rmax=None, alpha=3.0,ann=True,nstep=400):
        super(LoSIntegralFnFast,self).__init__(dp,dist,rmax,alpha,ann)

        self._nstep = nstep
        xedge = np.linspace(0,1.0,self._nstep+1)
        self._x = 0.5*(xedge[1:] + xedge[:-1])

    def __call__(self,psi,dhalo=None):
        """Evaluate the LoS integral at the offset angle psi for a halo
        located at the distance dhalo.

        Parameters
        ----------
        psi : array_like 
        Array of offset angles (in radians)

        dhalo : array_like
        Array of halo distances.
        """

        if dhalo is None: dhalo = np.array(self._dist,ndmin=1)
        else: dhalo = np.array(dhalo,ndmin=1)

        psi = np.array(psi,ndmin=1)

#        if dhalo.shape != psi.shape:
#            d = np.zeros(shape=psi.shape)
#            d[:] = dhalo
#            dhalo = d
#        elif dhalo.ndim == 0: dhalo = np.array([dhalo])
#        if psi.ndim == 0: psi = np.array([psi])
        
        v = np.zeros(shape=psi.shape)

        if self._ann: losfn = LoSFn(dhalo,psi,self._dp,self._alpha)
        else: losfn = LoSFnDecay(dhalo,psi,self._dp,self._alpha)

        # Closest approach to halo center
        rmin = dhalo*np.sin(psi)

        msk0 = self._rmax > dhalo
        msk1 = self._rmax > rmin

        # Distance between observer and point of closest approach
        xlim0 = np.power(np.abs(dhalo*np.cos(psi)),1./self._alpha)

        # Distance from point of closest approach to maximum
        # integration radius
        xlim1 = np.zeros(shape=psi.shape)
        xlim1[msk1] = np.power(np.sqrt(self._rmax**2 - rmin[msk1]**2),
                               1./self._alpha)

        # If observer inside the halo...
        if np.any(msk0):

            msk01 = msk0 & (psi < np.pi/2.)
            msk02 = msk0 & ~(psi < np.pi/2.)

            if np.any(msk01):

                dx0 = xlim0/float(self._nstep)
                dx1 = (xlim1-xlim0)/float(self._nstep)

                x0 = np.outer(self._x,xlim0)
                x1 = xlim0 + np.outer(self._x,xlim1-xlim0)

                s0 = 2*np.sum(losfn(x0)*dx0,axis=0)
                s1 = np.sum(losfn(x1)*dx1,axis=0)

                v[msk01] = s0[msk01]+s1[msk01]

            if np.any(msk02):
            
                dx1 = (xlim1-xlim0)/float(self._nstep)

                x1 = xlim0 + np.outer(self._x,xlim1-xlim0)
                s0 = np.sum(losfn(x1)*dx1,axis=0)
            
                v[msk02] = s0[msk02]
                
        # If observer outside the halo...
        if np.any(~msk0 & msk1):
            
            dx0 = xlim1/float(self._nstep)
            x0 = np.outer(self._x,xlim1)

            s0 = 2*np.sum(losfn(x0)*dx0,axis=0)

            v[~msk0 & msk1] = s0[~msk0 & msk1]


        return v

class LoSIntegralSplineFn(object):

    def __init__(self,dp=None,nx=40,ny=20):
        self.dp = copy.copy(dp)

        if self.dp is not None:
            nx = 40
            ny = 20
            dhalo, psi = np.mgrid[1:2:ny*1j,0.001:2.0:nx*1j]
            dhalo = np.power(10,dhalo)
            psi = np.radians(psi)            
            f = LoSIntegralFn(self.dp)
            self.z = f(dhalo,psi)
            self.init_spline(dhalo,psi,self.z)

    def init_spline(self,dhalo,psi,z):
        """Compute knots and coefficients of an interpolating spline
        given a grid of points in halo distance (dhalo) and offset
        angle (psi) at which the LoS integral has been computed.
        """

        kx = 2
        ky = 2
        self._psi_min = psi.min()
        self._tck = bisplrep(dhalo,psi,np.log10(z),s=0.0,kx=kx,ky=ky,
                             nxest=int(kx+np.sqrt(len(z.flat))),
                             nyest=int(ky+np.sqrt(len(z.flat))))

    def __call__(self,dhalo,psi,rho=1,rs=1):
        """Compute the LoS integral using a 2D spline table.

        Returns
        -------

        vals: LoS amplitude per steradian.
        """

        dhalo = np.asarray(dhalo)
        psi = np.asarray(psi)

        if dhalo.ndim == 0: dhalo = np.array([dhalo])
        if psi.ndim == 0: psi = np.array([psi])

        if psi.ndim == 2 and dhalo.ndim == 2:
            v = np.power(10,bisplev(dhalo[:,0],psi[0,:],self._tck))
        else:
            v = np.power(10,bisplev(dhalo,psi,self._tck))

        v *= rho*rho*rs
        return v


def SolidAngleIntegral(psi,pdf,angle):
    """ Compute the solid-angle integrated j-value
    within a given radius

    Parameters
    ----------
    psi : array_like 
    Array of offset angles (in radians)

    pdf : array_like
    Array of j-values at angle psi
    
    angle : array_like
    Maximum integration angle (in degrees)
    """
    angle = np.asarray(angle)
    if angle.ndim == 0: angle = np.array([angle])

    scale=max(pdf)
    norm_pdf = pdf/scale

    log_spline = UnivariateSpline(psi,np.log10(norm_pdf),k=1,s=0)
    spline = lambda r: 10**(log_spline(r))
    integrand = lambda r: spline(r)*2*np.pi*np.sin(r)

    integral = []
    for a in angle:
        integral.append(quad(integrand, 0, np.radians(a),full_output=True)[0])
    integral = np.asarray(integral)

    return integral*scale

class JProfile(object):
    def __init__(self,losfn):

        self._log_psi = np.linspace(np.log10(np.radians(0.001)),
                                    np.log10(np.radians(90.)),1000)
        self._psi = np.power(10,self._log_psi)

        self._jpsi = losfn(self._psi)

        self._jspline = UnivariateSpline(self._psi,self._jpsi,s=0,k=2)

    @staticmethod
    def create(dp,dist,rmax):
        losfn = LoSIntegralFn(dp,dist,rmax=rmax)        
        return JProfile(losfn)

    def __call__(self,psi):
        return self._jspline(psi)

    def integrate(self,psimax):

        xedge = np.linspace(0.0,np.radians(psimax),1001)
        x = 0.5*(xedge[1:] + xedge[:-1])
        domega = 2.0*np.pi*(-np.cos(xedge[1:])+np.cos(xedge[:-1]))
        return np.sum(self._jspline(x)*domega)

    def cumsum(self,psi):
        x = 0.5*(psi[1:]+psi[:-1])
        dcos = -np.cos(psi[1:])+np.cos(psi[:-1])
        return np.cumsum(self._jspline(x)*dcos)

class JIntegrator(object):

    def __init__(self,jspline,lat_cut,lon_cut,source_list=None):

        self._lat_cut = lat_cut
        self._lon_cut = lon_cut

        nbin_thetagc = 720
        thetagc_max = 180.

        self._phi_edges = np.linspace(0.,360.,721)
        self._theta_edges = np.linspace(0.,thetagc_max,nbin_thetagc+1)

        self._sources = None

        if not opts.source_list is None:
            source_list = np.loadtxt(opts.source_list,unpack=True,usecols=(1,2))
            self._sources = Vector3D.createLatLon(np.radians(source_list[0]),
                                                  np.radians(source_list[1]))

        self.compute()

    def compute(self):
        
        yaxis = Vector3D(np.pi/2.*np.array([0.,1.,0.]))

        costh_edges = np.cos(np.radians(self._theta_edges))
        costh_width = costh_edges[:-1]-costh_edges[1:]
    
        phi = 0.5*(self._phi_edges[:-1] + self._phi_edges[1:])
        self._theta = 0.5*(self._theta_edges[:-1] + self._theta_edges[1:])

        self._jv = []
        self._domega = []

        for i0, th in enumerate(self._theta):

            jtot = integrate(lambda t: jspline(t)*np.sin(t),
                             np.radians(self._theta_edges[i0]),
                             np.radians(self._theta_edges[i0+1]),100)

#    jval = jspline(np.radians(th))*costh_width[i0]
            v = Vector3D.createThetaPhi(np.radians(th),np.radians(phi))
            v.rotate(yaxis)

            lat = np.degrees(v.lat())
            lon = np.degrees(v.phi())

            src_msk = len(lat)*[True]

            if not self._sources is None:

                for k in range(len(v.lat())):
                    p = Vector3D(v._x[:,k])

                    sep = np.degrees(p.separation(self._sources))
                    imin = np.argmin(sep)
                    minsep = sep[imin]

                    if minsep < 0.62: src_msk[k] = False

            msk = ((np.abs(lat)>=self._lat_cut) |
                   ((np.abs(lat)<=self._lat_cut)&(np.abs(lon)<self._lon_cut)))

            msk &= src_msk
            dphi = 2.*np.pi*float(len(lat[msk]))/float(len(phi))

#    hc._counts[i0,msk] = 1

            jtot *= dphi
#            jsum += jtot
#            domegasum += costh_width[i0]*dphi

            self._jv.append(jtot)
            self._domega.append(costh_width[i0]*dphi)
    
        self._jv = np.array(self._jv)
        self._jv_cum = np.cumsum(self._jv)

        self._jv_cum_spline = UnivariateSpline(self._theta_edges[1:],
                                               self._jv_cum,
                                               s=0,k=1)

        self._domega = np.array(self._domega)
        self._domega_cum = np.cumsum(self._domega)
    
    def eval(self,rgc,decay=False):
        
        if decay:
            units0 = Units.gev_cm2
            units1 = (8.5*Units.kpc*0.4*Units.gev_cm3)
        else:
            units0 = Units.gev2_cm5
            units1 = (8.5*Units.kpc*np.power(0.4*Units.gev_cm3,2))

        jv = self._jv_cum_spline(rgc)
    
        i = np.argmin(np.abs(rgc-self._theta))

        print '%10.2f %20.6g %20.6g %20.6g %20.6g'%(rgc, jv, 
                                                    jv/units0, 
                                                    jv/units1,
                                                    self._domega_cum[i])


    def print_profile(self,decay=False):

        if decay:
            units0 = Units.gev_cm2
            units1 = (8.5*Units.kpc*0.4*Units.gev_cm3)
        else:
            units0 = Units.gev2_cm5
            units1 = (8.5*Units.kpc*np.power(0.4*Units.gev_cm3,2))


        for i, th in enumerate(self._theta_edges[1:]):

            jv = self._jv_cum[i]

            print '%10.2f %20.6g %20.6g %20.6g %20.6g'%(th, jv, 
                                                        jv/units0, 
                                                        jv/units1,
                                                        self._domega_cum[i])

class DensityProfile(object):
    """ DM density profile that truncates at a maximum DM density.
    
    rho(r) = rho(r) for rho(r) < rhomax AND r > rmin
           = rhomax for rho(r) >= rhomax
           = rho(rmin) for r <= rmin
    
    Parameters
    ----------
    rhos : Density normalization parameter.
    
    rmin : Inner radius interior to which the density will be fixed to
    a constant value. (rhomax = rho(rmin)).

    rhomax : Maximum DM density.  If rhomax and rmin are both defined
    the maximum DM density will be the lesser of rhomax and rho(rmin).
    
    """
    def __init__(self,rhos,rs,rmin=None,rhomax=None):
        self._name = 'profile'
        self._rmin=rmin
        self._rhomax=rhomax
        self._rhos = rhos
        self._rs = rs

    def setMassConcentration(self,mvir,c):

        rhoc = 9.9E-30*Units.g_cm3
        rvir = np.power(mvir*3.0/(177.7*4*np.pi*rhoc*0.27),1./3.)
        rs = rvir/c

        self._rs = rs
        mrvir = self.mass(rvir)
        self._rhos = self._rhos*mvir/mrvir

    def rho(self,r):

        r = np.array(r,ndmin=1)

        if self._rhomax is None and self._rmin is None: 
            return self._rho(r)
        elif self._rhomax is None:
            rho = self._rho(r)        
            rho[r<self._rmin] = self._rho(self._rmin)
            return rho
        elif self._rmin is None:
            rho = self._rho(r)        
            rho[rho>self._rhomax] = self._rhomax
            return rho
        else:
            rho = self._rho(r) 
            rhomax = min(self._rho(self._rmin),self._rhomax)
            rho[rho>rhomax] = rhomax
            return rho

#            return np.where(rho>self._rhomax,[self._rhomax],rho)
        
    def set_rho(self,rho,r):
        """Fix the density normalization at a given radius."""
        rhor = self._rho(r)
        self._rhos = rho*self._rhos/rhor

    @property
    def name(self):
        return self._name

    @property
    def rhos(self):
        return self._rhos

    @property
    def rs(self):
        return self._rs

    @staticmethod
    def create(opts):
        """Method for instantiating a density profile object given the
        profile name and a dictionary."""

        o = {}
        o.update(opts)

        name = opts['type']

        def extract(keys,d):
            od = {}
            for k in keys: 
                if k in d: od[k] = d[k]
            return od

        for k, v in o.iteritems():
            if v is None: continue
            elif isinstance(v,str): o[k] = Units.parse(v)
            elif k == 'dist': o[k] *= Units.kpc
            elif k == 'rs': o[k] *= Units.kpc
            elif k == 'rhos': o[k] *= Units.msun_kpc3
            elif k == 'rhor': o[k] = [o[k][0]*Units.gev_cm3,
                                      o[k][1]*Units.kpc]
            elif k ==' jval' : o[k] = o[k]*Units.gev2_cm5

        if o['rhos'] is None: o['rhos'] = 1.0

        if name == 'nfw':
            dp = NFWProfile(**extract(['rhos','rs','rmin'],o))
        elif name == 'gnfw':
            dp = GNFWProfile(**extract(['rhos','rs','rmin','gamma'],o))
        elif name == 'isothermal':
            dp = IsothermalProfile(**extract(['rhos','rs','rmin'],o))
        elif name == 'einasto':
            dp = EinastoProfile(**extract(['rhos','rs','rmin','alpha'],o))
        else:
            print 'No such halo type: ', name
            sys.exit(1)

        if 'rhor' in o: dp.set_rho(o['rhor'][0],o['rhor'][1])
        elif 'jval' in o: dp.set_jval(o['jval'],o['rs'],o['dist'])

        return dp


class BurkertProfile(DensityProfile):
    """ Burkert (1995)
        rho(r) = rhos/( (1+r/rs)(1+(r/rs)**2) )
    """
    def __init__(self,rhos=1,rs=1,rmin=None,rhomax=None):        
        super(BurkertProfile,self).__init__(rhos,rs,rmin,rhomax)
        self._name = 'burkert'

    def _rho(self,r):
        x = r/self._rs
        return self._rhos*np.power(1+x,-1)*np.power(1+x*x,-1)

    def _mass(self,r):
        x = r/self._rs        
        return 4*np.pi*self._rhos*np.power(self._rs,3)*(log(1+x)-x/(1+x))
    
class IsothermalProfile(DensityProfile):

    def __init__(self,rhos=1,rs=1,rmin=None,rhomax=None):        
        super(IsothermalProfile,self).__init__(rhos,rs,rmin,rhomax)
        self._name = 'isothermal'

    def _rho(self,r):
        x = r/self._rs
        return self._rhos*np.power(1+x,-2)

    def _mass(self,r):
        x = r/self._rs        
        return 4*np.pi*self._rhos*np.power(self._rs,3)*(log(1+x)-x/(1+x))

    
class NFWProfile(DensityProfile):
    """ Navarro, Frenk, and White (1996)
        rho(r) = rhos/( (r/rs)(1+r/rs)**2)
    """
    def __init__(self,rhos=1,rs=1,rmin=None,rhomax=None):
        super(NFWProfile,self).__init__(rhos,rs,rmin,rhomax)
        self._name = 'nfw'

    def set(self,rhos,rs):
        self._rs = rs
        self._rhos = rhos

    def set_jval(self,jval,rs,dist):
        rhos = np.sqrt(3./(4.*np.pi)*jval*dist**2/rs**3)
        self._rs = rs
        self._rhos = rhos

    def mass(self,r):
        x = r/self._rs
        return 4*np.pi*self._rhos*np.power(self._rs,3)*(np.log(1+x)-x/(1+x))

    def jval(self,r=None,rhos=None,rs=None):
        """Small angle approximation to halo Jvalue. """
        if rhos is None: rhos = self._rhos
        if rs is None: rs = self._rs

        if r is not None:
            x = r/rs
            return (4*np.pi/3.)*rhos**2*rs**3*(1.-np.power(1.+x,-3))
        else:
            return (4*np.pi/3.)*rhos**2*rs**3

#(4*M_PI/3.)*std::pow(a(0),2)*std::pow(a(1),3)*(1.-std::pow(1+x,-3));

    def _rho(self,r):
        x = r/self._rs
        return self._rhos*np.power(x,-1)*np.power(1+x,-2)        
    
class EinastoProfile(DensityProfile):
    """ Einasto profile
        rho(r) = rhos*exp(-2*((r/rs)**alpha-1)/alpha)
    """
    def __init__(self,rhos=1,rs=1,alpha=0.17,rmin=None,rhomax=None):
        self._alpha = alpha
        super(EinastoProfile,self).__init__(rhos,rs,rmin,rhomax)
        self._name = 'einasto'

    def set(self,rhos,rs):
        self._rs = rs
        self._rhos = rhos

    def mass(self,r):

        x = r/self._rs
        gamma = spfn.gamma(3./self._alpha)

        return 4*np.pi*self._rhos*np.power(self._rs,3)/self._alpha* \
            np.exp(2./self._alpha)* \
            np.power(2./self._alpha,-3./self._alpha)* \
            gamma*spfn.gammainc(3./self._alpha,
                                (2./self._alpha)*np.power(x,self._alpha))

    def _rho(self,r):
        x = r/self._rs
        return self._rhos*np.exp(-2./self._alpha*(np.power(x,self._alpha)-1))

class GNFWProfile(DensityProfile):
    """ Generalized NFW Profile
        rho(r) = rhos/( (r/rs)^g(1+r/rs)**(3-g))
    """
    def __init__(self,rhos=1,rs=1,gamma=1.0,rmin=None,rhomax=None):
        self._gamma = gamma
        super(GNFWProfile,self).__init__(rhos,rs,rmin,rhomax)
        self._name = 'nfw'

    def set(self,rhos,rs):
        self._rs = rs
        self._rhos = rhos

    def mass(self,r):
#        x = r/self._rs
#        return 4*np.pi*self._rhos*np.power(self._rs,3)*(np.log(1+x)-x/(1+x))
        return 0

    def _rho(self,r):
        x = r/self._rs
        return self._rhos*np.power(x,-self._gamma)* \
            np.power(1+x,-(3-self._gamma))    

class GeneralNFWProfile(DensityProfile):
    """ Strigari et al. (2007)
        rho(r) = rhos/( (r/rs)**a (1+(r/rs)**b )**(c-a)/b
        Default: NFW profile
    """
    def __init__(self,rhos=1,rs=1,a=1,b=1,c=3,rmin=None,rhomax=None):
        self._rs = rs
        self._a = a
        self._b = b
        self._c = c
        super(GeneralNFWProfile,self).__init__(rhos,rs,rmin,rhomax)
        self._name = 'general_nfw'

    def _rho(self,r):
        x = r/self._rs
        return self._rhos/(x**self._a*(1+x**self._b)**((self._c-self._a)/self._b))


class UniformProfile(object):
    """ Uniform spherical profile
        rho(r) = rhos for r < rs
        rho(r) = 0    otherwise
    """
    def __init__(self,rhos=1,rs=1):
        self._name = 'uniform'
        self._rhos = rhos
        self._rs = rs

    def _rho(self,r):
        return np.where(r<rs,rhos,0)


if __name__ == '__main__':
    print "Line-of-sight Integral Package..."

    import matplotlib.pyplot as plt

    psi = np.linspace(0.01,0.1,500)
    dp = NFWProfile(1,1)

    fn0 = LoSIntegralFnFast(dp,100,10)
    fn1 = LoSIntegralFn(dp,100,10)

    dhalo = np.linspace(100,100,500)
    v0 = fn0(dhalo,psi)

    v1 = fn1(dhalo,psi)

    delta = (v1-v0)/v0

    print delta

    plt.hist(delta,range=[min(delta),max(delta)],bins=100)

    plt.show()
    

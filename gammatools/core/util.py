import os
import errno
import numpy as np
import scipy.special as spfn
import re



def update_dict(d0,d1):
    """Recursively update the contents of a python dictionary from
    another python dictionary."""

    if d1 is None: return

    for k, v in d0.iteritems():
        
        if not k in d1: continue

        if isinstance(v,dict) and isinstance(d1[k],dict):
            update_dict(d0[k],d1[k])
        else: d0[k] = d1[k]
        
def clear_dict_by_vals(d,vals):

    if not isinstance(vals,list): vals = [vals]

    for k in d.keys(): 
        if d[k] in vals: del d[k]

def clear_dict_by_keys(d,keys,clear_if_present=True):

    if not isinstance(keys,list): keys = [keys]

    for k in d.keys(): 
        if clear_if_present and k in keys: 
            del d[k]
        if not clear_if_present and not k in keys: 
            del d[k]


def dispatch_jobs(exe,args,opts,queue='xlong',resources='rhel60'):
    for x in args:
        cmd = '%s %s '%(exe,x)

        if 'queue' in opts.__dict__ and not opts.queue is None:
            queue = opts.queue

        skip_keywords = ['queue','resources','batch']
        
        for k, v in opts.__dict__.iteritems():

            if k in skip_keywords: continue                
            if not v is None and k != 'batch': cmd += ' --%s=%s '%(k,v)

        print 'bsub -q %s -R %s %s'%(queue,resources,cmd)
        os.system('bsub -q %s -R %s %s'%(queue,resources,cmd))
    

def save_object(obj,outfile):

    import cPickle as pickle
    fp = open(outfile,'w')
    pickle.dump(obj,fp,protocol = pickle.HIGHEST_PROTOCOL)
    fp.close()


class Configurable(object):

    def __init__(self):

        self._config = {}

    def config(self,key=None):
        if key is None: return self._config
        else: return self._config[key]

    def set_config(self,key,value):
        self._config[key] = value
        
    def configure(self,default_config,config,subsection=None,**kwargs):
        
        if not default_config is None:
            self._config.update(default_config)

        if not config is None:

            for k, v in self._config.iteritems():
                if k in config: self._config[k] = config[k]

            if not subsection is None and subsection in config and not \
                    config[subsection] is None:

                for k, v in self._config.iteritems():
                    if k in config[subsection]:
                        self._config[k] = config[subsection][k]
                    
#            for k, v in config.iteritems():
#                if k in self._config: self._config[k] = v

        for k, v in kwargs.iteritems():
            if k in self._config: self._config[k] = v

        for k, v in self._config.iteritems():

            if v is None or not isinstance(v,str): continue            
            if os.path.isfile(v): self._config[k] = os.path.abspath(v)

def make_dir(d):
    try: os.makedirs(d)
    except os.error, e:
        if e.errno != errno.EEXIST: raise 

def get_parameters(expr):
    m = re.compile('([a-zA-Z])([a-zA-Z0-9\_\[\]]+)')
    pars = []

    if expr is None: return pars
    for t in m.finditer(expr):
        pars.append(t.group())

    return pars

def expand_aliases(aliases,expr):
    ignore = ['max','min','sqrt','acos','pow','log','log10']
    m = re.compile('([a-zA-Z])([a-zA-Z0-9\_\[\]]+)')

    if expr is None: return expr

    has_alias = False
    for t in m.finditer(expr):

        var = t.group()
        alias = ''
        if var in aliases: alias = aliases[var]
        
        if var not in ignore and alias != '':
            expr = re.sub(var + '(?![a-zA-Z0-9\_])',
                          '(' + alias + ')',expr)
            has_alias = True

    if has_alias: return expand_aliases(aliases,expr)
    else: return expr

def havsin(theta):
    return np.sin(0.5*theta)**2

def ahavsin(x):
    return 2.0*np.arcsin(np.sqrt(x))

def separation_angle(phiA,lamA,phiB,lamB):
    return ahavsin( havsin(lamA-lamB) + 
                    np.cos(lamA)*np.cos(lamB)*havsin(phiA-phiB) )

def dtheta(ref_ra,ref_dec,ra,dec):
    return np.arccos(np.sin(dec)*np.sin(ref_dec) + 
                     np.cos(dec)*np.cos(ref_dec)*
                     np.cos(ra-ref_ra))

#def dtheta(ref_ra,ref_dec,ra,dec):
#    return np.arccos(np.sin(dec)*np.sin(ref_dec) + 
#                     np.cos(dec)*np.cos(ref_dec)*
#                     np.cos(ra-ref_ra))

def integrate(fn,lo,hi,npoints):
    edges = np.linspace(lo,hi,npoints+1)
    x = 0.5*(edges[:-1] + edges[1:])
    w = edges[1:] - edges[:-1]
    return np.sum(fn(x)*w)

def interpolate(x0,z,x):
    """Perform linear interpolation in 1 dimension.

    Parameters
    ----------
    x0:  Array defining bin edges.

    z: Array defining the function value at the bin positions defined
    by x0.

    x: Point or set of points at which the interpolated function
    should be evaluated.

    """
    w = x0[1]-x0[0]

    dx = np.zeros(shape=(x0.shape[0]-1,x.shape[0]))
    dx[:] = x
    dx = np.abs(dx.T-x0[:-1]-0.5*w)

    ix = np.argmin(dx,axis=1)
    xs = (x - x0[ix])/w

    return (z[ix]*(1-xs) + z[ix+1]*xs)

def interpolate2d(x0,y0,z,x,y):
    """Perform linear interpolation in 2 dimensions.

    Parameters
    ----------
    x0:  Array defining bin edges in x dimension.

    y0:  Array defining bin edges in y dimension.

    z: Array with the function value evlauted at the bin positions
    defined by x0.

    x: X coordinates of point or set of points at which the
    interpolated function should be evaluated.

    y: Y coordinates of point or set of points at which the
    interpolated function should be evaluated.

    """


    y = np.array(y,ndmin=1)
    x = np.array(x,ndmin=1)

    wx = x0[1:]-x0[:-1]
    wy = y0[1:]-y0[:-1]

#    dx = np.zeros(shape=(x0.shape[0]-1,x.shape[0]))
#    dx[:] = x
#    dx = np.abs(dx.T-x0[:-1]-0.5*wx)
        
#    dy = np.zeros(shape=(y0.shape[0]-1,y.shape[0]))
#    dy[:] = y
#    dy = np.abs(dy.T-y0[:-1]-0.5*wy)
    
#    ix = np.argmin(dx,axis=1)
#    iy = np.argmin(dy,axis=1)

    ix = np.digitize(x,x0[:-1])-1
    iy = np.digitize(y,y0[:-1])-1

    ix[ix > x0.shape[0] -3 ] = x0.shape[0] - 3
    iy[iy > y0.shape[0] -3 ] = y0.shape[0] - 3

    xs = (x - x0[ix])/wx[ix]
    ys = (y - y0[iy])/wy[iy]

    return (z[ix,iy]*(1-xs)*(1-ys) + z[ix+1,iy]*xs*(1-ys) +
            z[ix,iy+1]*(1-xs)*ys + z[ix+1,iy+1]*xs*ys)


def convolve2d_gauss(fn,r,sig,rmax,nstep=200):
    """Evaluate the convolution f'(r) = f(r) * g(r) where f(r) is
    azimuthally symmetric function in two dimensions and g is a
    gaussian given by:

    g(r) = 1/(2*pi*s^2) Exp[-r^2/(2*s^2)]

    Parameters
    ----------

    fn : Input function that takes a single radial coordinate parameter.

    r :  Array of points at which the convolution is to be evaluated.

    sig : Width parameter of the gaussian.

    """
    r = np.array(r,ndmin=1,copy=True)
    sig = np.array(sig,ndmin=1,copy=True)

    if sig.shape[0] > 1:
        rp = np.ones(shape=(1,1,nstep))
        rp *= np.linspace(0,rmax,nstep)

        r = r.reshape((1,r.shape[0],1))
        sig = sig.reshape((sig.shape[0],1,1))
        
        dr = rmax/float(nstep)

        sig2 = sig*sig
        x = r*rp/(sig2)
        je = spfn.ive(0,x)
        fnrp = fn(rp[0,...])
        s = np.sum(rp*fnrp/(sig2)*
                   np.exp(np.log(je)+x-(r*r+rp*rp)/(2*sig2)),axis=2)*dr

        return s
    else:

        rp = np.zeros(shape=(r.shape[0],nstep))
        rp[:] = np.linspace(0,rmax,nstep)

        rp = rp.T

        dr = rmax/float(nstep)

        sig2 = sig*sig
        x = r*rp/(sig2)
        je = spfn.ive(0,x)
        fnrp = fn(np.ravel(rp))
        fnrp = fnrp.reshape(je.shape)

        return np.sum(rp*fnrp/(sig2)*
                      np.exp(np.log(je)+x-(r*r+rp*rp)/(2*sig2)),axis=0)*dr


def convolve1(fn,r,sig,rmax):

    r = np.asarray(r)

    if r.ndim == 0: r.resize((1))

    nr = 200
    rp = np.zeros(shape=(r.shape[0],nr))
    rp[:] = np.linspace(0,rmax,nr)

    rp = rp.T

    dr = rmax/float(nr)

    sig2 = sig*sig

    x = r*rp/(sig2)

    j = spfn.iv(0,x)
    fnrp = fn(np.ravel(rp))
    fnrp = fnrp.reshape(j.shape)

#    plt.figure()
#    plt.plot(np.degrees(rp[:,50]),x[:,50])

#    plt.figure()
#    plt.plot(np.degrees(rp[:,50]),
#             rp[:,50]*j[:,50]*fnrp[:,50]*
#             np.exp(-(r[50]**2+rp[:,50]**2)/(2*sig2)))
#    plt.show()

    return np.sum(rp*j*fnrp/(sig2)*np.exp(-(r*r+rp*rp)/(2*sig2)),axis=0)*dr

RA_NGP = np.radians(192.859508333333)
DEC_NGP = np.radians(27.1283361111111)
L_CP = np.radians(122.932)
L_0 = L_CP - np.pi / 2.
RA_0 = RA_NGP + np.pi / 2.
DEC_0 = np.pi / 2. - DEC_NGP

def gc2gal(phi,th):

    v = Vector3D.createThetaPhi(np.radians(th),np.radians(phi))
    v.rotatey(np.pi/2.)

    lat = np.degrees(v.lat())
    lon = np.degrees(v.phi())

    return lon, lat

def gal2eq(l, b):

    l = np.radians(l)
    b = np.radians(b)

    sind = np.sin(b) * np.sin(DEC_NGP) + np.cos(b) * np.cos(DEC_NGP) * np.sin(l - L_0)

    dec = np.arcsin(sind)

    cosa = np.cos(l - L_0) * np.cos(b) / np.cos(dec)
    sina = (np.cos(b) * np.sin(DEC_NGP) * np.sin(l - L_0) - np.sin(b) * np.cos(DEC_NGP)) / np.cos(dec)

    dec = np.degrees(dec)

    ra = np.arccos(cosa)
    ra[np.where(sina < 0.)] = -ra[np.where(sina < 0.)]

    ra = np.degrees(ra + RA_0)

    ra = np.mod(ra, 360.)
    dec = np.mod(dec + 90., 180.) - 90.

    return ra, dec


def eq2gal(ra, dec):

    ra, dec = np.radians(ra), np.radians(dec)

    np.sinb = np.sin(dec) * np.cos(DEC_0) - np.cos(dec) * np.sin(ra - RA_0) * np.sin(DEC_0)

    b = np.arcsin(np.sinb)

    cosl = np.cos(dec) * np.cos(ra - RA_0) / np.cos(b)
    sinl = (np.sin(dec) * np.sin(DEC_0) + np.cos(dec) * np.sin(ra - RA_0) * np.cos(DEC_0)) / np.cos(b)

    b = np.degrees(b)

    l = np.arccos(cosl)
    l[np.where(sinl < 0.)] = - l[np.where(sinl < 0.)]

    l = np.degrees(l + L_0)

    l = np.mod(l, 360.)
    b = np.mod(b + 90., 180.) - 90.

    return l, b

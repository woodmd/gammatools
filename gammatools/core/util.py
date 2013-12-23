import os
import errno
import numpy as np
import scipy.special as spfn
import re

class Units(object):

    pc = 3.08568e18   # pc to cm
    kpc = pc*1e3      # kpc to cm
    msun = 1.98892e33 # solar mass to g
    gev = 1.78266e-24 # gev to g    
    mev = 1E-3*gev
    tev = 1E3*gev
    ev = 1E-9*gev

    erg = 1./1.602177E-12*ev
    g = 1.0
    m2 = 1E4
    hr = 3600.
    deg2 = np.power(np.pi/180.,2)

    msun_pc3 = msun*np.power(pc,-3) 
    msun_kpc3 = msun*np.power(kpc,-3)
    msun2_pc5 = np.power(msun,2)*np.power(pc,-5)
    msun2_kpc5 = np.power(msun,2)*np.power(kpc,-5)
    gev2_cm5 = np.power(gev,2)
    gev_cm3 = np.power(gev,1)
    gev_cm2 = np.power(gev,1)
    g_cm3 = 1.0
    cm3_s = 1.0

def format_error(v, err, nsig=1, latex=False):
    if err > 0:
        logz = math.floor(math.log10(err)) - (nsig - 1)
        z = 10 ** logz
        err = round(err / z) * z
        v = round(v / z) * z

    if latex:
        return '$%s \pm %s$' % (v, err)
    else:
        return '%s +/- %s' % (v, err)

def update_dict(d0,d1,add_new_keys=False):
    """Recursively update the contents of a python dictionary from
    another python dictionary."""

    if d0 is None or d1 is None: return

    for k, v in d0.iteritems():
        
        if not k in d1: continue

        if isinstance(v,dict) and isinstance(d1[k],dict):
            update_dict(d0[k],d1[k])
        else: d0[k] = d1[k]

    for k, v in d1.iteritems():
        if not k in d0 and add_new_keys: d0[k] = d1[k]

        
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


def dispatch_jobs(exe,args,opts,queue=None,
                  resources='rhel60',split_args=True):

    skip_keywords = ['queue','resources','batch']
    
    if queue is None and 'queue' in opts.__dict__ and \
            not opts.queue is None:
        queue = opts.queue
        
    cmd_opts = ''
    for k, v in opts.__dict__.iteritems():
        if k in skip_keywords: continue                
        if isinstance(v,list): continue
        if not v is None: cmd_opts += ' --%s=\"%s\" '%(k,v)
        
    if split_args:

        for x in args:
            cmd = '%s %s '%(exe,x)
            batch_cmd = 'bsub -q %s -R %s '%(queue,resources)
            batch_cmd += ' %s %s '%(cmd,cmd_opts)        
            print batch_cmd
            os.system(batch_cmd)

    else:
        cmd = '%s %s '%(exe,' '.join(args))
        batch_cmd = 'bsub -q %s -R %s '%(queue,resources)
        batch_cmd += ' %s %s '%(cmd,cmd_opts)        
        print batch_cmd
        os.system(batch_cmd)
            

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
    x0: Array defining coordinate mesh.

    z: Array defining the a set of scalar values at the coordinates x0.

    x: Point or set of points at which the interpolation should be evaluated.

    """

    x = np.array(x,ndmin=1)

    if x0.shape != z.shape:
        raise('Coordinate and value arrays do not have equal dimension.')

    wx = x0[1:]-x0[:-1]

    ix = np.digitize(x,x0)-1
    ix[ix<0]=0
    ix[ix > x0.shape[0] -2 ] = x0.shape[0] - 2
    xs = (x - x0[:-1][ix])/wx[ix]

    return (z[ix]*(1-xs) + z[ix+1]*xs)

def interpolate2d(x0,y0,z,x,y):
    """Perform linear interpolation in 2 dimensions from a 2D mesh.

    Parameters
    ----------
    x0:  Array defining mesh coordinates in x dimension.

    y0:  Array defining mesh coordinates in y dimension.

    z: Array with the function value evlauted at the set of coordinates
    defined by x0, y0.  This must have the dimension N x M where N and M are
    the number of elements in x0 and y0 respectively.

    x: X coordinates of point or set of points at which the
    interpolated function should be evaluated.  Must have the same dimension
    as y.

    y: Y coordinates of point or set of points at which the
    interpolated function should be evaluated.  Must have the same dimension
    as x.

    """

    y = np.array(y,ndmin=1)
    x = np.array(x,ndmin=1)

    wx = x0[1:]-x0[:-1]
    wy = y0[1:]-y0[:-1]

    ix = np.digitize(x,x0)-1
    iy = np.digitize(y,y0)-1

    ix[ix<0]=0
    iy[iy<0]=0
    ix[ix > x0.shape[0] -2 ] = x0.shape[0] - 2
    iy[iy > y0.shape[0] -2 ] = y0.shape[0] - 2

    xs = (x - x0[:-1][ix])/wx[ix]
    ys = (y - y0[:-1][iy])/wy[iy]

    return (z[ix,iy]*(1-xs)*(1-ys) + z[ix+1,iy]*xs*(1-ys) +
            z[ix,iy+1]*(1-xs)*ys + z[ix+1,iy+1]*xs*ys)

def interpolatend(x0,z,x):
    """Perform linear interpolation over an N-dimensional mesh.

    Parameters
    ----------
    x0:  List of N arrays defining mesh coordinates in each of N dimensions.

    z: N-dimesional array of scalar values evaluated on the coordinate
    mesh defined by x0.  The number of elements along each dimension must
    equal to the corresponding number of mesh points in x0.

    x: NxM numpy array specifying the M points in N-dimensional space at
    which the interpolation should be evaluated.

    """

    x = np.array(x,ndmin=2)
    ndim = len(x0)

    index = np.zeros(shape=(2**ndim,ndim,len(x[0])),dtype=int)
    psum = np.ones(shape=(2**ndim,len(x[0])))

    for i, t in enumerate(x0):

        p = np.array(t,ndmin=1)
        w = p[1:]-p[:-1]
        ix = np.digitize(x[i],p)-1
        ix[ix<0]=0
        ix[ix > x0[i].shape[0] -2 ] = len(x0[i]) - 2
        xs = (x[i] - x0[i][:-1][ix])/w[ix]


        for j in range(len(psum)):
            if j & (1<<i):
                index[j][i] = ix+1
                psum[j] *= xs
            else:
                index[j][i] = ix
                psum[j] *= (1.0-xs)

#    print index
#    print index[0].shape
#    print z[np.ix_(index[0])]

    for j in range(len(psum)):

        idx = []
        for i in range(ndim): idx.append(index[j][i])

#        print idx

        psum[j] *= z[idx]

    return np.sum(psum,axis=0)

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

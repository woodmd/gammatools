import os
import errno
import numpy as np
import scipy.special as spfn
import re
import copy

try:
    import cPickle as pickle
except ImportError:
    import pickle

import gzip
import bisect
import inspect
from collections import OrderedDict

from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

def mkdir(dir):
    if not os.path.exists(dir):  os.makedirs(dir)
    return dir

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element.
    """
    from xml.dom import minidom
    import xml.etree.cElementTree as et

    rough_string = et.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")  
    
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

def common_prefix(strings):
    """ Find the longest string that is a prefix of all the strings.
    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix

def string_to_array(x,delimiter=',',dtype=float):
    return np.array([t for t in x.split(delimiter)],dtype=dtype)

def tolist(x):
    """
    convenience function that takes in a 
    nested structure of lists and dictionaries
    and converts everything to its base objects.
    This is useful for dupming a file to yaml.
    
        (a) numpy arrays into python lists

            >>> type(tolist(np.asarray(123))) == int
            True
            >>> tolist(np.asarray([1,2,3])) == [1,2,3]
            True

        (b) numpy strings into python strings.

            >>> tolist([np.asarray('cat')])==['cat']
            True

        (c) an ordered dict to a dict

            >>> ordered=OrderedDict(a=1, b=2)
            >>> type(tolist(ordered)) == dict
            True

        (d) converts unicode to regular strings

            >>> type(u'a') == str
            False
            >>> type(tolist(u'a')) == str
            True

        (e) converts numbers & bools in strings to real represntation,
            (i.e. '123' -> 123)

            >>> type(tolist(np.asarray('123'))) == int
            True
            >>> type(tolist('123')) == int
            True
            >>> tolist('False') == False
            True
    """
    if isinstance(x,list):
        return map(tolist,x)
    elif isinstance(x,dict):
        return dict((tolist(k),tolist(v)) for k,v in x.items())
    elif isinstance(x,np.ndarray) or \
            isinstance(x,np.number):
        # note, call tolist again to convert strings of numbers to numbers
        return tolist(x.tolist())
#    elif isinstance(x,PhaseRange):
#        return x.tolist(dense=True)
    elif isinstance(x,OrderedDict):
        return dict(x)
    elif isinstance(x,basestring) or isinstance(x,np.str):
        x=str(x) # convert unicode & numpy strings 
        try:
            return int(x)
        except:
            try:
                return float(x)
            except:
                if x == 'True': return True
                elif x == 'False': return False
                else: return x
    else:
        return x

def merge_dict(d0,d1,add_new_keys=False,append_arrays=False,skip_values=None):
    """Recursively merge the contents of python dictionary d0 with
    the contents of another python dictionary, d1.

    add_new_keys : Do not skip keys that only exist in d1.

    append_arrays : If an element is a numpy array set the value of
    that element by concatenating the two arrays.
    """
    
    if skip_values is None: skip_values = []
    
    if d1 is None: return d0
    elif d0 is None: return d1
    elif d0 is None and d1 is None: return {}

    od = {}
    
    for k, v in d0.items():

        if not k in d1 or k in d1 and d1[k] in skip_values:
            od[k] = copy.copy(d0[k])
        elif isinstance(v,dict) and isinstance(d1[k],dict):
            od[k] = merge_dict(d0[k],d1[k],add_new_keys,append_arrays)
        elif isinstance(v,list) and isinstance(d1[k],str):
            od[k] = d1[k].split(',')            
        elif isinstance(v,np.ndarray) and append_arrays:
            od[k] = np.concatenate((v,d1[k]))
        elif (d0[k] is not None and d1[k] is not None) and \
                (type(d0[k]) != type(d1[k])):
            raise Exception('Conflicting types in dictionary merge. ' + k + str(type(d0[k])) + str(type(d1[k])))
        else: od[k] = copy.copy(d1[k])

    if add_new_keys:
        for k, v in d1.items(): 
            if not k in d0: od[k] = copy.copy(d1[k])

    return od
    
def update_dict(d0,d1,add_new_keys=False,append=False):
    """Recursively update the contents of python dictionary d0 with
    the contents of python dictionary d1."""

    if d0 is None or d1 is None: return
    
    for k, v in d0.items():
        
        if not k in d1: continue

        if isinstance(v,dict) and isinstance(d1[k],dict):
            update_dict(d0[k],d1[k],add_new_keys,append)
        elif isinstance(v,np.ndarray) and append:
            d0[k] = np.concatenate((v,d1[k]))
        else: d0[k] = d1[k]

    if add_new_keys:
        for k, v in d1.items(): 
            if not k in d0: d0[k] = d1[k]
        
def clear_dict_by_vals(d,vals):

    if not isinstance(vals,list): vals = [vals]

    for k in d.keys(): 
        if d[k] in vals: del d[k]

def extract_dict_by_keys(d,keys,exclusive=False):
    """Extract a subset of the input dictionary.  If exclusive==False
    the output dictionary will contain all elements with keys in the
    input key list.  If exclusive==True the output dictionary will
    contain all elements with keys not in the key list."""

    if exclusive:
        return dict((k, d[k]) for k in d.keys() if not k in keys)
    else:
        return dict((k, d[k]) for k in d.keys() if k in keys)

def dispatch_jobs(exe,args,opts,W=300,
                  resources='rhel60',skip=None,split_args=True):

    skip_keywords = ['queue','resources','batch','W']

    if not skip is None: skip_keywords += skip
        
    cmd_opts = ''
    for k, v in opts.__dict__.items():
        if k in skip_keywords: continue                
        if isinstance(v,list): continue

        if isinstance(v,bool) and v: cmd_opts += ' --%s '%(k)
        elif isinstance(v,bool): continue
        elif not v is None: cmd_opts += ' --%s=\"%s\" '%(k,v)
        
    if split_args:

        for x in args:
            cmd = '%s %s '%(exe,x)
            batch_cmd = 'bsub -W %s -R %s '%(W,resources)
            batch_cmd += ' %s %s '%(cmd,cmd_opts)        
            print(batch_cmd)
            os.system(batch_cmd)

    else:
        cmd = '%s %s '%(exe,' '.join(args))
        batch_cmd = 'bsub -W %s -R %s '%(W,resources)
        batch_cmd += ' %s %s '%(cmd,cmd_opts)        
        print(batch_cmd)
        os.system(batch_cmd)
            

def save_object(obj,outfile,compress=False,protocol=pickle.HIGHEST_PROTOCOL):

    if compress:
        fp = gzip.GzipFile(outfile + '.gz', 'wb')
    else:
        fp = open(outfile,'wb')
    pickle.dump(obj,fp,protocol = protocol)
    fp.close()

def load_object(infile):

    if not re.search('\.gz?',infile) is None:
        fp = gzip.open(infile)
    else:
        fp = open(infile,'rb')

    o = pickle.load(fp)
    fp.close()
    return o

def expand_array(*x):
    """Reshape a list of arrays of dimension N such that size of each
    dimension is set to the largest size of any array in the list.
    Every output array has dimension NxM where M = Prod_i =
    max(N_i)."""

    ndim = len(x)
    
    shape = None
    for i in range(len(x)): 
        z = np.array(x[i])
        if shape is None: shape = z.shape
        shape = np.maximum(shape,z.shape)
    
    xv = np.zeros((ndim,np.product(shape)))
    for i in range(len(x)):
        xv[i] = np.ravel(np.array(x[i])*np.ones(shape))

    return xv, shape

def bitarray_to_int(x,big_endian=False):

    if x.dtype == 'int': return x
    elif x.dtype == 'float': return x.astype('int')
    
    o = np.zeros(x.shape[0],dtype=int)

    for i in range(x.shape[1]):
        if big_endian: o += (1<<i)*x[:,::-1][:,i]
        else: o += (1<<i)*x[:,i]

    return o

def bitfield_to_list(x):    
    o = []
    for i in range(32):
        if (x&(1<<i)): o.append(i)

            
def list_to_bitfield(x):

    if x is None: return x
    if not isinstance(x,list): return 2**x 
    if isinstance(x,list) and len(x) == 0: return None
    v = 0
    for t in x: v += 2**t        
    return v

def make_dir(d):
    try: os.makedirs(d)
    except OSError as e:
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

def separation_angle_havsin(phiA,lamA,phiB,lamB):
    return ahavsin( havsin(lamA-lamB) + 
                    np.cos(lamA)*np.cos(lamB)*havsin(phiA-phiB) )

def separation_angle(ref_ra,ref_dec,ra,dec):
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

def find_root(x,y,y0):
    """Solve for the x coordinate at which f(x)-y=0 where f(x) is
    a smooth interpolation of the histogram contents."""

    fn = UnivariateSpline(x,y-y0,k=2,s=0)
    return brentq(lambda t: fn(t),x[0],x[-1])

def find_fn_root(fn,x0,x1,y0):
    """Solve for the x coordinate at which f(x)-y=0 where f(x) is
    a smooth interpolation of the histogram contents."""

    return brentq(lambda t: fn(t)-y0,x0,x1)

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
    x0:  List of arrays defining mesh coordinates in each of N dimensions.

    z: N-dimesional array of scalar values evaluated on the coordinate
    mesh defined by x0.  The number of elements along each dimension must
    equal to the corresponding number of mesh points in x0 (N_tot = Prod_i N_i).

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

    for j in range(len(psum)):

        idx = []
        for i in range(ndim): idx.append(index[j][i])
        psum[j] *= z[idx]

    return np.sum(psum,axis=0)

def percentile(x,cdf,frac=0.68):
    """Given a cumulative distribution function C(x) find the value
    of x for which C(x) = frac."""
    indx = np.searchsorted(cdf, frac) - 1
    return ((frac - cdf[indx])/(cdf[indx+1] - cdf[indx])
            *(x[indx+1] - x[indx]) + x[indx])

def edge_to_center(edges):
    return 0.5*(edges[1:]+edges[:-1])

def edge_to_width(edges):
    return (edges[1:]-edges[:-1])

def convolve2d_king(fn,r,sig,gam,rmax,nstep=200):
    """Evaluate the convolution f'(x,y) = f(x,y) * g(x,y) where f(r) is
    azimuthally symmetric function in two dimensions and g is a
    King function given by:

    g(r,sig,gam) = 1/(2*pi*sig^2)*(1-1/gam)*(1+gam/2*(r/sig)**2)**(-gam)

    Parameters
    ----------

    fn  : Input function that takes a single radial coordinate parameter.

    r   :  Array of points at which the convolution is to be evaluated.

    sig : Width parameter of the King function.

    gam : Gamma parameter of the King function.

    """

    r = np.array(r,ndmin=1,copy=True)
    sig = np.array(sig,ndmin=1,copy=True)
    gam = np.array(gam,ndmin=1,copy=True)

    r2p = edge_to_center(np.linspace(0,rmax**2,nstep+1))
    r2w = edge_to_width(np.linspace(0,rmax**2,nstep+1))

    if sig.shape[0] > 1:        
        r2p = r2p.reshape((1,1,nstep))
        r2w = r2w.reshape((1,1,nstep))
        r = r.reshape((1,r.shape[0],1))
        sig = sig.reshape(sig.shape + (1,1))
        gam = sig.reshape(gam.shape + (1,1))
        saxis = 2
    else:
        r2p = r2p.reshape(1,nstep)
        r2w = r2w.reshape(1,nstep)
        r = r.reshape(r.shape + (1,))
        saxis = 1

    u = 0.5*(r/sig)**2
    v = 0.5*r2p*(1./sig)**2
    vw = 0.5*r2w*(1./sig)**2

    z = 4*u*v/(gam+u+v)**2
    hgfn = spfn.hyp2f1(gam/2.,(1.+gam)/2.,1.0,z)
    fnrp = fn(np.sqrt(r2p))
    s = np.sum(fnrp*(gam-1.0)/gam*np.power(gam/(gam+u+v),gam)*hgfn*vw,
               axis=saxis)

    return s

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

    rp = edge_to_center(np.linspace(0,rmax,nstep+1))
    dr = rmax/float(nstep)
    fnrp = fn(rp)

    if sig.shape[0] > 1:        
        rp = rp.reshape((1,1,nstep))
        fnrp = fnrp.reshape((1,1,nstep))
        r = r.reshape((1,r.shape[0],1))
        sig = sig.reshape(sig.shape + (1,1))
        saxis = 2
    else:
        rp = rp.reshape(1,nstep)
        fnrp = fnrp.reshape(1,nstep)
        r = r.reshape(r.shape + (1,))
        saxis = 1

    sig2 = sig*sig
    x = r*rp/(sig2)
    je = spfn.ive(0,x)

    s = np.sum(rp*fnrp/(sig2)*
               np.exp(np.log(je)+x-(r*r+rp*rp)/(2*sig2)),axis=saxis)*dr

    return s

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

    l = np.array(l,ndmin=1)
    b = np.array(b,ndmin=1)
    
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

    ra = np.array(ra,ndmin=1)
    dec = np.array(dec,ndmin=1)
    
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

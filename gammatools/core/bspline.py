import numpy as np
import copy

def fit_svd(x,y,yerr,n=2):

    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    
    A = np.zeros(shape=(len(x),n+1))

    for i in range(n+1):
        A[:,i] = np.power(x,i)/yerr

    b = y/yerr
    (u,s,v) = np.linalg.svd(A,full_matrices=False)
    ub = np.sum(u.T*b,axis=1)
    a = np.sum(ub*v.T/s,axis=1)

    vn = v.T/s
    cov = np.dot(vn,vn.T)

    return a, cov

class BSpline(object):
    """Class representing a 1-D B-spline."""
    
    m1 = np.array([[1.0]])

    m2 = np.array([[1.0,-1.0],[0.0,1.0]])

    m3 = 0.5*np.array([[1.0,-2.0,1.0],
                       [1.0,2.0,-2.0],
                       [0.0,0.0, 1.0]])

    m4 = (1./6.)*np.array([[1.0,-3.0, 3.0,-1.0],
                           [4.0, 0.0,-6.0, 3.0],
                           [1.0, 3.0, 3.0,-3.0],
                           [0.0, 0.0, 0.0, 1.0]])


    def __init__(self,k,w,nd):
        """Initialize a 1-D B-spline object.

        @param k Knot vector.
        @param w Weights vector.
        @param nd Order of spline (1=step-wise, 2=linear, 3=quadratic, 4=cubic)
        """
        self._k = k
        self._w = w
        self._nd = nd
        self._m = BSpline.get_matrix(nd)

    @staticmethod
    def get_matrix(nd):
        if nd == 1: return BSpline.m1
        elif nd == 2: return BSpline.m2
        elif nd == 3: return BSpline.m3
        elif nd == 4: return BSpline.m4
        else:
            raise Exception('Spline order %i not supported.'%nd)
        
    @staticmethod
    def fit(x,y,yerr,k,nd):

        m = BSpline.get_matrix(nd)

        if isinstance(k,int):
            k = np.linspace(x[0],x[-1],k)

        nrow = k.shape[0]
        ndata = x.shape[0]
        
        print nrow, ndata

        a = np.zeros(shape=(ndata,nrow))

        if yerr is None: yerr = np.ones(ndata)        
        b = y/yerr
        
        ix, px = BSpline.poly(x,k,nd)
        for i in range(ndata):
            for j in range(nd):
#                a[i,ix[i]:ix[i]+nd] += m[j]*px[j,i]
                a[i,ix[i]+j] += np.sum(m[j]*px[:,i]/yerr[i])

        (u,s,v) = np.linalg.svd(a,full_matrices=False)
        ub = np.sum(u.T*b,axis=1)
        w = np.sum(ub*v.T/s,axis=1)

        return BSpline(k,w,nd)

    @staticmethod
    def poly(x,k,nd,ndx=0):
        """Evaluate polynomial vector for a set of evaluation points
        (x), knots (k), and spline order (nd)."""

        import scipy.special as spfn

        kw = k[1] - k[0]

        dx = np.zeros(shape=(k.shape[0],x.shape[0]))
        dx[:] = x
        dx = np.abs(dx.T-k-0.5*kw)
        
        imax = k.shape[0]-nd

        ix = np.argmin(dx,axis=1)
        ix[ix>imax] = imax

        xp = (x-k[ix])/kw

        if ndx == 0:
            c = np.ones(nd)
        elif ndx == 1:
            c = np.zeros(nd)
            c[1:] = np.linspace(1,nd-1,nd-1)
        elif ndx > 0:
            for i in range(nd):
                
                j = i-ndx

                if i+2+ndx < nd: c[i] = 0.0
                else: c[i] = spfn.gamma(j)

            n = np.linspace(0,nd-1-ndx,nd)
            c = spfn.gamma(n)

        px = np.zeros(shape=(nd,x.shape[0]))
        for i in range(0,nd): 

            exp = max(i-ndx,0)
            px[i] = np.power(xp,exp)*c[i] 


        return ix, px*np.power(kw,-ndx)

    def __call__(self,x,ndx=0):
        return self.eval(x,ndx)

    def get_dict_repr(self):

        o = {}
        o['knots'] = self._k
        o['weights'] = self._w
        o['order'] = self._nd
        return o

    @staticmethod
    def create_from_dict(o):
        return BSpline(o['knots'], o['weights'], o['order'])
    
    def eval(self,x,ndx=0):

        x = np.asarray(x)
        if x.ndim == 0: x = x.reshape((1))
        
        ix, px = BSpline.poly(x,self._k,self._nd,ndx)
        wx = np.ones(shape=(self._nd,x.shape[0]))

        for i in range(self._nd): wx[i] = self._w[ix+i]

        s = np.zeros(x.shape[0])
        for i in range(self._nd):
            for j in range(self._nd):
                s += wx[i]*self._m[i,j]*px[j]

        return s

    def get_expr(self,xvar):
        """Return symbolic representation in ROOT compatible
        format."""

        cut = []
        ncut = len(self._k)-(self._nd-1)

        kw = self._k[1] - self._k[0]

        for i in range(ncut):

            if ncut == 1: cond = '1.0'
            elif i == 0:
                cond = '%s <= %f'%(xvar,self._k[i+1])
            elif i == ncut-1:
                cond = '%s > %f'%(xvar,self._k[i])
            else:
                cond = '(%s > %f)*(%s <= %f)'%(xvar,self._k[i],
                                               xvar,self._k[i+1])

            wexp = []

            for j in range(self._nd):

                ws = 0
                for k in range(self._nd):
#
#                    print i, j, k, ws
                    ws += self._w[i+k]*self._m[k,j]


                if j == 0: px = '(%f)*(1.0)'%(ws)
                else:
                    px = '(%f)*(pow((%s-%f)/%f,%i))'%(ws,xvar,
                                                      self._k[i],kw,j)

                wexp.append(px)
                

            cut.append('(%s)*(%s)'%(cond,'+'.join(wexp)))

        return '+'.join(cut)


class PolyFn(object):
    """Class representing a polynomial of the form:

    f(x) = Sum_{i=0}^{n+1} a_{i}*x^i
    """
    
    def __init__(self,n,a=None,cov=None):
        """Initialize 1-D polynomial function object.

        @param n Order of polynomial.        
        """

        self._n = n
        if a is None: self._a = np.zeros(n+1)
        else: self._a = np.array(a,copy=True,ndmin=1)

        if cov is None: self._cov = np.zeros((len(a),len(a)))
        else: self._cov = cov

    def coeff(self):
        return self._a

    def err(self):
        return np.sqrt(np.diag(self._cov))

    def cov(self):
        return self._cov
    
    def get_expr(self,xvar):
        """Return symbolic representation."""

        expr = ''
        
        for i in range(len(self._a)):
            if i == 0: expr += '%10.5g'%self._a[i]
            else: expr += '+ (%10.5g)*pow(%s,%i)'%(self._a[i],xvar,i)

        return expr

    def save_to_dict(self):
        o = {}
        o['order'] = self._n
        o['coeff'] = self._a.tolist()
        o['type'] = self.__class__.__name__
        return o

    @staticmethod
    def create_from_dict(o):
        return PolyFn(o['order'],o['coeff'])
    
    @staticmethod
    def fit_hist(h,n,rng):
        
        msk = (hq._center[0] > rng[0]) & (hq._center[0] < rng[1])
        a = fit_svd(hq._center[0][msk],
                    hq._counts[msk],np.sqrt(hq._var[msk]),n)

        return PolyFn(n,a)

    @staticmethod
    def fit(n,x,y,yerr=None):
        if yerr is None: yerr = np.ones(len(x))
        a, cov = fit_svd(x,y,yerr,n)            
        return PolyFn(n,a,cov)

    def roots(self,a=None,offset=0,imag=True):
        
        if a is None: a = copy.deepcopy(self._a)
        else: a = np.array(a,copy=True)
        
        a[0] -= offset
        roots = np.roots(a[::-1])
    
        if not imag:
            roots = roots[np.imag(roots)==0]
            return np.real(roots)
        else:
            return roots

    def __call__(self,x,a=None):

        if a is None: a = self._a        
        return np.polyval(a[::-1],x)

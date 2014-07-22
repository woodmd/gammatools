from parameter_set import *
from util import update_dict
from model_fn import ParamFn
import inspect
from gammatools.core.config import Configurable
#from iminuit import Minuit as Minuit2

class NLFitter(object):
    """Base class for non-linear fitting routines."""
    def __init__(self,objfn):
        self._objfn = objfn

class IMinuitFitter(object):

    def __init__(self,objfn,tol=1E-3):
        super(IMinuitFitter,self).__init__(objfn)

    def fit(self,pset=None):
        if pset is None: pset = self._objfn.param(True)

#        kwargs = {}

#        for p in pset:
#            kwargs { p.name() }

        m = Minuit2(lambda x: self._objfn.eval(x))

        print m.fit()


class MinuitFitter(object):
    """Wrapper class for performing function minimization with minuit."""

    def __init__(self,objfn,tol=1E-3):
        self._objfn = objfn
        self._fval = 0
        self._tol = tol
        self._maxcalls = 1000

    def rnd_scan(self,par_index=None,nscan=100,scale=1.3):

        pset = copy.copy(self._objfn.param())
        p = pset.array()

        if par_index is None: par_index = range(pset.npar())

        prnd = np.ones((p.shape[0],nscan,1))
        prnd *= p

        for i in par_index:
            rnd = np.random.uniform(0.0,1.0,(nscan,1))
            prnd[i] = p[i] - p[i]/scale + rnd*(scale*p[i] - (p[i] - p[i]/scale))

        lnl = self._objfn.eval(prnd)

        imin = np.argmin(lnl)

        pset.update(prnd[:,imin])

        print self._objfn.eval(pset)

        return pset

    def profile(self,pset,pname,pval,refit=True):

#        pset = copy.deepcopy(self._objfn.param())
        pset = copy.deepcopy(pset)

        fval = []
        pset.getParByName(pname).fix(True)

        if refit is True:

            for p in pval:  
         
                pset.setParByName(pname,p)
                pset_fit = self.fit(pset)
                fval.append(pset_fit.fval())
        else:
            for p in pval:           
                pset.setParByName(pname,p)

                v = self._objfn.eval(pset)

#                print p, v, pset.getParByName('agn_norm').value()

                fval.append(v)
            
        return np.array(fval)

    def minimize(self,pset=None):

        if pset is None: pset = self._objfn.param(True)

        npar = pset.npar()

        fixed = pset.fixed
        lo_lims = npar*[None]
        hi_lims = npar*[None]
        lims = []
        
        for i, p in enumerate(pset):
            if not p.lims is None: lims.append(p.lims)
            else: lims.append([0.0,0.0])
                            
        print pset.array()

        minuit = Minuit(lambda x: self._objfn.eval(x),
                        pset.array(),fixed=fixed,limits=lims,
                        tolerance=self._tol,strategy=1,
                        printMode=-1,
                        maxcalls=self._maxcalls)
        (pars,fval) = minuit.minimize()

        cov = minuit.errors()
        pset.update(pars)
            
        return FitResults(pset,fval,cov)

    def plot_lnl_scan(self,pset):
        print pset

        fig = plt.figure(figsize=(12,8))
        for i in range(9):

            j = i+4
            
            p = pset.makeParameterArray(j,
                                        np.linspace((pset[j].flat[0]*0.5),
                                                    (pset[j].flat[0]*2),50))
            y = self._objfn.eval(p)
            ax = fig.add_subplot(3,3,i+1)
            ax.set_title(p.getParByIndex(j).name())
            plt.plot(p[j],y-pset.fval())
            plt.axvline(pset[j])

    @staticmethod
    def fit(objfn,**kwargs):
        """Convenience method for fitting."""
        fitter = Fitter(objfn,**kwargs)
        return fitter.minimize()


class BFGSFitter(Configurable):

    default_config = { 'pgtol'   : 1E-5, 'factr' : 1E7 }

    def __init__(self,objfn,**kwargs):
        super(BFGSFitter,self).__init__(**kwargs)    
        self._objfn=objfn

    @property
    def objfn(self):
        return self._objfn

    @staticmethod
    def fit(fn,p0,**kwargs):        

        if not isinstance(fn,ParamFn):
            fn = ParamFn.create(fn,p0)

        fitter = BFGSFitter(fn,**kwargs)
        return fitter.minimize(**kwargs)

    def minimize(self,pset=None,**kwargs):

        if pset is None: pset = self._objfn.param(True)

        bounds = []
        for p in pset:
            if p.fixed:
                bounds.append([p.value.flat[0],p.value.flat[0]])
            else:
                bounds.append(p.lims)

        from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs

        bfgs_kwargs = self.config
#{'pgtol' : 1E-5, bounds=bounds, 'factr' : 1E7 }

        bfgs_kwargs['bounds'] = bounds
        update_dict(bfgs_kwargs,kwargs)

        res = fmin_bfgs(self._objfn,pset.array(),None,
                        approx_grad=1,**bfgs_kwargs)#,factr=1./self._tol)

        pset.update(res[0])        
        self._fit_results = FitResults(pset,res[1])

        # How to compute errors?
        
        return copy.deepcopy(self._fit_results)

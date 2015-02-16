"""
Provides a  convenience class to call Minuit, mimicking the interface to scipy.optimizers.fmin.

author: Eric Wallace <wallacee@uw.edu>
$Header: /nfs/slac/g/glast/ground/cvs/users/mdwood/python/minuit.py,v 1.1 2013/03/21 00:52:17 mdwood Exp $

"""
import sys, os
# normal CMT setup does not put ROOT.py in the python path
#if sys.platform == 'win32':
#    import win32api
#    console_title = win32api.GetConsoleTitle()

#try:
#    import ROOT
#except:
#    sys.path.append(os.path.join(os.environ['ROOTSYS'], 'bin'))
#    import ROOT

#if sys.platform == 'win32':
#    win32api.SetConsoleTitle(console_title) #restore title bar! 
#    import pyreadline # fix for tab completion
#    pyreadline.parse_and_bind('set show-all-if-ambiguous on')



import numpy as np
from numpy.linalg import inv

def rosenbrock(x):
    """Rosenbrock function, to use for testing."""
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def rosengrad(x):
    """Gradient of Rosenbrock function, for testing."""
    drdx = -2*((1-x[0])+200*x[0]*(x[1]-x[0]**2))
    drdy = 200*(x[1]-x[0]**2)
    return np.asarray([drdx,drdy])

class FCN(object):
    """Wrap a python callable as an FCN object passable to Minuit.

    __init__() params:
        fcn : A Python callable
        pars : A sequence of starting parameters for fcn
        args : An optional sequence of extra arguments to fcn
        gradient : An optional function to compute the function gradient.
                    Should take a list of parameters and return a list
                    of first derivatives. 
    """
    def __init__(self,fcn,pars,args=(),gradient = None):
        self.fcn = fcn
        self.p = pars
        self.args = args
        self.npars = len(self.p)
        self.iflag = 0
        self.fval = self.fcn(self.p,*self.args)
        self.grad_fcn = gradient

    def __call__(self,nargs,grads,fval,pars,iflag):
        self.p = np.asarray([pars[i] for i in xrange(self.npars)])
        self.iflag = iflag
        self.fval = fval[0] = self.fcn(self.p,*self.args)
        if self.grad_fcn:
            grad = self.grad_fcn(self.p,*self.args)
            for i in xrange(len(grad)):
                grads.__setitem__(i,grad[i])

class Minuit(object):
    """A wrapper class to initialize a minuit object with a numpy array.

    Positional args:
        myFCN : A python callable
        params : An array (or other python sequence) of parameters

    Keyword args:

        limits [None] : a nested sequence of (lower_limit,upper_limit) for each parameter.
        steps [[.1]*npars] : Estimated errors for the parameters, used as an initial step size.
        tolerance [.001] : Tolerance to test for convergence.  Minuit considers convergence to be
                            when the estimated distance to the minimum (edm) is <= .001*up*tolerance,
                            or 5e-7 by default.
        up [.5]  : Change in the objective function that determines 1-sigma error.  .5 if 
                          the objective function is -log(Likelihood), 1 for chi-squared.
        max_calls [10000] : Maximum number of calls to the objective function.
        param_names ['p0','p1',...] : a list of names for the parameters
        args [()] : a tuple of extra arguments to pass to myFCN and gradient.
        gradient [None] : a function that takes a list of parameters and returns a list of 
                          first derivatives of myFCN.  Assumed to take the same args as myFCN.
        force_gradient [0] : Set to 1 to force Minuit to use the user-provided gradient function.
        strategy[1] : Strategy for minuit to use, from 0 (fast) to 2 (safe) 
        fixed [False, False, ...] : If passed, an array of all the parameters to fix
    """


    def __init__(self,myFCN,params,**kwargs):

        from ROOT import TMinuit,Long,Double

        self.limits = np.zeros((len(params),2))
        self.steps = .04*np.ones(len(params)) # about 10 percent in log10 space
        self.tolerance = .001
        self.maxcalls = 10000
        self.printMode = 0
        self.up = 0.5
        self.param_names = ['p%i'%i for i in xrange(len(params))]
        self.erflag = Long()
        self.npars = len(params)
        self.args = ()
        self.gradient = None
        self.force_gradient = 0
        self.strategy = 1
        self.fixed = np.zeros_like(params)
        self.__dict__.update(kwargs)

        self.params = np.asarray(params,dtype='float')
        self.fixed=np.asarray(self.fixed,dtype=bool)
        self.fcn = FCN(myFCN,self.params,args=self.args,gradient=self.gradient)
        self.fval = self.fcn.fval
        self.minuit = TMinuit(self.npars)
        self.minuit.SetPrintLevel(self.printMode)
        self.minuit.SetFCN(self.fcn)
        if self.gradient:
            self.minuit.mncomd('SET GRA %i'%(self.force_gradient),self.erflag)
        self.minuit.mncomd('SET STR %i'%self.strategy,Long())


        for i in xrange(self.npars):

            if self.limits[i][0] is None: self.limits[i][0] = 0.0
            if self.limits[i][1] is None: self.limits[i][1] = 0.0
            
            self.minuit.DefineParameter(i,self.param_names[i],self.params[i],self.steps[i],self.limits[i][0],self.limits[i][1])

        self.minuit.SetErrorDef(self.up)

        for index in np.where(self.fixed)[0]:
            self.minuit.FixParameter(int(index))



    def minimize(self,method='MIGRAD'):

        from ROOT import TMinuit,Long,Double

        self.minuit.mncomd('%s %i %f'%(method, self.maxcalls,self.tolerance),self.erflag)
        for i in xrange(self.npars):
            val,err = Double(),Double()
            self.minuit.GetParameter(i,val,err)
            self.params[i] = val
        self.fval = self.fcn.fcn(self.params)
        return (self.params,self.fval)

    def errors(self,method='HESSE'):
        """method ['HESSE']   : how to calculate the errors; Currently, only 'HESSE' works."""
        if not np.any(self.fixed):
            mat = np.zeros(self.npars**2)
            if method == 'HESSE':
                self.minuit.mnhess()
            else:
                raise Exception("Method %s not recognized." % method)
            self.minuit.mnemat(mat,self.npars)
            return mat.reshape((self.npars,self.npars))
        else:
            # Kind of ugly, but for fixed parameters, you need to expand out the covariance matrix.
            nf=int(np.sum(~self.fixed))
            mat = np.zeros(nf**2)
            if method == 'HESSE':
                self.minuit.mnhess()
            else:
                raise Exception("Method %s not recognized." % method)
            self.minuit.mnemat(mat,nf)

            # Expand out the covariance matrix.
            cov = np.zeros((self.npars,self.npars))
            cov[np.outer(~self.fixed,~self.fixed)] = np.ravel(mat)
            return cov

    def uncorrelated_minos_error(self):
        """ Kind of a kludge, but a compromise between the speed of
            HESSE errors and the accuracy of MINOS errors.  It accounts
            for non-linearities in the likelihood by varying each function
            until the value has fallen by the desired amount using hte
            MINOS code. But does not account for correlations between
            the parameters by fixing all other parameters during the
            minimzation. 
            
            Again kludgy, but what is returned is an effective covariance
            matrix. The diagonal errors are calcualted by averaging the
            upper and lower errors and the off diagonal terms are set
            to 0. """

        cov = np.zeros((self.npars,self.npars))

        # fix all parameters
        for i in range(self.npars):
            self.minuit.FixParameter(int(i))

        # loop over free paramters getting error
        for i in np.where(self.fixed==False)[0]:
            self.minuit.Release(int(i))

            # compute error
            self.minuit.mnmnos() 
            low,hi,parab,gcc=ROOT.Double(),ROOT.Double(),ROOT.Double(),ROOT.Double()

            # get error
            self.minuit.mnerrs(int(i),low,hi,parab,gcc)
            cov[i,i]=((abs(low)+abs(hi))/2.0)**2
            self.minuit.FixParameter(int(i))

        for i,fixed in enumerate(self.fixed):
            if fixed:
                self.minuit.FixParameter(int(i))
            else:
                self.minuit.Release(int(i))

        return cov

def mycov(self,full_output=False,min_step=1e-5,max_step=1,max_iters=5,target=0.5,min_func=1e-4,max_func=4):
    """Perform finite differences on the _analytic_ gradient provided by user to calculate hessian/covariance matrix.

    Keyword args:

        full_output [False] : if True, return information about convergence, else just the covariance matrix
        min_step    [1e-5]  : the minimum step size to take in parameter space
        max_step    [1]     : the maximum step size to take in parameter sapce
        max_iters   [5]     : maximum number of iterations to attempt to converge on a good step size
        target      [0.5]   : the target change in the function value for step size
        min_func    [1e-4]  : the minimum allowable change in (abs) function value to accept for convergence
        max_func    [4]     : the maximum allowable change in (abs) function value to accept for convergence
    """

    if self.gradient is None:
        print('No analytic gradient found; using finite differences.')
        return self.errors(method='HESSE')
    step_size = np.diag(self.errors(method='MIGRAD'))**0.5 / 8.
    step_size = np.maximum(step_size,min_step*1.1)
    step_size = np.minimum(step_size,max_step*0.9)
    nparams   = len(self.params)
    hess      = np.zeros([nparams,nparams])
    par       = self.params
    min_flags = np.asarray([False]*nparams)
    max_flags = np.asarray([False]*nparams)

    def revised_step(delta_f,current_step,index):
        if (current_step == max_step):
            max_flags[i] = True
            return True,0
        elif (current_step == min_step):
            min_flags[i] = True
            return True,0
        else:
            adf = abs(delta_f)
            if adf < 1e-8:
                # need to address a step size that results in a likelihood change that's too
                # small compared to precision
                pass
                
            if (adf < min_func) or (adf > max_func):
                new_step = current_step/(adf/target)
                new_step = min(new_step,max_step)
                new_step = max(new_step,min_step)
                return False,new_step
            else:
                return True,0
    
    iters = np.zeros(nparams)
    for i in xrange(nparams):
        converged = False
        for j in xrange(max_iters):
            iters[i] += 1
            di = step_size[i]
            par[i] += di
            g_up    = self.gradient(par)
            par[i] -= 2*di
            g_dn    = self.gradient(par)
            par[i] += di
            delta_f = (g_up - g_dn)[i]
            converged,new_step = revised_step(delta_f,di,i)
            print('Parameter %d -- Iteration %d -- Step size: %.2e -- delta: %.2e'%(i,j,di,delta_f))
            if converged: break
            else: step_size[i] = new_step
        hess[i,:] = (g_up - g_dn) / (2*di)  # central difference
        if not converged:
            print('Warning: step size for parameter %d (%.2g) did not result in convergence.'%(i,di))
    
    try:
        cov = inv(hess)
    except:
        raise Exception
    if full_output:
        return cov,hess,step_size,iters,min_flags,max_flags
    else:
        return cov


"""
@file  parameter_set.py

@brief Python classes that encapsulate model parameters.

@author Matthew Wood       <mdwood@slac.stanford.edu>
"""
__source__   = "$Source: /nfs/slac/g/glast/ground/cvs/users/mdwood/python/parameter_set.py,v $"
__author__   = "Matthew Wood <mdwood@slac.stanford.edu>"
__date__     = "$Date: 2013/08/15 20:50:25 $"
__revision__ = "$Revision: 1.2 $, $Author: mdwood $"

import numpy as np
import copy
import re
from scipy.interpolate import UnivariateSpline
from histogram import Histogram
from util import expand_aliases, get_parameters

class Parameter(object):
    """This class encapsulates a single function parameter that can
    take a single value or an array of values.  The parameter is
    identified by a unique ID number and a name string."""

    def __init__(self,pid,value,name,fixed=False,lims=None):
        self._pid = pid
        self._name = name
        self._value = np.array(value,ndmin=1)
        self._err = 0

        if lims is None: self._lims = [None,None]
        else: self._lims = lims
        self._fixed = fixed

    def lims(self):
        return self._lims
        
    def name(self):
        return self._name

    def pid(self):
        return self._pid

    def value(self):
        return self._value
    
    def error(self):
        return self._err

    def fix(self,fix=True):
        self._fixed = fix

    def fixed(self):
        return self._fixed

    def size(self):
        return self._value.shape[0]

    def set(self,v):

        self._value = np.array(v,ndmin=1)
#        if isinstance(v,np.array): self._value = v
#        else: self._value[...] = v

    def setLoBound(self,v):
        self._lims[0] = v

    def setHiBound(self,v):
        self._lims[1] = v

    def __str__(self):
        return '%5i %5i %25s %s'%(self._pid,self._fixed,self._name,
                                  str(self._value.T))

class ParameterSet(object):
    """Class that stores a set of function parameters.  Each parameter
    is identified by a unique integer parameter id."""

    def __init__(self,pars=None):

        self._pars_dict = {}
        self._pars = []
        self._par_names = {}

        if isinstance(pars,ParameterSet):
            for p in pars: self.addParameter(p)
        elif not pars is None:
            for p in pars: self.addParameter(p)

    def __iter__(self):
        return iter(self._pars)

    def pids(self):
        return sorted(self._pars_dict.keys())
    
    def names(self):
        """Return the names of the parameters in this set."""
        return self._par_names.keys()

    def pid(self):
        """Return the sorted list of parameter IDs in this set."""
        return sorted(self._pars_dict.keys())

    def fixed(self):

        fixed = []
        for i, p in enumerate(self._pars):
            fixed.append(p.fixed())
        return fixed

    def array(self):
        """Return an NxM numpy array with the values of the parameters
        in this set."""

#        if self._pars[pkeys[0]].size() > 1:
        x = np.zeros((len(self._pars),self._pars[0].size()))
        for i, p in enumerate(self._pars):
            x[i] = p.value()

        return x

    def makeParameterArray(self,ipar,x):

        pset = copy.deepcopy(self)

        for i, p in enumerate(self._pars):
            if i != ipar: 
                pset[i].set(np.ones(len(x))*self._pars[i].value())
            else:
                pset[i].set(x)

        return pset

    def clear(self):

        self._pars.clear()
        self._pars_dict.clear()
        self._par_names.clear()

    def fix(self,ipar,fix=True):
        self._pars[ipar].fix(fix)

    def fixAll(self,fix=True,regex=None):
        """Fix or free all parameters in the set."""

        keys = []

        if not regex is None:            
            for k, v in self._par_names.iteritems():                
                if re.search(regex,k): keys.append(v)
        else:
            keys = self._pars_dict.keys()

        for k in keys: self._pars_dict[k].fix(fix)

    def update(self,pset):
        """Update parameter values from an existing parameter set or from
        a numpy array."""

        if pset is None: return
        elif isinstance(pset,ParameterSet):
            for p in pset:
                if p.pid() in self._pars_dict:
                    self._pars_dict[p.pid()].set(p.value())
        else:
            for i, p in enumerate(self._pars):
                self._pars[i].set(np.array(pset[i],ndmin=1))

    def createParameter(self,value,name,fixed=False,lims=None,pid=None):
        """Create a new parameter and add it to the set.  Returns a
        reference to the new parameter."""

        if pid is None:

            if len(self._pars_dict.keys()) > 0:
                pid = sorted(self._pars_dict.keys())[-1]+1
            else:
                pid = 0

        p = Parameter(pid,value,name,fixed,lims)
        self.addParameter(p)
        return p

    def addParameter(self,p):        

        if p.pid() in self._pars_dict.keys():
            raise Exception('Parameter with ID %i already exists.'%p.pid())
        elif p.pid() in self._pars and p.name() != self._pars[p.pid()].name():
            raise Exception('Parameter with name %s already exists.'%p.name())
#            print "Error : Parameter already exists: ", p.pid()
#            print "Error : Mismatch in parameter name: ", p.pid()
#            sys.exit(1)
#        if p.name() in self._par_names.keys():
#            print "Error : Parameter with name already exists: ", p.name()
#            sys.exit(1)

        par = copy.deepcopy(p)

        self._pars_dict[p.pid()] = par
#        self._pars.append(par)
        self._pars = []
        for k in sorted(self._pars_dict.keys()):
            self._pars.append(self._pars_dict[k])

        self._par_names[p.name()] = p.pid()

    def addSet(self,pset): 

        for p in pset:
            if not p.pid() in self._pars_dict.keys():
                self.addParameter(p)

    def __getitem__(self,ipar):

        if isinstance(ipar,str): 
            return self._pars_dict[self._par_names[ipar]]
        else: return self._pars[ipar]

#    def __setitem__(self,pid,val):
#        self._pars[pid] = val

    def getParByID(self,ipar):
        return self._pars_dict[ipar]

    def getParByIndex(self,ipar):
        return self._pars[ipar]

    def getParByName(self,name):
        pid = self._par_names[name]
        return self._pars_dict[pid]

    def setParByName(self,name,val):
        pid = self._par_names[name]
        self._pars_dict[pid]._value = np.array(val,ndmin=2)

    def set(self,*args):

        for i, v in enumerate(args):
            self._pars[i].set(v)

    def npar(self):
        return len(self._pars)

    def size(self):

        if len(self._pars) == 0: return 0
        else: return self._pars[0].size()

#    def set(self,p):
#        """Set the parameter list from an array."""
#        for i in range(len(p)):
#            self._pars[i]._value = np.array(p[i])*np.ones(1)
            
    def __str__(self):

        os = ''
        for p in self._pars: os += '%s\n'%(p)

        return os

class FitResults(ParameterSet):

    def __init__(self,pset,fval,cov=None):
        ParameterSet.__init__(self,pset)

        if cov is None: cov=np.zeros(shape=(pset.npar(),pset.npar()))
        else: self._cov = cov
        
        self._err = np.sqrt(np.diag(cov))
        self._fval = fval

    def fval(self):
        return self._fval

    def getParError(self,pid):

        if isinstance(pid,str):
            pid = self.getParByName(pid).pid()

        return self._err[pid]

    def __str__(self):

        os = ''
        for i, p in enumerate(self._pars):
            os += '%s %.6g\n'%(p,self._err[i])

        os += 'fval: %.3f\n'%(self._fval)
#        os += 'cov:\n %s'%(str(self._cov))

        return os


if __name__ == '__main__':

    pset = ParameterSet()
    pset.createParameter(0.0,'par0')
    
    for p in pset:
        print p

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
        self._value = np.array(value,ndmin=2)
        self._err = 0

        if lims is None: self._lims = [0,0]
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

        self._value = np.array(v,ndmin=2)
#        if isinstance(v,np.array): self._value = v
#        else: self._value[...] = v

    def __str__(self):
        return '%5i %5i %25s %s'%(self._pid,self._fixed,self._name,
                                  str(self._value.T))

class ParameterSet(object):
    """Class that stores a set of function parameters.  Each parameter
    is identified by a unique integer parameter id."""

    def __init__(self,pars=None):

        self._pars = {}
        self._par_names = {}

        if isinstance(pars,ParameterSet):
            self._pars = copy.deepcopy(pars._pars)
            self._par_names = copy.deepcopy(pars._par_names)
        else:
            if not pars is None:
                for p in pars: self.addParameter(p)

    def __iter__(self):
        return iter(self._pars.values())

    def pids(self):
        return sorted(self._pars.keys())
    
    def names(self):
        """Return the names of the parameters in this set."""
        return self._par_names.keys()

    def pid(self):
        """Return the sorted list of parameter IDs in this set."""

        pid = []
        pkeys = sorted(self._pars.keys())
        for i, k in enumerate(pkeys):
            pid.append(self._pars[k].pid())
        return pid

    def fixed(self):

        fixed = []
        pkeys = sorted(self._pars.keys())
        for i, k in enumerate(pkeys):
            fixed.append(self._pars[k].fixed())
        return fixed

    def array(self):
        """Return an NxM numpy array with the values of the parameters
        in this set."""
        pkeys = sorted(self._pars.keys())
#        if self._pars[pkeys[0]].size() > 1:
        x = np.zeros((len(self._pars),self._pars[pkeys[0]].size(),1))
        for i, k in enumerate(pkeys):
            x[i] = self._pars[k].value()

        return x
#        else:
#            x = np.zeros(len(self._pars))
#            for i, k in enumerate(pkeys):
#                x[i] = self._pars[k].value()[0]
#            return x

    def makeParameterArray(self,ipar,x):

        pset = copy.deepcopy(self)

        for k in pset._pars.keys():
            if k != ipar: 
                pset[k] = np.ones(len(x))*self._pars[k].value()
                pset[k] = pset[k].reshape((len(x),1))
            else:
                pset[k] = x
                pset[k].shape += (1,)

        return pset

    def clear(self):

        self._pars.clear()
        self._par_names.clear()

    def fixAll(self,fix=True,regex=None):
        """Fix or free all parameters in the set."""

        keys = []

        if not regex is None:            
            for k, v in self._par_names.iteritems():                
                if re.search(regex,k): keys.append(v)
        else:
            keys = self._pars.keys()

        for k in keys: self._pars[k].fix(fix)

    def setParam(self,p):
        """Set parameter values from an existing parameter set or from
        a numpy array."""
        
        if isinstance(p,ParameterSet):
            for k in self._pars.keys():
                self._pars[k] = p._pars[k]
        else:
            for i, k in enumerate(sorted(self._pars.keys())):
                self._pars[k]._value = np.array(p[i],ndmin=1)

                if self._pars[k]._value.ndim == 1:
                    self._pars[k]._value.shape += (1,)


    def createParameter(self,value,name,fixed=False,lims=None):
        """Create a new parameter and add it to the set.  Returns a
        reference to the new parameter."""

        if len(self._pars.keys()) > 0:
            pid = sorted(self._pars.keys())[-1]+1
        else:
            pid = 0

        p = Parameter(pid,value,name,fixed,lims)
        self.addParameter(p)
        return p

    def addParameter(self,p):        

        if p.pid() in self._pars and p.name() != self._pars[p.pid()].name():
#            print "Error : Parameter already exists: ", p.pid()
            print "Error : Mismatch in parameter name: ", p.pid()
            sys.exit(1)
#        if p.name() in self._par_names.keys():
#            print "Error : Parameter with name already exists: ", p.name()
#            sys.exit(1)

        self._pars[p.pid()] = copy.deepcopy(p)
        self._par_names[p.name()] = p.pid()

    def addSet(self,p):        
        self._pars.update(copy.deepcopy(p._pars))
        self._par_names.update(p._par_names)

    def __getitem__(self,pid):
        return self._pars[pid].value()

    def __setitem__(self,pid,val):
        self._pars[pid]._value = val

    def getParByID(self,ipar):
        return self._pars[ipar]

#    def getParIndex(self,pid):
#        pid = sorted(self._pars.keys())[ipar]
#        return self._pars[pid]

    def getParByIndex(self,ipar):
        pid = sorted(self._pars.keys())[ipar]
        return self._pars[pid]

    def getParByName(self,name):
        pid = self._par_names[name]
        return self._pars[pid]

    def setParByName(self,name,val):
        pid = self._par_names[name]
        self._pars[pid]._value = np.array(val,ndmin=2)

    def npar(self):
        return len(self._pars)

    def size(self):

        par = self.getParByIndex(0)
        return par._value.shape[0]

#    def set(self,p):
#        """Set the parameter list from an array."""
#        for i in range(len(p)):
#            self._pars[i]._value = np.array(p[i])*np.ones(1)
            
    def __str__(self):

        os = ''
        for k in sorted(self._pars.keys()):
            os += '%s\n'%(self._pars[k])

        return os


if __name__ == '__main__':

    pset = ParameterSet()
    pset.createParameter(0.0,'par0')
    
    for p in pset:
        print p

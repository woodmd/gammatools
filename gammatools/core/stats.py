"""
@author Matthew Wood <mdwood@slac.stanford.edu>
"""

__author__   = "Matthew Wood <mdwood@slac.stanford.edu>"
__date__     = "$Date: 2013/10/20 23:53:52 $"

import numpy as np
import copy
from scipy.interpolate import UnivariateSpline
import scipy.optimize as opt
import matplotlib.pyplot as plt
from gammatools.core.histogram import Histogram
from scipy.stats import norm

def limit_pval_to_sigma(p):
    """Convert the pval of a one-sided confidence interval to sigma."""
    return norm().isf(p)

def limit_sigma_to_pval(s):
    return norm().cdf(s)

def gauss_pval_to_sigma(p):
    """Convert the pval of a two-sided confidence interval to sigma."""
    return norm().isf(0.5+p*0.5)

def gauss_sigma_to_pval(s):
    return 2.0*(norm().cdf(s)-0.5)


def poisson_lnl(sc,bc,alpha):
        
    # total counts in each bin
    nc = sc + bc        

    # number of counts in control region
    cc = bc/alpha

    # model for total background counts in null hypothesis
    mub0 = (nc+cc)/(1.0+alpha)*alpha

    # model for total background counts in signal hypothesis
    mub1 = bc

    # model for signal counts
    mus = sc

    lnl0 = nc*np.log(mub0)-mub0 + cc*np.log(mub0/alpha) - mub0/alpha
    lnl1 = nc*np.log(mub1+mus) - mub1 - mus + \
        cc*np.log(mub1/alpha) - mub1/alpha
                

    return 2*(lnl1-lnl0)


    

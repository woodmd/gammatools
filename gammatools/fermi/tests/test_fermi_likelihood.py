import unittest
import numpy as np
from gammatools.core.model_fn import GaussFn
from gammatools.fermi.psf_likelihood import *
from gammatools.core.histogram import *

class TestFermiLikelihood(unittest.TestCase):


    def test_convolved_gauss(self):

        return

        pset = ParameterSet()
        
        sigma = 0.1
        ksigma = 0.2

        gpfn = Gauss2DProjFn.create(1.0,sigma,pset=pset)
        gfn = Gauss2DFn.create(1.0,0.0,0.0,sigma)
        gfn2 = Gauss2DFn.create(1.0,0.0,0.0,ksigma)

        cgfn = ConvolvedGaussFn.create(1.0,ksigma,gpfn,pset=pset,prefix='test')
        delta = np.array([0.2,0.1]).reshape((2,1))

        xaxis = Axis.create(-1.0,1.0,1000)
        yaxis = Axis.create(-1.0,1.0,1000)
        x, y = np.meshgrid(xaxis.center(),yaxis.center())
        x = np.ravel(x)
        y = np.ravel(y)
        xy = np.vstack((x,y))

        r = np.sqrt(x**2+y**2)
        s = np.sum(gfn.eval(xy+delta)*gfn2(xy))*(2.0/1000.)**2
        r0 = np.sqrt(delta[0][0]**2+delta[1][0]**2)
        x = np.array([0.1,0.2])


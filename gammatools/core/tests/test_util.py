import unittest
import numpy as np
import copy
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.util import *
from gammatools.core.config import *
from gammatools.core.histogram import Axis

class TestUtil(unittest.TestCase):

    def test_convolve2d_king(self):
        
        gfn = lambda r, s: np.power(2*np.pi*s**2,-1)*np.exp(-r**2/(2*s**2))
        kfn = lambda r, s, g: np.power(2*np.pi*s**2,-1)*(1.-1./g)* \
            np.power(1+0.5/g*(r/s)**2,-g)

        kfn0 = lambda x, y, mux, muy, s, g: kfn(np.sqrt((x-mux)**2+(y-muy)**2),s,g)

        xaxis = Axis.create(-3,3,501)
        yaxis = Axis.create(-3,3,501)

        x, y = np.meshgrid(xaxis.center,yaxis.center)
        xbin, ybin = np.meshgrid(xaxis.width,yaxis.width)

        r = np.sqrt(x**2+y**2)

        # Scalar Input

        mux = 0.5
        muy = -0.2
        mur = (mux**2+muy**2)**0.5

        gsig = 0.1
        ksig = 0.2
        kgam = 4.0

        fval0 = np.sum(kfn0(x,y,mux,muy,ksig,kgam)*gfn(r,gsig)*xbin*ybin)
        fval1 = convolve2d_king(lambda t: gfn(t,gsig),mur,ksig,kgam,3.0,
                                nstep=10000)
#        fval2 = convolve2d_gauss(lambda t: kfn(t,ksig,kgam),mur,gsig,3.0,
#                                 nstep=1000)
#        print fval0, fval1, fval2, fval1/fval0

        assert_almost_equal(fval0,fval1,4)

    def test_convolve2d_gauss(self):

        gfn0 = lambda x, y, mux, muy, s: np.power(2*np.pi*s**2,-1)* \
            np.exp(-((x-mux)**2+(y-muy)**2)/(2*s**2))

        gfn1 = lambda r, s: np.power(2*np.pi*s**2,-1)*np.exp(-r**2/(2*s**2))
        
        

        xaxis = Axis.create(-3,3,501)
        yaxis = Axis.create(-3,3,501)

        x, y = np.meshgrid(xaxis.center,yaxis.center)
        xbin, ybin = np.meshgrid(xaxis.width,yaxis.width)

        # Scalar Input

        sigma0 = 0.1
        sigma1 = 0.2

        mux = 0.5
        muy = -0.2
        mur = (mux**2+muy**2)**0.5

        fval0 = np.sum(gfn0(x,y,mux,muy,sigma1)*gfn0(x,y,0,0,sigma0)*xbin*ybin)
        fval1 = convolve2d_gauss(lambda t: gfn1(t,sigma0),mur,sigma1,3.0,
                                 nstep=1000)

        assert_almost_equal(fval0,fval1,4)

        # Vector Input for Gaussian Width

        sigma0 = 0.1
        sigma1 = np.array([0.1,0.15,0.2])

        mux = 0.5
        muy = -0.2
        mur = (mux**2+muy**2)**0.5

        fval0 = []
        for i in range(len(sigma1)):
            fval0.append(np.sum(gfn0(x,y,mux,muy,sigma1[i])*
                           gfn0(x,y,0,0,sigma0)*xbin*ybin))
        
        fval1 = convolve2d_gauss(lambda t: gfn1(t,sigma0),mur,sigma1,3.0,
                                 nstep=1000)

        assert_almost_equal(np.ravel(fval0),np.ravel(fval1),4)

        # Vector Input

        sigma0 = 0.1
        sigma1 = 0.2

        mux = np.array([0.3,0.4,0.5])
        muy = np.array([-0.2,-0.2,0.2])
        mur = (mux**2+muy**2)**0.5

        fval0 = []
        for i in range(len(mux)):
            fval0.append(np.sum(gfn0(x,y,mux[i],muy[i],sigma1)*
                                gfn0(x,y,0,0,sigma0)*
                                xbin*ybin))

        fval1 = convolve2d_gauss(lambda t: gfn1(t,sigma0),mur,sigma1,3.0,
                                 nstep=1000)

        assert_almost_equal(fval0,fval1,4)
        

import unittest
import numpy as np
import copy
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.util import *
from gammatools.core.config import *
from gammatools.core.algebra import *

def normalize_angle(angle):

    return np.fmod(angle+2*np.pi,2*np.pi)

class TestAlgebra(unittest.TestCase):

    def test_vector_norm(self):

        v1 = Vector3D.createThetaPhi(np.linspace(0,np.pi,10),
                                     np.linspace(0,2*np.pi,10))

        # Vector Norm        
        assert_almost_equal(v1.norm(),np.ones(10))

        v2 = v1*2.0

        v2.normalize()

        assert_almost_equal(v2.norm(),np.ones(10))


    def test_vector_rotation(self):

        v0 = Vector3D.createThetaPhi(np.pi/2.,0.0)
        v1 = Vector3D.createThetaPhi(np.linspace(np.pi/4.,np.pi/2.,10),0.0)
        v2 = Vector3D.createThetaPhi(np.linspace(np.pi/4.,np.pi/2.,10),0.0)

        r0 = np.pi/2.
        r1 = np.pi/2.*np.ones(10)
        r2 = np.linspace(np.pi/4.,3.*np.pi/4.,10)

        # Apply scalar rotation to scalar

        assert_almost_equal(v0.phi(),0.0)

        v0.rotatez(0.0*r0)
        assert_almost_equal(v0.phi(),0.0*r0)

        v0.rotatez(r0)
        assert_almost_equal(v0.phi(),r0)

        v0.rotatez(r0)
        assert_almost_equal(v0.phi(),2.*r0)

        v0.rotatez(r0)
        assert_almost_equal(normalize_angle(v0.phi()),3.*r0)


        # Apply scalar rotation to vector

        assert_almost_equal(v1.phi(),0.0*r0)

        v1.rotatez(r0)
        assert_almost_equal(v1.phi(),r0)

        v1.rotatez(r0)
        assert_almost_equal(v1.phi(),2.0*r0)

        v1.rotatez(r0)
        assert_almost_equal(normalize_angle(v1.phi()),3.*r0)

        # Apply vector rotation to vector

        assert_almost_equal(normalize_angle(v2.phi()),
                            normalize_angle(0.0*r2))

        v2.rotatez(r2)
        assert_almost_equal(normalize_angle(v2.phi()),
                            normalize_angle(1.0*r2))

        v2.rotatez(r2)
        assert_almost_equal(normalize_angle(v2.phi()),
                            normalize_angle(2.0*r2))

        v2.rotatez(r2)
        assert_almost_equal(normalize_angle(v2.phi()),
                            normalize_angle(3.0*r2))

        

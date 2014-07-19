import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.parameter_set import *
from gammatools.core.likelihood import *
from gammatools.core.nonlinear_fitting import *
from gammatools.core.model_fn import *
from gammatools.core.histogram import *

class TestParamFn(unittest.TestCase):

    def test_param_fn(self):

        def test_fn(x,y):
            return x**2 + 0.2*y**2 - 3.3*y + 4.4*x

        x0, y0 = 4.0, -1.7
        x1, y1 = -3.1, 2.3
        x2, y2 = [1.0,2.0,3.0], [-10.3,-1.4,2.2]
        
        pfn = ParamFn.create(test_fn,[x0,y0])

        # Test evaluation with internal parameter values
        self.assertEqual(pfn(),test_fn(x0,y0))

        # Test evaluation with parameter value scalar arguments
        self.assertEqual(pfn(x1,y1),test_fn(x1,y1))

        # Test evaluation with parameter value list arguments
        assert_array_equal(pfn(x2,y2),test_fn(np.array(x2),np.array(y2)))

        # Test evaluation with parameter value array arguments
        assert_array_equal(pfn(np.array(x2),np.array(y2)),
                           test_fn(np.array(x2),np.array(y2)))

        # Test evaluation with parameter value matrix arguments
        assert_array_equal(pfn(np.vstack((np.array(x2),np.array(y2)))),
                           test_fn(np.array(x2),np.array(y2)))

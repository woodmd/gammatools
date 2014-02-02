import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.parameter_set import *
from gammatools.core.likelihood import *
from gammatools.core.model_fn import *
from gammatools.core.histogram import *


class TestLikelihood(unittest.TestCase):


    def test_parameter_set_init(self):

        par0 = Parameter(0,3.0,'par0')
        par1 = Parameter(1,4.0,'par1')
        par4 = Parameter(4,6.0,'par4')

        pset = ParameterSet()
        pset.addParameter(par0)
        pset.addParameter(par1)
        pset.addParameter(par4)
        par2 = pset.createParameter(5.0,'par2',pid=2)

        pars = [par0,par1,par2,par4]

        for i, p in enumerate(pars):
            pname = p.name()
            self.assertEqual(pset[i].name(),pars[i].name())
            self.assertEqual(pset[i].value(),pars[i].value())
            self.assertEqual(pset[pname].name(),pars[i].name())
            self.assertEqual(pset[pname].value(),pars[i].value())

#        assert_array_equal(pset.array(),
#                           np.array([3.0,4.0,5.0,6.0],ndmin=2))



    def test_parameter_set_merge(self):

        par0 = Parameter(0,3.0,'par0')
        par1 = Parameter(1,3.0,'par1')

        pset0 = ParameterSet()

    def test_polyfn_eval(self):

        f = PolyFn.create(3,[3.0,-1.0,2.0])
        fn = lambda t, z=2.0: 3.0 - 1.0*t + z*t**2

        fni = lambda t, z=2.0: 3.0*t - 1.0*t**2/2. + z*t**3/3.0

        x = np.linspace(-2.0,2.0,12)
        a2 = np.linspace(0,10,10)
        pset = f.param()
        pset = pset.makeParameterArray(2,a2)

        self.assertEqual(f.eval(2.0),fn(2.0))
        assert_almost_equal(f.eval(x),fn(x))
        assert_almost_equal(f.eval(2.0,pset).flat,fn(2.0,a2))

        assert_almost_equal(f.integrate(0.0,2.0),fni(2.0)-fni(0.0))
        assert_almost_equal(f.integrate(0.0,2.0,pset).flat,
                            fni(2.0,a2)-fni(0.0,a2))


    def test_polyfn_fit(self):

        np.random.seed(1)

        f = PolyFn.create(2,[0,1.0])
        y = f.rnd(1000,0.0,1.0)
        f.set_norm(1000,0.0,1.0)
        pset = f.param()


        h = Histogram(np.linspace(0,1,10))
        h.fill(y)
        chi2_fn = Chi2HistFn(h,f)
        print chi2_fn.eval(pset)

        psetv = pset.makeParameterArray(1,pset[1].value()*np.linspace(0,2,10))
        chi2_fn.param()[1].set(chi2_fn._param[1].value()*1.5)

        fitter = MinuitFitter(chi2_fn)

        print fitter.fit()

    def test_hist_model_fit(self):

        pset0 = ParameterSet()
        
        fn0 = GaussFn.create(100.0,0.0,0.1,pset0)
        fn1 = GaussFn.create(50.0,1.0,0.1,pset0)

        hm0 = Histogram(Axis.create(-3.0,3.0,100))
        hm0.fill(hm0.axis().center(),fn0(hm0.axis().center()))

        hm1 = Histogram(Axis.create(-3.0,3.0,100))
        hm1.fill(hm1.axis().center(),fn1(hm1.axis().center()))
        
        hm2 = hm0*0.9 + hm1*0.8
        
        pset1 = ParameterSet()
        
        m0 = ScaledHistogramModel.create(hm0,pset=pset1,name='m0')
        m1 = ScaledHistogramModel.create(hm1,pset=pset1,name='m1')

        msum = CompositeSumModel([m0,m1])
        chi2_fn = Chi2HistFn(hm2,msum)
        fitter = BFGSFitter(chi2_fn)

        pset1[0].set(1.5)
        pset1[1].set(0.5)
        
        f = fitter.fit(pset1)
        
        assert_almost_equal(f[0].value(),0.9,4)
        assert_almost_equal(f[1].value(),0.8,4)

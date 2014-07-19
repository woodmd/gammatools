import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.util import *
from gammatools.core.config import *
from gammatools.core.histogram import Axis

class TestConfigurable(unittest.TestCase):

    def test_configurable_defaults(self):

        class BaseClass(Configurable):

            default_config = {'BaseClass.par0' : 0,
                              'BaseClass.par1' : 'x',
                              'BaseClass.par2' : None }

            def __init__(self,config=None,**kwargs):
                super(BaseClass,self).__init__(config,**kwargs)

        class DerivedClass(BaseClass):

            default_config = {'DerivedClass.par0' : 0, 
                              'DerivedClass.par1' : 'x', 
                              'DerivedClass.par2' : None }

            def __init__(self,config=None,**kwargs):
                super(DerivedClass,self).__init__(config,**kwargs)

        class DerivedClass2(DerivedClass):

            default_config = {'DerivedClass2.par0' : 0, 
                              'DerivedClass2.par1' : 'x', 
                              'DerivedClass2.par2' : None }

            def __init__(self,config=None,**kwargs):
                super(DerivedClass2,self).__init__(config,**kwargs)

        config = {'BaseClass.par0'     : 1, 
                  'BaseClass.par2'     : 'y', 
                  'DerivedClass.par0'  : 'z',
                  'DerivedClass2.par2' : 4 }

        kwargs = {'BaseClass.par0' : 2 }

 
        base_class1 = BaseClass(config)
        base_class2 = BaseClass(config,**kwargs)

        derived_class0 = DerivedClass()
        derived_class1 = DerivedClass(config)

        
        derived2_class1 = DerivedClass2(config)
        derived2_class1 = DerivedClass2(config)

        # Test no config input
        base_class = BaseClass()
        derived_class = DerivedClass()
        derived2_class = DerivedClass2()

        self.assertEqual(base_class.config,
                         BaseClass.default_config)
        self.assertEqual(derived_class.config,
                         dict(BaseClass.default_config.items()+
                              DerivedClass.default_config.items()))
        self.assertEqual(derived2_class.config,
                         dict(BaseClass.default_config.items()+
                              DerivedClass.default_config.items()+
                              DerivedClass2.default_config.items()))

        # Test dict input
        base_class = BaseClass(config)
        derived_class = DerivedClass(config)
        derived2_class = DerivedClass2(config)

        for k, v in config.iteritems():

            if k in base_class.default_config: 
                self.assertEqual(base_class.config[k],v)

            if k in derived_class.default_config: 
                self.assertEqual(derived_class.config[k],v)

            if k in derived2_class.default_config: 
                self.assertEqual(derived2_class.config[k],v)

        self.assertEqual(set(base_class.config.keys()),
                         set(BaseClass.default_config.keys()))
        self.assertEqual(set(derived_class.config.keys()),
                         set(BaseClass.default_config.keys()+
                             DerivedClass.default_config.keys()))
        self.assertEqual(set(derived2_class.config.keys()),
                         set(BaseClass.default_config.keys()+
                             DerivedClass.default_config.keys()+
                             DerivedClass2.default_config.keys()))
        
        # Test dict and kwarg input -- kwargs take precedence over dict
        base_class = BaseClass(config,**kwargs)
        derived_class = DerivedClass(config,**kwargs)
        derived2_class = DerivedClass2(config,**kwargs)

        config.update(kwargs)

        for k, v in config.iteritems():

            if k in base_class.default_config: 
                self.assertEqual(base_class.config[k],v)

            if k in derived_class.default_config: 
                self.assertEqual(derived_class.config[k],v)

            if k in derived2_class.default_config: 
                self.assertEqual(derived2_class.config[k],v)

        self.assertEqual(set(base_class.config.keys()),
                         set(BaseClass.default_config.keys()))
        self.assertEqual(set(derived_class.config.keys()),
                         set(BaseClass.default_config.keys()+
                             DerivedClass.default_config.keys()))
        self.assertEqual(set(derived2_class.config.keys()),
                         set(BaseClass.default_config.keys()+
                             DerivedClass.default_config.keys()+
                             DerivedClass2.default_config.keys()))

        return
        # Test update
        base_class = BaseClass()
        derived_class = DerivedClass()
        derived2_class = DerivedClass2()

        base_class.update_config(config)
        derived_class.update_config(config)
        derived2_class.update_config(config)

        self.assertEqual(base_class.config,base_class1.config)
        self.assertEqual(derived_class.config,derived_class1.config)

    def test_configurable_docstring(self):

        class BaseClass(Configurable):

            default_config = {'BaseClass.par0' : (0,'Doc for Option a'), 
                              'BaseClass.par1' : ('x','Doc for Option b'), 
                              'BaseClass.par2' : (None,'Doc for Option c')}

            def __init__(self,config=None):
                super(BaseClass,self).__init__()
                self.configure(config,default_config=BaseClass.default_config)

        base_class0 = BaseClass()


        self.assertEqual(base_class0.config_docstring('BaseClass.par0'),
                         BaseClass.default_config['BaseClass.par0'][1])

    def test_convolve2d_king(self):
        
        gfn = lambda r, s: np.power(2*np.pi*s**2,-1)*np.exp(-r**2/(2*s**2))
        kfn = lambda r, s, g: np.power(2*np.pi*s**2,-1)*(1.-1./g)* \
            np.power(1+0.5/g*(r/s)**2,-g)

        kfn0 = lambda x, y, mux, muy, s, g: kfn(np.sqrt((x-mux)**2+(y-muy)**2),s,g)

        xaxis = Axis.create(-3,3,501)
        yaxis = Axis.create(-3,3,501)

        x, y = np.meshgrid(xaxis.center(),yaxis.center())
        xbin, ybin = np.meshgrid(xaxis.width(),yaxis.width())

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

        x, y = np.meshgrid(xaxis.center(),yaxis.center())
        xbin, ybin = np.meshgrid(xaxis.width(),yaxis.width())

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
        

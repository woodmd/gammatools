import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.util import *


class TestConfigurable(unittest.TestCase):

    def test_configurable_defaults(self):

        class BaseClass(Configurable):

            defaults = {'a' : 0,
                        'b' : 'x',
                        'c' : None }

            def __init__(self,config=None):
                super(BaseClass,self).__init__()
                self.configure(config,default_config=BaseClass.defaults)

        class DerivedClass(BaseClass):

            defaults = {'d' : 0, 'e' : 'x', 'f' : None }

            def __init__(self,config=None):
                super(DerivedClass,self).__init__()
                self.configure(config,default_config=DerivedClass.defaults)
        

        config = {'a' : 1, 'c' : 'y', 'd' : 'z' }

        base_class0 = BaseClass()
        base_class1 = BaseClass(config)

        derived_class0 = DerivedClass()
        derived_class1 = DerivedClass(config)

        self.assertEqual(base_class0.config(),BaseClass.defaults)
        self.assertEqual(derived_class0.config(),
                         dict(BaseClass.defaults.items()+
                              DerivedClass.defaults.items()))


        self.assertEqual(base_class1.config()['a'],1)
        self.assertEqual(base_class1.config()['c'],'y')
        self.assertEqual(base_class1.config().keys(),
                         BaseClass.defaults.keys())


        base_class0.update_config(base_class1.config())
        derived_class0.update_config(derived_class1.config())

        self.assertEqual(base_class0.config(),base_class1.config())
        self.assertEqual(derived_class0.config(),derived_class1.config())


        base_class0.print_config()
        base_class1.print_config()

    def test_configurable_docstring(self):

        class BaseClass(Configurable):

            defaults = {'a' : (0,'Doc for Option a'), 
                        'b' : ('x','Doc for Option b'), 
                        'c' : (None,'Doc for Option c')}

            def __init__(self,config=None):
                super(BaseClass,self).__init__()
                self.configure(config,default_config=BaseClass.defaults)

        base_class0 = BaseClass()


        self.assertEqual(base_class0.config_docstring('a'),
                         BaseClass.defaults['a'][1])

import unittest
import numpy as np
import copy
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.util import *
from gammatools.core.config import *
from gammatools.core.histogram import Axis

class TestConfigurable(unittest.TestCase):

    def test_configurable_defaults(self):

        class BaseClass(Configurable):

            default_config = {'BaseClass_par0' : 0,
                              'BaseClass_par1' : 'x',
                              'BaseClass_par2' : None }

            def __init__(self,config=None,**kwargs):
                super(BaseClass,self).__init__(config,**kwargs)

        class DerivedClass(BaseClass):

            default_config = {'DerivedClass_par0' : 0, 
                              'DerivedClass_par1' : 'x', 
                              'DerivedClass_par2' : None }

            def __init__(self,config=None,**kwargs):
                super(DerivedClass,self).__init__(config,**kwargs)

        class DerivedClass2(DerivedClass):

            default_config = {'DerivedClass2_par0' : 0, 
                              'DerivedClass2_par1' : 'x', 
                              'DerivedClass2_par2' : None }

            def __init__(self,config=None,**kwargs):
                super(DerivedClass2,self).__init__(config,**kwargs)

        config = {'BaseClass_par0'     : 1, 
                  'BaseClass_par2'     : 'y', 
                  'DerivedClass_par0'  : 'z',
                  'DerivedClass2_par2' : 4 }

        kwargs = {'BaseClass_par0' : 2 }

 
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

    def test_configurable_nested_defaults(self):

        class BaseClass(Configurable):

            default_config = {'BaseClass_par0' : 0,
                              'BaseClass_par1' : 'x',
                              'BaseClass_group0.par0' : 'y',
                              'BaseClass_group0.par1' : 'z',
                              }

            def __init__(self,config=None,**kwargs):
                super(BaseClass,self).__init__(config,**kwargs)
        
        base_class_defaults = {
            'BaseClass_par0' : 0,
            'BaseClass_par1' : 'x',
            'BaseClass_group0' : {'par0' : 'y', 'par1' : 'z'}
            }

        extra_defaults = {
            'par0' : 'v0',
            'par1' : 'v1',
            'group0' : {'par0' : 'y', 'par1' : 'z'}
            }

        # Test no config input
        base_class = BaseClass()
        base_class.update_default_config(extra_defaults,'group1')

        test_dict = copy.deepcopy(base_class_defaults)
        test_dict['group1'] = extra_defaults

        self.assertEqual(base_class.config,test_dict)

        # Test dict input

        base_class_dict_input = { 
            'BaseClass_par0' : 1,
            'BaseClass_par1' : 'a',
            'BaseClass_group0' : {'par0' : 'c', 'par1' : 'd'}
            }

        base_class = BaseClass(base_class_dict_input)

        self.assertEqual(base_class.config,
                         base_class_dict_input)

        # Test dict and kwarg input

        base_class_dict_input = { 
            'BaseClass_par0' : 1,
            'BaseClass_par1' : 'a',
            'BaseClass_group0' : {'par0' : 'c', 'par1' : 'd'}
            }

        base_class_kwargs_input = { 
            'BaseClass_par0' : 2,
            'BaseClass_par1' : 'c',
            'BaseClass_group0' : {'par0' : 'e' }
            }

        test_dict = copy.deepcopy(base_class_defaults)
        update_dict(test_dict,base_class_dict_input)
        update_dict(test_dict,base_class_kwargs_input)

        base_class = BaseClass(base_class_dict_input,**base_class_kwargs_input)

        self.assertEqual(base_class.config,test_dict)


    def test_configurable_docstring(self):

        class BaseClass(Configurable):

            default_config = {'BaseClass_par0' : (0,'Doc for Option a'), 
                              'BaseClass_par1' : ('x','Doc for Option b'), 
                              'BaseClass_par2' : (None,'Doc for Option c')}

            def __init__(self,config=None):
                super(BaseClass,self).__init__()
                self.configure(config,default_config=BaseClass.default_config)

        base_class0 = BaseClass()


        self.assertEqual(base_class0.config_docstring('BaseClass_par0'),
                         BaseClass.default_config['BaseClass_par0'][1])

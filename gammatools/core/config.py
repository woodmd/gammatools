import inspect
import os
from util import update_dict

class Option(object):

    def __init__(self,value,docstring=''):
        self._value = value
        self._docstring = docstring

    def value(self):
        return self._value

    def docstring(self):
        return self._docstring

    @staticmethod
    def create(x):
        if isinstance(x,list): return Option(x[0],x[1])
        elif isinstance(x,tuple): return Option(x[0],x[1])
        else: return Option(x)

class Configurable(object):

    def __init__(self,config=None,**kwargs):

        self._config = {}
        self._default_config = {}
        self.register_default_config(self)
        self.configure(config,**kwargs)

    @property
    def config(self):
        return self._config
        
    @classmethod
    def register_default_config(cls,c,key='default_config'):
        """Register default configuration dictionaries for this class
        and all classes from which it inherits."""

        for base_class in inspect.getmro(cls):
#            print base_class

#            for k, v in base_class.__dict__.iteritems():
#                print k, v

            if key in base_class.__dict__:
                c.update_default_config(base_class.__dict__[key])
            
    def update_config(self,config):
        update_dict(self._config,config)

    def update_default_config(self,config):
        """Update configuration for the object adding keys for
        elements that are not present."""
        if config is None: return
        
        if not isinstance(config,dict) and issubclass(config,Configurable):
            config = config.default_config
            
        update_dict(self._default_config,config,True)
        for k in config.keys():
            if not isinstance(self._default_config[k],Option):
                self._default_config[k] = Option.create(config[k])
                
            self._config[k] = self._default_config[k].value()
       
    def print_config(self):
        
        print 'CONFIG'
        for k, v in self._default_config.iteritems():
            print '%20s %10s %10s %s'%(k,self._config[k],v.value(),
                                       v.docstring())

    @property
    def config(self):
        return self._config

    def config_docstring(self,key):
        return self._default_config[key].docstring()

    def set_config(self,key,value):
        self._config[key] = value
        
    def configure(self,config,opts=None,subsection=None,**kwargs):
        """Update the configuration of this object with the contents
        of 'config'."""
        
        if not config is None:
            
            if not subsection is None and subsection in config and not \
                    config[subsection] is None:                
                update_dict(self._config,config[subsection])
            else:
                update_dict(self._config,config)

        if not opts is None:
            for k,v in opts.__dict__.iteritems():
                if k in self._config and not v is None:
                    self.set_config(k,v)
                
        update_dict(self._config,kwargs)

        for k, v in self._config.iteritems():

            if v is None or not isinstance(v,str): continue            
            if os.path.isfile(v): self._config[k] = os.path.abspath(v)

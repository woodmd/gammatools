import inspect
import os
from util import update_dict

class Option(object):

    def __init__(self,name,value,docstring='',option_type=str,group=None):
        self._name = name
        self._value = value
        self._docstring = docstring
        self._option_type = option_type
        self._group = group

        if option_type == list and len(value):
            self._list_type = type(value[0])
        else:
            self._list_type = str
        
        
    @property
    def name(self):
        return self._name
        
    @property
    def value(self):
        return self._value

    @property
    def docstring(self):
        return self._docstring

    @property
    def type(self):
        return self._option_type

    @property
    def list_type(self):
        return self._list_type

    @property
    def group(self):
        return self._group
    
    @staticmethod
    def create(name,x,group=None):
        """Create an instance of an option from a tuple or a scalar."""

        if len(name.split('.')) > 1:        
            group, name = name.split('.')
            
        #isinstance(x,list): return Option(x[0],x[1])
        if isinstance(x,Option): return x
        elif isinstance(x,tuple):
            if len(x) == 1:
                value, docstring, option_type = x[0], '', str
            elif len(x) == 2:
                value, docstring, option_type = x[0], x[1], str
            elif len(x) == 3:
                value, docstring, option_type = x[0], x[1], x[2]
            else:
                raise Exception('Wrong size for option tuple.')

            if value is not None: option_type = type(value)
            
            return Option(name,value,docstring,option_type,group=group)
        else:
            if x is not None: option_type = type(x)
            else: option_type = str            
            return Option(name,x,option_type=option_type,group=group)

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
    def register_default_config(cls,c,key='default_config',
                                group_key='default_subsection'):
        """Register default configuration dictionaries for this class
        and all classes from which it inherits."""

        for base_class in inspect.getmro(cls):
            if key in base_class.__dict__:
                c.update_default_config(base_class.__dict__[key])
#            else:
#                raise Exception('No config dictionary with key %s '%key +
#                                'in %s'%str(cls))

    @classmethod
    def get_default_config(cls,key='default_config'):

        o = {}        
        for base_class in inspect.getmro(cls):            
            if key in base_class.__dict__:
                o.update(base_class.__dict__[key])

        for k in o.keys():
            o[k] = Option.create(k,o[k])

        return o
        
    @classmethod
    def add_arguments(cls,parser):
        config = cls.get_default_config()

        for k, v in config.iteritems():
            if v.type == bool:
                parser.add_argument('--' + k,default=v.value,
                                    action='store_true',
                                    help=v.docstring + ' [default: %s]'%v.value)
            else:

                if isinstance(v.value,list):
                    value=','.join(map(str,v.value))
                else: value = v.value
                parser.add_argument('--' + k,default=value,
                                    type=type(value),
                                    help=v.docstring + ' [default: %s]'%v.value)
        
                
    def update_config(self,config):
        update_dict(self._config,config)

    def update_default_config(self,default_dict,group=None):
        """Update configuration for the object adding keys for
        elements that are not present.  If group is defined then
        this configuration will be nested in a dictionary with that
        key."""
        if default_dict is None: return

        if not isinstance(default_dict,dict) and \
                issubclass(default_dict,Configurable):
            default_dict = default_dict.default_config
        elif not isinstance(default_dict,dict):
            raise Exception('Wrong type for default dict.')

        if group:
            default_config = self._default_config.setdefault(group,{})
            self._config.setdefault(group,{})
        else:
            default_config = self._default_config
            
        update_dict(default_config,default_dict,True)
        for k in default_dict.keys():
#            if not isinstance(self._default_config[k],Option):
            option = Option.create(k,default_dict[k],group=group) 
            default_config[option.name] = option

            print k, group, option.name, option.group

            if option.group:
                self._config.setdefault(option.group,{})
                self._config[option.group][option.name] = option.value
            else:
                self._config[option.name] = self._default_config[k].value
       
    def print_config(self):
        
        print 'CONFIG'
        for k, v in self._default_config.iteritems():
            print '%20s %10s %10s %s'%(k,self._config[k],v.value,
                                       v.docstring)

    @property
    def config(self):
        return self._config

    def config_docstring(self,key):
        return self._default_config[key].docstring

    def set_config(self,key,value):
        self._config[key] = value
        
    def configure(self,config=None,opts=None,group=None,**kwargs):
        """Update the configuration of this object with the contents
        of 'config'.  When the same option is defined in multiple
        inputs the order of precedence is config -> opts -> kwargs."""
        
        if not config is None:

            if group and not group in config:
                raise Exception('Exception')

            if group:
                update_dict(self._config,config[group])
            else:
                update_dict(self._config,config)
#            if not group is None and group in config and not \
#                    config[group] is None:                
#                update_dict(self._config,config[group])
#            else:
            
                
        if not opts is None:
            for k,v in opts.__dict__.iteritems():
                if k in self._config and not v is None:
                    if isinstance(self._config[k],list) and \
                            not isinstance(v,list):

                        value = v.split(',')
                        value = map(self._default_config[k].list_type,value)
                        self.set_config(k,value)
                    else:
                        self.set_config(k,v)
                
        update_dict(self._config,kwargs)
        
        for k, v in self._config.iteritems():

            if v is None or not isinstance(v,str): continue            
            if os.path.isfile(v): self._config[k] = os.path.abspath(v)

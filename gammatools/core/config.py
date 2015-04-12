import inspect
import os
import copy
from gammatools.core.util import update_dict, merge_dict

class Option(object):

    def __init__(self,name,value,docstring='',option_type=str,group=None,
                 list_type=None):
        self._name = name
        self._value = value
        self._docstring = docstring
        self._option_type = option_type
        self._group = group

        if list_type is not None:
            self._list_type = list_type
        elif option_type == list and value is not None and len(value):
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
    
    @property
    def argname(self):
        return self._name
    
    @staticmethod
    def create(name,x):
        """Create an instance of an option from a tuple or a scalar."""
            
        #isinstance(x,list): return Option(x[0],x[1])
        if isinstance(x,Option): return x
        elif isinstance(x,tuple):

            item_list = [None,'',None,str]
            item_list[:len(x)] = x
            value, docstring, groupname, option_type = item_list

            list_type = None

            if isinstance(option_type,tuple):
                option_type, list_type = option_type
            
#                raise Exception('Wrong size for option tuple.')
#            if value is None and (option_type == list or option_type == dict):
#                value = option_type()
                            
            if value is not None:
                option_type = type(value)
            
            return Option(name,value,docstring,option_type,list_type=list_type)
        else:
            if x is not None: option_type = type(x)
            else: option_type = str            
            return Option(name,x,option_type=option_type)

def create_options_dict(o):

    od = {}
            
    for k,v in o.items():
        if isinstance(v,dict):
            od[k] = create_options_dict(v)
        else:
            od[k] = Option.create(k,v)
            
    return od


class Configurable(object):

    def __init__(self,config=None,opts=None,register_defaults=True,**kwargs):
        
        self._config = {}
        if register_defaults:
            self._default_config = self.create_default_config()
            self._config = self.create_config(self._default_config)

        self.configure(config,opts,**kwargs)

    @property
    def config(self):
        return self._config
        
#    @classmethod
#    def create_default_config(cls,key='default_config'):
#        """Create default configuration dictionaries for this class
#        and all classes from which it inherits."""
#
#        config = {}        
#        for base_class in inspect.getmro(cls):
#
#            if key in base_class.__dict__:
#                c = Configurable.create_config(base_class.__dict__[key])
#                config = merge_dict(config,c,add_new_keys=True)
#        return config

    def update_default_config(self,default_dict):
        """Update the defaults dictionary for this class."""
        
        if not isinstance(default_dict,dict) and issubclass(default_dict,Configurable):
            default_dict = default_dict.default_config
        elif not isinstance(default_dict,dict):
            raise Exception('Wrong type for default dict.')

        new_default_config = create_options_dict(default_dict)
        
        default_config = merge_dict(self._default_config,new_default_config,
                                    add_new_keys=True)

#        default_config = create_options_dict(default_config)

        self._default_config = default_config
        self._config = self.create_config(self._default_config)
        
    @classmethod
    def create_default_config(cls,key='default_config'):
        
        o = {}        
        for base_class in inspect.getmro(cls):
            if key in base_class.__dict__:
                o = merge_dict(o,base_class.__dict__[key],add_new_keys=True)
                #            else:
#                raise Exception('No config dictionary with key %s '%key +
#                                'in %s'%str(cls))
        
        o = create_options_dict(o)
        return o

    @classmethod
    def get_class_config(cls,key='default_config'):
        
        o = copy.deepcopy(cls.__dict__[key])
        for k in o.keys():
            o[k] = Option.create(k,o[k],group=group)

        return o

    @classmethod
    def add_arguments(cls,parser,config=None,skip=None,prefix=None):
        
        if config is None:
            config = cls.create_default_config()

        groups = {}
#        for k, v in config.items():
#            if v.group is not None and not v.group in groups:
#                groups[v.group] = parser.add_argument_group(v.group)

        for k, v in sorted(config.items()):
            
            if skip is not None and v.name in skip: continue

            if isinstance(v,dict):
                Configurable.add_arguments(parser,config=v,prefix=k)
                continue

            if v.group is not None and not v.group in groups:
                groups[v.group] = parser.add_argument_group(v.group)
                group = groups[v.group]
            else:
                group=parser


            if prefix:
                argname = prefix + '.' + v.argname
            else:
                argname = v.argname
                
            if v.type == bool:
                group.add_argument('--' + argname,default=v.value,
                                   action='store_true',
                                   help=v.docstring + ' [default: %s]'%v.value)
            else:

                if v.type == list:
                    if v.value is not None:
                        value=','.join(map(str,v.value))
                    else:
                        value = v.value
                        
                    opt_type = str
                else:
                    value = v.value
                    opt_type = v.type
                    
                group.add_argument('--' + argname,default=value,
                                    type=opt_type,
                                    help=v.docstring + ' [default: %s]'%v.value)
        
                
    def update_config(self,config):
        self._config = merge_dict(self._config,config)

    @staticmethod
    def create_config(default_dict,options_dict=False):
        """Create a configuration dictionary from an options dictionary."""

        o = {}
        if not isinstance(default_dict,dict) and issubclass(default_dict,Configurable):
            default_dict = default_dict.default_config
        elif not isinstance(default_dict,dict):
            raise Exception('Wrong type for default dict.')
        
        for k, v in default_dict.items():

            if isinstance(v,dict):
                o[k] = Configurable.create_config(v)
            else:
                if not isinstance(v,Option):
                    option = Option.create(k,v)
                else:
                    option = v
                    
                if not options_dict:
                    o[option.name] = option.value
                else:
                    o[option.name] = option

        return o
       
    def print_config(self):
        
        for k, v in self._default_config.items():
            print('%20s %10s %10s %s'%(k,self._config[k],v.value,
                                       v.docstring))

    @property
    def config(self):
        return self._config

    def config_docstring(self,key):
        return self._default_config[key].docstring

    def set_config(self,key,value):
        self._config[key] = value

    def parse_opts(self,opts):

        
        for k,v in opts.__dict__.items():

            argname = k.split('.')
            if v is None: continue

            if len(argname) == 2:
                default_config = self._default_config[argname[0]][argname[1]]
                config = self._config[argname[0]][argname[1]]
            else:
                
                if not k in self._default_config: continue                
                default_config = self._default_config[k]
                config = self._config[k]

            if default_config.type == list and v is not None:
                value = v.split(',')
                value = map(default_config.list_type,value)
            else:
                value = v

                
            if len(argname) == 1:
                self._config[k] = value
            else:
                self._config[argname[0]][argname[1]] = value
                
        
    def configure(self,config=None,opts=None,**kwargs):
        """Update the configuration of this object with the contents
        of 'config'.  When the same option is defined in multiple
        inputs the order of precedence is config -> opts -> kwargs."""
        
        if not config is None:
            self._config = merge_dict(self._config,config)
                
        if not opts is None: self.parse_opts(opts)
                
        self._config = merge_dict(self._config,kwargs)
        
        for k, v in self._config.items():
            if v is None or not isinstance(v,str): continue            
            if os.path.isfile(v): self._config[k] = os.path.abspath(v)

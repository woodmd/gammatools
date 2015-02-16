import yaml

import gammatools
from gammatools.dm.jcalc import *

class HaloModelFactory(object):

    @staticmethod
    def create(src_name,model_file = None,rho_rsun = None, gamma = None,
               **kwargs):

        if model_file is None:
            model_file = os.path.join(gammatools.PACKAGE_ROOT,
                                      'data/dm_halo_models.yaml')

        halo_model_lib = yaml.load(open(model_file,'r'))
        
        if not src_name in halo_model_lib:
            raise Exception('Could not find profile: ' + src_name)

        src = halo_model_lib[src_name]
        src.update(kwargs)

        if rho_rsun is not None:
            src['rhor'] = [rho_rsun,8.5]

        if gamma is not None:
            src['gamma'] = gamma

        return HaloModel(src)

class HaloModel(object):

    def __init__(self,src,**kwargs):

        src.update(kwargs)

        self._losfn = LoSIntegralFnFast.create(src)
        self._dp = DensityProfile.create(src)
        self._jp = JProfile(self._losfn)
        self._dist = src['dist']*Units.kpc

    @property
    def dist(self):
        return self._dist

    @property
    def dp(self):
        return self._dp

    @property
    def jp(self):
        return self._jp

    @property
    def losfn(self):
        return self._losfn

    def jval(self,loge,psi):

        return self._jp(psi)

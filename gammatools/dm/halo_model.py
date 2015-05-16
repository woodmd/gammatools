import yaml
import copy

import gammatools
from gammatools.dm.jcalc import *
from gammatools.core.util import merge_dict

class HaloModelFactory(object):

    @staticmethod
    def create(src_name,model_file = None,**kwargs):

        if model_file is None:
            model_file = os.path.join(gammatools.PACKAGE_ROOT,
                                      'data/dm_halo_models.yaml')

        halo_model_lib = yaml.load(open(model_file,'r'))
        
        if not src_name in halo_model_lib:
            raise Exception('Could not find profile: ' + src_name)

        src = halo_model_lib[src_name]
        src = merge_dict(src,kwargs,skip_values=[None])

#        if rho_rsun is not None:
#            src['rhor'] = [rho_rsun,8.5]
#        if gamma is not None:
#            src['gamma'] = gamma
        hm = HaloModel(src)

        if 'jvalue' in src:
            
            if len(src['jvalue']) == 1:
                hm.set_jvalue(Units.parse(src['jvalue'][0]),0.5)
            else:
                hm.set_jvalue(Units.parse(src['jvalue'][0]),
                              Units.parse(src['jvalue'][1]))

        return hm

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

    def set_jvalue(self,jvalue,psi):

        jv = self._jp.cumsum(psi)
        scale = jvalue/jv
        self._dp._rhos *= scale**0.5
        self._losfn._dp = copy.deepcopy(self._dp)
        self._jp = JProfile(self._losfn)

        import matplotlib.pyplot as plt

        j90 = self._jp.cumsum(0.3597)

        print self._jp.cumsum(0.122)/self._jp.cumsum(0.5)


        return

        plt.figure()
        plt.plot(np.degrees(self._jp._psi),self._jp._jcum/self._jp._jcum[-1])
#        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.gca().grid(True)
        plt.gca().set_xlim(0.01,1.0)
#        plt.gca().set_ylim(1E18,1E20)

        plt.gca().axvline(0.5,color='k')
        plt.gca().axvline(0.3597,color='k')
        plt.gca().axvline(0.122,color='k')

        plt.show()


    def jvalue(self,loge,psi):

        return self._jp(psi)

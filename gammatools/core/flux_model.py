import numpy as np

class FluxModel(object):

    
    def e2flux(self,loge,psi):
        return np.power(10,2*loge)*self.flux(loge,psi)

    def eflux(self,loge,psi):
        return np.power(10,loge)*self.flux(loge,psi)

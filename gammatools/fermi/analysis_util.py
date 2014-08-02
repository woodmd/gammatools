import copy
from gammatools.core.histogram import Histogram
from data import PhotonData

def getHist(data,var_name,mask=None,edges=None):
    
    h = Histogram(edges)    

    if not mask is None:
        h.fill(data[var_name][mask])
    else:
        h.fill(data[var_name])

    return h

def getOnOffHist(data,var_name,phases,mask=None,edges=None):
    
    (on_phase,off_phase,alpha) = phases

    on_mask = PhotonData.get_mask(data,phases=on_phase)
    off_mask = PhotonData.get_mask(data,phases=off_phase)

    if not mask is None:
        on_mask &= mask
        off_mask &= mask

    hon = Histogram(edges)
    hon.fill(data[var_name][on_mask])

    hoff = Histogram(edges)
    hoff.fill(data[var_name][off_mask])

    hoffs = copy.deepcopy(hoff)
    hoffs *= alpha

    return (hon,hoff,hoffs)

def parse_phases(on_phase,off_phase):
    
    on_phases = []
    off_phases = []
    alpha = 0
    
    on_phase_range = 0
    phases = on_phase.split(',')
    for p in phases:
        (plo,phi) = p.split('/')
        plo = float(plo)
        phi = float(phi)
        on_phase_range += (phi-plo)
        on_phases.append([plo,phi])

    off_phase_range = 0
    phases = off_phase.split(',')
    for p in phases:
        (plo,phi) = p.split('/')
        plo = float(plo)
        phi = float(phi)
        off_phase_range += (phi-plo)
        off_phases.append([plo,phi])
    
    alpha = on_phase_range/off_phase_range

    return (on_phases,off_phases,alpha)


    
class ModelBuilder(object):

    default_config = { 'xml' : (None) }
    
    def __init__(self):

        pass

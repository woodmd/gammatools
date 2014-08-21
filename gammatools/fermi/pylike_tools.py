import numpy as np
from gammatools.fermi import units
from gammatools.core.util import tolist

from FluxDensity import FluxDensity
from SummedLikelihood import SummedLikelihood

from pyLikelihood import ParameterVector, SpatialMap_cast, PointSource_cast
import pyLikelihood

def GetCountsMap(binnedAnalysis):
    """ Get the shape of the observed counts map
    from a BinnedAnalysis object

    binnedAnalysis:    The BinnedAnalysis object 

    returns  np.ndarray( (nEBins, nPixX, nPixY), 'd' )
    """
    ll = binnedAnalysis.logLike    
    shape = GetCountsMapShape(ll.countsMap())
    a = np.ndarray( (shape[2],shape[1],shape[0]), 'f' )
    a.flat = binnedAnalysis.logLike.countsMap().data()
    return a


def GetCountsMapShape(countsMap):
    """ Get the shape of the observed counts map
    from a CountsMap object

    countsMap:    The CountsMap object     

    returns tuple ( nEBins, nPixX, nPixY )
    """
    n0 = countsMap.imageDimension(0)
    n1 = countsMap.imageDimension(1)
    try:
        n2 = countsMap.imageDimension(2)
    except:
        return (n0,n1)
    return (n0,n1,n2)

def get_gtlike_source(like,name):

    if isinstance(like,SummedLikelihood):
        return like.components[0].logLike.getSource(name)
    else:
        return like.logLike.getSource(name)

def name_to_spectral_dict(like, name, errors=False, minos_errors=False, covariance_matrix=False):

    print name

    source = get_gtlike_source(like,name)
    spectrum = source.spectrum()
    d=gtlike_spectrum_to_dict(spectrum, errors)
    if minos_errors:
        parameters=ParameterVector()
        spectrum.getParams(parameters)
        for p in parameters: 
            pname = p.getName()
            if p.isFree():
                lower,upper=like.minosError(name, pname)
                try:
                    d['%s_lower_err' % pname] = -1*lower*p.getScale()
                    d['%s_upper_err' % pname] = upper*p.getScale()
                except Exception, ex:
                    print 'ERROR computing Minos errors on parameter %s for source %s:' % (pname,name), ex
                    traceback.print_exc(file=sys.stdout)
                    d['%s_lower_err' % pname] = np.nan
                    d['%s_upper_err' % pname] = np.nan
            else:
                d['%s_lower_err' % pname] = np.nan
                d['%s_upper_err' % pname] = np.nan
    if covariance_matrix:
        d['covariance_matrix'] = get_covariance_matrix(like, name)
    return d

def get_full_energy_range(like):
    return like.energies[[0,-1]]

def gtlike_flux_dict(like,name, emin=None,emax=None,flux_units='erg', energy_units='MeV',
                     errors=True, include_prefactor=False, prefactor_energy=None):
    """ Note, emin, emax, and prefactor_energy must be in MeV """

    if emin is None and emax is None: 
        emin, emax = get_full_energy_range(like)

    cef=lambda e: units.convert(e,'MeV',flux_units)
    ce=lambda e: units.convert(e,'MeV',energy_units)
    f=dict(flux=like.flux(name,emin=emin,emax=emax),
           flux_units='ph/cm^2/s',
           eflux=cef(like.energyFlux(name,emin=emin,emax=emax)),
           eflux_units='%s/cm^2/s' % flux_units,
           emin=ce(emin),
           emax=ce(emax),
           energy_units=energy_units)

    if errors:
        try:
            # incase the errors were not calculated
            f['flux_err']=like.fluxError(name,emin=emin,emax=emax)
            f['eflux_err']=cef(like.energyFluxError(name,emin=emin,emax=emax))
        except Exception, ex:
            print 'ERROR calculating flux error: ', ex
            traceback.print_exc(file=sys.stdout)
            f['flux_err']=-1
            f['eflux_err']=-1

    if include_prefactor:
        assert prefactor_energy is not None
        source = get_gtlike_source(like,name)
#        source = like.logLike.getSource(name)
        spectrum = source.spectrum()
        cp = lambda e: units.convert(e,'1/MeV','1/%s' % flux_units)
        f['prefactor'] = cp(SpectrumPlotter.get_dnde_mev(spectrum,prefactor_energy))
        f['prefactor_units'] = 'ph/cm^2/s/%s' % flux_units
        f['prefactor_energy'] = ce(prefactor_energy)
    return tolist(f)

def energy_dict(emin, emax, energy_units='MeV'):
    ce=lambda e: units.convert(e,'MeV',energy_units)
    return dict(emin=ce(emin),
                emax=ce(emax),
                emiddle=ce(np.sqrt(emin*emax)),
                energy_units=energy_units)

def gtlike_ts_dict(like, name, verbosity=True):
    return dict(
        reoptimize=like.Ts(name,reoptimize=True, verbosity=verbosity),
        noreoptimize=like.Ts(name,reoptimize=False, verbosity=verbosity)
        )

def gtlike_spectrum_to_dict(spectrum, errors=False):
    """ Convert a pyLikelihood object to a python 
        dictionary which can be easily saved to a file. """
    parameters=ParameterVector()
    spectrum.getParams(parameters)
    d = dict(name = spectrum.genericName(), method='gtlike')
    for p in parameters: 
        d[p.getName()]= p.getTrueValue()
        if errors: 
            d['%s_err' % p.getName()]= p.error()*p.getScale() if p.isFree() else np.nan
        if d['name'] == 'FileFunction': 
            ff=pyLikelihood.FileFunction_cast(spectrum)
            d['file']=ff.filename()
    return d

def gtlike_name_to_spectral_dict(like, name, errors=False, minos_errors=False, covariance_matrix=False):
#    source = like.logLike.getSource(name)
    source = get_gtlike_source(like,name)
    spectrum = source.spectrum()
    d=gtlike_spectrum_to_dict(spectrum, errors)
    if minos_errors:
        parameters=ParameterVector()
        spectrum.getParams(parameters)
        for p in parameters: 
            pname = p.getName()
            if p.isFree():
                lower,upper=like.minosError(name, pname)
                try:
                    d['%s_lower_err' % pname] = -1*lower*p.getScale()
                    d['%s_upper_err' % pname] = upper*p.getScale()
                except Exception, ex:
                    print 'ERROR computing Minos errors on parameter %s for source %s:' % (pname,name), ex
                    traceback.print_exc(file=sys.stdout)
                    d['%s_lower_err' % pname] = np.nan
                    d['%s_upper_err' % pname] = np.nan
            else:
                d['%s_lower_err' % pname] = np.nan
                d['%s_upper_err' % pname] = np.nan
    if covariance_matrix:
        d['covariance_matrix'] = get_covariance_matrix(like, name)
    return d

def gtlike_source_dict(like, name, emin=None, emax=None, 
                       flux_units='erg', energy_units='MeV', 
                       errors=True, minos_errors=False, covariance_matrix=True,
                       save_TS=True, add_diffuse_dict=True,
                       verbosity=True):

    if emin is None and emax is None:
        emin, emax = get_full_energy_range(like)

    d=dict(
        logLikelihood=like.logLike.value(),
    )

    d['energy'] = energy_dict(emin=emin, emax=emax, energy_units=energy_units)
    
    d['spectrum']= name_to_spectral_dict(like, name, errors=errors, 
                                         minos_errors=minos_errors, covariance_matrix=covariance_matrix)

    if save_TS:
        d['TS']=gtlike_ts_dict(like, name, verbosity=verbosity)

    d['flux']=gtlike_flux_dict(like,name,
                               emin=emin, emax=emax,
                               flux_units=flux_units, energy_units=energy_units, errors=errors)


#    if add_diffuse_dict:
#        d['diffuse'] = diffuse_dict(like)

    return tolist(d)

def get_covariance_matrix(like, name):
    """ Get the covarince matrix. 

        We can mostly get this from FluxDensity, but
        the covariance matrix returned by FluxDensity
        is only for the free paramters. Here, we
        transform it to have the covariance matrix
        for all parameters, and set the covariance to 0
        when the parameter is free.
    """

#    source = like.logLike.getSource(name)
    source = get_gtlike_source(like,name)
    spectrum = source.spectrum()

    parameters=ParameterVector()
    spectrum.getParams(parameters)
    free = np.asarray([p.isFree() for p in parameters])
    scales = np.asarray([p.getScale() for p in parameters])
    scales_transpose = scales.reshape((scales.shape[0],1))

    cov_matrix = np.zeros([len(parameters),len(parameters)])

    try:
        fd = FluxDensity(like,name)
        cov_matrix[np.ix_(free,free)] = fd.covar

        # create absolute covariance matrix:
        cov_matrix = scales_transpose * cov_matrix * scales
    except RuntimeError, ex:
        if ex.message == 'Covariance matrix has not been computed.':
            pass
        else: 
            raise ex

    return tolist(cov_matrix)

def diffuse_dict(like):
    """ Save out all diffuse sources. """

    f = dict()
    bgs = get_background(like)
    for name in bgs:
        f[name] = name_to_spectral_dict(like, name, errors=True)
    return tolist(f)

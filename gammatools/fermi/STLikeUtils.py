#!/usr/bin/env python
#

# Description
"""
Utilities for dealing with ScienceTool likelihood analysis objects
"""


__facility__ = "LikeTools.STLikeUtils.py"
__abstract__ = __doc__
__author__    = "E. Charles & Geng Zhao"
__date__      = "$Date: 2014/07/24 19:59:42 $"
__version__   = "$Revision: 1.7 $, $Author: echarles $"
__release__   = "$Name:  $"

import numpy
import math
import pyLikelihood
import pyfits
#from pil import Pil
from scipy import misc
#from uw.darkmatter.spectral import DMFitFunction
from BinnedAnalysis import BinnedObs, BinnedAnalysis


def GetSourceMapDict(binnedAnalysis):
    """ Get the dictionary of name : SourceMap
    from a BinnedAnalysis object

    binnedAnalysis:    The BinnedAnalysis object 

    returns  dict{'SourceName':SourcMap}
    """
    from LikeTools import SourceHandle

    ll = binnedAnalysis.logLike
    cm = ll.countsMap()
    shape = GetCountsMapShape(cm)
    eVals = GetEnergyEdges(cm)
    
    srcNames = binnedAnalysis.sourceNames()
    d = {}
    for (idx,n) in enumerate(srcNames):
        srcHandle = SourceHandle.SourceHandle(binnedAnalysis,n,idx,shape,eVals)
        d[n] = srcHandle
        pass
    return d




def GetCountsMap(binnedAnalysis):
    """ Get the shape of the observed counts map
    from a BinnedAnalysis object

    binnedAnalysis:    The BinnedAnalysis object 

    returns  numpy.ndarray( (nEBins, nPixX, nPixY), 'd' )
    """
    ll = binnedAnalysis.logLike    
    shape = GetCountsMapShape(ll.countsMap())
    a = numpy.ndarray( (shape[2],shape[1],shape[0]), 'f' )
    a.flat = binnedAnalysis.logLike.countsMap().data()
    return a.swapaxes(0,2)


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


def GetEnergyEdges(countsMap):
    """ Get the edges of the energy bins 
    from a CountsMap object

    countsMap:    The CountsMap object     

    returns  numpy.ndarray( (nEBins+1), 'd' )    
    """
    import pyLikelihood
    n2 = countsMap.imageDimension(2)
    dv = pyLikelihood.DoubleVector(n2+1)
    countsMap.getAxisVector(2,dv)
    a = numpy.ndarray(n2+1,'d')
    a.flat = dv
    return a


def GetSpectralVals(source,eVals):
    """ Get the dF/dE values ( ph cm^-2 s^-1 MeV^-1 ) for a particular source

    source:            A Source object
    eVals:             Energy values at bin edges

    returns  numpy.ndarray( (nEBins+1), 'd' )  
    """
    import pyLikelihood
    aSpec = source.spectrum()
    retVals = numpy.ndarray( eVals.shape, 'd')
    n = len(eVals)
    for i in range(n):
        dArg = pyLikelihood.dArg(eVals[i])
        retVals[i] = aSpec(dArg)
        pass
    return retVals
    

def ExtractSrcMapVals(srcMap,shape):
    """ Extract the values from the gtsrcmap object for a particular source

    srcMap:            A SourceMap object
    shape:             Shape of the output array

    returns  numpy.ndarray( (shape), 'd' )      
    """
    a = numpy.ndarray( (shape[2],shape[1],shape[0]), 'f' )
    a.flat = srcMap.model()
    return a.swapaxes(0,2)



def MakeParameterTable(tableName,analysis):
    """ Store fit parameters to a fits table

    tableName:         The fits table name
    analysis:          The analysis object

    returns pyfits.BinTableHDU  
    """
    srcNames = []
    parNames = []
    parFree = []
    parScales = []
    parValues = []
    parErrors = []
    parMins = []
    parMaxs = []

    for srcName in analysis.sourceNames():
        spec = analysis[srcName].funcs["Spectrum"]
        pNames = spec.paramNames        
        for pName in pNames:
            param = spec.params[pName].parameter
            srcNames.append(srcName)
            parNames.append(param.getName())
            parFree.append(param.isFree())
            parScales.append(param.getScale())
            parValues.append(param.getValue())
            parErrors.append(param.error())
            parMins.append(param.getBounds()[0])
            parMaxs.append(param.getBounds()[1])
            pass
        pass

    srcNameCol = pyfits.Column(name="SourceName", format='26A',array=srcNames)
    parNameCol = pyfits.Column(name="ParamName", format='26A',array=parNames)
    parFreeCol = pyfits.Column(name="Free", format='L',array=parFree)
    parScaleCol = pyfits.Column(name="Scale", format='E',array=parScales)
    parValueCol = pyfits.Column(name="Value", format='E',array=parValues)
    parErrorCol = pyfits.Column(name="Error", format='E',array=parErrors)
    parMinCol = pyfits.Column(name="MinValue", format='E',array=parMins)
    parMaxCol = pyfits.Column(name="MaxValue", format='E',array=parMaxs)

    parColList = [srcNameCol,parNameCol,parFreeCol,parScaleCol,parValueCol,parErrorCol,parMinCol,parMaxCol]

    parTable = pyfits.new_table(parColList)
    parTable.name = tableName
    return parTable

    
def ExtractNPreds(srcMap,np):
    """ Extract the nPred values from the gtsrcmap object for a particular source

    srcMap:            A SourceMap object
    np:                Shape of the output array (Number of points)

    returns  numpy.ndarray( (np), 'd' )      
    """
    a = numpy.ndarray(np,'d')
    a.flat = srcMap.npreds()
    return a


def MakeModelMap_SpectrumVals(srcMapVals,specVals,eVals):
    """ Create the model map from the gtsrcmaps object, the spectral values and the energy bin edges

    srcMapVals:        A numpy.ndarray( nEBins+1, nPixX, nPixY ) of the SourceMap model values
    specVals:          A numpy.ndarray( nEBins+1 ) of the Spectrum flux values at the bin edges
    eVals:             Energy values at bin edges

    returns  numpy.ndarray( (nEBins, nPixX, nPixY), 'd' )          
    """
    yVals = srcMapVals * ( specVals * eVals )
    logEsteps = numpy.log( eVals[1:] / eVals[0:-1] )
    logEmeans = ( yVals[0:,0:,0:-1]  + yVals[0:,0:,1:] ) / 2.
    retVals = logEmeans * logEsteps
    return retVals
    

def ExtractSpectrumAndPredMap(srcMap,src,shape,eVals):
    """ Create the model map from the gtsrcmaps object, the spectral values and the energy bin edges

    srcMapVals:        A numpy.ndarray( nEBins+1, nPixX, nPixY ) of the SourceMap model values
    src:               A Source object
    shape:             Shape of the output array
    eVals:             Energy values at bin edges

    returns dict{'SrcMapVals':numpy.ndarray( (nEBins+1, nPixX, nPixY), 'd' ),
                 'SpecVals':numpy.ndarray( (nEBins), 'd' ),
                 'ModelVals':numpy.ndarray( (nEBins,nPixX, nPixY), 'd' )}
     
    """
    srcMapVals = ExtractSrcMapVals(srcMap,shape)
    specVals = GetSpectralVals(src,eVals)
    modelVals = MakeModelMap_SpectrumVals(srcMapVals,specVals,eVals)
    
    retDict = {"SrcMapVals":srcMapVals,
               "SpecVals":specVals,
               "ModelVals":modelVals}
    return retDict


def ExtractCovarianceMatrixFromAnalysis(analysis):
    """ Extract the covariance matrix from a binned analysis

    analysis:    The AnalysisBase object     

    returns numpy.matrix ( nFree, nFree )    
    """
    a = analysis.covariance
    n = len(a)
    arr = numpy.ndarray((n,n),'d')
    arr.flat = a
    retVal = numpy.matrix(arr)
    return retVal


def ExtractCorrelationFactors(covMatrix):
    """ Extract the correlation factors from a covariance matrix

    covMatrix:    The covariance matrix  

    returns numpy.matrix ( nFree, nFree )    
    """
    theCopy = covMatrix.copy()
    variances = numpy.sqrt(theCopy.diagonal())
    correl = ((theCopy / variances).T)/variances
    return correl
    

def ExtractLinearGradient(data_t,model_t,models,parScales=None):
    """ Extract the gradient for the normalization factors only for a set of models

    data_t:        The total counts map from data
    model_t:       The total model map
    models:        The individual model maps

    returns numpy.matrix ( nFree )    
    """
    npar = len(models)
    grad = numpy.ndarray( (npar), 'd')

    resid_t = (data_t - model_t) / model_t

    for ipar in range(npar):
        grad[ipar] = (resid_t * models[ipar]).sum()        
        pass
    if parScales is not None:
        grad *= parScales
    return numpy.matrix(grad)


def ExtractLinearHessian(data_t,model_t,models,parScales=None):
    """ Extract the Hessian for the normalization factors only for a set of models

    data_t:        The total counts map from data
    model_t:       The total model map
    models:        The individual model maps

    returns numpy.matrix ( nFree, nFree )    
    """
    npar = len(models)
    hessian = numpy.ndarray( (npar,npar), 'd')

    pref = data_t / ( model_t * model_t )

    for ipar in range(npar):
        for jpar in range(npar):
            hessian[ipar,jpar] = -1. * (pref * models[ipar] * models[jpar]).sum()
            if parScales is not None:
                hessian[ipar,jpar] *= (parScales[ipar] * parScales[jpar])
            pass
        pass
    return numpy.matrix(hessian)


def ExtractLinearCovMatrix(data_t,model_t,models):
    """ Extract the covariance matrix for the normalization factors only for a set of models

    data_t:        The total counts map from data
    model_t:       The total model map
    models:        The individual model maps

    returns numpy.matrix ( nFree, nFree )  
    """
    hesse = ExtractLinearHessian(data_t,model_t,models)
    cov = -1. * numpy.linalg.inv(hesse)
    return numpy.matrix(cov)


def ExtractModelTotals(models):
    """ Extract the total counts spectral from a set of models

    models:        The individual model maps

    returns numpy.ndarray ( nEBins,nModels )  
    """
    npar = len(models)
    nEBins = models[0].shape[0]
    tots = numpy.ndarray( (nEBins,npar), 'd')
    for ipar in range(npar):
        tots[0:,ipar] = models[ipar].sum(2).sum(1)
        pass
    return tots
        

def ExtractGradVersusEnergy(data_t,model_t,models):
    """ Extract the energy bin-by-bin Covariance matrices 

    data_t:        The total counts map from data
    model_t:       The total model map
    models:        The individual model maps

    returns numpy.ndarray ( nEBins, nFree )  
    """
    nEBins = data_t.shape[0]
    npar = len(models)
    grad = numpy.ndarray( (nEBins,npar), 'd')

    resid_t = (data_t - model_t) / model_t

    for ipar in range(npar):        
        grad[0:,ipar] = (resid_t * models[ipar]).sum(2).sum(1)
        pass
    return numpy.matrix(grad)
    

def ExtractCovVersusEnergy(data_t,model_t,models):
    """ Extract the energy bin-by-bin Covariance matrices 

    data_t:        The total counts map from data
    model_t:       The total model map
    models:        The individual model maps

    returns numpy.ndarray ( nEBins, nFree, nFree )  
    """
    nEBins = data_t.shape[0]
    npar = len(models)
    hessian = numpy.ndarray( (nEBins,npar,npar), 'd')
    cov = numpy.ndarray( (nEBins,npar,npar), 'd')
   
    pref = data_t / ( model_t * model_t )

    for ipar in range(npar):
        for jpar in range(npar):
            hessian[0:,ipar,jpar] = -1. * (pref * models[ipar] * models[jpar]).sum(2).sum(1) 
            pass
        pass

    for i in range(nEBins):
        cov[i] = -1. * numpy.linalg.inv(hessian[i])
        pass

    return cov,hessian



def ExtractDiagErrorVersusEnergy(hesse_v_energy):
    """ Extract the correlation factors from the energy bin-by-bin cov matrices

    hesse_v_energy:    The Hessian matrices

    returns numpy.ndarray ( nEBins, nFree )    
    """
    diag_err = numpy.ndarray( (hesse_v_energy.shape[0],hesse_v_energy.shape[1]), 'd')
    for i in range(hesse_v_energy.shape[0]):
        diag_err[i] = numpy.sqrt( -1. / hesse_v_energy[i].diagonal() )
        pass
    return diag_err



def ExtractCorrelVersusEnergy(cov_v_energy):
    """ Extract the correlation factors from the energy bin-by-bin cov matrices

    cov_v_energy:    The covariance matrices

    returns numpy.ndarray ( nEBins, nFree, nFree )    
    """
    correl = numpy.ndarray( (cov_v_energy.shape), 'd')
    for i in range(cov_v_energy.shape[0]):
        correl[i] = ExtractCorrelationFactors(cov_v_energy[i])
        pass
    return correl


def PropErrorsVersusEnergy(correl_v_energy,variances):
    """ Propagate the variances from the global fit to the bin-by-bin fits using the correlation matrices for the bin-by-bin fits

    correl_v_energy:   The covariance matrices for the bin-by-bin fits
    variances:         The variances from the global fits

    returns numpy.ndarray ( nEBins, nFree, nFree )    
    """
    nEbins = correl_v_energy.shape[0]
    propErrors = numpy.ndarray( (correl_v_energy.shape) , 'd')
    for iE in range(nEbins):        
        propErrors[iE] = variances.A*(correl_v_energy[iE]*variances.T.A).T
        pass
    return propErrors
        

def ExtractNormalizedDerivs(spec,eVals,normalizePar=False):
    """ Extract the derivatives for the flux with respect to the fit parameters at particular energies

    spec:          The spectrum function
    eVals:         The energy valeus
    normalizePar:  If true, normalize the derivative by muliplying but the parameter value / flux value

    return numpy.ndarray( nEBins, 1, nFree )
    """
    import pyLikelihood
    nFree = spec.getNumFreeParams()
    nE = len(eVals)
    result = numpy.ndarray((nE,1,nFree),'d')
    parVect = pyLikelihood.ParameterVector()
    pVals = pyLikelihood.DoubleVector()
    dVals = pyLikelihood.DoubleVector()
    spec.func.getFreeParamValues(pVals)
    spec.func.getFreeParams(parVect)
    for iE in range(nE):
        parE = pyLikelihood.dArg(eVals[iE])
        spec.func.getFreeParamValues(pVals)
        spec.func.getFreeParams(parVect)
        spec.getFreeDerivs(parE,dVals)
        val = spec.value(parE)
        for ipar in range(nFree):
            if normalizePar:
                result[iE,0,ipar] = dVals[ipar] * pVals[ipar] / val
            else:
                result[iE,0,ipar] = dVals[ipar]  #/ parVect[ipar].getScale()
            pass
        pass
    return result


def ExtractJacobians(analysis,eVals,normalizePar=False):
    """ Extract the derivatives of the flux with respect to the fit parameters at particular energy values
    This is the Jacobian of the transformation from the global covariance matrix to the bin-by-bin covariance matrices.

    analysis:       The analysis object
    eVals:          The energy values in question
    normalizePar:   If true, normalize the derivative by muliplying but the parameter value / flux value

    return numpy.ndarray( nEBins, nFreeSrc, nFreeSrc )
    """
    freeSpecList = []
    nFreeSrc = 0
    nFreePar = 0
    nE = len(eVals)
    freeParIdx = [0]

    for srcName in analysis.sourceNames():
        spec = analysis[srcName].funcs["Spectrum"]
        nFree = spec.getNumFreeParams()
        if nFree > 0:
            freeSpecList.append(spec)
            nFreeSrc += 1
            nFreePar += nFree
            freeParIdx.append(nFreePar)
        pass

    jacs = numpy.zeros( (nE,nFreeSrc,nFreePar), 'd')
    for iSrc in range(nFreeSrc):
        spec = freeSpecList[iSrc]
        normDerivs = ExtractNormalizedDerivs(spec,eVals,normalizePar)
        jacs[0:,iSrc,freeParIdx[iSrc]:freeParIdx[iSrc+1]] = normDerivs[0:,0]
        pass
    return jacs
    

def ExtractSpectralUncertainties(cov,jacs):
    """ Extract the covariance matrices of the flux at various energy bins by using the global covariance
    and the Jacobians at the various energies

    cov:    The global covariance matrix
    jacs:   The Jacobians at the various energies
    
    return numpy.ndarray( nEBins, nFreeSrc, nFreeSrc )
    """
    nE = jacs.shape[0]
    nFreeSrc = jacs.shape[1]
    nFreePar = jacs.shape[2]
    result = numpy.ndarray((nE,nFreeSrc,nFreeSrc),'d')
    for iE in range(nE):
        jacM = numpy.matrix(jacs[iE])
        result[iE] =  jacM * (cov * jacM.T)
        pass
    return result


def ExtractEnergyFactorsForComponent(matrices,idx):
    """ Extract the energy dependent factors for a single component from covariance or correlations matrices

    matrices:  The matrices
    idx:       The index of the component we care about

    return numpy.ndarray( nEBins, nFree )
    """
    return matrices[0:,0:,idx]


def ExtractVariancesVersusEnergy(cov_v_energy):
    """ Extract the correlation factors from the energy bin-by-bin cov matrices

    cov_V_energy:    The covariance matrices

    returns numpy.ndarray ( nEBins, nFree )    
    """
    vars = numpy.ndarray( (cov_v_energy.shape[0],cov_v_energy.shape[1]), 'd')
    for i in range(cov_v_energy.shape[0]):
        vars[i] = numpy.sqrt( cov_v_energy[i].diagonal() )
        pass
    return vars



def MakeBinnedSrcMap_FromPtSource(RA,DEC,obs):
    """ Make a binned Source map for a PointSource

    RA,DEC:  Coordinates of the point source
    obs:     The BinnedObservation object
    """
    import pyLikelihood
    source = pyLikelihood.PointSource(RA,DEC,obs.observation)
    applyPsfCorrections = True
    performConvolution = True
    resample = True
    resamp_factor = 2.0
    minbinsz = 0.1
    verbose = False
    srcMap = pyLikelihood.SourceMap(source,obs.countsMap,obs.observation,
                                    applyPsfCorrections,performConvolution,resample,resamp_factor,minbinsz,verbose)
    return (source,srcMap)


def MakeBinnedSrcMap_FromDiffuseSource(spatialProfile,obs):
    """ Make a binned Source map for a DiffuseSource

    spatialProfile:   The filename of the SpatialMap 
    obs:              The BinnedObservation object
    """
    import pyLikelihood
    spatialMap = pyLikelihood.SpatialMap(spatialProfile)
    source = pyLikelihood.DiffuseSource(spatialMap,obs.observation,False,False)
    applyPsfCorrections = True
    performConvolution = True
    resample = True
    resamp_factor = 2.0
    minbinsz = 0.1
    verbose = False
    srcMap = pyLikelihood.SourceMap(source,obs.countsMap,obs.observation,
                                    applyPsfCorrections,performConvolution,resample,resamp_factor,minbinsz,verbose)
    return (source,srcMap,spatialMap)

def AppendSourceMaps(baselineFile,extraFile,tempFile):
    """ Append source maps for a second file to a baseline file and write the output to a temp file
    """
    fin = pyfits.open(baselineFile)
    fextra = pyfits.open(extraFile)
    fin += fextra[3:]
    fin.writeto(tempFile,clobber=True)
    fin.close()


def MakeDMSource(options,obs):
    """ Make a DM Source object

    options:    Command line job options
    obs:        Observation object

    The following are used from the options object:
    
    For point sources (options.map_file = None):
    options.ra
    options.dec

    For diffuse sources (options.map_file != None):
    options.map_file

    Spectal Options:
    options.spectrum = "PowerLaw" or a DMFit Decay Channel
    
    For DMFit Channels:
    options.mass
    options.jvalue
    options.sigmav    

    return (Source,Spectrum,SpatialMap) for Diffuse sources or
    (Source,Spectrum,None) for Point Sources
    """
    if options.map_file is None:
        spatialMap = None
        source = pyLikelihood.PointSource(options.ra,options.dec,obs.observation) 
        pass
    else:
        spatialMap = pyLikelihood.SpatialMap(options.map_file)
        source = pyLikelihood.DiffuseSource(spatialMap,obs.observation,False,False)

    # Set the spectrum for the Model Map.   
    if options.spectrum == "PowerLaw":
        source.setSpectrum("PowerLaw")
        spec = source.spectrum()    
        spec.getParam("Prefactor").setValue(1.)
        spec.getParam("Prefactor").setScale(1.0e-7)
        spec.getParam("Prefactor").setBounds(1e-10,1e5)
        spec.getParam("Scale").setValue(100.)
        spec.getParam("Index").setValue(-2.)
        spec.getParam("Index").setFree(False)
        spec.getParam("Index").setBounds(-5.,5.)
    else:
        ch = DMFitFunction.channel2int(options.spectrum)
        spec = DMFitFunction(mass=options.mass,channel0=ch,norm=options.jvalue,sigmav=options.sigmav)
        spec.dmf.getParam("norm").setBounds(1e18,1e20)
        spec.dmf.getParam("norm").setFree(False)
        spec.dmf.getParam("sigmav").setBounds(1e-30,1e-20)
        spec.dmf.getParam("mass").setBounds(1.,1000.)
        spec.dmf.getParam("mass").setFree(False)
        spec.dmf.getParam("bratio").setBounds(0.0,1.0)
        spec.dmf.getParam("bratio").setFree(False)
        source.setSpectrum(spec.dmf)
    return (source,spec,spatialMap)


def ReplaceAnalysisComponent(analysis,obs,options,source_name=None,del_names=[]):
    """ Replace a particular source in an analysis

    analysis:    The BinnedAnalysis object
    obs:         The Observation object
    options:     Command line options (passed on to MakeDMSource
    source_name: Name of the new source.  None => Don't make new source
    del_names:   Name of sources to delete

    return (Source,Spectrum,SpatialMap) if a new source is made:
    (None,None,None) if no new source is made
    """
    for delSrc in del_names:
        analysis.deleteSource(delSrc)

    if source_name:
        (source,spec,spatialMap) = MakeDMSource(options,obs)
        source.setName(source_name)

        analysis.addSource(source)
        return (source,spec,spatialMap)

    return (None,None,None)


def SetupBinnedAnalysis(options,source_name=None,del_names=["lmc"]):
    """ Setup a standard Binned Likelihood analysis

    options:     Command line options (passed on to MakeDMSource
    source_name: Name of the new source.  None => Don't make new source
    del_names:   Name of sources to delete

    The following are used from the options object:
    options.input:  Input par file.
    
    The options are passed to 
    AppendSourceMaps()
    ReplaceAnalysisComponent()
    
    returns Observation,Analysis,Source,Spectrum,SpatialMap
    """
    # Read the par file
    pars = Pil(options.input,raiseKeyErrors=True,preserveQuotes=False)

    # Get parameters from the par file
    srcmaps = pars['cmap']
    
    try:
        if options.extraSrcMaps:
            AppendSourceMaps(pars['cmap'],options.extraSrcMaps,options.tempSrcMaps)
            srcmaps = options.tempSrcMaps
    except:
        pass

    expcube = pars['expcube']    
    expmap = pars['bexpmap']
    irfs = pars['irfs']
    phased_expmap = None

    # Make the Analysis objects
    obs = BinnedObs(srcMaps=srcmaps, expCube=expcube, binnedExpMap=expmap,
                    irfs=irfs, phased_expmap=phased_expmap)
    analysis = BinnedAnalysis(obs, pars['srcmdl'], pars['optimizer'])

    (source,spec,spatialMap) = ReplaceAnalysisComponent(analysis,obs,options,source_name,del_names)
    return (obs,analysis,source,spec,spatialMap)



if __name__=='__main__':

    pass

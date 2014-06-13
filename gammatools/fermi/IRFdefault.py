# @file IRFdefault.py
# @brief define default setup options
#
# $Header: /nfs/slac/g/glast/ground/cvs/ScienceTools-scons/irfs/handoff_response/python/IRFdefault.py,v 1.14 2014/06/04 19:34:32 jchiang Exp $
import os
p = os.getcwd().split(os.sep)
print 'loading setup from %s ' % os.getcwd()

#extract class name from file path
className = p[len(p)-1]
print 'eventClass is %s' % className

    
class Prune(object):
    """
    information for the prune step
    """
    fileName = 'goodEvent.root' # file to create
    branchNames ="""
        EvtRun    EvtEnergyCorr 
        McEnergy  McXDir  McYDir  McZDir   
        McXDirErr   McYDirErr  McZDirErr   
        McTkr1DirErr  McDirErr  
        GltWord   OBFGamStatus
        Tkr1FirstLayer VtxAngle 
        CTBVTX CTBCORE  CTBSummedCTBGAM  CTBBest*
        """.split()  # specify branch names to include
    cuts='(GltWord&10)>0 && (GltWord!=35) && (OBFGamStatus>0) && CTBBestEnergyProb>0.1 && CTBCORE>0.1'

class Data(object):
    files=['../all/'+Prune.fileName] # use pruned file in event class all by default
    # these correspond to the three runs at SLAC and UW
    generate_area = 6.0
    generated=[60e6,150e6, 97e6]
    logemin = [1.25, 1.25, 1.0]
    logemax = [5.75, 4.25, 2.75]
    

# define additional cuts based on event class: these are exclusive, add up to class 'all'
additionalCuts = {
    'all': '',
    'classA': '&&CTBSummedCTBGAM>=0.5 && CTBCORE>=0.8',
    'classB': '&&CTBSummedCTBGAM>=0.5 && CTBCORE>=0.5 && CTBCORE<0.8',
    'classC': '&&CTBSummedCTBGAM>=0.5 && CTBCORE<0.5',
    'classD': '&&CTBSummedCTBGAM>=0.1 && CTBSummedCTBGAM<0.5',
    'classF': '&&CTBSummedCTBGAM<0.1',
    'standard': '&&CTBSummedCTBGAM>0.5'
    }
if className in additionalCuts.keys():
    Prune.cuts += additionalCuts[className]
else:
    pass
    #print 'Event class "%s" not recognized: using cuts for class all' %className
  
try:
    import numarray as num
except ImportError:
    import numpy as num

#define default binning as attributes of object Bins
class Bins(object):

    @staticmethod
    def set_energy_range(logemin=None,logemax=None):

        if logemin is None: logemin = Bins.logemin
        if logemax is None: logemax = Bins.logemax
        
        Bins.energy_bins = int((logemax-logemin)/Bins.logedelta)
        Bins.energy_bin_edges = (num.arange(Bins.energy_bins+1)*
                                 Bins.logedelta+logemin).tolist()

        print 'Energy Bins ', Bins.energy_bin_edges
    
    logemin = 1.25
    logemax = 5.75
    logedelta = 0.25 #4 per decade
        
    deltaCostheta = 0.1
    cthmin = 0.2;
    angle_bins = int((1-cthmin)/deltaCostheta)
    
    angle_bin_edges = num.arange(angle_bins+1)*deltaCostheta+cthmin

    # no overlap with adjacent bins for energy dispersion fits
    edisp_energy_overlap = 0  
    edisp_angle_overlap = 0

    # no overlap with adjacent bins for psf fits
    psf_energy_overlap = 0  
    psf_angle_overlap = 0

Bins.set_energy_range()

    
    
class EffectiveAreaBins(Bins):
    """
    subclass of Bins for finer binning of effective area
    """
    logemin = Bins.logemin
    logemax = Bins.logemax
    ebreak = 4.25
    ebinfactor = 4
    ebinhigh = 2
    logedelta = Bins.logedelta 
    # generate list with different 
    anglebinfactor=4 # bins multiplier
    angle_bin_edges = num.arange(Bins.angle_bins*anglebinfactor+1)*Bins.deltaCostheta/anglebinfactor+Bins.cthmin

    @staticmethod
    def set_energy_range(logemin=None,logemax=None):

        if logemin is None: logemin = EffectiveAreaBins.logemin
        if logemax is None: logemax = EffectiveAreaBins.logemax
        
        EffectiveAreaBins.energy_bin_edges = []
        x = logemin
        factor = EffectiveAreaBins.ebinfactor
        while x<logemax+0.01:
            if x>= EffectiveAreaBins.ebreak: factor = EffectiveAreaBins.ebinhigh
            EffectiveAreaBins.energy_bin_edges.append(x)
            x += EffectiveAreaBins.logedelta/factor

        print 'Energy Bins ', EffectiveAreaBins.energy_bin_edges

EffectiveAreaBins.set_energy_range()
            
class PSF(object):
    pass

class Edisp(object):
    pass

# the log file - to cout if null
logFile = 'log.txt'



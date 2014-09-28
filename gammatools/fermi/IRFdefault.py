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

    @classmethod
    def set_energy_bins(cls,logemin=None,logemax=None,logedelta=None):

        if logemin is None: logemin = cls.logemin
        if logemax is None: logemax = cls.logemax
        if logedelta is None: logedelta = cls.logedelta
        
        cls.energy_bins = int((logemax-logemin)/logedelta)
        cls.energy_bin_edges = (num.arange(cls.energy_bins+1)*logedelta+logemin).tolist()

        print 'Energy Bins ', cls.energy_bin_edges

    @classmethod
    def set_angle_bins(cls,cthmin=None,cthdelta=None):
        if cthmin is None: cthmin = cls.cthmin
        if cthdelta is None: cthdelta = cls.cthdelta

        cls.angle_bins = int((1.0-cthmin)/cthdelta)    
        cls.angle_bin_edges = num.arange(cls.angle_bins+1)*cthdelta+cthmin
        
        
    logemin = 1.25
    logemax = 5.75
    logedelta = 0.25 #4 per decade

    cthmin = 0.2
    cthdelta = 0.1

    # no overlap with adjacent bins for energy dispersion fits
    edisp_energy_overlap = 0  
    edisp_angle_overlap = 0

    # no overlap with adjacent bins for psf fits
    psf_energy_overlap = 0  
    psf_angle_overlap = 0

Bins.set_energy_bins()
Bins.set_angle_bins()


class FisheyeBins(Bins):
    logemin = Bins.logemin
    logemax = Bins.logemax
    logedelta = 0.25

    cthmin = 0.2
    cthdelta = 0.1

FisheyeBins.set_energy_bins()
FisheyeBins.set_angle_bins()
    
    
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
    angle_bin_edges = num.arange(Bins.angle_bins*anglebinfactor+1)*Bins.cthdelta/anglebinfactor+Bins.cthmin

    @classmethod
    def set_energy_bins(cls,logemin=None,logemax=None):

        if logemin is None: logemin = cls.logemin
        if logemax is None: logemax = cls.logemax
        
        cls.energy_bin_edges = []
        x = logemin
        factor = cls.ebinfactor
        while x<logemax+0.01:
            if x>= cls.ebreak: factor = cls.ebinhigh
            cls.energy_bin_edges.append(x)
            x += cls.logedelta/factor

        print 'Energy Bins ', cls.energy_bin_edges

EffectiveAreaBins.set_energy_bins()
            
class PSF(object):
    pass

class Edisp(object):
    Version=2
    #Scale Parameters
    front_pars = [0.0195, 0.1831, -0.2163, -0.4434, 0.0510, 0.6621]
    back_pars = [0.0167, 0.1623, -0.1945, -0.4592, 0.0694, 0.5899]
    #Fit Parameters key=name, value=[pinit,pmin,max]
#    fit_pars = {"f":[0.96,0.5,1.],"s1":[1.5,0.1,5.], "k1":[1.2,0.1,5.], "bias":[0.,-3.,3.], "s2":[2.5,0.8,8],  "k2":[.8,.01,8],"bias2":[0.,-3.,3.], "pindex1":[1.8,0.01,2],"pindex2":[1.8,0.01,2]}
    fit_pars = {"f":[0.8,0.3,1.0],"s1":[1.5,0.1,5.0], "k1":[1.0,0.1,3.0], "bias":[0.0,-3,3], "bias2":[0.0,-3,3], "s2":[4.0,1.2,10], "k2":[1.0,0.1,3.0],"pindex1":[2.0,0.1,5],"pindex2":[2.0,0.1,5]}
    
# the log file - to cout if null
logFile = 'log.txt'



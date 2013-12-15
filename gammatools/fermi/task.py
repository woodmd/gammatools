import yaml
import os
import shutil
import copy
from tempfile import mkdtemp
#from uw.like.pointspec import SpectralAnalysis,DataSpecification
#from uw.like.pointspec_helpers import get_default_diffuse, PointSource, FermiCatalog, get_diffuse_source

from GtApp import GtApp
#from uw.like.roi_catalogs import SourceCatalog, Catalog2FGL, Catalog3Y
#from BinnedAnalysis import BinnedObs,BinnedAnalysis
#from UnbinnedAnalysis import UnbinnedObs, UnbinnedAnalysis
#from pyLikelihood import ParameterVector

from gammatools.fermi.catalog import Catalog, CatalogSource
from gammatools.core.util import Configurable


class TaskDispatcher(Configurable):

    default_config = { 'queue'   : 'xlong' }
    
    def __init__(self,config=None,**kwargs):
        super(TaskDispatcher,self).__init__()    
        self.configure(TaskDispatcher.default_config,config,**kwargs)

    
        
class Task(Configurable):

    default_config = { 'scratchdir'   : '/scratch',
                       'verbose'      : 0,
                       'overwrite'    : False,
                       'stage_inputs' : False }
    
    def __init__(self,config=None,**kwargs):
        super(Task,self).__init__()    
        self.configure(Task.default_config,config,**kwargs)

        self._input_files = []
        self._output_files = []

        if self.config('scratchdir') is None: self._scratchdir = os.getcwd()
        else: self._scratchdir = self.config('scratchdir')

        self._workdir=mkdtemp(prefix=os.environ['USER'] + '.',
                              dir=self._scratchdir)

    def register_output_file(self,outfile):
        self._output_files.append(outfile)
        
    def prepare(self):

        # stage input files
        self._cwd = os.getcwd()
        os.chdir(self._workdir)

    def run(self):

        if os.path.isfile(self._output_files[0]):
            print 'Output file exists: ', self._output_files[0]
            return
        
        self.prepare()

        self.run_task()
        
        self.cleanup()
        
    def cleanup(self):

        for f in self._output_files:
            
            if self.config('verbose'):
                print 'cp %s %s'%(os.path.basename(f),f)                
            os.system('cp %s %s'%(os.path.basename(f),f))

        os.chdir(self._cwd)
            
        if os.path.exists(self._workdir):
            shutil.rmtree(self._workdir)

    def __del__(self):
        if os.path.exists(self._workdir):
            if self.config('verbose'):
                print 'Deleting working directory ', self._workdir
            shutil.rmtree(self._workdir)

class LTSumTask(Task):

    default_config = { 'infile1' : None }

    def __init__(self,outfile,config=None,**kwargs):
        super(LTSumTask,self).__init__(config,**kwargs)
        self.configure(LTSumTask.default_config,config,**kwargs)

        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtltsum')
    
    def run_task(self):

        config = self.config()
        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**config)
            
class LTCubeTask(Task):

    default_config = { 'dcostheta' : 0.025,
                       'binsz' : 1.0,
                       'evfile' : None,
                       'scfile' : None,
                       'zmax' : 105.0 }

    def __init__(self,outfile,config=None,**kwargs):
        super(LTCubeTask,self).__init__(config,**kwargs)
        self.configure(LTCubeTask.default_config,config,**kwargs)

        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtltcube')
    
    def run_task(self):

        config = self.config()
        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**config)
    

class SrcModelTask(Task):

    default_config = { 'srcmaps'    : None,
                       'srcmdl'     : None,
                       'expcube'    : None,
                       'bexpmap'    : None,
                       'srcmdl'     : None,
                       'chatter'    : 2,
                       'irfs'       : 'P7REP_CLEAN_V15',
                       'outtype'    : 'ccube' }

    
    def __init__(self,outfile,config=None,**kwargs):
        super(SrcModelTask,self).__init__(config,**kwargs)
        self.configure(SrcModelTask.default_config,config,**kwargs)

        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtmodel')
    
    def run_task(self):

        config = self.config()
        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**config)
    
class SrcMapTask(Task):

    default_config = { 'scfile'  : None,
                       'expcube'  : None,
                       'bexpmap' : None,
                       'cmap'    : None,
                       'srcmdl'  : None,
                       'chatter' : 2,
                       'irfs'     : 'P7REP_CLEAN_V15',
                       'resample' : 'yes',
                       'rfactor'  : 2,
                       'minbinsz' : 0.1 }
    
    def __init__(self,outfile,config=None,**kwargs):
        super(SrcMapTask,self).__init__(config,**kwargs)
        self.configure(SrcMapTask.default_config,config,'gtsrcmaps',**kwargs)
        
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtsrcmaps','Likelihood')

    def run_task(self):

        config = self.config()
        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,emapbnds='no',**config)


        
class BExpTask(Task):

    default_config = { 'nxpix' : 360.,
                       'nypix' : 180.,
                       'allsky' : True,
                       'xref' : 0.0,
                       'yref' : 0.0,
                       'emin' : 1000.0,
                       'emax' : 100000.0,
                       'chatter' : 2,
                       'proj' : 'CAR',
                       'enumbins' : 16,
                       'infile'   : None,
                       'irfs'     : 'P7REP_CLEAN_V15',
                       'cmap' : 'none',
                       'coordsys' : 'CEL',
                       'ebinalg'  : 'LOG',
                       'binsz' : 1.0 }
    
    def __init__(self,outfile,config=None,**kwargs):
        super(BExpTask,self).__init__(config,**kwargs)
        self.configure(BExpTask.default_config,config,'gtexpcube',**kwargs)

        if self.config('allsky'):
            self.set_config('nxpix',360)
            self.set_config('nypix',180)
            self.set_config('xref',0.0)
            self.set_config('yref',0.0)
            self.set_config('binsz',1.0)
            self.set_config('proj','CAR')
             
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtexpcube2','Likelihood')
        

    def run_task(self):

        config = copy.deepcopy(self.config())
        del(config['allsky'])        
        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**config)

        
    
class BinnerTask(Task):

    default_config = { 'npix' : 140,
                       'xref' : 0.0,
                       'yref' : 0.0,
                       'emin' : 1000.0,
                       'emax' : 100000.0,
                       'scfile' : None,
                       'chatter' : 2,
                       'proj' : 'AIT',
                       'enumbins' : 16,
                       'algorithm' : 'ccube',
                       'binsz' : 0.1 }
    
    def __init__(self,infile,outfile,config=None,**kwargs):
        super(BinnerTask,self).__init__(config,**kwargs)
        self.configure(BinnerTask.default_config,config,'gtbin',**kwargs)
        
        self._infile = os.path.abspath(infile)
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtbin=GtApp('gtbin','evtbin')
        

    def run_task(self):

        config = self.config()

        outfile = os.path.basename(self._output_files[0])
        
        self._gtbin.run(algorithm=config['algorithm'],
                        nxpix=config['npix'],
                        nypix=config['npix'],
                        binsz=config['binsz'],
                        evfile=self._infile,
                        outfile=outfile,
                        scfile=config['scfile'],
                        xref=config['xref'],
                        yref=config['yref'],
                        axisrot=0,
                        proj=config['proj'],
                        ebinalg='LOG',
                        emin=config['emin'],
                        emax=config['emax'],
                        enumbins=config['enumbins'],
                        coordsys='CEL',
                        chatter=config['chatter'])

    
class SelectorTask(Task):

    default_config = { 'ra' : 0.0,
                       'dec' : 0.0,
                       'radius' : 10.0,
                       'tmin' : 239557414,
                       'tmax' : 365787814,
                       'zmax' : 100.,
                       'emin' : 1000.,
                       'emax' : 100000.,
                       'chatter' : 2,
                       'evclsmin' : 'INDEF',
                       'evclass' : 3,
                       'convtype' : -1 }               
    
    def __init__(self,infile,outfile,config=None,**kwargs):
        super(SelectorTask,self).__init__(config,**kwargs)        
        self.configure(SelectorTask.default_config,config,'gtselect',**kwargs)

        self._infile = os.path.abspath(infile)
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtselect=GtApp('gtselect','dataSubselector')


    def run_task(self):
        
        config = self.config()

        outfile = os.path.basename(self._output_files[0])
        
        self._gtselect.run(infile=self._infile,
                           outfile=outfile,
                           ra=config['ra'], dec=config['dec'], 
                           rad=config['radius'],
                           tmin=config['tmin'], tmax=config['tmax'],
                           emin=config['emin'], emax=config['emax'],
                           zmax=config['zmax'], chatter=config['chatter'],
                           evclass=config['evclass'],   # Only for Pass7
                           evclsmin=config['evclsmin'],
                           convtype=config['convtype']) # Only for Pass6



class MkTimeTask(Task):

    default_config = { 'roicut' : 'no',
                       'filter' : 'IN_SAA!=T&&DATA_QUAL==1&&LAT_CONFIG==1&&ABS(ROCK_ANGLE)<52',
                       'evfile' : None,
                       'scfile' : None }               
    
    def __init__(self,infile,outfile,config=None,**kwargs):
        super(MkTimeTask,self).__init__(config,**kwargs)        
        self.configure(MkTimeTask.default_config,config,'gtmktime',**kwargs)

        self._infile = os.path.abspath(infile)
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp=GtApp('gtmktime','dataSubselector')


    def run_task(self):
        
        config = self.config()

        outfile = os.path.basename(self._output_files[0])
        
        self._gtapp.run(evfile=self._infile,
                        outfile=outfile,
                        filter=config['filter'],
                        roicut=config['roicut'],
                        scfile=config['scfile'])

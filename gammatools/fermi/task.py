import yaml
import os
import shutil
import copy
from tempfile import mkdtemp
import re
import glob
from GtApp import GtApp
#from uw.like.roi_catalogs import SourceCatalog, Catalog2FGL, Catalog3Y
#from BinnedAnalysis import BinnedObs,BinnedAnalysis
#from UnbinnedAnalysis import UnbinnedObs, UnbinnedAnalysis
#from pyLikelihood import ParameterVector

from gammatools.fermi.catalog import Catalog, CatalogSource
from gammatools.core.config import Configurable
from gammatools.core.util import extract_dict_by_keys

class TaskDispatcher(Configurable):

    default_config = { 'queue'   : 'xlong' }
    
    def __init__(self,config=None,**kwargs):
        super(TaskDispatcher,self).__init__(config,**kwargs)    
        
class Task(Configurable):

    default_config = {
        'scratchdir'   : ('/scratch','Set the path under which temporary '
                          'working directories will be created.'),
        'workdir'      : (None,'Set the working directory.'),
        'verbose'      : 1,
        'overwrite'    : (True,'Overwrite the output file if it exists.'),
        'stage_inputs' : (False,'Copy input files to temporary working directory.') }
    
    def __init__(self,config=None,**kwargs):       
        super(Task,self).__init__(config,**kwargs)

        self._input_files = []
        self._output_files = []
        
        if self.config['scratchdir'] is None:
            self._scratchdir = os.getcwd()
        else:
            self._scratchdir = self.config['scratchdir']

        
        if self.config['workdir'] is None:
            self._savedata=False
            self._workdir=mkdtemp(prefix=os.environ['USER'] + '.',
                                  dir=self._scratchdir)

            if self.config['verbose']:
                print 'Created workdir: ', self._workdir
            
        else:
            self._savedata=False
            self._workdir= self.config['workdir']

            if self.config['verbose']:
                print 'Using workdir: ', self._workdir
                            
    def register_output_file(self,outfile):
        self._output_files.append(outfile)
        
    def prepare(self):

        # stage input files
        self._cwd = os.getcwd()
        os.chdir(self._workdir)

    def run(self):

        import pprint
        pprint.pprint(self.config)
        
        if len(self._output_files) and \
                os.path.isfile(self._output_files[0]) and \
                not self.config['overwrite']:
            print 'Output file exists: ', self._output_files[0]
            return
        
        self.prepare()

        self.run_task()
        
        self.cleanup()
        
    def cleanup(self):

        for f in self._output_files:
            
            if self.config['verbose']:
                print 'cp %s %s'%(os.path.basename(f),f)                
            os.system('cp %s %s'%(os.path.basename(f),f))

        os.chdir(self._cwd)
            
        if not self._savedata and os.path.exists(self._workdir):
            shutil.rmtree(self._workdir)

    def __del__(self):
        if not self._savedata and os.path.exists(self._workdir):
            if self.config['verbose']:
                print 'Deleting working directory ', self._workdir
            shutil.rmtree(self._workdir)

class LTSumTask(Task):

    default_config = { 'infile1' : None }

    def __init__(self,outfile,config=None,**kwargs):
        super(LTSumTask,self).__init__(config,**kwargs)

        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtltsum')
    
    def run_task(self):

        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**self.config)
            
class LTCubeTask(Task):

    default_config = { 'dcostheta' : 0.025,
                       'binsz' : 1.0,
                       'evfile' : None,
                       'scfile' : (None, 'spacecraft file'),
                       'tmin'   : 0.0,
                       'tmax'   : 0.0,
                       'zmax' : (100.0,'Set the maximum zenith angle.') }

    def __init__(self,outfile,config=None,opts=None,**kwargs):
        super(LTCubeTask,self).__init__(config,opts=opts,**kwargs)

        self._config['scfile'] = os.path.abspath(self._config['scfile'])
        
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtltcube')
    
    def run_task(self):

        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**self.config)
    

class SrcModelTask(Task):

    default_config = {
        'srcmaps'    : None,
        'srcmdl'     : None,
        'expcube'    : None,
        'bexpmap'    : None,
        'srcmdl'     : None,
        'chatter'    : 2,
        'irfs'       : None,
        'outtype'    : 'ccube' }

    
    def __init__(self,outfile,config=None,opts=None,**kwargs):
        super(SrcModelTask,self).__init__(config,opts=opts,**kwargs)

        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtmodel')
    
    def run_task(self):

        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**self.config)
    
class SrcMapTask(Task):

    default_config = { 'scfile'  : None,
                       'expcube'  : None,
                       'bexpmap' : None,
                       'cmap'    : None,
                       'srcmdl'  : None,
                       'chatter' : 2,
                       'irfs'     : None,
                       'resample' : 'yes',
                       'rfactor'  : 2,
                       'minbinsz' : 0.1 }

    def __init__(self,outfile,config=None,**kwargs):
        super(SrcMapTask,self).__init__()
        self.update_default_config(SrcMapTask)
        self.configure(config,subsection='gtsrcmaps',**kwargs)
        
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtsrcmaps','Likelihood')

    def run_task(self):

        config = self.config
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
                       'irfs'     : None,
                       'cmap' : 'none',
                       'coordsys' : 'CEL',
                       'ebinalg'  : 'LOG',
                       'binsz' : 1.0 }
    
    def __init__(self,outfile,config=None,**kwargs):
        super(BExpTask,self).__init__()
        self.update_default_config(BExpTask)
        self.configure(config,subsection='gtexpcube',**kwargs)

        if self.config['allsky']:
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

        config = copy.deepcopy(self.config)
        del(config['allsky'])        
        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(outfile=outfile,**config)

        
    
class BinTask(Task):

    default_config = { 'nxpix' : 140,
                       'nypix' : None,
                       'xref' : 0.0,
                       'yref' : 0.0,
                       'emin' : 1000.0,
                       'emax' : 100000.0,
                       'scfile' : None,
                       'chatter' : 2,
                       'proj' : 'AIT',
                       'hpx_order' : 3,
                       'enumbins' : 16,
                       'algorithm' : 'ccube',
                       'binsz' : 0.1,
                       'coordsys' : 'CEL'}
    
    def __init__(self,infile,outfile,config=None,opts=None,**kwargs):
        super(BinTask,self).__init__()
        self.configure(config,opts=opts,subsection='gtbin',**kwargs)
        
        self._infile = os.path.abspath(infile)
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)

        if re.search('^(?!\@)(.+)(\.txt|\.lst)$',self._infile):
            self._infile = '@'+self._infile
        
        self._gtbin=GtApp('gtbin','evtbin')
        

    def run_task(self):

        config = copy.deepcopy(self.config)

        outfile = os.path.basename(self._output_files[0])

        if config['nypix'] is None:
            config['nypix'] = config['nxpix']
        
        self._gtbin.run(algorithm=config['algorithm'],
                        nxpix=config['nxpix'],
                        nypix=config['nypix'],
                        binsz=config['binsz'],
                        hpx_order=config['hpx_order'],
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
                        coordsys=config['coordsys'],
                        chatter=config['chatter'])

    
class SelectorTask(Task):

    default_config = { 'ra' : 0.0,
                       'dec' : 0.0,
                       'radius' : 180.0,
                       'tmin' : 0.0,
                       'tmax' : 0.0,
                       'zmax' : 100.,
                       'emin' : 10.,
                       'emax' : 1000000.,
                       'chatter' : 2,
                       'evclsmin' : 'INDEF',
                       'evclass' : 'INDEF',
                       'evtype'  : 'INDEF',
                       'convtype' : -1 }               
    
    def __init__(self,infile,outfile,config=None,opts=None,**kwargs):
        super(SelectorTask,self).__init__()
        self.configure(config,opts=opts,**kwargs)
        
        self._infile = os.path.abspath(infile)
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)

        if re.search('^(?!\@)(.+)\.txt$',self._infile):
            self._infile = '@'+self._infile
            
        self._gtselect=GtApp('gtselect','dataSubselector')


    def run_task(self):
        
        config = self.config

        outfile = os.path.basename(self._output_files[0])


#        print self._infile
#        os.system('cat ' + self._infile[1:])
        
        self._gtselect.run(infile=self._infile,
                           outfile=outfile,
                           ra=config['ra'], dec=config['dec'], 
                           rad=config['radius'],
                           tmin=config['tmin'], tmax=config['tmax'],
                           emin=config['emin'], emax=config['emax'],
                           zmax=config['zmax'], chatter=config['chatter'],
                           evclass=config['evclass'],   # Only for Pass7
                           evtype=config['evtype'],
                           convtype=config['convtype']) # Only for Pass6



class MkTimeTask(Task):

    default_config = { 'roicut' : 'no',
                       'filter' : 'IN_SAA!=T&&DATA_QUAL==1&&LAT_CONFIG==1&&ABS(ROCK_ANGLE)<52',
                       'evfile' : None,
                       'scfile' : None }               
    
    def __init__(self,infile,outfile,config=None,**kwargs):
        super(MkTimeTask,self).__init__()
        self.update_default_config(MkTimeTask)
        self.configure(config,subsection='gtmktime',**kwargs)

        self._infile = os.path.abspath(infile)
        self._outfile = os.path.abspath(outfile)
        self.register_output_file(self._outfile)
        
        self._gtapp=GtApp('gtmktime','dataSubselector')


    def run_task(self):
        
        config = self.config

        outfile = os.path.basename(self._output_files[0])
        
        self._gtapp.run(evfile=self._infile,
                        outfile=outfile,
                        filter=config['filter'],
                        roicut=config['roicut'],
                        scfile=config['scfile'])

class ObsSimTask(Task):

    default_config = {
        'infile'     : None, 
        'srclist'    : None,
        'scfile'     : None,
        'ra'         : None,
        'dec'        : None,
        'radius'     : None,
        'emin'       : None,
        'emax'       : None,
        'irfs'       : None,
        'simtime'    : None,
        'evroot'     : 'sim',
        'use_ac'     : False,
        'seed'       : 1,
        'rockangle'  : 'INDEF'
        }

    
    def __init__(self,config=None,opts=None,**kwargs):
        super(ObsSimTask,self).__init__(config,opts=opts,**kwargs)

#        self._outfile = os.path.abspath(outfile)
#        self.register_output_file(self._outfile)
        
        self._gtapp = GtApp('gtobssim')
    
    def run_task(self):

        config = extract_dict_by_keys(self.config,
                                      ObsSimTask.default_config.keys())
#        outfile = os.path.basename(self._output_files[0])        
        self._gtapp.run(**config)


    def cleanup(self):

        # Copy files
        outfiles = glob.glob('sim*fits')

        for f in outfiles:

            print f
            
#            if os.path.dirname(f) != self._cwd:            
#                os.system('cp %s %s'%(f,self._cwd))

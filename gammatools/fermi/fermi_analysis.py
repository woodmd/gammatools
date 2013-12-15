import yaml
import os
import sys
import shutil
import copy
from tempfile import mkdtemp
from skymaps import SkyDir
from uw.like.pointspec import SpectralAnalysis,DataSpecification
from uw.like.pointspec_helpers import get_default_diffuse, PointSource, FermiCatalog, get_diffuse_source

from GtApp import GtApp
from uw.like.roi_catalogs import SourceCatalog, Catalog2FGL, Catalog3Y

from BinnedAnalysis import BinnedObs,BinnedAnalysis
from UnbinnedAnalysis import UnbinnedObs, UnbinnedAnalysis
from pyLikelihood import ParameterVector

from catalog import Catalog, CatalogSource

from util import Configurable
from task import *
from Composite2 import *

class BinnedAnalysisTool(Configurable):

    default_config = { 'convtype'   : -1,
                       'binsperdec' : 4,
                       'savedir'    : None,
                       'scratchdir' : None,
                       'target'     : None,
                       'evfile'     : None,
                       'scfile'     : None,
                       'srcmdl'     : None,
                       'ltcube'     : None,
                       'galdiff'    : None,
                       'isodiff'    : None,
                       'gtbin'      : None,
                       'gtexpcube'  : None,
                       'gtselect'   : None,
                       'gtsrcmap'   : None,
                       'catalog'    : '2FGL',
                       'optimizer'  : 'MINUIT',
                       'irfs'       : 'P7REP_CLEAN_V10' }
    
    def __init__(self,src,target_name,config,**kwargs):
        super(BinnedAnalysisTool,self).__init__()  
        self.configure(dict(BinnedAnalysisTool.default_config.items() +
                            SelectorTask.default_config.items() +
                            BinnerTask.default_config.items() +
                            SrcMapTask.default_config.items() +
                            BExpTask.default_config.items()),
                       config,**kwargs)

        savedir = self.config('savedir')
        
        self.ft1file = os.path.join(savedir,"%s_ft1.fits"%target_name)
        self.binfile = os.path.join(savedir,"%s_binfile.fits"%target_name)
        self.ccubefile = os.path.join(savedir,"%s_ccube.fits"%target_name)
        self.bexpfile = os.path.join(savedir,"%s_bexp.fits"%target_name)
        self.srcmdl = os.path.join(savedir,"%s_srcmdl_fit.xml"%target_name)
        
        self.srcmapfile = os.path.join(savedir,"%s_srcmap.fits"%target_name)
        self.srcmdlfile = os.path.join(savedir,"%s_srcmdl.fits"%target_name)
        
        self.skydir = SkyDir(src.ra(),src.dec())
        self.src = src
        
    def setup_gtlike(self):

        config = self.config()
        
        sel_task = SelectorTask(config['evfile'],self.ft1file,
                                ra=self.src.ra(),dec=self.src.dec(),
                                config=config)
        sel_task.run()

        bin_task = BinnerTask(self.ft1file,self.ccubefile,config,
                              xref=self.src.ra(),yref=self.src.dec())

        bin_task.run()

        
        bexp_task = BExpTask(self.bexpfile,infile=config['ltcube'],
                             config=config)
            
        bexp_task.run()

        srcmap_task = SrcMapTask(self.srcmapfile,bexpmap=self.bexpfile,
                                 srcmdl=config['srcmdl'],
                                 cmap=self.ccubefile,
                                 expcube=config['ltcube'],
                                 config=config)

        srcmap_task.run()

        self.obs = BinnedObs(srcMaps=self.srcmapfile,
                             expCube=config['ltcube'],
                             binnedExpMap=self.bexpfile,
                             irfs=config['irfs'])
        
        self.like = BinnedAnalysis(binnedData=self.obs,
                                   srcModel=config['srcmdl'],
                                   optimizer=config['optimizer'])

    def make_srcmodel(self,srcmdl=None):

        if srcmdl is None: srcmdl = self.srcmdl
        
        srcmdl_task = SrcModelTask(self.srcmdlfile,
                                   srcmaps=self.srcmapfile,
                                   bexpmap=self.bexpfile,
                                   srcmdl=srcmdl,
                                   expcube=self.config('ltcube'),
                                   config=self.config())
            
        srcmdl_task.run()

    def write_model(self):
        self.like.writeXml(self.srcmdl)


class AnalysisManager(Configurable):

    default_config = { 'convtype'   : -1,
                       'binsperdec' : 4,
                       'savedir'    : None,
                       'scratchdir' : None,
                       'target'     : None,
                       'evfile'     : None,
                       'scfile'     : None,
                       'ltcube'     : None,
                       'galdiff'    : None,
                       'isodiff'    : None,
                       'event_types': None,
                       'gtbin'      : None,
                       'catalog'    : '2FGL',
                       'optimizer'  : 'MINUIT',
                       'irfs'       : 'P7REP_CLEAN_V10' }
    
    def __init__(self,config,**kwargs):
        super(AnalysisManager,self).__init__()  
        self.configure(dict(AnalysisManager.default_config.items() +
                            SelectorTask.default_config.items()),
                       config,**kwargs)

    def setup_roi(self,**kwargs):

        target_name = self.config('target')
        
        cat = Catalog()
        src = CatalogSource(cat.get_source_by_name(target_name))

        if self.config('savedir') is None:
            self.set_config('savedir',target_name)

        if not os.path.exists(self.config('savedir')):
            os.makedirs(self.config('savedir'))
        
        config = self.config()

        self.savestate = os.path.join(config['savedir'],
                                    "%s_savestate.P"%target_name)
        
        self.ft1file = os.path.join(config['savedir'],
                                    "%s_ft1.fits"%target_name)
        self.ltcube = os.path.join(config['savedir'],
                                    "%s_ltcube.fits"%target_name)
        self.binfile = os.path.join(config['savedir'],
                                    "%s_binfile.fits"%target_name)
        self.srcmdl = os.path.join(config['savedir'],
                                   "%s_srcmdl.xml"%target_name)
        
        self.srcmdl_fit = os.path.join(config['savedir'],
                                       "%s_srcmdl_fit.xml"%target_name)
        
        self.evfile = config['evfile']
        self.skydir = SkyDir(src.ra(),src.dec())

        lt_task = LTSumTask(self.ltcube,infile1=config['ltcube'],
                            config=config)

        lt_task.run()

        self.set_config('ltcube',self.ltcube)
        
        sel_task = SelectorTask(self.evfile,self.ft1file,
                                ra=src.ra(),dec=src.dec(),
                                config=config)
        sel_task.run()

        self.setup_pointlike()

        self.components = []
        self.comp_like = Composite2()
        
        for i, t in enumerate(self.config('event_types')):

            print 'Setting up binned analysis ', i
            
            analysis = BinnedAnalysisTool(src,target_name + '_%02i'%(i),
                                          config,
                                          evfile=self.ft1file,
                                          srcmdl=self.srcmdl,
                                          evclass=t['evclass'],
                                          convtype=t['convtype'],
                                          irfs=t['irfs'])

            analysis.setup_gtlike()
            
            self.components.append(analysis)
            self.comp_like.addComponent(analysis.like)

#        for i, p in self.tied_pars.iteritems():
#            print 'Tying parameters ', i, p            
#            self.comp_like.tieParameters(p)
            
        for i, p in enumerate(self.components[0].like.params()):

            print i, p.srcName, p.getName()

            tied_params = []            
            for c in self.components:
                tied_params.append([c.like,p.srcName,p.getName()])
            self.comp_like.tieParameters(tied_params)
                
#        self.tied_pars = {}
#        for x in self.components:
        
#            for s in x.like.sourceNames():
#                p = x.like.normPar(s)                
#                pidx = x.like.par_index(s,p.getName())

#                if not pidx in self.tied_pars:
#                    self.tied_pars[pidx] = []

#                self.tied_pars[pidx].append([x.like,s,p.getName()])
                    
#                print s, p.getName()        
#                self.norm_pars.append([x.like,s,p.getName()])
#            self.norm_pars.append([self.analysis1.like,src,p.getName()])

    def run_gtlike(self):
        
        config = self.config()        
        print 'Fitting model'
        self.comp_like.fit(verbosity=2, covar=True)
        for c in self.components:
            c.write_model()
            c.make_srcmodel()

    def save(self):
        from util import save_object
        save_object(self,self.savestate)
            
    def setup_pointlike(self):

        if os.path.isfile(self.srcmdl): return
        
        config = self.config()
        
        self._ds = DataSpecification(ft1files = self.ft1file,
                                     ft2files = config['scfile'],
                                     ltcube   = config['ltcube'],
                                     binfile  = self.binfile)

        
        self._sa = SpectralAnalysis(self._ds,
                                    binsperdec = config['binsperdec'],
                                    emin       = config['emin'],
                                    emax       = config['emax'],
                                    irf        = config['irfs'],
                                    roi_dir    = self.skydir,
                                    maxROI     = config['radius'],
                                    minROI     = config['radius'],
                                    zenithcut  = config['zmax'],
                                    event_class= 0,
                                    conv_type  = config['convtype'])

        sources = []
        point_sources, diffuse_sources = [],[]

        galdiff = config['galdiff']        
        isodiff = config['isodiff']

        bkg_sources = self.get_background(galdiff,isodiff)
        sources += filter(None, bkg_sources)
        
        catalog = self.get_catalog(config['catalog'])
        catalogs = filter(None, [catalog])

        for source in sources:
            if isinstance(source,PointSource): point_sources.append(source)
            else: diffuse_sources.append(source)
        
        self._roi=self._sa.roi(roi_dir=self.skydir,
                               point_sources=point_sources,
                               diffuse_sources=diffuse_sources,
                               catalogs=catalogs,
                               fit_emin=config['emin'], 
                               fit_emax=config['emax'])

        # Create model file
        self._roi.toXML(self.srcmdl,
                        convert_extended=True,
                        expand_env_vars=True)
        
    @staticmethod
    def get_catalog(catalog=None, **kwargs):
        if catalog is None or isinstance(catalog,SourceCatalog):
            pass
        elif catalog == 'PSC3Y':
            catalog = Catalog3Y('/u/ki/kadrlica/fermi/catalogs/PSC3Y/gll_psc3yearclean_v1_assoc_v6r1p0.fit',
                                latextdir='/u/ki/kadrlica/fermi/catalogs/PSC3Y/',
                                prune_radius=0,
                                **kwargs)
        elif catalog == '2FGL':
            catalog = Catalog2FGL('/u/ki/kadrlica/fermi/catalogs/2FGL/gll_psc_v08.fit',
                               latextdir='/u/ki/kadrlica/fermi/catalogs/2FGL/Templates/',
                               prune_radius=0,
                               **kwargs)
        elif catalog == "1FGL":
            catalog = FermiCatalog('/u/ki/kadrlica/fermi/catalogs/gll_psc_v02.fit',
                               prune_radius=0,
                               **kwargs)
        else:
            raise Exception("Unknown catalog: %s"%catalog)

        return catalog

    @staticmethod
    def get_background(galdiff=None, isodiff=None, limbdiff=None):
        """ Diffuse backgrounds
        galdiff: Galactic diffuse counts cube fits file
        isodiff: Isotropic diffuse spectral text file
        limbdiff: Limb diffuse counts map fits file
        """
        backgrounds = []

        if galdiff is None: gal=None
        else:
            gfile = os.path.basename(galdiff)
            gal = get_diffuse_source('MapCubeFunction',galdiff,
                                     'PowerLaw',None,
                                     os.path.splitext(gfile)[0])
            gal.smodel.set_default_limits()
            gal.smodel.freeze('index')
        backgrounds.append(gal)

        if isodiff is None: iso=None
        else:
            ifile = os.path.basename(isodiff)
            iso = get_diffuse_source('ConstantValue',None,'FileFunction'
                                     ,isodiff,
                                     os.path.splitext(ifile)[0])
            iso.smodel.set_default_limits()
        backgrounds.append(iso)        

        if limbdiff is None: limb=None
        else:
            lfile = basename(limbdiff)
            dmodel = SpatialMap(limbdiff)
            smodel = PLSuperExpCutoff(norm=3.16e-6,index=0,
                                      cutoff=20.34,b=1,e0=200)
            limb = ExtendedSource(name=name,model=smodel,spatial_model=dmodel)
            for i in range(limb.smodel.npar): limb.smodel.freeze(i)
            backgrounds.append(limb)
        backgrounds.append(limb)

        return backgrounds

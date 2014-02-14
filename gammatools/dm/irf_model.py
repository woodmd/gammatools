import numpy as np
import yaml
from gammatools.core.histogram import *
from gammatools.core.model_fn import *
from gammatools.core.bspline import *

class IRFModel(object):
    def __init__(self,aeff,psf_r68,edisp_r68):

        self._aeff = aeff
        self._psf = psf_r68
        self._edisp = edisp_r68

        msk = (self._aeff.counts() > 0)&(self._aeff.axis().center()>4.0)
        self._aeff_fn = BSpline.fit(self._aeff.axis().center()[msk],
                              self._aeff.counts()[msk],
                              self._aeff.err()[msk],
                              np.linspace(4.0,8.0,16),4)

        self._eaxis = Axis.create(self._aeff.axis().lo_edge(),
                                  self._aeff.axis().hi_edge(),
                                  800)

        self._ematrix = Histogram2D(self._eaxis,self._eaxis)

        for i in range(self._eaxis.nbins()):
            
            ec = self._eaxis.center()[i]
            p = [1.0,ec,self._edisp.interpolate(ec)[0]]
            self._ematrix._counts[i] = GaussFn.evals(self._eaxis.center(),p)


        return

        self._cols = []

        for line in open(f):
            line = line.rstrip()
            m = re.search('#!',line)
            if m is None: continue
            else:
                self._cols = line.split()[1:]

        d = np.loadtxt(f,unpack=True)

        v = {}

        for i in range(len(d)):
            v[self._cols[i]] = d[i]

        self.__dict__.update(v)

        self.loge_edges = np.linspace(self.emin[0],self.emax[-1],
                                       len(self.emin)+1)

#        self._ebins = np.concatenate((self._emin,self._emax[-1]))
#        print self.__dict__

    def smooth_fn(self,x,fn):

        axis = self._eaxis
        
#        x0, y0 = np.meshgrid(axis.center(),axis.center(),ordering='ij')
        
#        m = self._ematrix.interpolate(np.ravel(x0),np.ravel(y0))
#        lobin = axis.valToBinBounded(self._ematrix.axis(0).edges()[0])

#        m = m.reshape((axis.nbins(),axis.nbins()))
#        m[:lobin,:] = 0

        m = self._ematrix.counts()

        cc = fn(axis.center())
        cm = np.dot(m,cc)*axis.width()
        return interpolate(axis.center(),cm,x)

    def aeff(self,x):

        return self._aeff_fn(x)

    @staticmethod
    def createCTAIRF(f):

        d = yaml.load(open(f,'r'))

        

        aeff = Histogram(d['aeff_ptsrc_rebin']['xedges']+3.0,
                         counts=d['aeff_ptsrc_rebin']['counts'],
                         var=d['aeff_ptsrc_rebin']['var'])

        aeff *= Units.m2


        psf = Histogram(d['th68']['xedges']+3.0,
                         counts=d['th68']['counts'],
                         var=0)

        edisp = Histogram(d['edisp68']['xedges']+3.0,
                         counts=np.log10(1.0+d['edisp68']['counts']),
                         var=0)

        return IRFModel(aeff,psf,edisp)


class CountsSpectrumModel(Model):

    ncall = 0

    def __init__(self,irf,spfn,fold_edisp=False):
        Model.__init__(self,spfn.param())
        self._irf = irf
        self._spfn = spfn
        self._fold_edisp = fold_edisp
        self._livetime = 100*Units.hr

    def _eval(self,x,pset):

        fn = lambda t: self._spfn(t,pset)*self._irf.aeff(t)* \
            self._livetime*np.log(10.)*10**t*Units.mev

        if self._fold_edisp: 
            return self._irf.smooth_fn(x,fn)
        else:
            return fn(x)

#        return c*exp*np.log(10.)*10**x*Units.mev

    def e2flux(self,h):

        livetime = 50*Units.hr
        exp = self._irf.aeff(h.axis().center())*self._livetime
        exp[exp<0] = 0

        msk = h.axis().center() < 4.5

        delta = 10**h.axis().edges()[1:]-10**h.axis().edges()[:-1]

        hf = copy.deepcopy(h)
        hf *= 10**(2*h.axis().center())/delta
        hf /= exp

        hf._counts[msk] = 0
        hf._var[msk] = 0

        return hf

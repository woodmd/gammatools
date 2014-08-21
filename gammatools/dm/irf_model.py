import numpy as np
import yaml
from gammatools.core.histogram import *
from gammatools.core.model_fn import *
from gammatools.core.bspline import *

class IRFModel(object):
    def __init__(self,aeff_ptsrc,aeff,bkg_ptsrc,bkg,psf_r68,edisp_r68):

        self._aeff = aeff
        self._aeff_ptsrc = aeff_ptsrc
        self._bkg = bkg
        self._bkg_ptsrc = bkg_ptsrc

        self._bkg /= self._bkg.axis().width
        self._bkg_ptsrc /= self._bkg_ptsrc.axis().width
        
        self._psf = psf_r68
        self._edisp = edisp_r68

        

        msk = ((self._aeff.counts > 0)&
               (self._aeff.axis().center>4.0))

        aeff_err = np.log10(1.0 + self._aeff.err[msk]/
                            self._aeff.counts[msk])

        self._aeff_fn = BSpline.fit(self._aeff.axis().center[msk],
                                    np.log10(self._aeff.counts[msk]),
                                    aeff_err,
                                    np.linspace(4.2,8.0,8),4)

        

        msk = ((self._aeff_ptsrc.counts > 0)&
               (self._aeff_ptsrc.axis().center>4.0))


        if np.sum(self._aeff_ptsrc.err[msk]) < 1E-4:
            err = np.ones(np.sum(msk))
        else:
            err = self._aeff_ptsrc.err[msk]


        self._aeff_ptsrc_fn = BSpline.fit(self._aeff_ptsrc.axis().center[msk],
                                          self._aeff_ptsrc.counts[msk],err,
                                          np.linspace(4.0,8.0,16),4)

        
#        plt.figure()
#        x = self._aeff_ptsrc.axis().center
#        plt.plot(x,10**self._aeff_fn(x))
#        plt.plot(x,self._aeff_ptsrc_fn(x))
#        self._aeff.plot()
#        plt.gca().set_yscale('log')
#        plt.show()


        msk = ((self._bkg_ptsrc.counts > 0)&
               (self._bkg_ptsrc.axis().center>4.0))
        bkg_err = np.log10(1.0 + 
                           self._bkg_ptsrc.err[msk]/
                           self._bkg_ptsrc.counts[msk])
        
        self._log_bkg_ptsrc_fn = \
            BSpline.fit(self._bkg_ptsrc.axis().center[msk],
                        np.log10(self._bkg_ptsrc.counts[msk]),
                        bkg_err,
                        np.linspace(4.0,8.0,8),4)

        msk = (self._bkg.counts > 0)&(self._bkg.axis().center>4.0)
        bkg_err = np.log10(1.0 + self._bkg.err[msk]/self._bkg.counts[msk])

        self._log_bkg_fn = BSpline.fit(self._bkg.axis().center[msk],
                                       np.log10(self._bkg.counts[msk]),
                                       bkg_err,
                                       np.linspace(4.0,8.0,8),4)

        
#        plt.figure()
#        self._bkg.plot()
#        x = np.linspace(4,8,100)        
#        plt.plot(x,10**self._bkg_fn(x))        
#        plt.gca().set_yscale('log')        
#        plt.show()
        
        self._eaxis = Axis.create(self._aeff_ptsrc.axis().lo_edge(),
                                  self._aeff_ptsrc.axis().hi_edge(),
                                  800)

        self._ematrix = Histogram2D(self._eaxis,self._eaxis)

        for i in range(self._eaxis.nbins):
            
            ec = self._eaxis.center[i]
            p = [1.0,ec,self._edisp.interpolate(ec)[0]]
            self._ematrix._counts[i] = GaussFn.evals(self._eaxis.center,p)


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
        
#        x0, y0 = np.meshgrid(axis.center,axis.center,ordering='ij')
        
#        m = self._ematrix.interpolate(np.ravel(x0),np.ravel(y0))
#        lobin = axis.valToBinBounded(self._ematrix.axis(0).edges()[0])

#        m = m.reshape((axis.nbins,axis.nbins))
#        m[:lobin,:] = 0

        m = self._ematrix.counts

        cc = fn(axis.center)
        cm = np.dot(m,cc)*axis.width
        return interpolate(axis.center,cm,x)

    def aeff(self,x):
        return 10**self._aeff_fn(x)

    def aeff_ptsrc(self,x):
        return self._aeff_ptsrc_fn(x)

    def bkg(self,x):
        return 10**self._log_bkg_fn(x)

    def bkg_ptsrc(self,x):
        return 10**self._log_bkg_ptsrc_fn(x)

    def fill_bkg_histogram(self,axis,livetime):

        h = Histogram(axis)
        h.fill(axis.center,
               10**self._log_bkg_fn(axis.center)*axis.width*livetime)

        return h
        
    @staticmethod
    def createCTAIRF(f):

        d = yaml.load(open(f,'r'))

        
        aeff_ptsrc = Histogram(d['aeff_ptsrc_rebin']['xedges']+3.0,
                         counts=d['aeff_ptsrc_rebin']['counts'],
                         var=d['aeff_ptsrc_rebin']['var'])

        aeff_ptsrc *= Units.m2

        if not 'aeff_diffuse' in d:
            d['aeff_diffuse'] = d['aeff_erec_ptsrc']

        aeff = Histogram(d['aeff_diffuse']['xedges']+3.0,
                         counts=d['aeff_diffuse']['counts'],
                         var=d['aeff_diffuse']['var'])

        aeff *= Units.m2

        bkg_ptsrc = Histogram(d['bkg_wcounts_rate']['xedges']+3.0,
                              counts=d['bkg_wcounts_rate']['counts'],
                              var=d['bkg_wcounts_rate']['var'])

        bkg = Histogram(d['bkg_wcounts_rate_density']['xedges']+3.0,
                        counts=d['bkg_wcounts_rate_density']['counts'],
                        var=d['bkg_wcounts_rate_density']['var'])

        bkg *= Units._deg2


        psf = Histogram(d['th68']['xedges']+3.0,
                         counts=d['th68']['counts'],
                         var=0)

        edisp = Histogram(d['edisp68']['xedges']+3.0,
                         counts=np.log10(1.0+d['edisp68']['counts']),
                         var=0)

        return IRFModel(aeff_ptsrc,aeff,bkg_ptsrc,bkg,psf,edisp)


class BkgSpectrumModel(PDF):

    def __init__(self,irf,livetime):
        Model.__init__(self)
        self._irf = irf
        self._livetime = livetime

    def _eval(self,x,pset):

        return 10**self._irf._log_bkg_ptsrc_fn(x)*self._livetime
        
class CountsSpectrumModel(PDF):

    ncall = 0

    def __init__(self,irf,spfn,livetime,fold_edisp=False):
        Model.__init__(self,spfn.param())
        self._irf = irf
        self._spfn = spfn
        self._fold_edisp = fold_edisp
        self._livetime = livetime

    def _eval(self,x,pset):

        fn = lambda t: self._spfn(t,pset)*self._irf.aeff(t)* \
            self._livetime*np.log(10.)*10**t*Units.mev

        if self._fold_edisp: 
            return self._irf.smooth_fn(x,fn)
        else:
            return fn(x)

#        return c*exp*np.log(10.)*10**x*Units.mev

    def e2flux(self,h):

        exp = self._irf.aeff(h.axis().center)*self._livetime
        exp[exp<0] = 0

        msk = h.axis().center < 4.5

        delta = 10**h.axis().edges()[1:]-10**h.axis().edges()[:-1]

        hf = copy.deepcopy(h)
        hf *= 10**(2*h.axis().center)/delta
        hf /= exp

        hf._counts[msk] = 0
        hf._var[msk] = 0

        return hf

    def e2flux2(self,h):

#        exp = self._irf.aeff(h.axis().center)*self._livetime
#        exp[exp<0] = 0

        exp_fn = lambda t: self._irf.aeff(t)*self._livetime*self._spfn(t)
        
        exp2 = self._irf.smooth_fn(h.axis().center,exp_fn)
        flux = self._spfn(h.axis().center)
        
        msk = h.axis().center < 4.5

        delta = 10**h.axis().edges()[1:]-10**h.axis().edges()[:-1]

        hf = copy.deepcopy(h)
        hf *= 10**(2*h.axis().center)/delta

        hf *= flux
        hf /= exp2

        hf._counts[msk] = 0
        hf._var[msk] = 0

        return hf

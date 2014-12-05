__author__ = 'Matthew Wood <mdwood@slac.stanford.edu>'
__date__ = '11/14/13'

import matplotlib

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, ScalarFormatter, MultipleLocator
from matplotlib.ticker import LogLocator, AutoLocator
import numpy as np
from numpy import ma
import matplotlib.cbook as cbook

class MPLUtil(object):

    scatter_kwargs = ['marker','color','edgecolor','label']    
    imshow_kwargs = ['interpolation','origin','vmin','vmax']
    pcolormesh_kwargs = ['shading','origin','vmin','vmax']
    contour_kwargs = ['levels','origin','cmap','colors']
    fill_kwargs = ['alpha','where']
    errorbar_kwargs = ['marker','markersize','color','markerfacecolor',
                       'markeredgecolor','linestyle','linewidth','label',
                       'drawstyle']
    hist_kwargs = ['color','alpha','histtype','label']

class PowerNorm(matplotlib.colors.Normalize):
    """
    Normalize a given value to the ``[0, 1]`` interval with a power-law
    scaling. This will clip any negative data points to 0.
    """
    def __init__(self, gamma, vmin=None, vmax=None, clip=True):
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = ma.getmask(result)
                val = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            resdat = result.data
            resdat -= vmin
            np.power(resdat, gamma, resdat)
            resdat /= (vmax - vmin) ** gamma
            result = np.ma.array(resdat, mask=result.mask, copy=False)
            result[(value < 0)&~result.mask] = 0
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = ma.asarray(value)
            return ma.power(value, 1. / gamma) * (vmax - vmin) + vmin
        else:
            return pow(value, 1. / gamma) * (vmax - vmin) + vmin

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        self.vmin = ma.min(A)
        if self.vmin < 0:
            self.vmin = 0
            warnings.warn("Power-law scaling on negative values is "
                          "ill-defined, clamping to 0.")

        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is None and np.size(A) > 0:
            self.vmin = ma.min(A)
            if self.vmin < 0:
                self.vmin = 0
                warnings.warn("Power-law scaling on negative values is "
                              "ill-defined, clamping to 0.")

        if self.vmax is None and np.size(A) > 0:
            self.vmax = ma.max(A)

class PowerNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, power=2., clip=False):
        self.power = power
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):

        print 'call ', type(value)

#        print 'value ', value
#        print 'clip ', clip

        return np.ma.masked_array(np.power((value-self.vmin)/self.vmax,
                                           1./self.power))

        if isinstance(value,np.ma.masked_array):
            print 'here'
            mask = value.mask
            v = np.ma.masked_array(value,copy=True)
            v[~v.mask] = np.power((v[~v.mask]-self.vmin)/self.vmax,
                                 1./self.power)

        else:
            print 'here2'
            v = np.ma.masked_array(np.power((value-self.vmin)/self.vmax,
                                            1./self.power))
#        import healpy as hp
#        v[v.mask]=10

        return v

    
class SqrtScale(mscale.ScaleBase):
    """
    Scales data using the function x^{1/2}.
    """

    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.

        thresh: The degree above which to crop the data.
        """

        exp = kwargs.pop('exp', 2.0)

        mscale.ScaleBase.__init__(self)

#        if thresh >= np.pi / 2.0:
#            raise ValueError("thresh must be less than pi/2")
        self.thresh = 0.0 #thresh
        self.exp = exp

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.
        """
        return self.SqrtTransform(self.thresh,exp=self.exp)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in ``ticker.py``.
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
        return


    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, self.thresh), max(vmax, self.thresh)

    class SqrtTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, thresh, exp):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.exp = exp

        def transform_non_affine(self, a):

            masked = np.ma.masked_where(a < self.thresh, a)
            return np.power(masked,1./self.exp)

 #           if masked.mask.any():
 #               return ma.log(np.abs(ma.tan(masked) + 1.0 / ma.cos(masked)))
 #           else:
 #               return np.log(np.abs(np.tan(a) + 1.0 / np.cos(a)))

        def inverted(self):
            return SqrtScale.InvertedSqrtTransform(self.thresh,self.exp)

    class InvertedSqrtTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self,thresh,exp):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.exp = exp

        def transform_non_affine(self, a):
            return np.power(a,self.exp)

        def inverted(self):
            return SqrtScale.SqrtTransform(self.thresh,self.exp)

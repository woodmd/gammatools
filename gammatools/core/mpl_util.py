__author__ = 'Matthew Wood <mdwood@slac.stanford.edu>'
__date__ = '11/14/13'

import matplotlib

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, ScalarFormatter, MultipleLocator
from matplotlib.ticker import LogLocator, AutoLocator
import numpy as np

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

class PowerNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, power=2., clip=False):
        self.power = power
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.power((value-self.vmin)/self.vmax,
                                  1./self.power))
    
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

import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from gammatools.core.histogram import *


class TestHistogram(unittest.TestCase):
    
    def test_histogram_init(self):

        axis = Axis(np.linspace(0,1,6))

        # Initialize with constant value
        h = Histogram(axis,counts=1.0,var=2.0)
        assert_almost_equal(h.counts(),1.0)
        assert_almost_equal(h.var(),2.0)
        
        # Initialize with vector of values
        h = Histogram(axis,counts=axis.center(),var=2.0*axis.center())
        assert_almost_equal(h.counts(),axis.center())
        assert_almost_equal(h.var(),2.0*axis.center())


    def test_histogram_fill(self):

        h = Histogram(np.linspace(0,1,6))

        # Test filling from scalar, list, and array input

        xs = 0.1
        xv = h.axis().center()
        xl = xv.tolist()
        unitw = np.ones(h.axis().nbins())

        # Scalar w/ unit weight
        for x in xv: h.fill(x)
        assert_almost_equal(h.counts(),unitw)
        assert_almost_equal(h.var(),unitw)
        h.clear() 

        # List w/ unit weight
        h.fill(xl)  
        assert_almost_equal(h.counts(),unitw)
        assert_almost_equal(h.var(),unitw)
        h.clear()

        # Array w/ unit weight
        h.fill(xv) 
        assert_almost_equal(h.counts(),unitw)
        assert_almost_equal(h.var(),unitw)
        h.clear()

        # Scalar w/ scalar weight
        wv = np.cos(xv)
        wl = np.cos(xv).tolist()
        for x, w in zip(xv,wv): h.fill(x,w)

        assert_almost_equal(h.counts(),wv)
        assert_almost_equal(h.var(),wv)
        h.clear() 

        # List w/ list weight

        h.fill(xv,wl)
        assert_almost_equal(h.counts(),wv)
        assert_almost_equal(h.var(),wv)
        h.clear() 

        # Scalar w/ scalar weight and variance
        wv = np.cos(xv)
        wl = np.cos(xv).tolist()
        vv = wv*np.abs(np.sin(xv))

        for x, w,v in zip(xv,wv,vv): h.fill(x,w,v)

        assert_almost_equal(h.counts(),wv)
        assert_almost_equal(h.var(),vv)
        h.clear() 

        # Test Overflow and Underflow

        xv_shift = np.concatenate((xv,xv+0.5,xv-0.5))
        h.fill(xv_shift) 
        assert (h.overflow()==np.sum(xv_shift>=1.0))
        assert (h.underflow()==np.sum(xv_shift<0.0))

    def test_histogram_rebin(self):

        h = Histogram(np.linspace(0,1,6))
        w = [1,2,3,4,5]
        v = [5,4,3,2,1]
        b = h.axis().bins()

        h.fill(h.axis().binToVal(b),w,v)
        h = h.rebin(2)

        assert_almost_equal(h.counts(),[3,7,5])
        assert_almost_equal(h.var(),[9,5,1])

        h = Histogram(np.linspace(0,1,7))
        w = [1,2,3,4,5,1]
        v = [5,4,3,2,1,1]
        b = h.axis().bins()

        h.fill(h.axis().binToVal(b),w,v)
        h = h.rebin_mincount(4)

        assert_almost_equal(h.counts(),[6,4,5,1])
        assert_almost_equal(h.var(),[12,2,1,1])

    def test_histogram2d_fill(self):

        h = Histogram2D(np.linspace(0,1,6),np.linspace(0,1,6))

        unitw = np.ones((h.xaxis().nbins(),h.yaxis().nbins()))

        xv, yv = np.meshgrid(h.xaxis().center(), h.yaxis().center(),indexing='ij')
        xv = np.ravel(xv)
        yv = np.ravel(yv)

        # Scalar w/ unit weight
        for x, y in zip(xv,yv): h.fill(x,y)

        assert_almost_equal(h.counts(),unitw)
        assert_almost_equal(h.var(),unitw)
        h.clear()

        # Vector w/ unit weight
        h.fill(xv,yv)
        assert_almost_equal(h.counts(),unitw)
        assert_almost_equal(h.var(),unitw)
        h.clear()

        # Scalar w/ scalar weight
        wv = np.cos(xv)*np.sin(yv+0.5*np.pi)
        wl = wv.tolist()
        vv = wv*np.abs(np.sin(xv))

        for x, y, w in zip(xv,yv,wv): h.fill(x,y,w)
        assert_almost_equal(h.counts(),wv.reshape(5,5))
        assert_almost_equal(h.var(),wv.reshape(5,5))
        h.clear()

        # Vector w/ vector weight
        h.fill(xv,yv,wv)
        assert_almost_equal(h.counts(),wv.reshape(5,5))
        assert_almost_equal(h.var(),wv.reshape(5,5))
        h.clear()
        
        # Vector w/ vector weight
        h.fill(xv,yv,wv,vv)
        assert_almost_equal(h.counts(),wv.reshape(5,5))
        assert_almost_equal(h.var(),vv.reshape(5,5))
        h.clear()

    def test_histogram_operators(self):

        axis = Axis(np.linspace(0,1,11))
        xc = axis.center()

        # Addition
        h0 = Histogram(axis)
        h1 = Histogram(axis)


        h0.fill(xc,1.0+np.cos(xc)**2,np.ones(axis.nbins()))
        h1.fill(xc,1.0+np.sin(xc)**2,np.ones(axis.nbins()))

        h2 = h0 + h1 + 1.0

        assert_almost_equal(h2.counts(),4.0)
        assert_almost_equal(h2.var(),2.0)

        h2.clear()

        # Subtraction by Histogram

        h2.fill(xc,3.0)
        h2 -= h0

        assert_almost_equal(h2.counts(),h1.counts())
        assert_almost_equal(h2.var(),h2.var())

        # Multiplication by Histogram

        h2.clear()

        h2 += h1
        h2 *= h0

        assert_almost_equal(h2.counts(),h1.counts()*h0.counts())

        # Division by Histogram

        h2.clear()

        h2 += h1
        h2 /= h0

        assert_almost_equal(h2.counts(),h1.counts()/h0.counts())

        # Division by Scalar Float

        h2.clear()
        h2 += h1
        h2 /= 2.0

        assert_almost_equal(h2.counts(),h1.counts()/2.)
        assert_almost_equal(h2.var(),h1.var()/4.)

        # Division by Vector Float

        h2.clear()
        h2 += h1
        h2 /= xc

        assert_almost_equal(h2.counts(),h1.counts()/xc)
        assert_almost_equal(h2.var(),h1.var()/xc**2)

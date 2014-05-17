#!/usr/bin/env python

from distutils.core import setup

setup(name='gammatools',
      version='1.0.0',
      author='Matthew Wood',
      author_email='mdwood@slac.stanford.edu',
      packages=['gammatools',
                'gammatools.core',
                'gammatools.fermi',
                'gammatools.dm'],
     )

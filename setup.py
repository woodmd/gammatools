#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(name='gammatools',
      version='1.0.0',
      author='Matthew Wood',
      author_email='mdwood@slac.stanford.edu',
      packages=['gammatools',
                'gammatools.core',
                'gammatools.fermi',
                'gammatools.dm'],
      url = "https://github.com/woodmd/gammatools",
      download_url = "https://github.com/woodmd/gammatools/tarball/master",
      scripts = ['scripts/gtmktime.py','scripts/calc_dmflux.py'],
      data_files=[('gammatools/data',
                   ['gammatools/data/dm_halo_models.yaml',
                    'gammatools/data/gammamc_dif.dat'])],
      install_requires=[#'pywcsgrid2',
                        'numpy >= 1.8.0',
                        'matplotlib >= 1.2.0',
                        'astropy >= 0.3',
                        'scipy >= 0.13'])

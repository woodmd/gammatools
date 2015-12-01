#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

from gammatools.version import get_git_version

setup(name='gammatools',
      version=get_git_version(),
      author='Matthew Wood',
      author_email='mdwood@slac.stanford.edu',
      packages=['gammatools'],
      url = "https://github.com/woodmd/gammatools",
      download_url = "https://github.com/woodmd/gammatools/tarball/master",
      scripts = ['scripts/gtmktime.py','scripts/calc_dmflux.py'],
      include_package_data = True, 
      package_data = {
        'gammatools' : ['data/gll_psc_v11.fit',
                        'data/gll_psc_v08.fit',
                        'data/dm_halo_models.yaml',
                        'data/gammamc_dif.dat'] },
      
      install_requires=['wcsaxes',
                        'numpy >= 1.8.0',
                        'matplotlib >= 1.4.0',
                        'astropy >= 0.4',
                        'scipy >= 0.13'])

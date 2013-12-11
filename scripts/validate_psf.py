#!/usr/bin/env python

import os

os.environ['CUSTOM_IRF_DIR'] = '/u/gl/mdwood/ki10/analysis/custom_irfs/'
os.environ['CUSTOM_IRF_NAMES'] = 'P7SOURCE_V6,P7SOURCE_V6MC,P7SOURCE_V9,P7CLEAN_V6,P7CLEAN_V6MC,P7ULTRACLEAN_V6,' \
        'P7ULTRACLEAN_V6MC,P6_v11_diff,P7SOURCE_V6MCPSFC,P7CLEAN_V6MCPSFC,P7ULTRACLEAN_V6MCPSFC'

import sys
import copy
import re
import pickle
import argparse

import math
import numpy as np
import matplotlib.pyplot as plt
from gammatools.core.histogram import Histogram, Histogram2D
from matplotlib import font_manager

from gammatools.fermi.psf_model import *
import gammatools.core.stats as stats
from gammatools.fermi.catalog import Catalog
from gammatools.fermi.validate import *

if __name__ == '__main__':
    usage = "%(prog)s [options] [pickle file ...]"
    description = """Perform PSF validation analysis on agn or
pulsar data samples."""
    parser = argparse.ArgumentParser(usage=usage, description=description)

    parser.add_argument('files', nargs='+')

    PSFValidate.configure(parser)

    args = parser.parse_args()

    psfv = PSFValidate(args)
    psfv.run()

#!/usr/bin/env python

import os, sys
import re
import tempfile
import logging
import shutil
import yaml
import numpy as np
import argparse
from gammatools.fermi.task import ObsSimTask
from gammatools.core.util import dispatch_jobs

usage = "%(prog)s [options]"
description = """Run gtobssim."""
parser = argparse.ArgumentParser(usage=usage, description=description)

#parser.add_argument('files', nargs='+')
#parser.add_argument('--output',required=True)
parser.add_argument('--config',default=None)

ObsSimTask.add_arguments(parser)

args = parser.parse_args()

config = None

if args.config:
    config = yaml.load(open(args.config))

gtobssim = ObsSimTask(config,opts=args)

gtobssim.run()

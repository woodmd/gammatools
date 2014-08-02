#!/usr/bin/env python

import os, sys
import re
import tempfile
import logging
import shutil
from GtApp import GtApp
import numpy as np
import argparse
from gammatools.fermi.task import SrcModelTask
from gammatools.core.util import dispatch_jobs

usage = "%(prog)s [options]"
description = """Run gtmodel."""
parser = argparse.ArgumentParser(usage=usage, description=description)

#parser.add_argument('files', nargs='+')

parser.add_argument('--output',required=True)

SrcModelTask.add_arguments(parser)

args = parser.parse_args()

gtmodel = SrcModelTask(args.output,opts=args)


gtmodel.run()

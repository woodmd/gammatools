#! /usr/bin/env python

import sys
import healpy
import matplotlib.pyplot as plt
import numpy as np
from gammatools.core.histogram import *
from gammatools.core.algebra import Vector3D
from gammatools.core.util import integrate
from gammatools.dm.jcalc import *
from gammatools.dm.halo_model import *
from scipy.interpolate import UnivariateSpline
import argparse

usage = "usage: %(prog)s [options]"
description = """Compute the J-factor within the chosen ROI around a DM halo."""
parser = argparse.ArgumentParser(usage=usage,description=description)

parser.add_argument('--halo_model', default = 'gc_nfw', 
                  help = 'Set the profile name.')

parser.add_argument('--profile', default = 'nfw', 
                  help = 'Set the profile name.')

parser.add_argument('--prefix', default = 'nfw', 
                  help = 'Set the output file prefix.')

parser.add_argument('--alpha', default = None, type=float,
                  help = 'Set the alpha parameter of the DM halo profile.')

parser.add_argument('--gamma', default = None, type=float,
                  help = 'Set the gamma parameter of the DM halo profile.')

parser.add_argument('--lon_cut', default = 6.0, type=float,
                  help = 'Set the longitude cut value.')

parser.add_argument('--lat_cut', default = 5.0, type=float,
                  help = 'Set the latitude cut value.')

parser.add_argument('--radius_cut', default = None, type=float,
                  help = 'Set the latitude cut value.')

parser.add_argument('--rmin', default = 0.001, type=float,
                  help = 'Set the profile name.')

parser.add_argument('--rho_rsun', default = None, type=float,
                  help = 'Set the profile name.')

parser.add_argument('--rhos', default = None, type=str,
                  help = 'Set the profile normalization.')

parser.add_argument('--decay', default = False, action='store_true',
                  help = 'Set the profile name.')

parser.add_argument('--source_list', default = None, 
                  help = 'Set the profile name.')

args = parser.parse_args()

hm = HaloModelFactory.create(args.halo_model,**args.__dict__)

print hm.jp.cumsum(args.radius_cut)/Units.gev2_cm5

print hm.dp.rhos/Units.msun_kpc3

#print hm.set_jvalue(1E18*Units.gev2_cm5,0.5)
#sys.exit(0)



jint = ROIIntegrator(hm.jp,args.lat_cut,args.lon_cut,args.source_list)
#jint.print_profile(args.decay)

if not args.radius_cut is None:
    jint.eval(args.radius_cut,args.decay)



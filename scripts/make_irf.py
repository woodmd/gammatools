#! /usr/bin/env python

import os
import sys
import copy
import glob
import yaml
import re
import ROOT
import array

os.environ['PYTHONPATH'] += ':%s'%(os.path.join(os.environ['INST_DIR'],
                                                'irfs/handoff_response/python'))


def getGenerateEvents(f):

    rf = ROOT.TFile(f)
    if not rf.GetListOfKeys().Contains("jobinfo"): return None
    chain = rf.Get('jobinfo')

    
    NGen_sum = 0
    vref = {}
    
    vref['trigger'] = array.array('i',[0])
    vref['generated'] = array.array('i',[0])
    vref['version'] = array.array('f',[0.0])
    vref['revision'] = array.array('f',[0.0])
    vref['patch'] = array.array('f',[0.0])
    chain.SetBranchAddress('trigger',vref['trigger'])
    chain.SetBranchAddress('generated',vref['generated'])
    chain.SetBranchAddress('version',vref['version'])

    if chain.GetListOfBranches().Contains('revision'):
        chain.SetBranchAddress('revision',vref['revision'])

    if chain.GetListOfBranches().Contains('patch'):
        chain.SetBranchAddress('patch',vref['patch'])

    for i in range(chain.GetEntries()):
        chain.GetEntry(i)        

        ver = int(vref['version'][0])
        rev = int(vref['revision'][0])
        patch = int(vref['patch'][0])
        
        NGen = 0

        if ver == 20 and rev == 4: NGen = vref['trigger'][0]
        else: NGen = vref['generated'][0]

        if (ver == 20 and rev == 6) or (ver == 20 and rev == 8 and patch < 8):
            NGen *= 0.85055

        NGen_sum += NGen

    return NGen_sum

def get_branches(expr):
    ignore = ['max','min','sqrt','acos','pow','log','log10']
    m = re.compile('([a-zA-Z])([a-zA-Z0-9\_]+)')

    branches = []
    for t in m.finditer(expr):
        var = t.group()
        if var not in ignore:
            branches.append(var)

    return branches

def expand_aliases(aliases,expr):
    ignore = ['max','min','sqrt','acos','pow','log','log10']
    m = re.compile('([a-zA-Z])([a-zA-Z0-9\_]+)')

    has_alias = False
    for t in m.finditer(expr):

        var = t.group()
        alias = ''
        if var in aliases: alias = aliases[var]
        
        if var not in ignore and alias != '':
            expr = re.sub(var + '(?![a-zA-Z0-9\_])',
                          '(' + alias + ')',expr)
            has_alias = True

    if has_alias: return expand_aliases(aliases,expr)
    else: return expr



    
from optparse import OptionParser
usage = "Usage: %prog  [MC file] [options]"
description = """Generate a set of IRFs from an input merit tuple and
event selection."""
parser = OptionParser(usage=usage,description=description)

parser.add_option("--selection",default=None,type='string',
                  help=".")

parser.add_option("--friends",default=None,type='string',
                  help=".")

parser.add_option("--cuts_file",default=None,type='string',
                  help=".")

parser.add_option("--class_name",default=None,type='string',
                  help="Set the class name.")

parser.add_option("--irf_scaling",default=None,
                  help="Set the class name.")

parser.add_option("--psf_scaling",default=None,
                  help="Set the class name.")

parser.add_option("--psf_overlap",default='1/1',
                  help="Set the PSF overlap parameters for energy/angle.")

parser.add_option("--edisp_overlap",default='1/1',
                  help="Set the class name.")

parser.add_option("--generated",default=None,type='float',
                  help="Set the number of generated events in each file.  "
                  "If this option is not given the number of "
                  "generated events will be automatically determined from "
                  "the jobinfo tree if it exists.")


(opts, args) = parser.parse_args()




c = yaml.load(open(args[0],'r'))

input_file_strings = []
emin_array = []
emax_array = []
generated_array = []

for d in c['datasets']:

    o = copy.deepcopy(d)
    o['generated'] = 0
    o['files'] = glob.glob(o['files'])

    emin_array.append('%f'%o['emin'])
    emax_array.append('%f'%o['emax'])
    
    for f in o['files']:
        o['generated'] += getGenerateEvents(f)
        input_file_strings.append('\'%s\''%f)

    generated_array.append('%f'%o['generated'])

#input_file_path = os.path.abspath(args[0])
#f = ROOT.TFile(input_file_path)

#if not opts.generated is None: generated = opts.generated
#elif f.GetListOfKeys().Contains("jobinfo"):
#    jobinfo = f.Get('jobinfo')
#    generated = getGenerateEvents(jobinfo)
#else:
#    print 'Number of generated events not defined.'
#    sys.exit(1)
    
if opts.class_name is None:
    print 'No class name given.'
    sys.exit(1)
    
irf_dir = opts.class_name
irf_output_dir = 'custom_irfs'

if not os.path.isdir(irf_dir):
    os.system('mkdir %s'%(irf_dir))
    
if not os.path.isdir(irf_output_dir):
    os.system('mkdir %s'%(irf_output_dir))

irf_output_dir = os.path.abspath(irf_output_dir)
    
cut_defs = {}
if not opts.cuts_file is None:
    cut_defs = yaml.load(open(opts.cuts_file,'r'))

cut_expr = ''
if not opts.selection is None:
    cut_expr = expand_aliases(cut_defs,opts.selection)

branch_names = get_branches(cut_expr)
branch_names = list(set(branch_names))

friends = ''
if not opts.friends is None:

    friends_list = []
    for s in opts.friends.split(','):
        friend_path = os.path.abspath(s)
        friends_list.append('\'%s\''%friend_path) 
    
    friends = '\'%s\' : [ %s ]'%(input_file_path,','.join(friends_list))

    
if opts.irf_scaling is None:
    psf_pars_string = '[5.81e-2, 3.77e-4, 9.6e-2, 1.3e-3, -0.8]'
    edisp_front_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    edisp_back_pars_string = '[0.0215, 0.0507, -0.22, -0.243, 0.065, 0.584]'
elif opts.irf_scaling == 'front':
    psf_pars_string = '[5.81e-2, 3.77e-4, 5.81e-2, 3.77e-4, -0.8]'
    edisp_front_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    edisp_back_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
elif opts.irf_scaling == 'back':
    psf_pars_string = '[9.6e-2, 1.3e-3, 9.6e-2, 1.3e-3, -0.8]'
    edisp_front_pars_string = '[0.0215, 0.0507, -0.22, -0.243, 0.065, 0.584]'
    edisp_back_pars_string = '[0.0215, 0.0507, -0.22, -0.243, 0.065, 0.584]'
elif opts.irf_scaling == 'psf3':
    psf_pars_string = '[4.97e-02,6.13e-04,4.97e-02,6.13e-04,-0.8]'
    edisp_front_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    edisp_back_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
elif opts.irf_scaling == 'psf2':
    psf_pars_string = '[7.02e-02,1.07e-03,7.02e-02,1.07e-03,-0.8]'
    edisp_front_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    edisp_back_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
elif opts.irf_scaling == 'psf1':
    psf_pars_string = '[9.64e-02,1.78e-03,9.64e-02,1.78e-03,-0.8]'
    edisp_front_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    edisp_back_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
elif opts.irf_scaling == 'psf0':
    psf_pars_string = '[1.53e-01,5.70e-03,1.53e-01,5.70e-03,-0.8]'
    edisp_front_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    edisp_back_pars_string = '[0.0210, 0.058, -0.207, -0.213, 0.042, 0.564]'
    
if not opts.psf_scaling is None: psf_pars_string = opts.psf_scaling
    
edisp_energy_overlap, edisp_angle_overlap = opts.edisp_overlap.split('/')
psf_energy_overlap, psf_angle_overlap = opts.psf_overlap.split('/')


x = '''
from gammatools.fermi.IRFdefault import *

className="%s"
selectionName="front"

Prune.fileName = 'skim.root'
Prune.cuts = '%s'
Prune.branchNames = """
McEnergy  McLogEnergy McXDir  McYDir  McZDir
Tkr1FirstLayer
WP8Best*
EvtRun
%s
""".split()

Data.files = [%s]
Data.generated = [%s]
Data.logemin = [%s]
Data.logemax = [%s]
#Bins.logemin = 0.75
#Bins.logemax = 6.5
Bins.set_energy_range(0.75,6.5)
EffectiveAreaBins.set_energy_range(0.75,6.5)

Data.friends = { %s }

Data.var_xdir = 'WP8BestXDir'
Data.var_ydir = 'WP8BestYDir'
Data.var_zdir = 'WP8BestZDir'
Data.var_energy = 'WP8BestEnergy'

Bins.edisp_energy_overlap = %s
Bins.edisp_angle_overlap = %s

Bins.psf_energy_overlap = %s
Bins.psf_angle_overlap = %s

PSF.pars = %s     # there must be 5 parameters
 
Edisp.front_pars = %s    # each edisp set must have six parameters
Edisp.back_pars = %s

parameterFile = 'parameters.root'
'''%(opts.class_name,cut_expr,' '.join(branch_names),
     ','.join(input_file_strings), ','.join(generated_array),
     ','.join(emin_array), ','.join(emax_array), 
     friends,edisp_energy_overlap,edisp_angle_overlap,
     psf_energy_overlap,psf_angle_overlap,
     psf_pars_string,edisp_front_pars_string,edisp_back_pars_string)

f = open(os.path.join(opts.class_name,'setup.py'),'w')
f.write(x)
f.close()

os.chdir(irf_dir)

if not os.path.isfile('skim.root'):
    cmd = 'prune setup.py'
    print cmd
    os.system(cmd)

#
    
#cmd = 'makeirf %s'%irf_dir
if not os.path.isfile('parameters.root'):
    cmd = 'makeirf setup.py'
    print cmd
    os.system(cmd)

#if not os.path.isfile(os.path.join(irf_dir,'skim.root')):
#    cmd = 'prune %s'%irf_dir
#    print cmd
#    os.system(cmd)

#cmd = 'makeirf %s'%irf_dir
#print cmd
#os.system(cmd)
    
#os.chdir(irf_dir)

cmd = 'makefits %s %s'%('parameters.root',opts.class_name)
print cmd
os.system(cmd)


if not re.search('psf',opts.irf_scaling) is None:

    os.system('rm *_front.fits')
    os.system('rm *_back.fits')

    for s in glob.glob('*%s*fits'%opts.class_name):

        print s
        
        sf = os.path.splitext(s)[0] + '_front.fits'
        sb = os.path.splitext(s)[0] + '_back.fits'

        os.system('cp %s %s'%(s,sf))
        os.system('cp %s %s'%(s,sb))
        os.system('rm %s'%(s))
        
os.system('tar cfz %s.tar.gz *%s*fits'%(opts.class_name,opts.class_name))

os.system('cp *%s*fits %s'%(opts.class_name,irf_output_dir))
os.system('cp %s.tar.gz %s'%(opts.class_name,irf_output_dir))


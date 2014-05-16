#! /usr/bin/env python

import os
import sys
import yaml
import re
import ROOT
import array

os.environ['PYTHONPATH'] += ':%s'%(os.path.join(os.environ['INST_DIR'],
                                                'irfs/handoff_response/python'))


def getGenerateEvents(chain):
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

parser.add_option("--generated",default=None,type='float',
                  help="Set the number of generated events in each file.  "
                  "If this option is not given the number of "
                  "generated events will be automatically determined from "
                  "the jobinfo tree if it exists.")


(opts, args) = parser.parse_args()

if len(args) == 0:
    print 'No input ROOT file.'
    sys.exit(1)
elif not os.path.isfile(args[0]):
    print 'Input file does not exist.'
    sys.exit(1)

input_file_path = os.path.abspath(args[0])

f = ROOT.TFile(input_file_path)

if not opts.generated is None: generated = opts.generated
elif f.GetListOfKeys().Contains("jobinfo"):
    jobinfo = f.Get('jobinfo')
    generated = getGenerateEvents(jobinfo)
else:
    print 'Number of generated events not defined.'
    sys.exit(1)
    
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


friends = ''
if not opts.friends is None:

    friends_list = []
    for s in opts.friends.split(','):
        friend_path = os.path.abspath(s)
        friends_list.append('\'%s\''%friend_path) 
    
    friends = '\'%s\' : [ %s ]'%(input_file_path,','.join(friends_list))

x = '''
from IRFdefault import *

className = "source"

Prune.fileName = 'skim.root'
Prune.cuts = '%s'
Prune.branchNames = """
McEnergy  McLogEnergy McXDir  McYDir  McZDir
Tkr1FirstLayer
CTBBest*
WP8Best*
EvtRun
%s
""".split()

Data.files = ['%s']
Data.generated = [%f]
Data.logemin = [1.25]
Data.logemax = [5.75]
Bins.logemin = 1.25
Bins.logemax = 5.75

Data.friends = { %s }

Data.var_xdir = 'WP8BestXDir'
Data.var_ydir = 'WP8BestYDir'
Data.var_zdir = 'WP8BestZDir'
Data.var_energy = 'WP8BestEnergy'

Bins.edisp_energy_overlap = 2
Bins.edisp_angle_overlap = 2

Bins.psf_energy_overlap = 1
Bins.psf_angle_overlap = 1

parameterFile = 'parameters.root'
'''%(cut_expr,' '.join(branch_names),input_file_path,
     generated,friends)

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
cmd = 'makeirf setup.py'
print cmd
os.system(cmd)

sys.exit(0)
    
#os.chdir(irf_dir)


    

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

os.system('tar cfz %s.tar.gz *%s*fits'%(opts.class_name,opts.class_name))

os.system('cp *%s*fits %s'%(opts.class_name,irf_output_dir))
os.system('cp %s.tar.gz %s'%(opts.class_name,irf_output_dir))


#!/usr/bin/env python

import os
import sys
from optparse import OptionParser
import tempfile
import re
import ROOT
import shutil
from gammatools.core.util import dispatch_jobs

def getEntries(inFile):

    FP = ROOT.TFile.Open(inFile)
    tree = FP.Get('MeritTuple')
    return tree.GetEntries()

def skimMerit(inFile, outfilename, selection,
              nentries, firstentry, enableB = None, disableB = None):
    print 'Preparing merit chunk from %s' % inFile

    print 'Opening input file %s' % inFile  
    oldFP = ROOT.TFile.Open(inFile)
    oldTree = oldFP.Get('MeritTuple')
    oldTree.SetBranchStatus('*',1)
    oldTree.SetBranchStatus('Pulsar_Phase', 0)

#    for branch in enableB:
#      oldTree.SetBranchStatus(branch, 1)
#    for branch in disableB:
#      oldTree.SetBranchStatus(branch, 0)
   
    newFP = ROOT.TFile(outfilename, "recreate")
    newTree = oldTree.CopyTree(selection,"fast",nentries, firstentry)
    newTree.AutoSave()
    nevents = newTree.GetEntries()
    print 'Skimmed events ', nevents
    newFP.Close()
    print 'Closing output file %s' % outfilename
    oldFP.Close()
    return nevents

def phase_ft1(ft1File,outFile,logFile,stagedFT2,stagedEphem):
    cmd = '$TEMPO2ROOT/tempo2 '
    cmd += ' -gr fermi -ft1 %s '%(ft1File)
    cmd += ' -ft2 %s '%(stagedFT2)
    cmd += ' -f %s -phase '%(stagedEphem)

    print cmd
    os.system(cmd)

    print 'mv %s %s'%(ft1File,outFile)
    os.system('mv %s %s'%(ft1File,outFile))

def phase_merit(meritFile,outFile,logFile,stagedFT2,stagedEphem):
    nevent_chunk = 30000  # number of events to process per chunk
    mergeChain=ROOT.TChain('MeritTuple')

    skimmedEvents = getEntries(meritFile)
    
    for firstEvent in range(0, skimmedEvents,nevent_chunk):

        filename=os.path.splitext(os.path.basename(meritFile))[0]
        meritChunk=filename + '_%s.root'%firstEvent
        nevts = skimMerit(meritFile, meritChunk, 
                          '', nevent_chunk, firstEvent)

        cmd = '$TEMPO2ROOT/tempo2 -gr root -inFile %s -ft2 %s -f %s -graph 0 -nobs 32000 -npsr 1 -addFriend -phase'%(meritChunk, stagedFT2, stagedEphem)

        print cmd
        os.system(cmd + ' >> %s 2>> %s'%(logFile,logFile))

#        print tempo

        mergeChain.Add(meritChunk)

    mergeFile = ROOT.TFile('merged.root', 'RECREATE')
# Really bad coding
    if mergeChain.GetEntries()>0: mergeChain.CopyTree('')

    mergeFile.Write()
    print 'merged events %s' %mergeChain.GetEntries()
    mergeFile.Close()

    os.system('mv merged.root %s'%(outFile))
    

usage = "usage: %prog [options] "
description = "Run tempo2 application on one or more FT1 files."
parser = OptionParser(usage=usage,description=description)

parser.add_option('--par_file', default = None, type = "string", 
                  help = 'Par File')

parser.add_option('--ft2_file', default = None, type = "string", 
                  help = 'FT2 file')

parser.add_option("--batch",action="store_true",
                  help="Split this job into several batch jobs.")

parser.add_option('--queue', default = None,
                  type='string',help='Set the batch queue.')

parser.add_option('--phase_colname', default='J0835_4510_Phase',
                  type='string',help='Set the name of the phase column.')
                  
(opts, args) = parser.parse_args()

if opts.par_file is None:
    print 'No par file.'
    sys.exit(1)

if opts.ft2_file is None:
    print 'No FT2 file.'
    sys.exit(1)


if not opts.queue is None:
    
    dispatch_jobs(os.path.abspath(__file__),args,opts)
#    for x in args:
#        cmd = 'run_tempo.py %s '%(x)
        
#        for k, v in opts.__dict__.iteritems():
#            if not v is None and k != 'batch': cmd += ' --%s=%s '%(k,v)

#        print 'bsub -q %s -R rhel60 %s'%(opts.queue,cmd)
#        os.system('bsub -q %s -R rhel60 %s'%(opts.queue,cmd))

    sys.exit(0)
    
par_file = os.path.abspath(opts.par_file)
ft2_file = os.path.abspath(opts.ft2_file)
    
input_files = []
for x in args: input_files.append(os.path.abspath(x))

    
cwd = os.getcwd()
user = os.environ['USER']
tmpdir = tempfile.mkdtemp(prefix=user + '.', dir='/scratch')

print 'tmpdir ', tmpdir

os.chdir(tmpdir)

for x in input_files:

    outFile = x
    inFile = os.path.basename(x)
    logFile=os.path.splitext(x)[0] + '_tempo2.log'

    staged_ft2_file = os.path.basename(ft2_file)

    print 'cp %s %s'%(ft2_file,staged_ft2_file)
    os.system('cp %s %s'%(ft2_file,staged_ft2_file))
    
    if os.path.isfile(logFile):
        os.system('rm %s'%logFile)
    
    print 'cp %s %s'%(x,inFile)
    os.system('cp %s %s'%(x,inFile))
    
    if not re.search('\.root?',x) is None:
        phase_merit(inFile,outFile,logFile,staged_ft2_file,par_file)
    elif not re.search('\.fits?',x) is None:
        phase_ft1(inFile,outFile,logFile,staged_ft2_file,par_file)
    else:
        print 'Unrecognized file extension: ', x


os.chdir(cwd)
shutil.rmtree(tmpdir)
        
sys.exit(0)

for x in args:

#    x = os.path.abspath(x)
    
    cmd = '$TEMPO2ROOT/tempo2 '
    cmd += ' -gr fermi -ft1 %s '%(x)
    cmd += ' -ft2 %s '%(os.path.abspath(opts.ft2_file))
    cmd += ' -f %s -phase '%(os.path.abspath(opts.par_file))

    cwd = os.getcwd()

    script_file = tempfile.mktemp('.sh',os.environ['USER'] + '.',cwd)
    ftemp = open(script_file,'w')

    ftemp.write('#!/bin/sh\n')
    ftemp.write('cd %s\n'%(cwd))
    ftemp.write('TMPDIR=`mktemp -d /scratch/mdwood.XXXXXX` || exit 1\n')
    ftemp.write('cp %s $TMPDIR\n'%(x))
    ftemp.write('cd $TMPDIR\n')
    ftemp.write(cmd + '\n')
    ftemp.write('cp %s %s\n'%(x,cwd))
    ftemp.write('cd %s\n'%(cwd))
    ftemp.write('rm -rf $TMPDIR\n')
    ftemp.close()

    os.system('chmod u+x %s'%(script_file))
    
#    cwd = os.getcwd()
#    tmp_dir = tempfile.mkdtemp(prefix=os.environ['USER'] + '.',
#                               dir='/scratch')
    
#    print cmd
    os.system('bsub -q kipac-ibq %s'%(script_file))

#tempo2 -gr fermi -ft1 vela_239557417_302629417_ft1_10.fits -ft2 ../all-sky_239557417_302629417_ft2-30s.fits -f vela.par -phase



    


import fnmatch
import numpy as np
import numexpr
import subprocess
import os
import difflib
import glob
import sys
import json
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-p', '--process', help='process', dest='process')
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-d', '--dataset', help='dataset', dest='dataset', default=False)
(options, args) = parser.parse_args()

year = options.year

with open('metadata/KITv3_UL_ALL'+year+'_v1.json') as f:
    samplefiles = json.load(f)

totaljob = list(samplefiles.keys())
if options.dataset:
    f = open("./missJobs_"+str(options.process)+"_"+str(options.dataset)+".txt","w")
else:
    f = open("./missJobs_"+str(options.process)+".txt","w")

for job in samplefiles.keys():
    if options.dataset:
        if not str(options.dataset) in job: totaljob.remove(job)
#    f.write(job+"\n")

    dlist = glob.glob('./hists/'+options.process+'/'+job+'*')
    for i in dlist:
        if options.dataset:
            if not str(options.dataset) in i: continue
        ijob = i.split('/')[3].split('.')[0]
        totaljob.remove(ijob)
        #f.write(ijob+"\n")

for miss in totaljob:
    f.write(miss+"\n")

f.close()

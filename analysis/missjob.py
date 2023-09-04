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
parser.add_option('-y', '--year', help='year', dest='year')
(options, args) = parser.parse_args()

## year is 2018, 2017, 2016_preVFP or 2016_postVFP
# KITv3_UL_ALL2017_v1.json
year = options.year
run_name = {
    '2018': 'manyplot_2018_Run1',
    '2017': 'runSYS_2017_Run4',
    '2016_preVFP' : 'runSYS_2016preVFP_Run4',
    '2016_postVFP': 'runSYS_2016postVFP_Run4'
}

with open('metadata/KITv3_UL_ALL'+year+'_v1.json') as f:
    samplefiles = json.load(f)

totaljob = list(samplefiles.keys())
f = open("./missJobs_"+year+".txt","w")
for job in samplefiles.keys():
    #if not 'EG' in job: totaljob.remove(job)
#    f.write(job+"\n")

    dlist = glob.glob('./hists/'+run_name[year]+'/'+job+'*')
    for i in dlist:
        #if not 'EG' in i: continue
        #print(i)
        ijob = i.split('/')[3].split('.')[0]
        totaljob.remove(ijob)
        #f.write(ijob+"\n")

for miss in totaljob:
    f.write(miss+"\n")
        
#cnt = 0        
#for d in lst:
#    totaljob = list()
#    for nj in range(njob[cnt]):
#        jobname = d + '____' + str(nj+1) + '_'
#        totaljob.append(jobname)
#
#    print(d, len(totaljob))
#    for miss in totaljob:
#        f.write(miss+"\n")
#    cnt = cnt+1
f.close()

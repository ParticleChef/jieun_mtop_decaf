import uproot
import json
import pyxrootd.client
import fnmatch
import numpy as np
import numexpr
import subprocess
import concurrent.futures
import warnings
import os
import difflib
from optparse import OptionParser
import glob
import yaml
import sys

with open("metadata/KITv3_UL_ALL2018_v1.json") as fin:
    samplefiles = json.load(fin)

lst = list()
with open('./missJobs_Run6.txt', 'r') as ff:
    filelist = ff.readlines()
    for fff in filelist:
        file = fff.split('\n')[0]
        lst.append(file)

datadict_obj = {}
cnt = 0
print(len(lst))
for missjob in lst:
    lstobj = []
    filelst = samplefiles[missjob]['files']
    for filename in filelst:
        lstobj.append(filename)

    n = int(1)
            
    output=[lstobj[j:j + n] for j in range(0, len(lstobj), n)] 
 
    for i, lt in enumerate(output):
        key_name = missjob[:-1] + str(i+100)+'_'
        datadict_obj[key_name] = {'files': lt, 'xs': samplefiles[missjob]['xs']}
        cnt = cnt +1
        #print(key_name, samplefiles[missjob]['xs'], lt)


for key in datadict_obj.keys():
    print(key, datadict_obj[key])

folder = "metadata/KITv3_ALLrun_rerun6_v1.json"
with open(folder, "w") as fout:
    json.dump(datadict_obj, fout, indent=4)

print(cnt)
fout.close()

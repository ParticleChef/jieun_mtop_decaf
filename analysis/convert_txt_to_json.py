#!/usr/bin/env python

#usage: python convert_txt_to_json.py -y [enter the year you want to process] -p [enter the pack size]
#  python convert_txt_to_json.py -y 2016_postVFP -p 3
#  python convert_txt_to_json.py -y 2016_preVFP -p 3
#  python convert_txt_to_json.py -y 2017 -p 3
#  python convert_txt_to_json.py -y 2018 -p 3
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
#from datasets import *

parser = OptionParser()
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()

# a dict for dataset and cross section reference
if '2016' in options.year:
    year = options.year.split('_')[0]

else:
    year = options.year
    
yml_file = {}
#yml_file['2016'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2016.yml'
#yml_file['2017'] = 'UL_data_processing/dataset_yml_files/MonoTop_UL_2017.yml'
yml_file['2018'] = 'MonoTop_customUL_2018.yml' # 'monotop_customul_2018.yml'
with open(yml_file['2018'], 'r') as file:
    data = yaml.safe_load(file)
    print('data: ', data)

lst = list()
#with open('metadata/nanolist/datasetlist.txt', 'r') as ff:
with open('metadata/nanolist2018/datasetlist.txt', 'r') as ff:
#with open('metadata/nanolist_v3/UL2018SIG/signal2018.txt', 'r') as ff:
    filelist = ff.readlines()
    for fff in filelist:
#        if 'Z1Jet' in fff or 'Z2Jet' in fff:
#            file = fff.split('\n')[0]
#            lst.append(file)
        file = fff.split('\n')[0]
        lst.append(file)

xs_reference = {}
for dataset in lst:
#    print(dataset.split('/')[1])
    tmp = dataset.split('/')[1]
    if 'TPhi' in dataset:
        tmp = tmp.split('_TuneCP5')[0]
    print('tmp', tmp)
    #print(dataset,':',data['2018'][dataset]['xs'])
    if 'EGamma' in dataset or 'SingleMuon' in dataset or 'MET' in dataset or 'SingleElectron' in dataset or 'SinglePhoton' in dataset:
        print(dataset)
        #xs_reference[dataset] = -1
        xs_reference[tmp] = -1
    elif 'MPhi' in dataset:
        print('signal!', dataset)
        xs_reference[tmp] = 4.315 # 9999999
    else: 
        #xs_reference[tmp] = data[tmp]['xs']
        xs_reference[tmp] = data['2018'][tmp]['xs']
        #xs_reference[dataset] = data[year][tmp]['xs']
        #print(dataset)
        #print(xs_reference[dataset])
        #xs_reference[dataset] = data[year][dataset]['xs']
        #xs_reference[dataset] = 1.0
        #xs_reference[dataset] = datasets[year][dataset]


    
#flist = glob.glob('metadata/UL_'+options.year+'/*.txt')
#flist = glob.glob('metadata/customNano/UL_'+options.year+'_txtfiles/*.txt')
#flist = glob.glob('metadata/nanolist/new_UL'+options.year+'/*.txt')

#flist = glob.glob('metadata/nanolist_v3/UL'+options.year+'_v3/*.txt')
#flist = glob.glob('./METB.txt')
#flist = glob.glob('metadata/nanolist_v3/UL2018SIG/TPhiTo2Chi_MPhi1000_MChi1000_TuneCP5_13TeV-amcatnlo-pythia8.txt')
#flist = glob.glob('metadata/nanolist_v3/addWJet0J/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8.txt')
flist = glob.glob('metadata/nanolist2018/txtFiles/*.txt')
#metadata/customNano/KIT_list2018_Z
#metadata/customNano/UL_2018_txtfiles/
#metadata/nanolist_v3/UL2018_v3/
datadict_obj = {}
for txt_file in flist:
    print('txt_file: ', txt_file)
    if 'SingleMuon' in txt_file:
        continue
#    if 'MET' not in txt_file:
#        continue
#    if not 'TPhi' in txt_file:
#        continue

    print('txt file', txt_file)
    
    lstobj = []
    n = int(options.pack)
    with open(txt_file) as f:
        data = f.readlines()
        for filename in data:
            file = filename.split('\n')[0]
            lstobj.append(file)
            
    if not len(lstobj) == len(set(lstobj)):
        print("your dataset has duplicate root files")

    output=[lstobj[i:i + n] for i in range(0, len(lstobj), n)] 
    print('split txt_file ',txt_file.split('/')[1].split('.')[0])
 
    for i, lst in enumerate(output):
        if 'EGamma' in txt_file or 'SingleMuon' in txt_file or 'MET' in txt_file or 'SingleElectron' in txt_file or 'SinglePhoton' in txt_file:
            key_name = txt_file.split('/')[3].split('.')[0].split('_')[0] + '____'+str(i+1)+'_'
            dataset_name = txt_file.split('/')[3].split('.')[0]
        else:
            key_name = txt_file.split('/')[3].split('.')[0] + '____'+str(i+1)+'_'
            if 'preVFP' in key_name:
                dataset_name = txt_file.split('/')[3].split('.')[0][:-6]
            elif 'postVFP' in key_name:
                dataset_name = txt_file.split('/')[3].split('.')[0][:-7]
            else:
                dataset_name = txt_file.split('/')[3].split('.')[0]
        datadict_obj[key_name] = {'files': lst, 'xs': xs_reference[dataset_name]}
       
import json
#folder = "metadata/UL_"+options.year+'.json'
folder = "metadata/KITv3_UL_ALL"+options.year+'_v2.json'
#folder = "metadata/KITv3_UL_SIG"+options.year+'_v1.json'
#folder = "metadata/KITv3_UL_WJetsToLNu_0J_"+opt ions.year+'.json'
with open(folder, "w") as fout:                  
    json.dump(datadict_obj, fout, indent=4)      

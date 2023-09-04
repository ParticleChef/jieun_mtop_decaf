from coffea import hist,  util
import awkward as ak
import numpy as np
import glob as glob
import re
import itertools


import json

import pickle
import os
import mplhep as hep


import matplotlib.pyplot as plt

from coffea.util import load, save

year = 2018
vfp = 'post'
version = 1
path = '/uscms_data/d1/jhong/working/decaf/analysis/hists/'


if year == 2016:
    hists = load(path+'runSYS_'+str(year)+str(vfp)+'VFP_Run'+str(version)+'.scaled')
    print(path+'runSYS_'+str(year)+str(vfp)+'VFP_Run'+str(version)+'.scaled')
else:
    hists = load(path+'runSYS_'+str(year)+'_Run'+str(version)+'.scaled') #Run all
    print(path+'runSYS_'+str(year)+'_Run'+str(version)+'.scaled')

data_hists = hists['data']
bkg_hists = hists['bkg']
sig_hists = hists['sig'] 

lumi = {'2018':59.83,
        '2017': 41.48,
        '2016' : {'pre': 19.52,
                'post': 16.81} 
        }

dat = {'2018': 
        {'e': 'EGamma',
        'm': 'MET',
        'g': 'EGamma'},
    '2017': {'e': 'SingleElectron',
        'm': 'MET',
        'g': 'SinglePhoton'},
    '2016': {'e': 'SingleElectron',
        'm': 'MET',
        'g': 'SinglePhoton'}}
for i in range(18):
    print(data_hists['cutflow'].integrate('region','tmcr').integrate('process','MET').values()[()][i])



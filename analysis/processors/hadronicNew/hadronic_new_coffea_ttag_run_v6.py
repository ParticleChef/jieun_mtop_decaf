#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from coffea import hist, nanoevents, util
from coffea.util import load, save
import coffea.processor as processor
import awkward as ak
import numpy as np
import glob as glob
import re
import itertools
# import vector as vec
from coffea.nanoevents.methods import vector, candidate
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea.lumi_tools import LumiMask
# for applying JECs
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import JetResolution, JetResolutionScaleFactor
# from jmeCorrections import JetMetCorrections
import json


import coffea.processor as processor
from coffea import hist
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod

NanoAODSchema.warn_missing_crossrefs = False
from optparse import OptionParser
import pickle
np.errstate(invalid='ignore', divide='ignore')

def METXYCorr_Met_MetPhi(events, year, metpt, metphi):
    if 'genWeight' in events.fields:
        isData = False
    else: isData = True
    dataset=events.metadata['dataset']

    originalMet = metpt
    originalMet_phi = metphi

    npv = events.PV.npvsGood
    mask = np.asarray(npv>100)
    npv = np.asarray(npv)
    npv[mask==True] = 100

    if not isData:
        if (year == '2016') and ("preVFP" in dataset):
            METxcorr = -(-0.188743*npv +0.136539)
            METycorr = -(0.0127927*npv +0.117747)

        if (year == '2016') and ("postVFP" in dataset):
            METxcorr = -(-0.153497*npv -0.231751)
            METycorr = -(0.00731978*npv +0.243323)

        elif year == '2017':
            METxcorr = -(-0.300155*npv +1.90608)
            METycorr = -(0.300213*npv +-2.02232)

        elif year == '2018':
            METxcorr = -(0.183518*npv +0.546754)
            METycorr = -(0.192263*npv +-0.42121)

    if isData:
        if '2018' in year:
            runNMax = max(events.run)
            runNMin = min(events.run)

            if (runNMin > 315252) and (runNMax <= 316995):
                era = "A"

            elif (runNMin > 316998) and (runNMax <= 319312):
                era = "B"

            elif (runNMin > 319313) and (runNMax <= 320393):
                era = "C"

            elif (runNMin > 320394) and (runNMax <= 325273):
                era = "D"

            if "A" in era:
                METxcorr = -(0.263733*npv +-1.91115)
                METycorr = -(0.0431304*npv +-0.112043)

            elif "B" in era:
                METxcorr = -(0.400466*npv +-3.05914)
                METycorr = -(0.146125*npv +-0.533233)

            elif "C" in era:
                METxcorr = -(0.430911*npv +-1.42865)
                METycorr = -(0.0620083*npv +-1.46021)

            elif "D" in era:
                METxcorr = -(0.457327*npv +-1.56856)
                METycorr = -(0.0684071*npv +-0.928372)

        if '2017' in year:
            runNMax = max(events.run)
            runNMin = min(events.run)

            if (runNMin >=297020) and (runNMax <=299329):
                era = "B"

            elif (runNMin >=299337) and (runNMax <=302029):
                era = "C"

            elif (runNMin >=302030) and (runNMax <=303434):
                era = "D"

            elif (runNMin >=303435) and (runNMax <=304826):
                era = "E"

            elif (runNMin >=304911) and (runNMax <=306462):
                era = "F"

            else: era = "X"

            if "B" in era:
                METxcorr = -(-0.211161*npv +0.419333)
                METycorr = -(0.251789*npv +-1.28089)

            elif "C" in era:
                METxcorr = -(-0.185184*npv +-0.164009)
                METycorr = -(0.200941*npv +-0.56853)

            elif "D" in era:
                METxcorr = -(-0.201606*npv +0.426502)
                METycorr = -(0.188208*npv +-0.58313)

            elif "E" in era:
                METxcorr = -(-0.162472*npv +0.176329)
                METycorr = -(0.138076*npv +-0.250239)

            elif "F" in era:
                METxcorr = -(-0.210639*npv +0.72934)
                METycorr = -(0.198626*npv +1.028)

            elif "X" in era:
                METxcorr = np.zeros(len(events))
                METycorr = np.zeros(len(events))

    CorrectedMET_x = np.multiply(originalMet, np.cos(originalMet_phi)) + (np.array(METxcorr))
    CorrectedMET_y = np.multiply(originalMet, np.sin(originalMet_phi)) + (np.array(METycorr))

    CorrectedMET = np.sqrt((CorrectedMET_x)**2 + (CorrectedMET_y)**2)
    CorrectedMETPhi = np.zeros(len(events))


    mask1 = (CorrectedMET_x ==0 )& (CorrectedMET_y > 0)
    CorrectedMETPhi[mask1==True] = np.pi

    mask2 = (CorrectedMET_x ==0 )& (CorrectedMET_y < 0)
    CorrectedMETPhi[mask2==True] = -np.pi

    mask3 = (CorrectedMET_x <0 )& (CorrectedMET_y > 0)
    CorrectedMETPhi[mask3==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x) + np.pi )[mask3==True]

    mask4 = (CorrectedMET_x <0 )& (CorrectedMET_y < 0)
    CorrectedMETPhi[mask4==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x) - np.pi )[mask4==True]

    mask5 = (CorrectedMET_x >0 )
    CorrectedMETPhi[mask5==True] = (np.arctan(CorrectedMET_y/CorrectedMET_x))[mask5==True]

    mask6 = (~mask1 & ~mask2 & ~mask3 & ~mask4 & ~mask5)
    CorrectedMETPhi[mask6==True] = 0

    return (CorrectedMET, CorrectedMETPhi)


class AnalysisProcessor(processor.ProcessorABC):

    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
        '2016': 19.52, #preVFP
#         '2016': 16.81, #postVFP
        '2017': 41.48,
        '2018': 59.83
    }

    met_filter_flags = {

        '2016': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter'
                 ],

        '2017': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilter'
                 ],

        '2018': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'BadPFMuonDzFilter',
                 'eeBadScFilter',
                 'ecalBadCalibFilter'
                 ]
    }

    def __init__(self, year, xsec, corrections, correctionsUL, correctionsBtag, ids, common):


        self._year = year

        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])

        self._xsec = xsec
        
        self._samples = {
            'sr':('Z1Jets','Z2Jets','WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','MET','Mphi'),
            'wmcr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','MET'),  #ZJet and GJet
            'tmcr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','MET'),  #ZJet and GJet
            'wecr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma'), #ZJet and GJet
            'tecr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma'), #ZJet and GJet
            'zmcr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','MET'),   #ZJet and GJet
            'zecr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma'), #ZJet and GJet
            'gcr':('G1Jet','QCD','SinglePhoton','EGamma')
        }

        self._gentype_map = {
            'xbb':      1,
            'tbcq':     2,
            'tbqq':     3,
            'zcc':      4,
            'wcq':      5,
            'vqq':      6,
            'bb':       7,
            'bc':       8,
            'b':        9,
            'cc':     10,
            'c':       11,
            'other':   12
            # 'garbage': 13
        }

        self._TvsQCDwp = {
            '2016': 0.53,
            '2017': 0.61,
            '2018': 0.65
        }

        self._met_triggers = {
            '2016': [
                'PFMETNoMu90_PFMHTNoMu90_IDTight',
                'PFMETNoMu100_PFMHTNoMu100_IDTight',
                'PFMETNoMu110_PFMHTNoMu110_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight'
            ],
            '2017': [
                'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                'PFMETNoMu120_PFMHTNoMu120_IDTight'
            ],
            '2018': [
                'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                'PFMETNoMu120_PFMHTNoMu120_IDTight'
            ]
        }

        self._singlephoton_triggers = {
            '2016': [
                'Photon175',
                'Photon165_HE10'
            ],
            '2017': [
                'Photon200'
            ],
            '2018': [
                'Photon200'
            ]
        }

        self._singleelectron_triggers = {  # 2017 and 2018 from monojet, applying dedicated trigger weights
            '2016': [
                'Ele27_WPTight_Gsf',
                 'Ele105_CaloIdVT_GsfTrkIdT'
#                'Ele115_CaloIdVT_GsfTrkIdT'
#                 'Ele50_CaloIdVT_GsfTrkIdT_PFJet165'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Photon200'
            ]
        }
        self._singlemuon_triggers = {
            '2016': [
                'IsoMu24',
                'IsoTkMu24',
                'Mu50',
                'TkMu50'

            ],
            '2017':
                [
                'IsoMu27',
                'Mu50',
                'OldMu100',
                'TkMu100'
            ],
            '2018':
                [
                'IsoMu24',
                'Mu50',
                'OldMu100',
                'TkMu100'
            ]
        }

        self._jec = {

            '2016': [ 
                'Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFPuppi',
                'Summer16_07Aug2017_V11_MC_L2L3Residual_AK4PFPuppi',
                'Summer16_07Aug2017_V11_MC_L2Relative_AK4PFPuppi',
                'Summer16_07Aug2017_V11_MC_L2Residual_AK4PFPuppi',
                'Summer16_07Aug2017_V11_MC_L3Absolute_AK4PFPuppi'
            ],

            '2017': [
                'Fall17_17Nov2017_V32_MC_L1FastJet_AK4PFPuppi',
                'Fall17_17Nov2017_V32_MC_L2L3Residual_AK4PFPuppi',
                'Fall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi',
                'Fall17_17Nov2017_V32_MC_L2Residual_AK4PFPuppi',
                'Fall17_17Nov2017_V32_MC_L3Absolute_AK4PFPuppi'
            ],

            '2018': [
                'Autumn18_V19_MC_L1FastJet_AK4PFPuppi',
                'Autumn18_V19_MC_L2L3Residual_AK4PFPuppi',
                'Autumn18_V19_MC_L2Relative_AK4PFPuppi',  # currently broken
                'Autumn18_V19_MC_L2Residual_AK4PFPuppi',
                'Autumn18_V19_MC_L3Absolute_AK4PFPuppi'
            ]
        }
# not updated JUNC
        self._junc = {

            '2016': 
                'Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFPuppi',

            '2017': [
                'Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi'
            ],

            '2018': [
                'Autumn18_V19_MC_Uncertainty_AK4PFPuppi'
            ]
        }

        self._jr = {

            '2016': 
                'Summer16_25nsV1b_MC_PtResolution_AK4PFPuppi',


            '2017': [
                'Fall17_V3b_MC_PtResolution_AK4PFPuppi'
            ],

            '2018': [
                'Autumn18_V7b_MC_PtResolution_AK4PFPuppi'
            ]
        }

        self._jersf = {

            '2016': 
                'Summer16_25nsV1b_MC_SF_AK4PFPuppi',
            

            '2017': [
                'Fall17_V3b_MC_SF_AK4PFPuppi'
            ],

            '2018': [
                'Autumn18_V7b_MC_SF_AK4PFPuppi'
            ]
        }

        self._corrections = corrections
        self._correctionsUL = correctionsUL
        self._correctionsBtag = correctionsBtag
        self._ids = ids
        self._common = common

        self._accumulator = processor.dict_accumulator({
            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])),
            
            'template': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('recoil', 'Hadronic recoil', [250,310,370,470,590,840,1020,1250,3000])),
            
            'mT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mT', '$m_{T}$ [GeV]', 20, 0, 600)),
            
            'recoil': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('recoil', 'Hadronic Recoil', [250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0, 3000])),

            'eT_miss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('eT_miss', '$E^T_{miss}$[GeV]', 20, 0, 600)),
            'metphi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('metphi','MET phi',35,-3.5,3.5)),

            'ele_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)),
            'ele_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_eta', 'Leading Electron Eta', 48, -2.4, 2.4)),
            'ele_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_phi', 'Leading Electron Phi', 64, -3.2, 3.2)),

            'mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_pT', 'Tight Muon $p_{T}$ [GeV]', 10, 0, 200)),
            'mu_eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_eta', 'Leading Muon Eta', 48, -2.4, 2.4)),
            'mu_phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_phi', 'Leading Muon Phi', 64, -3.2, 3.2)),
            
            'ak8pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ak8pt', 'FatJet pT [GeV]', 20, 0, 2000)),

            'ak8phi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ak8phi', 'FatJet phi', 35, -3.5, 3.5)),

            'ak8eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ak8eta', 'FatJet eta', 35, -3.5, 3.5)),

            'ak8_tvsqcd': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ak8_tvsqcd', 'FatJet particleNet_TvsQCD', 100, 0, 1)),

            'j1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1pt','AK4 Leading Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'j1eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1eta','AK4 Leading Jet Eta',35,-3.5,3.5)),
            
            'j1phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('j1phi','AK4 Leading Jet Phi',35,-3.5,3.5)),

            'fj1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('fj1pt','AK15 Leading Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),
            'fj1eta': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('fj1eta','AK15 Leading Jet Eta',35,-3.5,3.5)),
            
            'fj1phi': hist.Hist(
                'Events', 
                hist.Cat('dataset', 'Dataset'), 
                hist.Cat('region', 'Region'), 
                hist.Bin('fj1phi','AK15 Leading Jet Phi',35,-3.5,3.5)),

            'dphi_e_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi_e_etmiss', '$\Delta\phi (e, E^T_{miss} )$', 30, 0, 3.5)),
            
            'dphi_mu_etmiss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi_mu_etmiss','$\Delta\phi (\mu, E^T_{miss} )$', 30, 0, 3.5)),
            
            'ndflvL': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndflvL', 'AK4 Number of deepFlavor Loose Jets', 6, -0.5, 5.5)),
            'ndflvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndflvM', 'AK4 Number of deepFlavor Medium Jets', 6, -0.5, 5.5)),
            'njets': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('njets', 'AK4 Number of Jets', 7, -0.5, 6.5)),

            'nfatjets': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('nfatjets', 'AK15 Number of Jets', 7, -0.5, 6.5)),

            'TvsQCD': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('TvsQCD', 'TvsQCD', 15, 0., 1)),

            'ndcsvM': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ndcsvM', 'AK4 Number of deepCSV Medium Jets', 6, -0.5, 5.5)),
            
            'dphi_Met_LJ': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dphi_Met_LJ', '$\Delta \Phi (E^T_{miss}, Leading Jet)$', 30, 0, 3.5)),
            'dr_e_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dr_e_lj', '$\Delta r (Leading e, Leading Jet)$', 30, 0, 5.0)),
            'dr_mu_lj': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('dr_mu_lj', '$\Delta r (Leading \mu, Leading Jet)$', 30, 0, 5.0)),
            'leading_tvsqcd': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin("leading_tvsqcd","TvsQCD (leading ak15 jet)",100,0,1)),
            'drfjtop': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin("drfjtop","dR(leading ak15jet, top)",100,0,10)),
            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                #hist.Bin('cut', 'Cut index', 11, 0, 11),
                hist.Bin('cut', 'Cut index', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])),
 
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata['dataset']
            
        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample not in dataset:
                    continue
                selected_regions.append(region)

        isData = 'genWeight' not in events.fields
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()

        ###
        # Getting corrections, ids from .coffea files
        ###
        if ("preVFP" in dataset) and (self._year == '2016'):
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_preVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_preVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_preVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_preVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_preVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_preVFP_below20'][self._year]
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_preVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            get_mu_loose_err_sf = self._corrections['get_mu_loose_id_err_preVFP'][self._year]            
            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_preVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_preVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_preVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_preVFP'][self._year]            

            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_preVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_weight_preVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_preVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_preVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_preVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_preVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_preVFP']['deepflav'][self._year]
            
        elif ("postVFP" in dataset) and (self._year == '2016'):
            get_ele_loose_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            get_ele_tight_id_sf = self._corrections['get_ele_tight_id_sf_postVFP'][self._year]
            
            get_ele_reco_sf = self._corrections['get_ele_reco_sf_postVFP_above20'][self._year]
            get_ele_reco_err = self._corrections['get_ele_reco_err_postVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._corrections['get_ele_reco_sf_postVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._corrections['get_ele_reco_err_postVFP_below20'][self._year]
            
            
            get_mu_tight_id_sf = self._corrections['get_mu_tight_id_sf_postVFP'][self._year]
            get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]            
            
            get_mu_tight_iso_sf = self._corrections['get_mu_tight_iso_sf_postVFP'][self._year]
            get_mu_loose_iso_sf = self._corrections['get_mu_loose_iso_sf_postVFP'][self._year]
            get_mu_tight_iso_err = self._corrections['get_mu_tight_iso_err_postVFP'][self._year]
            get_mu_loose_iso_err = self._corrections['get_mu_loose_iso_err_postVFP'][self._year]
            
            
            get_mu_trig_weight = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_mu_trig_err = self._corrections['get_mu_trig_weight_postVFP'][self._year]
            get_ele_loose_id_err = self._corrections['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._corrections['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._corrections['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._corrections['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]

        else:
            get_ele_trig_weight = self._correctionsUL['get_ele_trig_weight_preVFP'][self._year]
            get_ele_trig_err = self._correctionsUL['get_ele_trig_err_preVFP'][self._year]
            get_ele_loose_id_sf = self._correctionsUL['get_ele_tight_id_sf_postVFP'][self._year]
            get_ele_tight_id_sf = self._correctionsUL['get_ele_tight_id_sf_postVFP'][self._year]
            
            get_ele_reco_sf = self._correctionsUL['get_ele_reco_sf_postVFP_above20'][self._year]
            get_ele_reco_err = self._correctionsUL['get_ele_reco_err_postVFP_above20'][self._year]
            
            get_ele_reco_lowet_sf = self._correctionsUL['get_ele_reco_sf_postVFP_below20'][self._year]
            get_ele_reco_lowet_err = self._correctionsUL['get_ele_reco_err_postVFP_below20'][self._year]
            
            
            get_mu_tight_id_sf = self._correctionsUL['get_mu_tight_id_sf_postVFP'][self._year]
            get_mu_loose_id_sf = self._correctionsUL['get_mu_loose_id_sf_postVFP'][self._year]
            get_mu_tight_id_err = self._correctionsUL['get_mu_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._correctionsUL['get_mu_loose_id_err_postVFP'][self._year]            
            
            get_mu_tight_iso_sf = self._correctionsUL['get_mu_tight_iso_sf_postVFP'][self._year]
            get_mu_loose_iso_sf = self._correctionsUL['get_mu_loose_iso_sf_postVFP'][self._year]
            get_mu_tight_iso_err = self._correctionsUL['get_mu_tight_iso_err_postVFP'][self._year]
            get_mu_loose_iso_err = self._correctionsUL['get_mu_loose_iso_err_postVFP'][self._year]
            
            
            get_mu_trig_weight = self._correctionsUL['get_mu_trig_weight_postVFP'][self._year]
            get_mu_trig_err = self._correctionsUL['get_mu_trig_weight_postVFP'][self._year]
            get_ele_loose_id_err = self._correctionsUL['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._correctionsUL['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._correctionsUL['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._correctionsUL['get_mu_tight_id_err_postVFP'][self._year]
            
            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]
            #get_deepflav_weight = self._correctionsBtag['get_btag_weight_preVFP']['deepflav'][self._year]

            get_pu_nom_weight = self._correctionsUL['get_pu_nom_weight_postVFP'][self._year]

        get_msd_weight = self._correctionsUL['get_msd_weight']
        get_ttbar_weight = self._correctionsUL['get_ttbar_weight']
        get_nnlo_nlo_weight = self._correctionsUL['get_nnlo_nlo_weight'][self._year]
        get_nlo_qcd_weight = self._correctionsUL['get_nlo_qcd_weight'][self._year]
        get_nlo_ewk_weight = self._correctionsUL['get_nlo_ewk_weight'][self._year]

        #get_pu_weight = self._corrections['get_pu_weight'][self._year]

        get_met_trig_weight = self._correctionsUL['get_met_trig_weight'][self._year]
        #get_ele_trig_weight = self._correctionsUL['get_ele_trig_weight'][self._year]
        get_ele_trig_err    = self._correctionsUL['get_ele_trig_err'][self._year]
        get_pho_trig_weight = self._correctionsUL['get_pho_trig_weight'][self._year]
        get_pho_tight_id_sf = self._correctionsUL['get_pho_tight_id_sf'][self._year]
        get_pho_csev_sf = self._correctionsUL['get_pho_csev_sf'][self._year]
        get_ecal_bad_calib = self._correctionsUL['get_ecal_bad_calib']
#        Jetevaluator = self._corrections['Jetevaluator'] ###

        isLooseElectron = self._ids['isLooseElectron']
        isTightElectron = self._ids['isTightElectron']
        isLooseMuon = self._ids['isLooseMuon']
        isTightMuon = self._ids['isTightMuon']
        isLooseTau = self._ids['isLooseTau']
        isLoosePhoton = self._ids['isLoosePhoton']
        isTightPhoton = self._ids['isTightPhoton']
        isGoodJet = self._ids['isGoodJet']
        isGoodFatJet = self._ids['isGoodFatJet']
        isHEMJet = self._ids['isHEMJet']

        match = self._common['match']
        # to calculate photon trigger efficiency
        sigmoid = self._common['sigmoid']
        deepflavWPs = self._common['btagWPs']['deepflav'][self._year]
        deepcsvWPs = self._common['btagWPs']['deepcsv'][self._year]

        ###
        # Derive jet corrector for JEC/JER
        ###

#        if not isData:
#            if (self._year == '2016'):
#                if ("preVFP" in dataset):
#                    list_of_jec = self._jec[self._year]['apv']
#                    list_of_jr = self._jr[self._year]['apv']
#                    list_of_junc = self._junc[self._year]['apv']
#                    list_of_jersf = self._jersf[self._year]['apv']
#                else:
#                    list_of_jec = self._jec[self._year]['no_apv']
#                    list_of_jr = self._jr[self._year]['no_apv']
#                    list_of_junc = self._junc[self._year]['no_apv']
#                    list_of_jersf = self._jersf[self._year]['no_apv']
#
#                for name in list_of_jec:
#                    jec_names.append(name)
#                for name in list_of_jr:
#                    jec_names.append(name)
#                for name in list_of_junc:
#                    jec_names.append(name)
#                for name in list_of_jersf:
#                    jec_names.append(name)
#            else:
#                for name in self._jec[self._year]:
#                    jec_names.append(name)
#                for name in self._jr[self._year]:
#                    jec_names.append(name)
#                for name in self._junc[self._year]:
#                    jec_names.append(name)
#                for name in self._jersf[self._year]:
#                    jec_names.append(name)
#
#        #if not isData:
#            print('jec names ', jec_names)
#            jec_inputs = {name: Jetevaluator[name] for name in jec_names}
#            print('jec inputs ', jec_inputs)
#            jec_stack = JECStack(jec_inputs)
#
#            jec_name_map = jec_stack.blank_name_map
#            jec_name_map = {
#                'JetPt': 'pt',
#                'JetMass': 'mass',
#                'JetEta': 'eta',
#                'JetA': 'area',
#                'ptGenJet': 'pt_gen',
#                'ptRaw': 'pt_raw',
#                'massRaw': 'mass_raw',
#                'Rho': 'rho',
#                'METpt': 'pt',
#                'METphi': 'phi',
#                'JetPhi': 'phi',
#                'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
#                'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
#            }
#
#            jet_factory = CorrectedJetsFactory(jec_name_map, jec_stack)
#            j["T"] = ak.zip({"pt": j.pt, "phi": j.phi},
#                        with_name="PolarTwoVector",
#                        behavior=vector.behavior
#            )
#            j['p4'] = ak.zip({
#                "pt": j.pt,
#                "eta": j.eta,
#                "phi": j.phi,
#                "mass": j.mass},
#                with_name="PtEtaPhiMLorentzVector",
#            )
#            j["pt_raw"] =  j.pt*(1 - j.rawFactor)
#            j["mass_raw"] =  j.mass*(1 - j.rawFactor)
#            j["pt_gen"] = ak.values_astype(
#                ak.fill_none(j.matched_gen.pt, 0), np.float32
#            )
#            j["pt"] = j.pt_raw
#            j["mass"] = j.mass_raw
#            j["p4","pt"] = j.pt_raw
#            j["p4","mass"] = j.mass_raw
#            j["rho"] = ak.broadcast_arrays( events.fixedGridRhoFastjetAll, j.pt)[0]
#
#            events_cache = events.caches[0]
#            met_factory = CorrectedMETFactory(jec_name_map)
#            corrected_jets = jet_factory.build(j, lazy_cache=events_cache)
#            corrected_mets = met_factory.build(events.MET,corrected_jets, {})
#
#            shift_syst = None
#            if shift_syst == "JESUp":
#                j = corrected_jets['JES_jes'].up
#            elif shift_syst == "JESDown":
#                j = corrected_jets['JES_jes'].down
#            elif shift_syst == "JERUp":
#                j = corrected_jets['JER'].up
#            elif shift_syst == "JERDown":
#                j = corrected_jets['JER'].down
#            elif shift_syst == "METunclUp":
#                met = corrected_mets.MET_UnclusteredEnergy.up
#                j   = events.Jet
#            elif shift_syst == "METunclDown":
#                met = corrected_mets.MET_UnclusteredEnergy.down
#                j   = events.Jet
#            else:
#                j   = corrected_jets
#                met = corrected_mets

        ###
        # Initialize global quantities (MET ecc.)
        ###

        met = events.MET
        met["T"] = ak.zip({"pt": met.pt, "phi": met.phi}, 
                          with_name="PolarTwoVector", 
                          behavior=vector.behavior)
        calomet = events.CaloMET

        XY_correctedmet_pt , XY_correctedmet_phi = METXYCorr_Met_MetPhi(events, self._year, met["T"]['pt'], met["T"]['phi'])
        print('XY met pt: ', XY_correctedmet_pt, len(XY_correctedmet_pt))
        print('XY met phi:', XY_correctedmet_phi, len(XY_correctedmet_phi))

        corrected_met = met
        corrected_met["T"] = ak.zip({"pt": XY_correctedmet_pt, "phi": XY_correctedmet_phi},
                            with_name="PolarTwoVector",
                            behavior=vector.behavior)
        ###
        # Initialize physics objects
        ###

        mu = events.Muon
        mu['isloose'] = isLooseMuon(mu.pt, mu.eta, mu.pfIsoId, mu.looseId, self._year, events.Muon.isPFcand, events.Muon.isGlobal, events.Muon.isTracker, mu.pfIsoId)
        mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.pfIsoId, mu.tightId, self._year, events.Muon.isPFcand, events.Muon.isGlobal, events.Muon.isTracker, mu.pfIsoId)
        mu["T"] = ak.zip({"pt": mu.pt, "phi": mu.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        mu['p4'] = ak.zip({
                            "pt": mu.pt,
                            "eta": mu.eta,
                            "phi": mu.phi,
                            "mass": mu.mass},
                            with_name="PtEtaPhiMLorentzVector",
        )

        mu_loose = mu[ak.values_astype(mu.isloose, np.bool)]
        mu_tight = mu[ak.values_astype(mu.istight, np.bool)]

        mu_ntot = ak.num(mu, axis=1)
        mu_nloose = ak.num(mu_loose, axis=1)
        mu_ntight = ak.num(mu_tight, axis=1)
        leading_mu = mu_tight[:,:1]
        leading_mu["T"] = ak.zip({"pt": leading_mu.pt, "phi": leading_mu.phi},
                with_name = "PolarTwoVector",
                behavior=vector.behavior)
        
        e = events.Electron
        event_size = len(events)
        print('event_size: ', event_size)
        
        e['isclean'] = ak.all(e.metric_table(mu_loose) >= 0.3, axis=-1)
        e['isloose'] = isLooseElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['istight'] = isTightElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e["T"] = ak.zip({"pt": e.pt, "phi": e.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        e['p4'] = ak.zip({
                            "pt": e.pt,
                            "eta": e.eta,
                            "phi": e.phi,
                            "mass": e.mass},
                            with_name="PtEtaPhiMLorentzVector",
        )
        e_clean = e[ak.values_astype(e.isclean, np.bool)]
        #e_loose = e_clean[ak.values_astype(e_clean.isloose, np.bool)]
        #e_loose = e[ak.values_astype(e_clean.isloose, np.bool)]
        e_loose = e[ak.values_astype(e.isloose, np.bool)]
        e_tight = e_clean[ak.values_astype(e_clean.istight, np.bool)]

        e_ntot = ak.num(e, axis=1)
        e_nloose = ak.num(e_loose, axis=1)
        e_ntight = ak.num(e_tight, axis=1)

        leading_e = e_tight[:,:1]
        leading_e["T"] = ak.zip({"pt": leading_e.pt, "phi": leading_e.phi},
                with_name = "PolarTwoVector",
                behavior=vector.behavior)
        
        tau = events.Tau
        tau['isclean'] = ak.all(tau.metric_table(mu_loose) > 0.4, axis=-1) & ak.all(tau.metric_table(e_loose) > 0.4, axis=-1)
        try:
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayMode,tau.idDeepTau2017v2p1VSjet,self._year)
        except:
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayModeOldDMs,tau.idDeepTau2017v2p1VSjet,self._year)
        else: 
            tau['isloose']=isLooseTau(tau.pt,tau.eta,tau.idDecayModeNewDMs,tau.idDeepTau2017v2p1VSjet,self._year)

        tau_clean = tau[ak.values_astype(tau.isclean, np.bool)]
        tau_loose = tau_clean[ak.values_astype(tau_clean.isloose, np.bool)]

        tau_ntot = ak.num(tau, axis=1)
        tau_nloose = ak.num(tau_loose, axis=1)

        pho = events.Photon
        pho['isclean'] = ak.all(pho.metric_table(mu_loose) >= 0.5, axis=-1) & ak.all(pho.metric_table(e_loose) >= 0.5, axis=-1)
        _id = 'cutBased'
        if self._year == '2016':
            _id = 'cutBased'
        #pho['isloose'] = isLoosePhoton(pho.pt, pho.eta, pho[_id], self._year) & (pho.electronVeto)  # added electron veto flag
        pho['isloose'] = isLoosePhoton(pho.pt, pho.eta, pho[_id], self._year)  ## no electronveto version
        pho['istight'] = isTightPhoton(pho.pt, pho[_id], self._year) & (pho.isScEtaEB) & (pho.electronVeto)  # tight photons are barrel only

        pho["T"] = ak.zip({"pt": pho.pt, "phi": pho.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
    
        pho_clean = pho[ak.values_astype(pho.isclean, np.bool)]
        pho_loose = pho_clean[ak.values_astype(pho_clean.isloose, np.bool)]
        pho_tight = pho_clean[ak.values_astype(pho_clean.istight, np.bool)]

        pho_ntot = ak.num(pho,axis=1)
        pho_nloose = ak.num(pho_loose, axis=1)
        pho_ntight = ak.num(pho_tight, axis=1)
        leading_pho = pho_tight[:,:1]
        #leading_pho = pho[:,:1] #new way to define leading photon
        #leading_pho = leading_pho[ak.values_astype(leading_pho.isclean, np.bool)]
        
        #leading_pho = leading_pho[ak.values_astype(leading_pho.istight, np.bool)]

        if not isData:
            gen = events.GenPart
            top = events.GenPart[abs(events.GenPart.pdgId) == 6]
        
        ak8 = events.FatJet
        ak8['TvsQCD'] = events.FatJet['particleNet_TvsQCD']
        ak8['p4'] = ak.zip({
            "pt"  : ak8.pt,
            "eta" : ak8.eta,
            "phi" : ak8.phi,
            "mass": ak8.mass},
            with_name="PtEtaPhiMCollection",
        )

        ak8EleMask = ak.all(ak8.p4.metric_table(e_loose) > 1.5, axis=-1)
        ak8MuMask = ak.all(ak8.p4.metric_table(mu_loose) > 1.5, axis=-1)
        ak8PhoMask = ak.all(ak8.p4.metric_table(pho_loose) > 1.5, axis=-1)

        ak8_isclean_mask = (ak8MuMask & ak8EleMask & ak8PhoMask)
        ak8_isgood_mask = isGoodFatJet(ak8.pt, ak8.eta, ak8.jetId)
        ak8_good_clean = ak8[ak8_isclean_mask & ak8_isgood_mask]
        ak8_clean = ak8[ak8_isclean_mask]
        ak8_nclean = ak.num(ak8_clean)
        ak8_good = ak8[ak8_isgood_mask]
        ak8_ngood = ak.num(ak8_good)
        leading_ak8 = ak8[:,:1]
        leading_ak8_good = ak8[ak8_isgood_mask][:,:1]
        leading_ak8_ngood = ak.num(leading_ak8_good)

        fj = events.AK15PFPuppi
        fj['pt'] = events.AK15PFPuppi['Jet_pt']
        fj['phi'] = events.AK15PFPuppi['Jet_phi']
        fj['eta'] = events.AK15PFPuppi['Jet_eta']
        fj['mass'] = events.AK15PFPuppi['Jet_mass']
        fj["T"] = ak.zip({"pt": fj.Jet_pt, "phi": fj.Jet_phi},
                                with_name="PolarTwoVector",
                                behavior=vector.behavior)
        fj['p4'] = ak.zip({
            "pt"  : fj.Jet_pt,
            "eta" : fj.Jet_eta,
            "phi" : fj.Jet_phi,
            "mass": fj.Jet_mass},
            with_name="PtEtaPhiMCollection",
        )
        fj['sd'] = ak.zip({
            "pt"  : fj.Subjet_pt,
            "phi" : fj.Subjet_phi,
            "eta" : fj.Subjet_eta,
            "mass": fj.Subjet_mass},
            with_name="PtEtaPhiMCollection",
        )
        fj['probQCDothers'] = events.AK15PFPuppi['Jet_particleNetAK15_QCDothers']
        fj['probQCDb'] = events.AK15PFPuppi['Jet_particleNetAK15_QCDb']
        fj['probQCDbb'] = events.AK15PFPuppi['Jet_particleNetAK15_QCDbb']
        fj['probQCDc'] = events.AK15PFPuppi['Jet_particleNetAK15_QCDc']
        fj['probQCDcc'] = events.AK15PFPuppi['Jet_particleNetAK15_QCDcc']
        fj['probTbcq'] = events.AK15PFPuppi['Jet_particleNetAK15_Tbcq']
        fj['probTbqq'] = events.AK15PFPuppi['Jet_particleNetAK15_Tbqq']
        fj['probQCDothers'] = events.AK15PFPuppi['Jet_particleNetAK15_QCDothers']
        probQCD=fj.probQCDbb+fj.probQCDcc+fj.probQCDb+fj.probQCDc+fj.probQCDothers
        probT=fj.probTbcq+fj.probTbqq
        fj['TvsQCD'] = probT/(probT+probQCD)

        fjEleMask = ak.all(fj.p4.metric_table(e_loose) > 1.5, axis=-1)
        fjMuMask = ak.all(fj.p4.metric_table(mu_loose) > 1.5, axis=-1)
        fjPhoMask = ak.all(fj.p4.metric_table(pho_loose) > 1.5, axis=-1)

        fj_isclean_mask = (fjMuMask & fjEleMask & fjPhoMask)
        fj_isgood_mask = isGoodFatJet(fj.pt, fj.eta, fj.Jet_jetId)

        fj_good = fj[fj_isgood_mask]
        fj_clean = fj[fj_isclean_mask]
        fj_good_clean = fj[fj_isclean_mask & fj_isgood_mask]
        fj_nclean = ak.num(fj_clean)

        fj_ngood = ak.num(fj_good)
        leading_fj = fj_good_clean[:,:1]
        leading_fj_good = fj[fj_isgood_mask][:,:1] # original one: good & clean ak15jet
        leading_fj_ngood = ak.num(leading_fj_good)
        
#        notuse = False
#        #if not isData: # For gen top, top eff
#        if notuse: # For gen top, top eff
#            qFromW = gen[
#                (abs(gen.pdgId) < 5) & # 1: d, 2: u, 3: s, 4: c
#                gen.hasFlags(['fromHardProcess', 'isFirstCopy']) &
#                (abs(gen.distinctParent.pdgId) == 24)  # 24: W
#            ]
#            bFromTop = gen[
#                (abs(gen.pdgId) == 5) & # 5: b
#                gen.hasFlags(['fromHardProcess','isFirstCopy']) & 
#                (gen.distinctParent.pdgId == 6) # 6: t
#            ]
#            qFromWFromTop = qFromW[qFromW.distinctParent.distinctParent.pdgId == 6]
#            qWmatch = ak.to_numpy(ak.num(qFromWFromTop) > 0)
#
#            leading_Lak8_top = ak.cartesian({"ak8jet":ak8.p4[:,:1], "gentop": top})
#            leading_DeltaR_Lak8_top = abs(leading_Lak8_top.ak8jet.delta_r(leading_Lak8_top.gentop))
#            ak8_TvsQCD_dR = ak.cartesian({"tvsqcd": leading_ak8.particleNet_TvsQCD, "dr": leading_DeltaR_Lak8_top})
#            ak8_leading_passdr = ak.all(leading_DeltaR_Lak8_top < 0.8, axis=-1)
#
#            leading_LJ_top = ak.cartesian({"ak15jet":fj.p4[:,:1], "gentop": top})
#            leading_DeltaR_LJ_top = abs(leading_LJ_top.ak15jet.delta_r(leading_LJ_top.gentop))
#            TvsQCD_dR = ak.cartesian({"tvsqcd": leading_fj.Jet_particleNetAK15_TvsQCD, "dr": leading_DeltaR_LJ_top})
#            leading_passdr = ak.all(leading_DeltaR_LJ_top < 0.8, axis=-1)


        j = events.Jet
        #j['isgood'] = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j['isHEM'] = isHEMJet(j.pt, j.eta, j.phi)
        j['isdcsvL'] = (j.btagDeepB>deepcsvWPs['loose'])
        j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose'])
        j['isdflvM'] = (j.btagDeepFlavB > deepflavWPs['medium']) ## from Rishab
        j['isdcsvM'] = (j.btagDeepB > deepcsvWPs['medium']) ## from Rishabh

        j["T"] = ak.zip({"pt": j.pt, "phi": j.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        j['p4'] = ak.zip({
        "pt": j.pt,
        "eta": j.eta,
        "phi": j.phi,
        "mass": j.mass},
        with_name="PtEtaPhiMLorentzVector",
        )
        
#         https://coffeateam.github.io/coffea/modules/coffea.nanoevents.methods.vector.html#
#         j['ptRaw'] =j.pt * (1-j.rawFactor)
#         j['massRaw'] = j.mass * (1-j.rawFactor)
#         j['rho'] = j.pt.ones_like()*events.fixedGridRhoFastjetAll.array

#         j_dflvL = j_clean[j_clean.isdflvL.astype(np.bool)]
        #jetMuMask = ak.all(j.metric_table(mu_loose) > 0.4, axis=-1)
        #jetEleMask = ak.all(j.metric_table(e_loose) > 0.4, axis=-1)
        #jetPhoMask = ak.all(j.metric_table(pho_loose) > 0.4, axis=-1)
        #jetisoMask = ak.all(j.metric_table(fj_clean) > 1.5, axis=-1)

        #j_isclean_mask = (jetMuMask & jetEleMask & jetPhoMask)
        j_isgood_mask = isGoodJet(j["T"]["pt"], j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)

        #j_good_clean = j[j_isclean_mask & j_isgood_mask]
        j_good = j[j_isgood_mask]
        jetMuMask = ak.all(j_good.metric_table(mu_loose) > 0.4, axis=-1)
        jetEleMask = ak.all(j_good.metric_table(e_loose) > 0.4, axis=-1)
        jetPhoMask = ak.all(j_good.metric_table(pho_loose) > 0.4, axis=-1)
        j_good_clean = j_good[jetMuMask & jetEleMask & jetPhoMask]

        jetisoMask = ak.all(j_good_clean.metric_table(fj_clean) > 1.5, axis=-1)
        j_iso = j_good_clean[jetisoMask]

        #j_iso = j[j_isclean_mask & j_isgood_mask & jetisoMask]
        j_niso = ak.num(j_iso, axis=1)
        j_ngood_clean = ak.num(j_good_clean, axis=1)

        #j_good_clean_dflvB = j_good_clean.isdflvM
        #atleat_two_flvjet = ak.num(j_good_clean_dflvB)>=2
        #j_good_clean_dflvB = j_good_clean_dflvB[atleat_two_flvjet]

        #j_iso_dflvL = j_iso.isdflvL
        #j_ndflvL = ak.num(j[j_iso_dflvL])

        j_iso_dflvL = j_iso[ak.values_astype(j_iso.isdflvL, np.bool)]
        j_ndflvL = ak.num(j_iso_dflvL, axis=1)
        j_iso_dflvM = j_iso[ak.values_astype(j_iso.isdflvM, np.bool)]
        j_ndflvM = ak.num(j_iso_dflvM, axis=1)

        #j_ndflvM = ak.num(j[j_good_clean_dflvB])
        leading_j = j_good_clean[:,:1] # new way to define leading jet
        j_HEM = j[ak.values_astype(j.isHEM, np.bool)]       
        j_nHEM = ak.num(j_HEM, axis=1)

        #atleast_one_jet_with_pt_grt_50 = ((ak.num(j_good_clean)>=1) & ak.any(j_good_clean.pt>=50, axis=-1))

        # *****btag
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X#Supported_Algorithms_and_Operati
        # medium     0.4184
#         btagWP_medium = 0.4184
#         Jet_btag_medium = j_clean[j_clean['btagDeepB'] > btagWP_medium]
        ###
        # Calculating derivatives
        ###

        # ************ calculate delta phi( leading ak4jet, met) > 1.5***********
########################################################
##  calculate delta phi( leading ak4jet, met) > 1.5
########################################################

        met_LJ_Pairs = ak.cartesian({"met":met, "lj": leading_j})
        Delta_Phi_Met_LJ_var = (abs(met_LJ_Pairs.met.delta_phi(met_LJ_Pairs.lj)))
        Delta_Phi_Met_LJ = (abs(met_LJ_Pairs.met.delta_phi(met_LJ_Pairs.lj))>1.5)

#######################################################
##  calculate deltaR( leading ak4jet, e/mu) < 3.4
#######################################################

        LJ_Ele = ak.cartesian({"leading_j":leading_j, "e_loose": e_loose})
        DeltaR_LJ_Ele = abs(LJ_Ele.leading_j.delta_r(LJ_Ele.e_loose))
        
        DeltaR_LJ_Ele_mask = ak.any(DeltaR_LJ_Ele < 3.4, axis=-1)

        LJ_Mu = ak.cartesian({"leading_j":leading_j, "mu_loose": mu_loose})
        DeltaR_LJ_Mu = abs(LJ_Mu.leading_j.delta_r(LJ_Mu.mu_loose))
        
        DeltaR_LJ_Mu_mask = ak.any(DeltaR_LJ_Mu < 3.4, axis=-1)

#################################################
##  Calculate leading dilepton
#################################################
        ele_pairs = ak.cartesian({"one_e":e_loose[:,:1], "two_e":e_loose[:,1:2]})
        diele = ele_pairs.one_e + ele_pairs.two_e

        mu_pairs = ak.cartesian({"one_mu":mu_loose[:,:1], "two_mu":mu_loose[:,1:2]})
        dimu = mu_pairs.one_mu + mu_pairs.two_mu

#################################################

        ###
        # Calculate Recoil
        ###
        recoil_e = corrected_met + leading_e
        recoil_m = corrected_met + leading_mu

        mete_j = ak.cartesian({"mete": recoil_e, "j": j_good_clean})
        metm_j = ak.cartesian({"metm": recoil_m, "j": j_good_clean})
        mete_fj = ak.cartesian({"mete": recoil_e, "fj": leading_fj})
        metm_fj = ak.cartesian({"metm": recoil_m, "fj": leading_fj})

        u = { # recoil
            #'sr': met + np.ones_like(leading_e),
            'sr': met + np.zeros_like(met), # + np.zeros_like(leading_e),
            'wmcr': recoil_m,
            'tmcr': recoil_m,
            'wecr': recoil_e,
            'tecr': recoil_e,
            'zmcr': recoil_m,
            'zecr': recoil_e,
            'gcr': met
        }

        mT = {
            'sr': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'wmcr': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
            'tmcr': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
            'wecr': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'tecr': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'zmcr': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
            'zecr': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'gcr': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu))))
        }

        ###
        # Calculating weights
        ###
        if not isData:

            ###
            # JEC/JER
            ###

            #gen = events.GenPart
            ###
            # Fat-jet top matching at decay level
            ###
#            qFromW_ = gen[
#                (abs(gen.pdgId) < 4) &
#                gen.hasFlags(['fromHardProcess', 'isFirstCopy']) &
#                (abs(gen.distinctParent.pdgId) == 24)
#            ]
#            cFromW = gen[
#                (abs(gen.pdgId) == 4) &
#                gen.hasFlags(['fromHardProcess', 'isFirstCopy']) &
#                (abs(gen.distinctParent.pdgId) == 24)
#            ]
#            bFromTop = gen[(
#                abs(gen.pdgId) == 5) & 
#                gen.hasFlags(['fromHardProcess','isFirstCopy']) & 
#                (gen.distinctParent.pdgId == 6)
#            ]
#            qFromWFromTop = qFromW[qFromW.distinctParent.distinctParent.pdgId == 6]
#            qFromWFromTop['p4'] = ak.zip({
#                "pt"  : qFromWFromTop.pt,
#                "eta" : qFromWFromTop.eta,
#                "phi" : qFromWFromTop.phi,
#                "mass": qFromWFromTop.mass}, 
#                with_name="PtEtaPhiMCollection",)
#            jetgenWq = ak.cartesian({'fj': fj.sd, 'qWT': qFromWFromTop.p4})
#            dr_jetgenWq = jetgenWq.fj.delta_r(jetgenWq.qWT)

#            def tbqqmatch(topid, dR=1.5):
#                qFromWFromTop = qFromW[qFromW.distinctParent.distinctParent.pdgId == topid]
#                bFromTop = gen[
#                    (abs(gen.pdgId) == 5) &
#                    gen.hasFlags(['fromHardProcess','isFirstCopy']) &
#                    (gen.distinctParent.pdgId == topid)
#                ]
#                jetgenWq = fj.sd.cross(qFromWFromTop, nested=True)

            gen['isb'] = (abs(gen.pdgId) == 5) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])

            gen['isc'] = (abs(gen.pdgId) == 4) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])

            gen['isTop'] = (abs(gen.pdgId) == 6) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            ttjet_weights = np.ones(event_size)
            if('TTJets' in dataset):
                ttjet_weights = np.sqrt(get_ttbar_weight(genTops.pt.sum()) * get_ttbar_weight(genTops.pt.sum()))

            gen['isW'] = (abs(gen.pdgId) == 24) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isZ'] = (abs(gen.pdgId) == 23) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isA'] = (abs(gen.pdgId) == 22) & gen.hasFlags(['isPrompt', 'fromHardProcess', 'isLastCopy']) & (gen.status == 1)

            ###
            # Calculating gen photon dynamic isolation as in https://arxiv.org/pdf/1705.04664.pdf
            ###
                
            epsilon_0_dyn = 0.1
            n_dyn = 1
            gen['R_dyn'] = (91.1876/(gen.pt * np.sqrt(epsilon_0_dyn))) * ak.values_astype((gen.isA), np.int) + (-999)*ak.values_astype((~gen.isA), np.int)
            gen['R_0_dyn'] = gen.R_dyn * ak.values_astype((gen.R_dyn < 1.0), np.int) + ak.values_astype((gen.R_dyn >= 1.0), np.int)   
#
            genWs = gen[gen.isW & (gen.pt > 100)]
            genZs = gen[gen.isZ & (gen.pt > 100)]
            genDYs = gen[gen.isZ & (gen.mass > 30)]
#            # Based on photon weight distribution
#            genIsoAs = gen[gen.isIsoA & (gen.pt > 100)]

            nnlo_nlo = {}
            nlo_qcd = np.ones(event_size)
            nlo_ewk = np.ones(event_size)
#             if('GJets' in dataset):
#                 if self._year == '2016':
#                     nlo_qcd = get_nlo_qcd_weight['a'](genIsoAs.pt.max())
#                     nlo_ewk = get_nlo_ewk_weight['a'](genIsoAs.pt.max())
#                 for systematic in get_nnlo_nlo_weight['a']:
#                     nnlo_nlo[systematic] = get_nnlo_nlo_weight['a'][systematic](genIsoAs.pt.max())*ak.values_astype((ak.num(genIsoAs,axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genIsoAs, axis=1) > 0), np.int)

            if ('WJetsToLNu' in dataset) & ('HT' in dataset):
                nlo_qcd = get_nlo_qcd_weight['w'](ak.max(genWs.pt, axis=1))
                nlo_ewk = get_nlo_ewk_weight['w'](ak.max(genWs.pt, axis=1))
                for systematic in get_nnlo_nlo_weight['w']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['w'][systematic](ak.max(genWs.pt))*ak.values_astype((ak.num(genWs,axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genWs, axis=1) > 0), np.int)

            elif('DY' in dataset):
                nlo_qcd = get_nlo_qcd_weight['dy'](ak.max(genDYs.pt, axis=1))
                nlo_ewk = get_nlo_ewk_weight['dy'](ak.max(genDYs.pt, axis=1))
                for systematic in get_nnlo_nlo_weight['dy']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['dy'][systematic](ak.max(genDYs.pt))*ak.values_astype((ak.num(genZs, axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genZs, axis=1) > 0), np.int)
            elif('Z1Jets' in dataset or 'Z2jets' in dataset):
                nlo_qcd = get_nlo_qcd_weight['z'](ak.max(genZs.pt, axis=1))
                nlo_ewk = get_nlo_ewk_weight['z'](ak.max(genZs.pt, axis=1))
                for systematic in get_nnlo_nlo_weight['z']:
                    nnlo_nlo[systematic] = get_nnlo_nlo_weight['z'][systematic](ak.max(genZs.pt))*ak.values_astype((ak.num(genZs, axis=1) > 0), np.int) + ak.values_astype(~(ak.num(genZs, axis=1) > 0), np.int)

            ###
            # Calculate PU weight and systematic variations
            ###

            pu = get_pu_nom_weight(events.Pileup.nTrueInt)
            #pu = get_pu_weight(events.Pileup.nTrueInt)

            ###
            # Trigger efficiency weight
            ###

#             e1sf = get_ele_trig_weight(leading_ele_pair.i0.eta.sum()+leading_ele_pair.i0.deltaEtaSC.sum(), leading_ele_pair.i0.pt.sum())*(leading_ele_pair.i0.pt.sum() > 40).astype(np.int)
#             e2sf = get_ele_trig_weight(leading_ele_pair.i1.eta.sum()+leading_ele_pair.i1.deltaEtaSC.sum(), leading_ele_pair.i1.pt.sum())*(leading_ele_pair.i1.pt.sum() > 40).astype(np.int)

#             if self._year == '2016':
#                 sf = get_pho_trig_weight(leading_pho.pt.sum())
#             elif self._year == '2017':  # Sigmoid used for 2017 and 2018, values from monojet
#                 sf = sigmoid(leading_pho.pt.sum(), 0.335, 217.91, 0.065, 0.996) / sigmoid(leading_pho.pt.sum(), 0.244, 212.34, 0.050, 1.000)
#                 sf[np.isnan(sf) | np.isinf(sf)] == 1
#             elif self._year == '2018':
#                 sf = sigmoid(leading_pho.pt.sum(), 1.022, 218.39, 0.086, 0.999) / sigmoid(leading_pho.pt.sum(), 0.301, 212.83, 0.062, 1.000)
#                 sf[np.isnan(sf) | np.isinf(sf)] == 1

            trig = {
                'sr': get_met_trig_weight(met.pt),
                #'wmcr': get_met_trig_weight(ak.sum(recoil_m.phi, axis= -1), ak.sum(recoil_m.pt, axis= -1)),
                #'tmcr': get_met_trig_weight(ak.sum(recoil_m.phi, axis= -1), ak.sum(recoil_m.pt, axis= -1)),
                'wmcr': get_met_trig_weight(ak.sum(recoil_m.pt, axis= -1)),
                'tmcr': get_met_trig_weight(ak.sum(recoil_m.pt, axis= -1)),
                'wecr': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'tecr': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'zmcr': np.ones_like(get_met_trig_weight(met.pt)),
                'zecr': np.ones_like(get_met_trig_weight(met.pt)),
                'gcr': np.ones_like(get_met_trig_weight(met.pt))
            }
            print('ak.sum(recoil.pt)', ak.sum(recoil_m.pt, axis= -1))
            print('trig weight', get_met_trig_weight(ak.sum(recoil_m.phi, axis= -1), ak.sum(recoil_m.pt, axis= -1)))
            trig_err = {
                'sr': np.ones_like(trig['sr']),
                'wmcr': get_mu_trig_err(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
                'tmcr': get_mu_trig_err(abs(ak.sum(leading_mu.eta, axis= -1)), ak.sum(leading_mu.pt, axis=-1)),
                'wecr': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'tecr': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'zmcr': np.ones_like(trig['zmcr']),
                'zecr': np.ones_like(trig['zmcr']),
                'gcr': np.ones_like(trig['gcr'])
            }

            ###
            # Calculating electron and muon ID weights
            ###

            if self._year == '2016':
                sf = get_pho_tight_id_sf(leading_pho.eta, leading_pho.pt)
            else:  # 2017/2018 monojet measurement depends only on abs(eta)
                sf = get_pho_tight_id_sf(abs(leading_pho.eta))

            ids = {
                'sr': np.ones(event_size),
                'wmcr': get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'tmcr': get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wecr': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'tecr': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'zmcr': get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),  ##!
                'zecr': get_ele_tight_id_sf(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),  ##!
                'gcr': sf
            }
            print('id sf', get_mu_tight_id_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)))
            ids_err = {
                'sr': np.ones_like(ids['sr']),
                'wmcr': get_mu_tight_id_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'tmcr': get_mu_tight_id_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wecr': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'tecr': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'zmcr': get_mu_tight_id_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)), ##!
                'zecr': get_ele_tight_id_err(ak.sum(leading_e.eta, axis=-1), ak.sum(leading_e.pt, axis=-1)),  ##!
                'gcr': np.ones_like(ids['gcr'])
            }
            ###
            # Reconstruction weights for electrons
            ###

            # 2017 has separate weights for low/high pT (threshold at 20 GeV)
            def ele_reco_sf(pt, eta): 
                return get_ele_reco_sf(eta, pt)*ak.values_astype((pt > 20), np.int) + get_ele_reco_lowet_sf(eta, pt)*ak.values_astype((~(pt > 20)), np.int)
            
            def ele_reco_err(pt, eta):
                return get_ele_reco_err(eta, pt)*ak.values_astype((pt > 20), np.int) + get_ele_reco_lowet_err(eta, pt)*ak.values_astype((~(pt > 20)), np.int)

            #look at this rRISHABH
            if self._year == '2017' or self._year == '2018' or self._year == '2016':
                sf = ele_reco_sf
            else:
                sf = get_ele_reco_sf

            reco = {
                'sr': np.ones(event_size),
                'wmcr': np.ones(event_size),
                'tmcr': np.ones(event_size),
                'wecr': sf(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'tecr': sf(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'zmcr': np.ones(event_size),
                'zecr': sf(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'gcr': np.ones(event_size)
            }
            print('reco sf',np.ones(event_size))
            reco_err = {
                'sr':   np.ones_like(reco['sr']),
                'wmcr': np.ones_like(reco['wmcr']),
                'tmcr': np.ones_like(reco['tmcr']),
                'wecr': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'tecr': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'zmcr': np.ones_like(reco['zmcr']),
                'zecr': ele_reco_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt, axis=-1)),
                'gcr':  np.ones_like(reco['gcr'])
            }
            ###
            # Isolation weights for muons
            ###

            isolation = {
                'sr': np.ones(event_size),
                'wmcr': get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'tmcr': get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wecr': np.ones(event_size),
                'tecr': np.ones(event_size),
                'zmcr': get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'zecr': np.ones(event_size),
                'gcr': np.ones(event_size)
            }
            print('iso sf', get_mu_tight_iso_sf(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)))
            isolation_err = {
                'sr':   np.ones_like(isolation['sr']),
                'wmcr': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'tmcr': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wecr': np.ones_like(isolation['wecr']),
                'tecr': np.ones_like(isolation['tecr']),
                'zmcr': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'zecr': np.ones_like(isolation['zecr']),
                'gcr':  np.ones_like(isolation['gcr'])
            }
            ###
            # CSEV weight for photons: https://twiki.cern.ch/twiki/bin/view/CMS/EgammaIDRecipesRun2#Electron_Veto_CSEV_or_pixel_seed
            ###

#             if self._year == '2016':
#                 csev_weight = get_pho_csev_sf(abs(ak.sum(leading_pho.eta, axis=-1)), ak.sum(leading_pho.pt))
#             elif self._year == '2017':
#                 csev_sf_index = 0.5*(ak.values_astype(ak.sum(leading_pho.isScEtaEB, axis=-1), np.int))+3.5*(~(ak.values_astype(ak.sum(leading_pho.isScEtaEB, axis=-1), np.int)))+1*(ak.values_astype(ak.sum(leading_pho.r9, axis=-1) > 0.94), np.int)+2*(ak.values_astype(ak.sum(leading_pho.r9, axis=-1) <= 0.94), np.int)
#                 csev_weight = get_pho_csev_sf(csev_sf_index)
#             elif self._year == '2018':
#                 csev_weight = get_pho_csev_sf(ak.sum(leading_pho.pt, axis=-1), abs(ak.sum(leading_pho.eta, axis=-1)))
#             csev_weight[csev_weight == 0] = 1



            ###
            # AK4 b-tagging weights
            ###
            '''
            if in a region you are asking for 0 btags, you have to apply the 0-btag weight
            if in a region you are asking for at least 1 btag, you need to apply the -1-btag weight

            it’s “-1” because we want to reserve “1" to name the weight that should be applied when you ask for exactly 1 b-tag

            that is different from the weight you apply when you ask for at least 1 b-tag
            '''
            print('j good clean', j_good_clean, ak.type(j_good_clean), type(j_good_clean))
            print('get_deepflav weight', get_deepflav_weight['loose'])

            btag = {}
            btagUp = {}
            btagDown = {}
            
            btag['sr'], btagUp['sr'], btagDown['sr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '0', )  
            btag['wmcr'], btagUp['wmcr'], btagDown['wmcr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '0', )  
            #btag['tmcr'], btagUp['tmcr'], btagDown['tmcr'] = get_deepflav_weight['loose'](j_good_clean, j_good_clean["T"]["pt"], j_good_clean.eta, j_good_clean.hadronFlavour, '-1', )  
            btag['wecr'], btagUp['wecr'], btagDown['wecr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '0', )  
            btag['tecr'], btagUp['tecr'], btagDown['tecr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '-1', )  
            #btag['zmcr'], btagUp['zmcr'], btagDown['zmcr'] = np.ones_like(btag['sr']), np.ones_like(btag['sr']), np.ones_like(btag['sr'])  
            #btag['zecr'], btagUp['zecr'], btagDown['zecr'] = np.ones_like(btag['sr']), np.ones_like(btag['sr']), np.ones_like(btag['sr'])  
            #btag['gcr'], btagUp['gcr'], btagDown['gcr'] = np.ones_like(btag['sr']), np.ones_like(btag['sr']), np.ones_like(btag['sr'])  

            btag['tmcr'], btagUp['tmcr'], btagDown['tmcr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '-1', )  


        ###
        # Selections
        ###

        met_filters = np.ones(event_size, dtype=np.bool)
        # this filter is recommended for data only
        if isData:
            met_filters = met_filters & events.Flag['eeBadScFilter']
        for flag in AnalysisProcessor.met_filter_flags[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters', ak.to_numpy(met_filters, np.bool))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._met_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('met_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singleelectron_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_electron_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singlephoton_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_photon_triggers', ak.to_numpy(triggers))

        triggers = np.zeros(event_size, dtype=np.bool)
        for path in self._singlemuon_triggers[self._year]:
            if path not in events.HLT.fields:
                continue
            triggers = triggers | events.HLT[path]
        selection.add('single_muon_triggers', ak.to_numpy(triggers))

        noHEMj = np.ones(event_size, dtype=np.bool)
        if self._year == '2018':
            noHEMj = (j_nHEM == 0)

        noHEMmet = np.ones(event_size, dtype=np.bool)
        if self._year == '2018':
            noHEMmet = (met.pt > 470) | (met.phi > -0.62) | (met.phi < -1.62)

        '''
        what the next 6 lines of code do:

        main object is to exclude events from JetHt sample with W_pT b/w 70-100 GeV

        events.metadata['dataset'] = 'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8____27_'
        dataset = 'WJetsToLNu'

        see if the 'HT' is in the name of the sample
        so, it first goes to genpart,
        figures out if the genlevel process is hardprocess and firstcopy and there are genlevel particle with
        abs(pdgID)= 24

        ad selects only those events for the pT of W was > 100 GeV

        '''

        # predeclration just in cas I don't want the filter
        # selection.add("exclude_low_WpT_JetHT", np.full(len(events), True))
#         if ('WJetsToLNu' in dataset) & (events.metadata['dataset'].split('-')[0].split('_')[1] == 'HT'):

#             GenPart = events.GenPart
#             remove_overlap = (GenPart[GenPart.hasFlags(['fromHardProcess', 'isFirstCopy', 'isPrompt']) &
#                                       ((abs(GenPart.pdgId) == 24))].pt > 100).all()
#             selection.add("exclude_low_WpT_JetHT", remove_overlap)

#         else:
#             selection.add("exclude_low_WpT_JetHT", np.full(event_size, True))
        if ('WJetsToLNu' in dataset) & ('Pt' in dataset):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 24))].pt >= 200)
            selection.add("exclude_wjets_greater_200", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_wjets_greater_200", np.full(event_size, True))
            
        if ('WJetsToLNu' in dataset) & (not ('Pt' in dataset)):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 24))].pt < 200)
            selection.add("exclude_wjets_less_200", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_wjets_less_200", np.full(event_size, True))

        if ('DYJetsToLL' in dataset) & (not ('Pt' in dataset)):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 23))].pt < 400)
            selection.add("exclude_dyjets_less_400", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_dyjets_less_400", np.full(event_size, True))

        if ('DYJetsToLL' in dataset) & ('Pt' in dataset):
            remove_overlap = (gen[gen.hasFlags(['fromHardProcess', 'isFirstCopy']) & ((abs(gen.pdgId) == 23))].pt >= 400)
            selection.add("exclude_dyjets_greater_400", ak.to_numpy(ak.sum(remove_overlap, axis=1)>0))
        else:
            selection.add("exclude_dyjets_greater_400", np.full(event_size, True))

        selection.add('DeltaR_LJ_mask',ak.to_numpy(DeltaR_LJ_Ele_mask | DeltaR_LJ_Mu_mask))
        selection.add('isoneM', ak.to_numpy((e_nloose == 0) & (mu_ntight == 1) & ( mu_nloose == 1) ))
        selection.add('isoneM_noE', ak.to_numpy((e_nloose == 0)))
        selection.add('isoneM_onetightMu', ak.to_numpy((mu_ntight == 1)))
        selection.add('isoneM_onelooseMu', ak.to_numpy(( mu_nloose == 1)))
        selection.add('isoneM_notau',ak.to_numpy((tau_nloose == 0)))
        selection.add('isoneM_nopho',ak.to_numpy((pho_nloose == 0)))
        selection.add('isoneE', ak.to_numpy((e_ntight == 1) & (e_nloose == 1) & (mu_nloose == 0) & (tau_nloose == 0) & (pho_nloose == 0)))
        selection.add('onelooseEle', ak.to_numpy((e_nloose == 1)))
        selection.add('onetightEle', ak.to_numpy((e_ntight == 1)))
        selection.add('nolooseMu', ak.to_numpy((mu_ntight == 0)))
        selection.add('iszeroL', ak.to_numpy((tau_nloose == 0) & (pho_nloose == 0) & (e_nloose == 0) & (mu_nloose == 0)))

        selection.add('exactly_1_medium_btag', ak.to_numpy(j_ndflvM == 1))
        selection.add('atleast_2_medium_btag', ak.to_numpy(j_ndflvM >= 2))
        selection.add('zero_medium_btags', ak.to_numpy(j_ndflvM == 0))

        selection.add('leading_e_pt', ak.to_numpy(ak.sum(e_loose.pt, axis=1) > 40)) ##

        selection.add('noHEMj', ak.to_numpy(noHEMj))
        selection.add('noHEMmet', ak.to_numpy(noHEMmet))
        selection.add('met80', ak.to_numpy(met.pt < 80))
        selection.add('met100', ak.to_numpy(met.pt > 100))
        selection.add('met120', ak.to_numpy(met.pt < 120))
        selection.add('Delta_Phi_Met_LJ', ak.to_numpy(ak.sum(abs(met_LJ_Pairs.met.delta_phi(met_LJ_Pairs.lj))>1.5, axis=1)>0))
        selection.add('DeltaR_LJ_Ele_mask', ak.to_numpy((DeltaR_LJ_Ele_mask)>0))

        selection.add('one_muon', ak.to_numpy(ak.num(mu_tight, axis=1) == 1))
        selection.add('zero_loose_electron', ak.to_numpy(ak.num(e_loose, axis=1) == 0))
        selection.add('DeltaR_LJ_Mu_mask', ak.to_numpy((DeltaR_LJ_Mu_mask)>0))

        selection.add('fj_clean', ak.to_numpy(ak.num(fj_clean)>0))

        selection.add('fj_ngood_clean', ak.to_numpy(ak.num(fj_good_clean)>0))

        selection.add('fatjet_good_leading', ak.to_numpy(leading_fj_ngood > 0))

        selection.add('fatjet_leading', ak.to_numpy(ak.num(leading_fj) > 0)) # good clean leading fatjet
        selection.add('noextrab', ak.to_numpy(j_ndflvL==0))
        selection.add('extrab', ak.to_numpy(j_ndflvL>0))

        selection.add('leading_j>70',ak.to_numpy(ak.sum(leading_j.pt, axis=1) >70))# from the monotop paper
        selection.add('fatjetpt160',ak.to_numpy(ak.sum(leading_fj.pt, axis=1) >= 160))# from kit
        #selection.add('atleast_one_jet_with_pt_grt_50',ak.to_numpy(atleast_one_jet_with_pt_grt_50))

        selection.add('dPhi_recoil_j_e', ak.to_numpy(ak.sum(abs(mete_j.mete.delta_phi(mete_j.j))>0.5, axis=1)>0))
        selection.add('dPhi_recoil_j_m', ak.to_numpy(ak.sum(abs(metm_j.metm.delta_phi(metm_j.j))>0.5, axis=1)>0))
        #selection.add('dPhi_recoil_fj_e', ak.to_numpy(abs(mete_fj.mete.delta_phi(mete_fj.fj))>1.5))
        #selection.add('dPhi_recoil_fj_m', ak.to_numpy(abs(metm_fj.metm.delta_phi(metm_fj.fj))>1.5))
        selection.add('dPhi_recoil_fj_e', ak.to_numpy(ak.sum(abs(mete_fj.mete.delta_phi(mete_fj.fj))>1.5, axis=1)>0))
        selection.add('dPhi_recoil_fj_m', ak.to_numpy(ak.sum(abs(metm_fj.metm.delta_phi(metm_fj.fj))>1.5, axis=1)>0))

        #selection.add('2bjet', ak.to_numpy(ak.num(j_good_clean_dflvB)>2))

        selection.add('recoil_tmcr', ak.to_numpy(ak.sum(u['tmcr'].pt, axis = 1) > 250)) # 250
        selection.add('recoil_tecr', ak.to_numpy(ak.sum(u['tecr'].pt, axis = 1) > 250)) # 250
        selection.add('recoil_wmcr', ak.to_numpy(ak.sum(u['wmcr'].pt, axis = 1) > 250)) # 250

        regions = {
            'sr': [ 'met_filters', 'met_triggers','fatjet_leading', 'fatjet_good_leading', 'iszeroL', 'noextrab', 'noHEMj', 'noHEMmet'],
            'wmcr': ['met_filters','met_triggers','isoneM_nopho','isoneM_onelooseMu','isoneM_onetightMu','isoneM_noE', 'fj_ngood_clean', 'recoil_wmcr','dPhi_recoil_fj_m', 'dPhi_recoil_j_m', 'noHEMj', 'noextrab'],
            ##'wmcr': ['met_filters','met_triggers','isoneM_nopho','isoneM_onelooseMu','isoneM_onetightMu','isoneM_noE','fj_ngood_clean'],

            'wecr': ['met_filters','single_electron_triggers','isoneE', 'fj_ngood_clean', 'dPhi_recoil_j_e', 'dPhi_recoil_fj_e', 'noextrab', 'met100','noHEMj'],
            ##'wecr': ['met_filters','single_electron_triggers','isoneM_nopho','onelooseEle','onetightEle','nolooseMu','fj_ngood_clean'],

            'tmcr': ['met_filters', 'met_triggers','isoneM_nopho','isoneM_onelooseMu','isoneM_onetightMu','isoneM_noE', 'fj_ngood_clean','recoil_tmcr', 'dPhi_recoil_fj_m','dPhi_recoil_j_m', 'noHEMj'],
            ##'tmcr': ['met_filters','met_triggers','isoneM_nopho','isoneM_onelooseMu','isoneM_onetightMu','isoneM_noE','fj_ngood_clean'],

            'tecr': ['met_filters', 'single_electron_triggers','isoneE', 'fj_ngood_clean','recoil_tecr','dPhi_recoil_fj_e', 'dPhi_recoil_j_e', 'noHEMj','extrab', 'met100'],
            ##'tecr': ['met_filters','single_electron_triggers','isoneM_nopho','onelooseEle','onetightEle','nolooseMu','fj_ngood_clean'],

            'zmcr': ['isoneM', 'fatjet', 'noHEMj', 'met_filters', 'met_triggers'],
            'zecr': ['isoneE', 'fatjet', 'noHEMj', 'met_filters', 'single_electron_triggers'],
            'gcr': ['isoneE', 'fatjet', 'noHEMj', 'met_filters',]
        }

        isFilled = False
        for region, cuts in regions.items():
            if region not in selected_regions: continue
            print('Considering region:', region)
            print('cuts: ', cuts)
            if region == 'sr' or  region == 'gcr' or region == 'zecr' or region == 'zmcr':
                print('only tt, w CR')
#            if not region == 'tmcr':
#                print('only ttbar mu region')
                continue

            ###
            # Adding recoil and minDPhi requirements
            ###

            CaloMinusPfOverRecoil = abs(calomet.pt - met.pt) / u[region].pt
#            if region == 'sr':
#                selection.add('recoil_'+region, ak.to_numpy(u[region].pt > 250)) # 250
#                selection.add('calo_'+region, ak.to_numpy(CaloMinusPfOverRecoil < 0.5))
#            else:
#                selection.add('recoil_'+region, ak.to_numpy(ak.sum(u[region].pt, axis = 1) > 250)) # 250
#                selection.add('calo_'+region, ak.to_numpy(ak.sum(CaloMinusPfOverRecoil, axis = 1) < 0.5))
#            regions[region].insert(7, 'recoil_'+region)
            #regions[region].insert(6, 'calo_'+region)
            print('region after add: ', regions)
            # regions[region].update({'recoil_'+region,'mindphi_'+region})
            variables = {

                'recoil':                 u[region].pt,
                'eT_miss':              met.pt,
                'metphi':                 met.phi,
                'ele_pT':              e_tight.pt,
                'ele_phi'     : leading_e.phi,
                'ele_eta'     : leading_e.eta,
                'mu_pT':              mu_tight.pt,
                'mu_phi' : leading_mu.phi,
                'mu_eta' : leading_mu.eta,
                'ak8phi':                  leading_ak8.phi,
                'ak8pt':                  leading_ak8.pt,
                'ak8eta':                  leading_ak8.eta,
                'ak8_tvsqcd':                  leading_ak8.particleNet_TvsQCD,
                'dphi_Met_LJ':             Delta_Phi_Met_LJ_var,
                'j1pt':                   leading_j.pt,
                'j1eta':                  leading_j.eta,
                'j1phi':                  leading_j.phi,
                'fj1pt':                   leading_fj.pt,
                'fj1eta':                  leading_fj.eta,
                'fj1phi':                  leading_fj.phi,
                #'drfjtop':              leading_DeltaR_LJ_top,
                # 'njets':                  j_nclean,
                # 'ndflvL':                 j_ndflvL,
                # 'ndcsvL':     j_ndcsvL,
                # 'dielemass' : leading_diele.mass,
                # 'dielept'   : leading_diele.pt,
                # 'dimumass' : leading_dimu.mass,
                'dphi_e_etmiss':          abs(met.delta_phi(leading_e)),
                'dphi_mu_etmiss':         abs(met.delta_phi(leading_mu)),
                'ndflvM':                 j_ndflvM,
                'njets':                  j_ngood_clean,
                'nfatjets':                  fj_nclean,
                'dr_e_lj': DeltaR_LJ_Ele,
                'dr_mu_lj': DeltaR_LJ_Mu,
                'TvsQCD':                 leading_fj.TvsQCD,
                }
            if region in mT:
                variables['mT'] = mT[region]
#                 print(mT[region])
#                 if 'e' in region[-1]:
#                     WRF = leading_e.T.sum()-met.T
#                 else:
#                     pass
#                     WRF = leading_mu.T.sum()-met.T
#                 variables['recoilphiWRF'] = abs(u[region].delta_phi(WRF))
#             print('Variables:', variables.keys())

            def fill(dataset, weight, cut):
                #print('in fill weight[cut]', weight[cut])

                flat_variables = {k: ak.flatten(v[cut], axis=None) for k, v in variables.items()}
                flat_weight = {k: ak.flatten(~np.isnan(v[cut])*weight[cut], axis=None) for k, v in variables.items()}

                for histname, h in hout.items():
                    #print('fill hist: ', histname, h)
                    if not isinstance(h, hist.Hist):
                        continue
                    if histname not in variables:
                        print('histname not in variables: ', histname)
                        continue
                    elif histname == 'sumw':
                        continue
                    elif histname == 'template':
                        continue
#                     elif histname == 'scale_factors':
#                         flat_variable = {histname: flat_weight[histname]}
#                         h.fill(dataset=dataset,
#                                region=region,
#                                **flat_variable)

                    else:
                        flat_variable = {histname: flat_variables[histname]}
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable,
                               weight=flat_weight[histname])

            if isData:
                if not isFilled:
                    hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                    isFilled = True
                weights = Weights(len(events))
                cut = selection.all(*regions[region])
                hout['template'].fill(dataset=dataset,
                                      region=region,
                                      systematic='nominal',
                                      recoil = ak.sum(u[region].pt, axis=-1),
                                      weight=np.ones(event_size)*cut)
                vcut=np.zeros(event_size, dtype=np.int)
                noweight = np.ones_like(weights.weight())
                #hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=noweight)
                hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=np.ones(event_size))
                allcuts = set()
                for i, icut in enumerate(cuts):
                    allcuts.add(icut)
                    jcut = selection.all(*allcuts)
                    vcut = (i+1)*jcut
                    #print('i', i)
                    #print('icut', icut)
                    #hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=noweight*jcut)
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=np.ones(event_size)*jcut)
                fill(dataset, np.ones(event_size), cut)

#############################################################
###             Print The Events (in if isData loop)
#############################################################
#                allcuts_ = set()
#                print('cuts line: ', cuts)
#                for j, jcut in enumerate(cuts):
#                    if 'dPhi' in jcut: break
#                    print('consider cut ', jcut)
#                    allcuts_.add(jcut)
#                kcut = selection.all(*allcuts_)
#                print('kcut', kcut, len(kcut))
#                selected_met = met[kcut]
#                print('met after kcut: ', selected_met.pt, len(selected_met.pt))
#
#                print('Check Jet ID')
#                allcuts__ = set()
#                for jj, jjcut in enumerate(cuts): # Check only met_filter to fj
#                    print('jj and jjcut', jj, jjcut)
#                    if jj > 6: continue
#                    allcuts__.add(jjcut)
#                kkcut = selection.all(*allcuts__)
#
#                tmp_counts = 0
#                for i in range(event_size):
#                    if kkcut[i]:
#                        tmp = False
#                        for j in range(len(fj.Jet_jetId[i])):
#                            if fj.Jet_jetId[i][j] < 6:
#                                #print('(',events.event[i],') ak15 id ', fj.Jet_jetId[i])
#                                tmp = True
#                        if tmp: tmp_counts = tmp_counts+1
#                        #print('jet id', fj.Jet_jetId[i], fj.Jet_jetId[i]&6, (fj.Jet_jetId[i]&6)==6)
#                print("Number of less than 6 ID events After cuts: ", tmp_counts)
#                for n in range(200, 1000):
#                    #print('(',n,') events')
#                    if kcut[n]:
#                        print('(',n,') events after selection')
#                        print('event number -->', events.event[n])
#                        print(' met pt :', XY_correctedmet_pt[n])
#                        print(' met phi:', XY_correctedmet_phi[n])
#                        print(' recoil :', u[region].pt[n])
#                        print(' ak15jet pt :', fj.pt[n])
#                        print(' ak15jet phi:', fj.phi[n])
#                        print(' ak15jet eta:', fj.eta[n])
#                        print(' clean ak15jet pt :', fj_good_clean.pt[n])
#                        print(' clean ak15jet phi :', fj_good_clean.phi[n])
#                        print(' clean ak15jet eta :', fj_good_clean.eta[n])
#                        print('----------Next event-----------')
################################################################
###
################################################################

            else: # If not isData (for mc)
                weights = Weights(len(events))
                if 'L1PreFiringWeight' in events.fields:
                    weights.add('prefiring', events.L1PreFiringWeight.Nom)
                weights.add('genw', events.genWeight)
                #weights.add('nlo_qcd', nlo_qcd)
                #www#weights.add('nlo_ewk', nlo_ewk)
                #www#weights.add('ttjet_weights', ttjet_weights)
                #www#if 'cen' in nnlo_nlo:
                    #www#weights.add('nnlo_nlo', nnlo_nlo['cen'])
                    #www#weights.add('qcd1', np.ones(event_size), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
                    #www#weights.add('qcd2', np.ones(event_size), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
                    #www#weights.add('qcd3', np.ones(event_size), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
                    #www#weights.add('ew1', np.ones(event_size), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
                    #www#weights.add('ew2G', np.ones(event_size), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
                    #www#weights.add('ew3G', np.ones(event_size), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
                    #www#weights.add('ew2W', np.ones(event_size), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
                    #www#weights.add('ew3W', np.ones(event_size), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
                    #www#weights.add('ew2Z', np.ones(event_size), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
                    #www#weights.add('ew3Z', np.ones(event_size), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
                    #www#weights.add('mix', np.ones(event_size), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
                    #www#weights.add('muF', np.ones(event_size), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
                    #www#weights.add('muR', np.ones(event_size), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
                #www#weights.add('pileup', pu)
                
                trig_name = str()
                ids_name = str()
                reco_name = str()
                isolation_name = str()
                if 'e' in region:
                    trig_name = 'trig_e'
                elif 'm' in region:
                    trig_name = 'trig_m'
                print('trig: ', trig[region])
                #www#weights.add(trig_name, trig[region],trig[region]+trig_err[region], trig[region]-trig_err[region])
                if 'e' in region:
                    ids_name = 'id_e'
                elif 'm' in region:
                    ids_name = 'id_m'                
                print('ids: ', ids[region])
                #www#weights.add(ids_name, ids[region], ids[region]+ids_err[region], ids[region]-ids_err[region])
                
                if 'e' in region:
                    reco_name = 'reco_e'
                elif 'm' in region:
                    reco_name = 'reco_m'
                print('reco: ', reco[region])
                #www#weights.add(reco_name, reco[region], reco[region]+reco_err[region], reco[region]-reco_err[region])
                
                if 'e' in region:
                    isolation_name = 'isolation_e'
                elif 'm' in region:
                    isolation_name = 'isolation_m'           
                print('isolation ', isolation[region])
                #www#weights.add(isolation_name, isolation[region], isolation[region]+isolation_err[region], isolation[region]-isolation_err[region])
#                 #www#weights.add('csev', csev[region])
                #www#weights.add('btag', btag[region],btagUp[region], btagDown[region])

                if 'WJets' in dataset or 'DY' in dataset or 'Z1Jets' in dataset or 'G1Jet' in dataset:
                    if not isFilled:
                        print('events.genWeight: ',events.genWeight)
                        print('len evnets.genWeight: ',len(events.genWeight))
                        print('ak.sum(events.genWeight): ', ak.sum(events.genWeight))
                        hout['sumw'].fill(dataset='HF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        hout['sumw'].fill(dataset='LF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        isFilled = True
                    whf = ak.values_astype(((ak.num(gen[gen.isb],axis=1) > 0) | (ak.num(gen[gen.isc], axis=1) > 0)), np.int)
                    wlf = ak.values_astype(~(ak.values_astype(whf,np.bool)), np.int)
                    cut = ak.to_numpy(selection.all(*regions[region]))
                    if 'WJets' in dataset:
                        systematics = [None,
#www#                                    'btagUp',
#www#                                    'btagDown',
#www#                                       trig_name+'Up', trig_name+'Down',
#www#                                       ids_name+'Up', ids_name+'Down',
#www#                                       reco_name+'Up', reco_name+'Down',
#www#                                       isolation_name+'Up', isolation_name+'Down',
                                      ]
                    else:
                        systematics = [None,
#www#                                       'btagUp',
#www#                                       'btagDown',
#www#                                       'qcd1Up',
#www#                                       'qcd1Down',
#www#                                       'qcd2Up',
#www#                                       'qcd2Down',
#www#                                       'qcd3Up',
#www#                                       'qcd3Down',
#www#                                       'muFUp',
#www#                                       'muFDown',
#www#                                       'muRUp',
#www#                                       'muRDown',
#www#                                       'ew1Up',
#www#                                       'ew1Down',
#www#                                       'ew2GUp',
#www#                                       'ew2GDown',
#www#                                       'ew2WUp',
#www#                                       'ew2WDown',
#www#                                       'ew2ZUp',
#www#                                       'ew2ZDown',
#www#                                       'ew3GUp',
#www#                                       'ew3GDown',
#www#                                       'ew3WUp',
#www#                                       'ew3WDown',
#www#                                       'ew3ZUp',
#www#                                       'ew3ZDown',
#www#                                       'mixUp',
#www#                                       'mixDown',
#www#                                       trig_name+'Up', trig_name+'Down',
#www#                                       ids_name+'Up', ids_name+'Down',
#www#                                       reco_name+'Up', reco_name+'Down',
#www#                                       isolation_name+'Up', isolation_name+'Down',
                                      ]
                    for systematic in systematics:
                        sname = 'nominal' if systematic is None else systematic
                        hout['template'].fill(dataset='HF--'+dataset,
                                              region=region,
                                              systematic=sname,
                                              #mT = ak.sum(mT[region], axis=-1),
                                              recoil = ak.sum(u[region].pt, axis=-1),
                                              weight=weights.weight(modifier=systematic)*whf*cut)
    
                        hout['template'].fill(dataset='LF--'+dataset,
                                              region=region,
                                              systematic=sname,
                                              #mT = ak.sum(mT[region], axis=-1),
                                              recoil = ak.sum(u[region].pt, axis=-1),
                                              weight=weights.weight(modifier=systematic)*wlf*cut)
                    ## Cutflow loop
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*whf)
                    hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*wlf)
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*whf)
                        hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*wlf)
                    fill('HF--'+dataset, weights.weight()*whf, cut)
                    fill('LF--'+dataset, weights.weight()*wlf, cut)

                else: #if not WJ, DY, Z1, Z2, G1 in datasets
                    if not isFilled:
                        print('if not isFilled if not WJ DY Z G in datasets?')
                        hout['sumw'].fill(dataset=dataset, sumw=1, weight=ak.sum(events.genWeight, axis=-1))

                        isFilled = True
                    cut = selection.all(*regions[region])
#                     for systematic in [None]:
                    for systematic in [None,
#www#                                       'btagUp',
#www#                                       'btagDown',
#www#                                       trig_name+'Up', trig_name+'Down',
#www#                                       ids_name+'Up', ids_name+'Down',
#www#                                       reco_name+'Up', reco_name+'Down',
#www#                                       isolation_name+'Up', isolation_name+'Down',
                                      ]:
                        #print('systematic ', systematic, 'weight->', weights.weight(modifier=systematic), ak.sum(weights.weight(modifier=systematic)))
                        sname = 'nominal' if systematic is None else systematic
                        hout['template'].fill(dataset=dataset,
                                              region=region,
                                              systematic=sname,
                                              #mT = ak.sum(mT[region], axis=-1),
                                              recoil = ak.sum(u[region].pt, axis=-1),
                                              weight=weights.weight(modifier=systematic)*cut)
                        ## Cutflow loopweights.weight(modifier=systematic)*cut
                    vcut=np.zeros(event_size, dtype=np.int)
                    noweight = np.ones_like(weights.weight())
                    #hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=noweight)
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight())
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        #print('jcut', jcut)
                        #print('CUTS ', icut, '->', weights.weight()*jcut, ak.sum(weights.weight()*jcut))
                        hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight()*jcut)
                        #hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=noweight*jcut)
                        print("fill cutflow")
                        #allcuts.remove(icut)
                    #print('allcuts: ', allcuts)
                    print('weights.weight()', weights.weight())
                    print('SUM weights.weight()', ak.sum(weights.weight()))
                    fill(dataset, weights.weight(), cut)
#         time.sleep(0.5)
        return hout

    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
            print('Scaling:', d.name)
            dataset = d.name
            if '--' in dataset:
                dataset = dataset.split('--')[1]
            print('Cross section:', self._xsec[dataset])
            if self._xsec[dataset] != -1:
#                scale[d.name] = 1
                scale[d.name] = self._lumi*self._xsec[dataset]
                print('lumi * xsec: ', self._lumi, '*', self._xsec[dataset], '= ', scale[d.name])
            else:
                scale[d.name] = 1

        for histname, h in accumulator.items():
            if histname == 'sumw':
                continue
            if isinstance(h, hist.Hist):
                h.scale(scale, axis='dataset')

        return accumulator


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    (options, args) = parser.parse_args()

    #with open('metadata/onefile.json') as fin:
    with open('metadata/KIT_UL_2018_v4.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k, v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    correctionsUL = load('data/corrections_UL.coffea')
    correctionsBtag = load('data/correction_btag.coffea')
    ids = load('data/ids.coffea')
    common = load('data/common.coffea')

    processor_instance = AnalysisProcessor(year=options.year,
                                           xsec=xsec,
                                           corrections=corrections,
                                           correctionsUL=correctionsUL,
                                           correctionsBtag=correctionsBtag,
                                           ids=ids,
                                           common=common)

    save(processor_instance, 'data/UL_had_'+options.year+'_eventcheck_v1.processor')
    print("processor name: UL_had_{}_eventcheck_v1".format(options.year))
    #save(processor_instance, 'data/UL_had_ttag_cutflow_'+options.year+'_New_v6_fatjet.processor')
    #print("processor name: UL_had_ttag_cutflow_{}_New_v6_fatjet".format(options.year))

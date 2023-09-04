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

from XYcorrMET import *


class AnalysisProcessor(processor.ProcessorABC):

    lumis = {  # Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
        '2016':{'preVFP':19.52,'postVFP':16.81},
        '2017': 40.66,
        '2018': 59.74
    }

    met_filter_flags = {

        '2016': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter'
                 'BadPFMuonDzFilter',
                 #'eeBadScFilter'
                 ],

        '2017': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilter'
                 'BadPFMuonDzFilter',
                 #'eeBadScFilter',
                 'ecalBadCalibFilter'
                 ],

        '2018': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'BadPFMuonDzFilter',
                 #'eeBadScFilter',
                 'ecalBadCalibFilter'
                 ]
    }


    def __init__(self, year, xsec,corrections,  correctionsUL, ids, common, vfp):

        self._year = year
        if self._year == "2016":
            self._lumi = 1000.*float(AnalysisProcessor.lumis[year][vfp])
        else:
            self._lumi = 1000.*float(AnalysisProcessor.lumis[year])

        self._xsec = xsec

        self._samples = {
            'sr':('Z1Jets','Z2Jets','WJets','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','MET','Mphi'),
            'wmcr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','MET'),
            'tmcr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','MET'),
            'wecr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma'), 
            'tecr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma'), 
            'zmcr':('WJets','TT','ST','WW','WZ','ZZ','QCD','MET'), 
            'zecr':('WJets','Z1Jets','Z2Jets', 'G1Jet','DY','TT','ST','WW','WZ','ZZ','QCD','SingleElectron','EGamma'), 
            'gcr':('G1Jet','QCD','SinglePhoton','EGamma')
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
                'Ele27_WPTight_Gsf', 'Photon175'
            ],
            '2017': [
                'Ele35_WPTight_Gsf',
                'Photon200'
            ],
            '2018': [
                'Ele32_WPTight_Gsf',
                'Photon200'
            ]
        }
        self._singlemuon_triggers = {
            '2016': [
                'IsoMu24',
                'IsoTkMu24',
            ],
            '2017':
                [
                'IsoMu27',
            ],
            '2018':
                [
                'IsoMu24',
            ]
        }


        self._ids = ids
        self._common = common
        self._corrections = corrections
        self._correctionsUL = correctionsUL
        #self._correctionsBtag = correctionsBtag

        self._accumulator = processor.dict_accumulator({

            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])
            ),

            'template': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('recoil', 'Hadronic recoil', [250,310,370,470,590,840,1020,1250,3000]),
                hist.Bin('fjmass','AK15 Jet Mass', [40,50,60,70,80,90,100,110,120,130,150,160,180,200,220,240,300]),
                hist.Bin('TvsQCD','TvsQCD', [0, self._TvsQCDwp[self._year], 1])
            ),

            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('cut', 'Cut index', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
            ),

            'runnumber': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                #hist.Cat('select','Selct'),
                hist.Bin('cut', 'Cut index', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
                hist.Bin('runnumber', 'Run Number', 500000,0,500000)
            ),


            'recoil': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('recoil', 'Hadronic Recoil', [250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0, 3000])
            ),

            'eT_miss': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('eT_miss', '$E^T_{miss}$[GeV]', 20, 0, 1500)
            ),

            'metphi': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('metphi','MET phi',35,-3.5,3.5)
            ),

            'ele_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ele_pT', 'Tight electron $p_{T}$ [GeV]', 10, 0, 200)
            ),

            'mu_pT': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('mu_pT', 'Tight muon $p_{T}$ [GeV]', 10, 0, 200)
            ),

            'nphoton': hist.Hist(
                'Events',
                hist.Cat('dataset', 'dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('nphoton', 'Number of photon', 10, 0, 100)
            ),

            'j1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('j1pt','AK4 Leading Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),

            'fj1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('fj1pt','AK15 Leading Jet Pt',[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0])
            ),

            'TvsQCD': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('TvsQCD', 'TvsQCD', 15, 0., 1)
            )
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata['dataset']
            
        isData = 'genWeight' not in events.fields
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()

        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample not in dataset:
                    continue
                selected_regions.append(region)

        ###
        # Getting corrections, ids from .coffea files
        ###
        if ("preVFP" in dataset) and (self._year == '2016'):
            print('pass 2016 now')
        elif ("postVFP" in dataset) and (self._year == '2016'):
            print('pass 2016 now')
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
            get_met_trig_weight = self._correctionsUL['get_met_trig_weight'][self._year]
            get_met_trig_err = self._correctionsUL['get_met_trig_err'][self._year]
            get_ele_loose_id_err = self._correctionsUL['get_ele_loose_id_err_postVFP'][self._year]
            get_ele_tight_id_err = self._correctionsUL['get_ele_tight_id_err_postVFP'][self._year]
            get_mu_loose_id_err = self._correctionsUL['get_mu_loose_id_err_postVFP'][self._year]
            get_mu_tight_id_err = self._correctionsUL['get_mu_tight_id_err_postVFP'][self._year]

            get_pu_nom_weight = self._correctionsUL['get_pu_nom_weight_postVFP'][self._year]
            get_pu_up_weight = self._correctionsUL['get_pu_up_weight_postVFP'][self._year]
            get_pu_down_weight = self._correctionsUL['get_pu_down_weight_postVFP'][self._year]

            get_deepflav_weight = self._corrections['get_btag_weight_postVFP']['deepflav'][self._year]

        
        get_pho_tight_id_sf = self._correctionsUL['get_pho_tight_id_sf'][self._year]
        get_pho_csev_sf = self._correctionsUL['get_pho_csev_sf'][self._year]
        get_ecal_bad_calib = self._correctionsUL['get_ecal_bad_calib']
        get_nlo_qcd_weight = self._correctionsUL['get_nlo_qcd_weight'][self._year]
        get_nnlo_nlo_weight = self._correctionsUL['get_nnlo_nlo_weight'][self._year]
        get_ttbar_weight = self._correctionsUL['get_ttbar_weight']
        get_nlo_ewk_weight = self._correctionsUL['get_nlo_ewk_weight'][self._year]
        Jetevaluator = self._correctionsUL['Jetevaluator'] ###

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

        deepflavWPs = self._common['btagWPs']['deepflav'][self._year]


        event_size = len(events)

        met = events.MET

        met["T"] = ak.zip({"pt": met.pt, "phi": met.phi}, 
                          with_name="PolarTwoVector", 
                          behavior=vector.behavior)

        XY_correctedmet_pt , XY_correctedmet_phi = METXYCorr_Met_MetPhi(events, self._year, met["T"]['pt'], met["T"]['phi'])

        corrected_met = met
        corrected_met['pt'] = XY_correctedmet_pt
        corrected_met['phi'] = XY_correctedmet_phi
        corrected_met["T"] = ak.zip({"pt": XY_correctedmet_pt, "phi": XY_correctedmet_phi},
                            with_name="PolarTwoVector",
                            behavior=vector.behavior)


        mu = events.Muon
        mu['isloose'] = isLooseMuon(mu.pt, mu.eta, mu.pfIsoId, mu.looseId, self._year, mu.isPFcand, mu.isGlobal, mu.isTracker)
        mu['istight'] = isTightMuon(mu.pt, mu.eta, mu.pfIsoId, mu.tightId, self._year, mu.isPFcand, mu.isGlobal, mu.isTracker)
        mu["T"] = ak.zip({"pt": mu.pt, "phi": mu.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        n_mu = ak.num(mu)

        mu_loose = mu[ak.values_astype(mu.isloose, np.bool)]
        mu_tight = mu[ak.values_astype(mu.istight, np.bool)]
        mu_nloose = ak.num(mu_loose)
        mu_ntight = ak.num(mu_tight)

        leading_mu = mu_tight[:,:1]


        e = events.Electron
        e["T"] = ak.zip({"pt": e.pt, "phi": e.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)
        n_e = ak.num(e)

        #e['isclean'] = ak.all(e.metric_table(mu_loose) >= 0.3, axis=-1)
        e['isclean'] = ~(ak.any(e.metric_table(mu_loose) < 0.3, axis=2))
        e['isloose'] = isLooseElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)
        e['istight'] = isTightElectron(e.pt, e.eta+e.deltaEtaSC, e.dxy, e.dz, e.cutBased, self._year)

        e_clean = e[ak.values_astype(e.isclean, np.bool)]

        e_loose = e[ak.values_astype(e.isloose, np.bool)]
        e_tight = e[ak.values_astype(e.istight, np.bool)]
        ###e_tight = e_clean[ak.values_astype(e_clean.istight, np.bool)]
        
        e_nloose = ak.num(e_loose, axis=1)
        e_ntight = ak.num(e_tight, axis=1)

        leading_e = e_tight[:,:1]


        tau = events.Tau
        tau['isclean'] = ~(ak.any(tau.metric_table(mu_loose) < 0.4, axis=2)) & ~(ak.any(tau.metric_table(e_loose) < 0.4, axis=2))
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

        pho['isclean'] = ~(ak.any(pho.metric_table(mu_loose) < 0.4, axis=2)) & ~(ak.any(pho.metric_table(e_loose) < 0.4, axis=2))
        pho['isloose'] = isLoosePhoton(pho.pt, pho.eta, pho['cutBased'], self._year) & (pho.electronVeto)  ## no electronveto version
        pho['istight'] = isTightPhoton(pho.pt, pho['cutBased'], self._year) & (pho.isScEtaEB) & (pho.electronVeto)  # tight photons are barrel only
        pho_clean = pho[ak.values_astype(pho.isclean, np.bool)]
        pho_loose = pho_clean[ak.values_astype(pho_clean.isloose, np.bool)]
        pho_tight = pho_clean[ak.values_astype(pho_clean.istight, np.bool)]
        pho_n = ak.num(pho,axis=1)
        pho_nloose = ak.num(pho_loose, axis=1)
        pho_ntight = ak.num(pho_tight, axis=1)
        leading_pho = pho_tight[:,:1]


        fj = events.AK15PFPuppi
        fj['pt'] = events.AK15PFPuppi['Jet_pt']
        fj['phi'] = events.AK15PFPuppi['Jet_phi']
        fj['eta'] = events.AK15PFPuppi['Jet_eta']
        fj['mass'] = events.AK15PFPuppi['Jet_mass']

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
        nfj = ak.num(fj)

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

        fjEleMask = ak.any(fj.p4.metric_table(e_loose) < 1.5, axis=-1)
        fjMuMask  = ak.any(fj.p4.metric_table(mu_loose) < 1.5, axis=-1)
        fjPhoMask = ak.any(fj.p4.metric_table(pho_loose) < 1.5, axis=-1)

        fj_isclean_mask = (~fjMuMask & ~fjEleMask & ~fjPhoMask)
        fj_isgood_mask = isGoodFatJet(fj.pt, fj.eta, fj.Jet_jetId)

        fj_good = fj[fj_isgood_mask]
        fj_clean = fj[fj_isclean_mask]
        fj_good_clean = fj[fj_isclean_mask & fj_isgood_mask]
        fj_nclean = ak.num(fj_clean)
        fj_ngood = ak.num(fj_good)
        fj_ngood_clean = ak.num(fj_good_clean, axis=1)

        ak15Mask = ak.all(fj_good_clean.pt > 250, axis=-1)
        cutak15 = fj[fj_isclean_mask & fj_isgood_mask & ak15Mask]
        cutak15_leading = cutak15[:,:1]

        ak15qualityMask = (fj.Jet_chHEF>0.1)&(fj.Jet_neHEF<0.8)

        fj_leading = fj_good_clean[:,:1]


        j = events.Jet
        nj = ak.num(j)

        j["T"] = ak.zip({"pt": j.pt, "phi": j.phi}, 
                  with_name="PolarTwoVector", 
                  behavior=vector.behavior)

        j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose'])
        j['isdflvM'] = (j.btagDeepFlavB > deepflavWPs['medium']) ## from Rishab
        j['isHEM'] = isHEMJet(j.pt, j.eta, j.phi)
        j_isgood_mask = isGoodJet(j["T"]["pt"], j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j_good = j[j_isgood_mask]

        jetMuMask  = ak.any(j_good.metric_table(mu_loose) < 0.4, axis=-1)
        jetEleMask = ak.any(j_good.metric_table(e_loose) < 0.4, axis=-1)
        jetPhoMask = ak.any(j_good.metric_table(pho_loose) < 0.4, axis=-1)

        j_good_clean = j_good[~jetMuMask & ~jetEleMask] # & jetPhoMask]
        j_ngood_clean = ak.num(j_good_clean, axis=1)


        jetisoMask = ak.all(j_good_clean.metric_table(fj_good_clean) > 1.5, axis=-1)

        j_iso = j_good_clean[jetisoMask]
        j_niso = ak.num(j_iso, axis=1)

        j_leading = j_iso[:,:1]
        nj_leading = ak.num(j_leading)

        j_iso_dflvL = j_iso[ak.values_astype(j_iso.isdflvL, np.bool)]
        j_ndflvL = ak.num(j_iso_dflvL, axis=1)

        j_HEM = j[ak.values_astype(j.isHEM, np.bool)]       
        j_nHEM = ak.num(j_HEM, axis=1)


        recoil_e = corrected_met + leading_e
        recoil_m = corrected_met + leading_mu

        mete_j = ak.cartesian({"mete": recoil_e, "j": j_good_clean})
        metm_j = ak.cartesian({"metm": recoil_m, "j": j_good_clean})
        mete_fj = ak.cartesian({"mete": recoil_e, "fj": fj_leading})
        metm_fj = ak.cartesian({"metm": recoil_m, "fj": fj_leading})

        u = { # recoil
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
            'wmcr': np.sqrt(2*leading_mu.pt*corrected_met.pt*(1-np.cos(corrected_met.delta_phi(leading_mu)))),
            'tmcr': np.sqrt(2*leading_mu.pt*corrected_met.pt*(1-np.cos(corrected_met.delta_phi(leading_mu)))),
            'wecr': np.sqrt(2*leading_e.pt*corrected_met.pt*(1-np.cos(corrected_met.delta_phi(leading_e)))),
            'tecr': np.sqrt(2*leading_e.pt*corrected_met.pt*(1-np.cos(corrected_met.delta_phi(leading_e)))),
            'zmcr': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu)))),
            'zecr': np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.delta_phi(leading_e)))),
            'gcr': np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.delta_phi(leading_mu))))
        }



        #######
        ## Calculate weights
        #######
        if not isData:

            gen = events.GenPart

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

            genWs = gen[gen.isW & (gen.pt > 100)]
            genZs = gen[gen.isZ & (gen.pt > 100)]
            genDYs = gen[gen.isZ & (gen.mass > 30)]
#            # Based on photon weight distribution
#            genIsoAs = gen[gen.isIsoA & (gen.pt > 100)]

            nnlo_nlo = {}
            nlo_qcd = np.ones(event_size)
            nlo_ewk = np.ones(event_size)

            if ('WJetsToLNu' in dataset):
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

            #pu = np.minimum(10, GetPUSF(IOV, (events.Pileup.nTrueInt)))
            pu = get_pu_nom_weight(events.Pileup.nTrueInt)

            #######
            ## Trigger efficiency weight
            #######
            trig = {
                'sr': get_met_trig_weight(met.pt),
                #'wmcr': get_met_trig_weight(ak.sum(recoil_m.pt, axis= -1)),
                #'tmcr': get_met_trig_weight(ak.sum(recoil_m.pt, axis= -1)),
                'wmcr': np.ones(event_size),
                'tmcr': np.ones(event_size),
                'wecr': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'tecr': get_ele_trig_weight(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'zmcr': np.ones(event_size),
                'zecr': np.ones(event_size),
                'gcr':  np.ones(event_size)
            }
            trig_err = {
                'sr': np.ones_like(trig['sr']),
                #'wmcr': get_met_trig_err(ak.sum(recoil_m.pt, axis= -1)),
                #'tmcr': get_met_trig_err(ak.sum(recoil_m.pt, axis= -1)),
                'wmcr': np.ones(event_size),
                'tmcr': np.ones(event_size),
                'wecr': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'tecr': get_ele_trig_err(ak.sum(leading_e.eta+leading_e.deltaEtaSC, axis=-1), ak.sum(leading_e.pt,axis=-1)),
                'zmcr': np.ones(event_size),
                'zecr': np.ones(event_size),
                'gcr':  np.ones(event_size)
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
            isolation_err = {
                'sr':   np.zeros(event_size),
                'wmcr': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'tmcr': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'wecr': np.zeros(event_size),
                'tecr': np.zeros(event_size),
                'zmcr': get_mu_tight_iso_err(abs(ak.sum(leading_mu.eta, axis=-1)), ak.sum(leading_mu.pt, axis=-1)),
                'zecr': np.zeros(event_size),
                'gcr':  np.zeros(event_size)
            }


            ###
            # AK4 b-tagging weights
            ###
            '''
            if in a region you are asking for 0 btags, you have to apply the 0-btag weight
            if in a region you are asking for at least 1 btag, you need to apply the -1-btag weight

            it’s “-1” because we want to reserve “1" to name the weight that should be applied when you ask for exactly 1 b-tag

            that is different from the weight you apply when you ask for at least 1 b-tag
            '''

            btag = {}
            btagUp = {}
            btagDown = {}
            
            #btag['sr'], btagUp['sr'], btagDown['sr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '0', )  
            btag['sr'], btagUp['sr'], btagDown['sr'] = get_deepflav_weight['loose'](j_good_clean, j_good_clean.pt, j_good_clean.eta, j_good_clean.hadronFlavour, '0', )  
            btag['wmcr'], btagUp['wmcr'], btagDown['wmcr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '0', )  
            btag['tmcr'], btagUp['tmcr'], btagDown['tmcr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '-1', )  
            #btag['wecr'], btagUp['wecr'], btagDown['wecr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '0', )  
            #btag['tecr'], btagUp['tecr'], btagDown['tecr'] = get_deepflav_weight['loose'](j_iso, j_iso.pt, j_iso.eta, j_iso.hadronFlavour, '-1', )  
            #btag['zmcr'], btagUp['zmcr'], btagDown['zmcr'] = np.ones_like(btag['sr']), np.ones_like(btag['sr']), np.ones_like(btag['sr'])  
            #btag['zecr'], btagUp['zecr'], btagDown['zecr'] = np.ones_like(btag['sr']), np.ones_like(btag['sr']), np.ones_like(btag['sr'])  
            #btag['gcr'], btagUp['gcr'], btagDown['gcr'] = np.ones_like(btag['sr']), np.ones_like(btag['sr']), np.ones_like(btag['sr'])  

        ###
        # Selections
        ###

        met_filters = np.ones(event_size, dtype=np.bool)
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

        #print(events.run == 317626)
        #selection.add('RunNumber', ak.to_numpy(events.run == 317626)) ## Biggest events 
        selection.add('RunNumber', ak.to_numpy(events.run == 318877)) ## Biggest trigger efficiency event
        selection.add('isoneM', ak.to_numpy((e_nloose == 0) & (mu_ntight == 1) & ( mu_nloose == 1) ))
        selection.add('isoneE', ak.to_numpy((e_ntight == 1) & (e_nloose == 1) & (mu_nloose == 0)))
        selection.add('istwoM', ak.to_numpy(( mu_nloose == 2) & (e_nloose == 0)))
        selection.add('istwoE', ak.to_numpy(( mu_nloose == 0) & (e_nloose == 1)))
        selection.add('isoneG', ak.to_numpy((pho_ntight == 1) & (e_nloose == 0) & (mu_nloose == 0)))

        selection.add('iszeroG',ak.to_numpy((pho_nloose == 0) & (pho_ntight == 0)))
#        selection.add('iszeroL', ak.to_numpy((tau_nloose == 0) & (pho_nloose == 0) & (e_nloose == 0) & (mu_nloose == 0)))

######### For Cutflow one by one  #########
        selection.add('no_loose_pho', ak.to_numpy((pho_nloose == 0)))
        selection.add('one_loose_mu', ak.to_numpy((mu_nloose == 1)))
        selection.add('one_tight_mu', ak.to_numpy((mu_ntight == 1)))
        selection.add('one_loose_e', ak.to_numpy((e_nloose == 1)))
        selection.add('one_tight_e', ak.to_numpy((e_ntight == 1)))
        selection.add('no_loose_e', ak.to_numpy((e_nloose == 0)))
        selection.add('no_tight_e', ak.to_numpy((e_ntight == 0)))
        selection.add('no_loose_mu', ak.to_numpy((mu_nloose == 0)))
        selection.add('no_tight_mu', ak.to_numpy((mu_ntight == 0)))
        selection.add('no_loose_tau', ak.to_numpy(tau_nloose == 0))
        selection.add('one_clean_j', ak.to_numpy(j_ngood_clean > 0))
###########################################

        #selection.add('fatjet_leading', ak.to_numpy(ak.num(fj_leading) > 0)) # good clean leading fatjet
        selection.add('fatjet_leading', ak.to_numpy(ak.num(cutak15_leading) > 0)) # good clean leading fatjet
        selection.add('one_ak4', ak.to_numpy(j_ngood_clean > 0)) # good clean ak4jet
        selection.add('one_ak15', ak.to_numpy(fj_ngood_clean > 0)) # good clean ak4jet

        selection.add('leading_fj250', ak.to_numpy(ak.sum(fj_leading.pt, axis = 1)> 250)) # 250
        selection.add('fj_quality', ak.to_numpy((ak.sum(fj.Jet_chHEF, axis = 1)>0.1) & (ak.sum(fj.Jet_neHEF, axis = 1)<0.8))) # 250

        selection.add('recoil_tmcr', ak.to_numpy(ak.sum(u['tmcr'].pt, axis = 1) > 250)) # 250
        selection.add('recoil_wmcr', ak.to_numpy(ak.sum(u['wmcr'].pt, axis = 1) > 250)) # 250
        selection.add('recoil_tecr', ak.to_numpy(ak.sum(u['tecr'].pt, axis = 1) > 250)) # 250
        selection.add('recoil_wecr', ak.to_numpy(ak.sum(u['wecr'].pt, axis = 1) > 250)) # 250

        #dphi05 = ak.sum(abs(metm_j.metm.delta_phi(metm_j.j))>0.5, axis=1)
        #dphi05TF = ak.to_numpy(ak.sum(abs(metm_j.metm.delta_phi(metm_j.j))>0.5, axis=1)>0) 
        #cnt = 0
        #for i in range(len(dphi05)):
        #    if not dphi05[i] == 0:
        #        print('(',i,')',metm_j.metm.delta_phi(metm_j.j)[i], ak.min(abs(metm_j.metm.delta_phi(metm_j.j)[i])), dphi05TF[i])
        #        print('       ',ak.to_numpy(ak.min(abs(metm_j.metm.delta_phi(metm_j.j)), axis = 1)>0.5)[i])

        #        cnt = cnt +1
        #    if cnt == 100: break
        #selection.add('dPhi_recoil_j_e', ak.to_numpy(ak.sum(abs(mete_j.mete.delta_phi(mete_j.j))>0.5, axis=1)>0))
        #selection.add('dPhi_recoil_j_m', ak.to_numpy(ak.sum(abs(metm_j.metm.delta_phi(metm_j.j))>0.5, axis=1)>0))
        selection.add('dPhi_recoil_j_e', ak.to_numpy(ak.min(abs(mete_j.mete.delta_phi(mete_j.j)), axis = 1)>0.5))
        selection.add('dPhi_recoil_j_m', ak.to_numpy(ak.min(abs(metm_j.metm.delta_phi(metm_j.j)), axis = 1)>0.5))
        selection.add('dPhi_recoil_fj_e', ak.to_numpy(ak.sum(abs(mete_fj.mete.delta_phi(mete_fj.fj))>1.5, axis=1)>0))
        selection.add('dPhi_recoil_fj_m', ak.to_numpy(ak.sum(abs(metm_fj.metm.delta_phi(metm_fj.fj))>1.5, axis=1)>0))

        selection.add('noextrab', ak.to_numpy(j_ndflvL==0))
        selection.add('extrab', ak.to_numpy(j_ndflvL>0))
        selection.add('oneb', ak.to_numpy(j_ndflvL==1))

        selection.add('noHEMj', ak.to_numpy(noHEMj))

        selection.add('mt_tmcr', ak.to_numpy(ak.sum(mT['tmcr'], axis = 1) < 150))
        selection.add('mt_wmcr', ak.to_numpy(ak.sum(mT['wmcr'], axis = 1) < 150))
        selection.add('mt_tecr', ak.to_numpy(ak.sum(mT['tecr'], axis = 1) < 150))
        selection.add('mt_wecr', ak.to_numpy(ak.sum(mT['wecr'], axis = 1) < 150))

        selection.add('met150', ak.to_numpy(corrected_met.pt>150))


        regions = {
            'sr': [ 
                    'met_filters' , 'met_triggers', #'iszeroL',
                    ],

            'wmcr': [ 
                    'met_filters' ,'met_triggers',
                    'no_loose_pho','one_loose_mu', 'one_tight_mu', 'no_loose_e',
                    'one_ak4', 'one_ak15',
                    'leading_fj250',
                    'fj_quality',
                    'recoil_wmcr', 
                    'dPhi_recoil_fj_m', 'dPhi_recoil_j_m', 
                    'noextrab',
                    'met150',
                    'mt_wmcr',
                    ],

            'wecr': [
                    'met_filters', 'single_electron_triggers',#'isoneE', 'iszeroG', 
                    'no_loose_pho','one_loose_e', 'one_tight_e', 'no_loose_mu',
                    'one_ak4', 'one_ak15',
                    'leading_fj250',
                    'fj_quality',
                    'recoil_wecr', 
                    'dPhi_recoil_fj_e', 'dPhi_recoil_j_e', 
                    'noextrab',
                    'met150',
                    'mt_wecr',
                    ],

            'tmcr': [
                    'met_filters' , 'met_triggers',
                    'no_loose_pho', 'one_loose_mu', 'one_tight_mu','no_loose_e',
                    'one_ak4', 'one_ak15',
                    'leading_fj250',
                    'fj_quality',
                    'recoil_tmcr', 
                    'dPhi_recoil_fj_m', 'dPhi_recoil_j_m', 
                    'mt_tmcr',
                    ],

            'tecr': [
                    'met_filters', 'single_electron_triggers', #'isoneE', 'iszeroG',
                    'no_loose_pho', 'one_loose_e', 'one_tight_e','no_loose_mu',
                    'one_ak4', 'one_ak15',
                    'leading_fj250',
                    'fj_quality',
                    'recoil_tecr', 
                    'dPhi_recoil_fj_e', 'dPhi_recoil_j_e', 
                    'mt_tecr',
                    ],

            'zmcr': [
                    'met_filters' , 'met_triggers', #'istwoM', 'iszeroG',
                    ],

            'zecr': [
                    'met_filters' #, 'met_triggers','istwoE', 'iszeroG',
                    ],

            'gcr': [
                    'met_filters' #,'isoneG','iszeroG', 
                    ]
        }



        isFilled = False
        print('TotalEvents', event_size)
        for region, cuts in regions.items():
            if region not in selected_regions: continue
            if region == 'sr' or  region == 'gcr' or region == 'zecr' or region == 'zmcr':
#or region == 'wecr' or region == 'tecr':
                print('only tt, W muon CR')
                continue
            print('Considering region:', region, '--> events: ',event_size)

            variables = {

                'recoil':                 u[region].pt,
                'eT_miss':              corrected_met.pt,
                'ele_pT':              e_tight.pt,
                'mu_pT':              mu_tight.pt,
                'nphoton':              pho_nloose,
                'j1pt':                   j_leading.pt,
                'fj1pt':                   fj_leading.pt,
                'njets':                  nj,
                'nfatjets':                  nfj,
                'TvsQCD':                 fj_leading.TvsQCD
                }

            def fill(dataset, weight, cut):

                flat_variables = {k: ak.flatten(v[cut], axis=None) for k, v in variables.items()}
                print("v[cut", variables.items())
                flat_weight = {k: ak.flatten(~np.isnan(v[cut])*weight[cut], axis=None) for k, v in variables.items()}
                print("weight[cut]", weight[cut])

                for histname, h in hout.items():
                    if not isinstance(h, hist.Hist):
                        continue
                    if histname not in variables:
                        continue
                    elif histname == 'sumw':
                        continue
                    elif histname == 'template':
                        continue

                    else:
                        print("Fill histogram!!!!!!")
                        flat_variable = {histname: flat_variables[histname]}
                        h.fill(dataset=dataset,
                               region=region,
                               **flat_variable,
                               weight=flat_weight[histname])


            if isData:
                if not isFilled:
                    hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                    isFilled = True

                cut = selection.all(*regions[region])

                hout['template'].fill(dataset=dataset,
                                      region=region,
                                      systematic='nominal',
                                      recoil = ak.sum(u[region].pt, axis=-1),
                                      fjmass = ak.sum(fj.mass), 
                                      TvsQCD = ak.sum(fj.TvsQCD),
                                      weight = np.ones(event_size)*cut
                                      )
                #hout['runnumber'].fill(dataset=dataset,
                #                      region=region,
                #                      select='no_cut',
                #                      runnumber = events.run,
                #                      weight = np.ones(event_size)#*cut
                #                      )
                #hout['runnumber'].fill(dataset=dataset,
                #                      region=region,
                #                      select='cut',
                #                      runnumber = events.run,
                #                      weight = np.ones(event_size)*cut
                #                      )

                vcut=np.zeros(event_size, dtype=np.int)
                hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=np.ones(event_size))
                allcuts = set()
                for i, icut in enumerate(cuts):
                    allcuts.add(icut)
                    jcut = selection.all(*allcuts)
                    print('(',i,')', icut, jcut)
                    vcut = (i+1)*jcut
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=np.ones(event_size)*jcut)
                    hout['runnumber'].fill(dataset=dataset, region=region, cut=vcut, runnumber = events.run, weight=np.ones(event_size)*jcut)
                fill(dataset, np.ones(event_size), cut)


            else: # If not isData (for mc)
                print('This is MC part!')
                weights = Weights(len(events))
                print('weight.weight EMPTY', weights.weight())
                weights.add('none', np.ones(event_size))
                if 'L1PreFiringWeight' in events.fields:
                    weights.add('prefiring', events.L1PreFiringWeight.Nom)
                    print('weight.weight PreFiring', weights.weight())
                weights.add('genw', events.genWeight)
                print('weight.weight GENW', weights.weight())
                weights.add('nlo_ewk', nlo_ewk)
                print('weight.weighti NLO_EWK', weights.weight())
                weights.add('pileup', pu)
                print('weight.weight PU', weights.weight())

                if 'e' in region:
                    trig_name = 'trig_e'
                elif 'm' in region:
                    trig_name = 'trig_m'
                weights.add(trig_name, trig[region],trig[region]+trig_err[region], trig[region]-trig_err[region])
                print('weight.weight TRIG', weights.weight())

                if 'e' in region:
                    ids_name = 'id_e'
                elif 'm' in region:
                    ids_name = 'id_m'                
                weights.add(ids_name, ids[region], ids[region]+ids_err[region], ids[region]-ids_err[region])
                print('weight.weight ID', weights.weight())
                
                if 'e' in region:
                    reco_name = 'reco_e'
                elif 'm' in region:
                    reco_name = 'reco_m'
                weights.add(reco_name, reco[region], reco[region]+reco_err[region], reco[region]-reco_err[region])
                print('weight.weight RECO', weights.weight())
                
                if 'e' in region:
                    isolation_name = 'isolation_e'
                elif 'm' in region:
                    isolation_name = 'isolation_m'           
                weights.add(isolation_name, isolation[region], isolation[region]+isolation_err[region], isolation[region]-isolation_err[region])
                print('weight.weight ISO', weights.weight())

                #weights.add('btag', btag[region],btagUp[region], btagDown[region])
                #print('weight.weight BTAG', weights.weight())


                if 'WJets' in dataset or 'DY' in dataset or 'Z1Jets' in dataset or 'G1Jets' in dataset:
                    if not isFilled:
                        hout['sumw'].fill(dataset='HF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        hout['sumw'].fill(dataset='LF--'+dataset, sumw=1, weight=ak.sum(events.genWeight))
                        isFilled = True
                    whf = ak.values_astype(((ak.num(gen[gen.isb],axis=1) > 0) | (ak.num(gen[gen.isc], axis=1) > 0)), np.int)
                    wlf = ak.values_astype(~(ak.values_astype(whf,np.bool)), np.int)
                    cut = ak.to_numpy(selection.all(*regions[region]))
                    if 'WJets' in dataset:
                        systematics = [None,]
                    else:
                        systematics = [None,]
                    for systematic in systematics:
                        sname = 'nominal' if systematic is None else systematic
                        hout['template'].fill(dataset='HF--'+dataset,
                                                region=region,
                                                systematic=sname,
                                                recoil = ak.sum(u[region].pt, axis=-1),
                                                fjmass = ak.sum(fj.mass),
                                                TvsQCD = ak.sum(fj.TvsQCD),
                                                weight = weights.weight(modifier=systematic)*whf*cut)
                        hout['template'].fill(dataset='LF--'+dataset,
                                                region=region,
                                                systematic=sname,
                                                recoil = ak.sum(u[region].pt, axis=-1),
                                                fjmass = ak.sum(fj.mass),
                                                TvsQCD = ak.sum(fj.TvsQCD),
                                                weight = weights.weight(modifier=systematic)*wlf*cut)

                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*whf)
                    hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*wlf)
                    hout['runnumber'].fill(dataset='HF--'+dataset, region=region, cut=vcut, runnumber = events.run, weight=weights.weight()*whf)
                    hout['runnumber'].fill(dataset='LF--'+dataset, region=region, cut=vcut, runnumber = events.run, weight=weights.weight()*wlf)
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        hout['cutflow'].fill(dataset='HF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*whf)
                        hout['cutflow'].fill(dataset='LF--'+dataset, region=region, cut=vcut, weight=weights.weight()*jcut*wlf)
                        hout['runnumber'].fill(dataset='HF--'+dataset, region=region, cut=vcut, runnumber = events.run, weight=weights.weight()*jcut*whf)
                        hout['runnumber'].fill(dataset='LF--'+dataset, region=region, cut=vcut, runnumber = events.run, weight=weights.weight()*jcut*wlf)
                    fill('HF--'+dataset, weights.weight()*whf, cut)
                    fill('LF--'+dataset, weights.weight()*wlf, cut)

                else: ## if not Wjet, DY, ZJet, or GJet
                    if not isFilled:
                        hout['sumw'].fill(dataset=dataset, sumw=1, weight=ak.sum(events.genWeight, axis=-1))
                        isFilled = True
                    #cut = selection.all(*regions[region])
                    cut = ak.to_numpy(selection.all(*regions[region]))
                    print('weight.weight', weights.weight())
                    for systematic in [None,]:
                        sname = 'nominal' if systematic is None else systematic
                        hout['template'].fill(dataset=dataset,
                                                region=region,
                                                systematic=sname,
                                                recoil = ak.sum(u[region].pt, axis=-1),
                                                fjmass = ak.sum(fj.mass),
                                                TvsQCD = ak.sum(fj.TvsQCD),
                                                weight = weights.weight(modifier=systematic)*cut)
                    ## Cutflow
                    vcut=np.zeros(event_size, dtype=np.int)
                    hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight())
                    hout['runnumber'].fill(dataset=dataset, region=region, cut=vcut, runnumber = events.run,weight=weights.weight())
                    allcuts = set()
                    for i, icut in enumerate(cuts):
                        allcuts.add(icut)
                        jcut = selection.all(*allcuts)
                        vcut = (i+1)*jcut
                        hout['cutflow'].fill(dataset=dataset, region=region, cut=vcut, weight=weights.weight()*jcut)
                        hout['runnumber'].fill(dataset=dataset, region=region, cut=vcut, runnumber = events.run, weight=weights.weight()*jcut)
                    fill(dataset, weights.weight(), cut)



        return hout


    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
            dataset = d.name
            if '--' in dataset:
                dataset = dataset.split('--')[1]
            print('Cross section:', self._xsec[dataset])
            if self._xsec[dataset] != -1:
                scale[d.name] = self._lumi*self._xsec[dataset]
                print('lumi * xsec: ', self._lumi, '*', self._xsec[dataset], '= ', scale[d.name])
            else:
                scale[d.name] = 1

        for histname, h in accumulator.items():
            if histname == 'sumw':
                print(h.values())
                continue
            if isinstance(h, hist.Hist):
                h.scale(scale, axis='dataset')

        return accumulator


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-r', '--run', help='number of run', dest='run')
    parser.add_option('-v', '--vfp', help='vfp', dest='vfp', default=None)
    (options, args) = parser.parse_args()

    #with open('metadata/KITv3_ALLrun_rerun6_v1.json') as fin:
    with open('metadata/KITv3_UL_ALL2018_v1.json') as fin:
    #with open('metadata/onefile.json') as fin:
    #with open('metadata/KITv3_UL_DATA2018_vone.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k, v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    correctionsUL = load('data/corrections_UL.coffea')
#    correctionsBtag = load('data/correction_btag.coffea')
    #ids = load('data/ids_runDATA_run'+options.run+'.coffea')
    ids = load('data/ids_runDATA_run3.coffea')
    common = load('data/common.coffea')

    processor_instance = AnalysisProcessor(year=options.year,
                                           xsec=xsec,
                                           corrections=corrections,
                                           correctionsUL=correctionsUL,
 #                                          correctionsBtag=correctionsBtag,
                                           ids=ids,
                                           common=common,
                                           vfp=options.vfp
                                           )

    save(processor_instance, 'data/runALLv3_'+options.year+'_Run'+options.run+'.processor')
    print("processor name: runALLv3_{}_Run{}".format(options.year,options.run))

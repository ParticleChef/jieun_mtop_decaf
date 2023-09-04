import hist
import awkward as ak
import numpy as np

from coffea.nanoevents.methods import vector, candidate 
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup

events = NanoEventsFactory.from_root(
        "./TTHadronic_Nanov9_2018.root",
        #"./METv2_sample.root",
        entry_stop=100_000,
        schemaclass=NanoAODSchema,
        ).events()

_jec = {
    '2018':[
        'Summer19UL18_V5_MC_L1FastJet_AK4PFchs', 
        'Summer19UL18_V5_MC_L2Relative_AK4PFchs', 
        'Summer19UL18_V5_MC_L2L3Residual_AK4PFchs', 
        'Summer19UL18_V5_MC_L3Absolute_AK4PFchs',

        'Summer19UL18_V5_MC_L1FastJet_AK8PFPuppi', 
        'Summer19UL18_V5_MC_L2Relative_AK8PFPuppi', 
        'Summer19UL18_V5_MC_L2L3Residual_AK8PFPuppi', 
        'Summer19UL18_V5_MC_L3Absolute_AK8PFPuppi',
        ]
}

_junc = {
    '2018':[
        'Summer19UL18_V5_MC_Uncertainty_AK4PFchs',
        'Summer19UL18_V5_MC_Uncertainty_AK8PFPuppi' ]
}

_jr = {
    '2018':[
        'Summer19UL18_JRV2_MC_PtResolution_AK4PFchs',
        'Summer19UL18_JRV2_MC_PtResolution_AK8PFPuppi' ]
}

_jersf = {
    '2018':[
        'Summer19UL18_JRV2_MC_SF_AK4PFchs',
        'Summer19UL18_JRV2_MC_SF_AK8PFPuppi', ]
}

year = '2018'
mu = events.Muon

j = events.Jet
fj = events.AK15PFPuppi

print(j.fields)

j["T"] = ak.zip({"pt": j.pt, "phi": j.phi},
        with_name="PolarTwoVector",
        behavior=vector.behavior)
j['p4'] = ak.zip({
    "pt": j.pt,
    "eta": j.eta,
    "phi": j.phi,
    "mass": j.mass},
    with_name="PtEtaPhiMLorentzVector"
    )
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
fj['pt'] = events.AK15PFPuppi['Jet_pt']
fj['phi'] = events.AK15PFPuppi['Jet_phi']
fj['eta'] = events.AK15PFPuppi['Jet_eta']
fj['mass'] = events.AK15PFPuppi['Jet_mass']
fj['area'] = events.AK15PFPuppi['Jet_area']

jec_names = []
jec_names_ak15 = []

for name in _jec[year]:
    if 'AK4PFchs' in name:
        jec_names.append(name)
    if 'AK8PF' in name:
        jec_names_ak15.append(name)
for name in _jr[year]:
    if 'AK4PFchs' in name:
        jec_names.append(name)
    if 'AK8PF' in name:
        jec_names_ak15.append(name)
for name in _junc[year]:
    if 'AK4PFchs' in name:
        jec_names.append(name)
    if 'AK8PF' in name:
        jec_names_ak15.append(name)
for name in _jersf[year]:
    if 'AK4PFchs' in name:
        jec_names.append(name)
    if 'AK8PF' in name:
        jec_names_ak15.append(name)





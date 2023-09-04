import hist
import awkward as ak
import numpy as np

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup

events = NanoEventsFactory.from_root(
        "../../TTHadronic_Nanov9_2018.root",
        #"./METv2_sample.root",
        entry_stop=100_000,
        schemaclass=NanoAODSchema,
        ).events()

print(events.Muon.pfIsoId)

met = events.MET
mu = events.Muon

#recoil = met + mu
#
#print(recoil.pt)
#
#
#met_x = met.pt * np.cos(met.phi)
#met_y = met.pt * np.sin(met.phi)
#
#mu_x = mu.pt * np.cos(mu.phi)
#mu_y = mu.pt * np.sin(mu.phi)
#
#recoil_x = met_x + mu_x
#recoil_y = met_y + mu_y
#
#recoil_pt = np.hypot(recoil_x,recoil_y)
#
#print(recoil_pt)

fj = events.AK15PFPuppi
print(fj.fields)

#j_flav = j.btagDeepFlavB
#bc_jets = events.Jet[(events.Jet.pt > 25)
#        & (abs(events.Jet.eta) < 2.4)
#        & (events.Jet.btagDeepFlavB > 0)]
#
#import correctionlib
## btagging
#cset = correctionlib.CorrectionSet.from_file("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz")
##print(f'medium b-tagging deepJet WP 2017: {cset["deepJet_wp_value"].evaluate("M")}')
#bc_jets, nj = ak.flatten(bc_jets), ak.num(bc_jets)
#print(bc_jets.pt)
#sf = cset["deepJet_comb"].evaluate("central","M", 
#        np.array(bc_jets.btagDeepFlavB), 
#        np.array(bc_jets.eta), 
#        np.array(bc_jets.pt))
#
#ak.unflatten(sf,nj)
#print(sf)





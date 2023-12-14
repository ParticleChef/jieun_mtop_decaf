#! /usr/bin/env python
import correctionlib
import os
#import uproot
#import uproot_methods
import awkward as ak

import numpy as np
from coffea import hist, lookup_tools
from coffea.lookup_tools import extractor, dense_lookup

import uproot3
from coffea import util
from coffea.util import save, load
import json


############
## BTag
## Btag recommend: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation#Recommendation_for_13_TeV_Data



from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup

class BTagCorrector:

    def __init__(self, tagger, year, workingpoint):
        self._year = year
        common = load('data/common.coffea')
        self._wp = common['btagWPs'][tagger][year][workingpoint]
        
        btvjson = correctionlib.CorrectionSet.from_file('data/BtagSF/'+year+'_UL/btagging.json.gz')
        self.sf = btvjson # ["deepJet_comb", "deepCSV_comb"]

        files = {
#            '2016preVFP': 'btageff2016.merged',
#            '2016postVFP': 'btageff2016.merged',
#            '2017': 'btageff2017.merged',
            '2018': 'btageff2018.merged',
        }
        filename = 'hists/'+files[year]
        btag = load(filename)
        bpass = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag', 'pass').values()[()]
        ball = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag').values()[()]
        ball[ball<=0.]=1.
        nom = bpass / np.maximum(ball, 1.)
        self.eff = lookup_tools.dense_lookup.dense_lookup(nom, [ax.edges() for ax in btag[tagger].axes()[3:]])

    def btag_weight(self, pt, eta, flavor, istag):
        abseta = abs(eta)
        
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
        def P(eff):
            weight = eff.ones_like()
            weight[istag] = eff[istag]
            weight[~istag] = (1 - eff[~istag])
            return weight.prod()

        '''
        Correction deepJet_comb has 5 inputs
        Input systematic (string): 
        Input working_point (string): L/M/T
        Input flavor (int): hadron flavor definition: 5=b, 4=c, 0=udsg
        Input abseta (real):
        Input pt (real):
        '''

        bc = flavor > 0
        light = ~bc
        
        eff = self.eff(flavor, pt, abseta)
        
        #sf_nom = self.sf.eval('central', flavor, abseta, pt)
        sf_nom = self.sf["deepJet_comb"].evaluate('central','M', flavor, abseta, pt)

        bc_sf_up_correlated = pt.ones_like()
        bc_sf_up_correlated[~bc] = sf_nom[~bc]
        bc_sf_up_correlated[bc] = self.sf["deepJet_comb"].evaluate('up_correlated', 'M', flavor, eta, pt)[bc]
        
        bc_sf_down_correlated = pt.ones_like()
        bc_sf_down_correlated[~bc] = sf_nom[~bc]
        bc_sf_down_correlated[bc] = self.sf["deepJet_comb"].evaluate('down_correlated', 'M', flavor, eta, pt)[bc]

        bc_sf_up_uncorrelated = pt.ones_like()
        bc_sf_up_uncorrelated[~bc] = sf_nom[~bc]
        bc_sf_up_uncorrelated[bc] = self.sf["deepJet_comb"].evaluate('up_uncorrelated', 'M', flavor, eta, pt)[bc]

        bc_sf_down_uncorrelated = pt.ones_like()
        bc_sf_down_uncorrelated[~bc] = sf_nom[~bc]
        bc_sf_down_uncorrelated[bc] = self.sf["deepJet_comb"].evaluate('down_uncorrelated', 'M', flavor, eta, pt)[bc]

        light_sf_up_correlated = pt.ones_like()
        light_sf_up_correlated[~light] = sf_nom[~light]
        light_sf_up_correlated[light] = self.sf["deepJet_comb"].evaluate('up_correlated', 'M', flavor, abseta, pt)[light]

        light_sf_down_correlated = pt.ones_like()
        light_sf_down_correlated[~light] = sf_nom[~light]
        light_sf_down_correlated[light] = self.sf["deepJet_comb"].evaluate('down_correlated', 'M', flavor, abseta, pt)[light]

        light_sf_up_uncorrelated = pt.ones_like()
        light_sf_up_uncorrelated[~light] = sf_nom[~light]
        light_sf_up_uncorrelated[light] = self.sf["deepJet_comb"].evaluate('up_uncorrelated', 'M', flavor, abseta, pt)[light]

        light_sf_down_uncorrelated = pt.ones_like()
        light_sf_down_uncorrelated[~light] = sf_nom[~light]
        light_sf_down_uncorrelated[light] = self.sf["deepJet_comb"].evaluate('down_uncorrelated', 'M', flavor, abseta, pt)[light]

        eff_data_nom  = np.minimum(1., sf_nom*eff)
        bc_eff_data_up_correlated   = np.minimum(1., bc_sf_up_correlated*eff)
        bc_eff_data_down_correlated = np.minimum(1., bc_sf_down_correlated*eff)
        bc_eff_data_up_uncorrelated   = np.minimum(1., bc_sf_up_uncorrelated*eff)
        bc_eff_data_down_uncorrelated = np.minimum(1., bc_sf_down_uncorrelated*eff)
        light_eff_data_up_correlated   = np.minimum(1., light_sf_up_correlated*eff)
        light_eff_data_down_correlated = np.minimum(1., light_sf_down_correlated*eff)
        light_eff_data_up_uncorrelated   = np.minimum(1., light_sf_up_uncorrelated*eff)
        light_eff_data_down_uncorrelated = np.minimum(1., light_sf_down_uncorrelated*eff)
       
        nom = P(eff_data_nom)/P(eff)
        bc_up_correlated = P(bc_eff_data_up_correlated)/P(eff)
        bc_down_correlated = P(bc_eff_data_down_correlated)/P(eff)
        bc_up_uncorrelated = P(bc_eff_data_up_uncorrelated)/P(eff)
        bc_down_uncorrelated = P(bc_eff_data_down_uncorrelated)/P(eff)
        light_up_correlated = P(light_eff_data_up_correlated)/P(eff)
        light_down_correlated = P(light_eff_data_down_correlated)/P(eff)
        light_up_uncorrelated = P(light_eff_data_up_uncorrelated)/P(eff)
        light_down_uncorrelated = P(light_eff_data_down_uncorrelated)/P(eff)

        return np.nan_to_num(nom, nan=1.), np.nan_to_num(bc_up_correlated, nan=1.), np.nan_to_num(bc_down_correlated, nan=1.), np.nan_to_num(bc_up_uncorrelated, nan=1.), np.nan_to_num(bc_down_uncorrelated, nan=1.), np.nan_to_num(light_up_correlated, nan=1.), np.nan_to_num(light_down_correlated, nan=1.), np.nan_to_num(light_up_uncorrelated, nan=1.), np.nan_to_num(light_down_uncorrelated, nan=1.)


get_btag_weight = {
    'deepflav': {
#        '2016preVFP': {
#            'loose'  : BTagCorrector('deepflav','2016preVFP','loose').btag_weight,
#            'medium' : BTagCorrector('deepflav','2016preVFP','medium').btag_weight,
#            'tight'  : BTagCorrector('deepflav','2016preVFP','tight').btag_weight
#        },
#        '2016postVFP': {
#            'loose'  : BTagCorrector('deepflav','2016postVFP','loose').btag_weight,
#            'medium' : BTagCorrector('deepflav','2016postVFP','medium').btag_weight,
#            'tight'  : BTagCorrector('deepflav','2016postVFP','tight').btag_weight
#        },
#        '2017': {
#            'loose'  : BTagCorrector('deepflav','2017','loose').btag_weight,
#            'medium' : BTagCorrector('deepflav','2017','medium').btag_weight,
#            'tight'  : BTagCorrector('deepflav','2017','tight').btag_weight
#        },
        '2018': {
            'loose'  : BTagCorrector('deepflav','2018','loose').btag_weight,
            'medium' : BTagCorrector('deepflav','2018','medium').btag_weight,
            'tight'  : BTagCorrector('deepflav','2018','tight').btag_weight
            }
        }
    }
corrections = {
    'get_btag_weight':           get_btag_weight
}


save(corrections, 'data/btag_playground.coffea')






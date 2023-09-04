#!/usr/bin/env python
import uproot3
import numpy as np
import os
from coffea import hist, lookup_tools
# from coffea.lookup_tools import extractor, dense_lookup
from coffea import util
from coffea.lookup_tools import dense_lookup
from coffea.util import save, load
from coffea.btag_tools import BTagScaleFactor
import awkward as ak
import json
from coffea.lookup_tools import extractor


class BTagCorrectorPreVFP(object):

    def __init__(self, tagger, year, workingpoint):
        self._year = year
        files = {
           'deepflav': {
               '2016': 'wp_deepJet_106XUL16preVFP_v2.csv',
               '2017': 'wp_deepJet_106XUL17_v3.csv',
               '2018': 'wp_deepJet_106XUL18_v2.csv'
           },
           'deepcsv': {
                '2016': 'wp_deepCSV_106XUL16preVFP_v2.csv',
                '2017': 'wp_deepCSV_106XUL17_v3.csv',
                '2018': 'wp_deepCSV_106XUL18_v2.csv'
           }
        }
        common = load('data/common.coffea')
        self._wp = common['btagWPs'][tagger][year][workingpoint]
        filename = 'data/btag_SF/UL/'+files[tagger][year]
        self.sf = BTagScaleFactor(filename, workingpoint)
        files = {
            '2016': 'btagUL2018.merged',
            '2017': 'btag2017.merged',
            '2018': 'btag2016.merged',
            #'2016': 'btagUL2018.merged',
            #'2017': 'btagUL2018.merged',
            #'2018': 'btagUL2018.merged',
        }
        filename = 'hists/'+files[year]
        btag = load(filename)
        bpass = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag', 'pass').values()[()]
        ball = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag').values()[()]
        nom = bpass / np.maximum(ball, 1.)
        self.eff = dense_lookup.dense_lookup(nom, [ax.edges() for ax in btag[tagger].axes()[3:]])

    def btag_weight(self, events, pt, eta, flavor, tag):
        abseta = abs(eta)
        tightJet = events
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
        def zerotag(eff):
            return ak.prod(1.0 - eff[tightJet.isdflvM], axis=-1)
        
        def onetag(eff):
            output = ak.prod(eff[tightJet.isdflvM], axis=-1) * ak.prod((1.0 - eff[np.invert(tightJet.isdflvM)]), axis=-1)
            return(output)
        
        eff = self.eff(flavor, pt, abseta)
        sf_nom = self.sf.eval('central', flavor, abseta, pt)
        sf_up = self.sf.eval('up', flavor, abseta, pt)
        sf_down = self.sf.eval('down', flavor, abseta, pt)

        eff_data_nom  = np.minimum(1., sf_nom*eff)
        eff_data_up   = np.minimum(1., sf_up*eff)
        eff_data_down = np.minimum(1., sf_down*eff)


        if '-1' in tag: 
            nom = (1 - zerotag(eff_data_nom)) / (1 - zerotag(eff))
            up = (1 - zerotag(eff_data_up)) / (1 - zerotag(eff))
            down = (1 - zerotag(eff_data_down)) / (1 - zerotag(eff))
        elif '2' in tag:
            nom = (1- zerotag(eff_data_nom) - onetag(eff_data_nom))/(1- zerotag(eff) - onetag(eff))
            up =(1- zerotag(eff_data_up) - onetag(eff_data_up))/(1- zerotag(eff) - onetag(eff))
            down =(1- zerotag(eff_data_down) - onetag(eff_data_down))/(1- zerotag(eff) - onetag(eff))
        elif '+1' in tag:
            nom = onetag(eff_data_nom)/onetag(eff)
            up= onetag(eff_data_up)/onetag(eff)
            down = onetag(eff_data_down)/onetag(eff)
        else:
            nom = zerotag(eff_data_nom)/zerotag(eff)
            up = zerotag(eff_data_up)/zerotag(eff)
            down = zerotag(eff_data_down)/zerotag(eff)

        return np.nan_to_num(nom), np.nan_to_num(up), np.nan_to_num(down)

get_btag_weight_preVFP = {
    'deepflav': {
        '2016': {
            'loose'  : BTagCorrectorPreVFP('deepflav','2016','loose').btag_weight,
            'medium' : BTagCorrectorPreVFP('deepflav','2016','medium').btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepflav','2016','tight').btag_weight,

        },
        '2017': {
            'loose'  : BTagCorrectorPreVFP('deepflav','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepflav','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepflav','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPreVFP('deepflav','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepflav','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepflav','2018','tight' ).btag_weight
        }
    },
    'deepcsv' : {
        '2016': {
            'loose'  : BTagCorrectorPreVFP('deepcsv','2016','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepcsv','2016','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepcsv','2016','tight' ).btag_weight
        },
        '2017': {
            'loose'  : BTagCorrectorPreVFP('deepcsv','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepcsv','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepcsv','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPreVFP('deepcsv','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPreVFP('deepcsv','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPreVFP('deepcsv','2018','tight').btag_weight
        }
    }
}

class BTagCorrectorPostVFP:

    def __init__(self, tagger, year, workingpoint):
        self._year = year
        files = {
           'deepflav': {
               '2016': 'wp_deepJet_106XUL16preVFP_v2.csv',
               '2017': 'wp_deepJet_106XUL17_v3.csv',
               '2018': 'wp_deepJet_106XUL18_v2.csv'
           },
           'deepcsv': {
                '2016': 'wp_deepCSV_106XUL16preVFP_v2.csv',
                '2017': 'wp_deepCSV_106XUL17_v3.csv',
                '2018': 'wp_deepCSV_106XUL18_v2.csv'
           }
        }

        common = load('data/common.coffea')
        self._wp = common['btagWPs'][tagger][year][workingpoint]
        filename = 'data/btag_SF/UL/'+files[tagger][year]
        self.sf = BTagScaleFactor(filename, workingpoint)
        files = {
            '2016': 'btag2018.merged',
            '2017': 'btag2017.merged',
            '2018': 'btag2016.merged',
            #'2016': 'btagUL2018.merged',
            #'2017': 'btagUL2018.merged',
            #'2018': 'btagUL2018.merged',
        }
        filename = 'hists/'+files[year]
        btag = util.load(filename)
        bpass = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag', 'pass').values()[()]
        ball = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag').values()[()]
        nom = bpass / np.maximum(ball, 1.)
        self.eff = dense_lookup.dense_lookup(nom, [ax.edges() for ax in btag[tagger].axes()[3:]])

    def btag_weight(self, events, pt, eta, flavor, tag):
        abseta = abs(eta)
        tightJet = events
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
        def zerotag(eff):
            return ak.prod(1.0 - eff[tightJet.isdflvM], axis=-1)
        
        def onetag(eff):
            output = ak.prod(eff[tightJet.isdflvM], axis=-1) * ak.prod((1.0 - eff[np.invert(tightJet.isdflvM)]), axis=-1)
            return(output)
        
        eff = self.eff(flavor, pt, abseta)
        sf_nom = self.sf.eval('central', flavor, abseta, pt)
        sf_up = self.sf.eval('up', flavor, abseta, pt)
        sf_down = self.sf.eval('down', flavor, abseta, pt)

        eff_data_nom  = np.minimum(1., sf_nom*eff)
        eff_data_up   = np.minimum(1., sf_up*eff)
        eff_data_down = np.minimum(1., sf_down*eff)


        if '-1' in tag: 
            nom = (1 - zerotag(eff_data_nom)) / (1 - zerotag(eff))
            up = (1 - zerotag(eff_data_up)) / (1 - zerotag(eff))
            down = (1 - zerotag(eff_data_down)) / (1 - zerotag(eff))
        elif '2' in tag:
            nom = (1- zerotag(eff_data_nom) - onetag(eff_data_nom))/(1- zerotag(eff) - onetag(eff))
            up =(1- zerotag(eff_data_up) - onetag(eff_data_up))/(1- zerotag(eff) - onetag(eff))
            down =(1- zerotag(eff_data_down) - onetag(eff_data_down))/(1- zerotag(eff) - onetag(eff))
        elif '+1' in tag:
            nom = onetag(eff_data_nom)/onetag(eff)
            up= onetag(eff_data_up)/onetag(eff)
            down = onetag(eff_data_down)/onetag(eff)
        else:
            nom = zerotag(eff_data_nom)/zerotag(eff)
            up = zerotag(eff_data_up)/zerotag(eff)
            down = zerotag(eff_data_down)/zerotag(eff)

        return np.nan_to_num(nom), np.nan_to_num(up), np.nan_to_num(down)

get_btag_weight_postVFP = {
    'deepflav': {
        '2016': {
            'loose'  : BTagCorrectorPostVFP('deepflav','2016','loose').btag_weight,
            'medium' : BTagCorrectorPostVFP('deepflav','2016','medium').btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepflav','2016','tight').btag_weight,

        },
        '2017': {
            'loose'  : BTagCorrectorPostVFP('deepflav','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepflav','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepflav','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPostVFP('deepflav','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepflav','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepflav','2018','tight' ).btag_weight
        }
    },
    'deepcsv' : {
        '2016': {
            'loose'  : BTagCorrectorPostVFP('deepcsv','2016','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepcsv','2016','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepcsv','2016','tight' ).btag_weight
        },
        '2017': {
            'loose'  : BTagCorrectorPostVFP('deepcsv','2017','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepcsv','2017','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepcsv','2017','tight' ).btag_weight
        },
        '2018': {
            'loose'  : BTagCorrectorPostVFP('deepcsv','2018','loose' ).btag_weight,
            'medium' : BTagCorrectorPostVFP('deepcsv','2018','medium' ).btag_weight,
            'tight'  : BTagCorrectorPostVFP('deepcsv','2018','tight').btag_weight
        }
    }
}


corrections = {
    'get_btag_weight_preVFP':get_btag_weight_preVFP,
    'get_btag_weight_postVFP':get_btag_weight_postVFP,
}
save(corrections, 'data/correctionsB.coffea')


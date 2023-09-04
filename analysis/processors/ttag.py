import os
import numpy
import json
import math
import awkward as ak
from coffea import processor, hist, util
from coffea.util import save, load
from optparse import OptionParser
from coffea.lookup_tools.dense_lookup import dense_lookup
from uproot_methods import TVector2Array, TLorentzVectorArray

class TTagEfficiency(processor.ProcessorABC):

    def __init__(self, year,wp):
        self._year = year
        self._btagWPs = wp
        self._TvsQCDwp = { ### ?
            '2016': 0.53,
            '2017': 0.61,
            '2018': 0.65
        }
        self._accumulator = processor.dict_accumulator({
            'TvsQCD' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('TvsQCD','TvsQCD',100,0,1)
            ),
            'fjTvsQCD' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('fjTvsQCD','TvsQCD',100,0,1)
            ),
            'dRfjtop' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('dRfjtop','dR (fatjet, top)',200,0,10)
            ),
            'tmatchj' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('tag', 'Top matching Fatjet'),
                #hist.Bin('dRfjtop','dR (fatjet, top)',200,0,10),
                hist.Bin('TvsQCD','TvsQCD',100,0,1)
            ),
            'qWmatchedJetpt' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('tag', 'qW matching Fatjet'),
                hist.Bin('qWmatchedpt','qW matched fatjet pt',200,0,2000)
            ),
            'toppt' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('toppt','Top pT',200,0,2000)
            ),
            'met' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('met','MET',200,0,2000)
            ),
#            'ak15jetpt' :
#            hist.Hist(
#                'Events',
#                hist.Cat('dataset', 'Dataset'),
#                hist.Bin('ak15jetpt','AK15 Jet pT',200,0,2000)
#            ),
            'qFromToppt' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('qFromToppt','q from Top pT',200,0,2000)
            ),
            'qWmatchedJet' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('tag', 'Hadronic Top matching Fatjet'),
                hist.Bin('qWmatchedJet','TvsQCD',100,0,1)
            ),
            'ttag' :
            hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('wp', 'Working point'),
                hist.Cat('ttag', 'TTag WP pass/fail'),
                hist.Bin('pt', 'AK8 pT', [200, 300, 500, 700, 1000, 1400, 2000]),
                hist.Bin('abseta', 'AK8 Jet abseta', [0, 1.4, 2.0, 2.5]),
 #               hist.Bin('TvsQCD','TvsQCD', [0, self._TvsQCDwp['2018'], 1])
            )
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        
        dataset = events.metadata['dataset']

        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & (events.Jet.jetId & 2)  # tight id
        ]

        fatjet = events.FatJet[events.FatJet.pt > 250.]
        tvsqcd = fatjet['particleNet_TvsQCD']
        ### AK15 JET
        ak15jet = events.AK15PFPuppi
        fjtvsqcd = ak15jet['particleNetAK15_TvsQCD']

        gen = events.GenPart

        qFromW = gen[
            (abs(gen.pdgId) < 5) & # 1: d, 2: u, 3: s, 4: c
            gen.hasFlags(['fromHardProcess', 'isFirstCopy']) &
            (abs(gen.distinctParent.pdgId) == 24)  # 24: W
        ]
        qFromWFromTop = qFromW[qFromW.distinctParent.distinctParent.pdgId == 6]
        jetgenWq = fatjet.cross(qFromWFromTop, nested=True)
        #qWmatch = ((jetgenWq.i0.delta_r(jetgenWq.i1) < 0.8).sum()==2) & (qFromWFromTop.counts>0)
        qWmatch = qFromWFromTop.counts>0

        top = gen[gen.pdgId==6]
        met = gen[gen.pdgId==5000001]


        topgen = fatjet.cross(top, nested=True)

        dRfjtoT= (topgen.i0.delta_r(topgen.i1)).min()
        
#### Fill out['variable'] ####
        out = self.accumulator.identity()

        out['dRfjtop'].fill(
            dataset=dataset,
            dRfjtop=dRfjtoT.flatten()
        )
        out['toppt'].fill(
            dataset=dataset,
            toppt=top.pt.flatten()
        )
        out['met'].fill(
            dataset=dataset,
            met=met.pt.flatten()
        )
        out['qFromToppt'].fill(
            dataset=dataset,
            qFromToppt=qFromWFromTop.pt.flatten()
        )
        out['qWmatchedJetpt'].fill(
            dataset=dataset,
            tag='pass',
            qWmatchedpt=fatjet[qWmatch].pt.flatten()
        )
        out['qWmatchedJetpt'].fill(
            dataset=dataset,
            tag='fail',
            qWmatchedpt=fatjet[~qWmatch].pt.flatten()
        )
#        out['ak15jetpt'].fill(
#            dataset=dataset,
#            ak15jetpt=ak15jet['Jet_pt'].sum().flatten()
#        )

        passdr = dRfjtoT < 0.8
        passttag = fatjet['particleNet_TvsQCD'] > self._TvsQCDwp['2018']
        out['tmatchj'].fill(
            dataset=dataset,
            tag='pass',
            #dRfjtop=dRfjtoT[passdr].flatten(),
            #TvsQCD=fatjet['particleNet_TvsQCD'][passdr&passttag].flatten()
            TvsQCD=fatjet['particleNet_TvsQCD'][passdr].flatten()
        )
        out['tmatchj'].fill(
            dataset=dataset,
            tag='fail',
            #dRfjtop=dRfjtoT[passdr].flatten(),
            #TvsQCD=fatjet['particleNet_TvsQCD'][~passdr&passttag].flatten()
            TvsQCD=fatjet['particleNet_TvsQCD'][~passdr].flatten()
        )
        out['qWmatchedJet'].fill(
            dataset=dataset,
            tag='pass',
            qWmatchedJet=fatjet['particleNet_TvsQCD'][qWmatch&passdr].flatten()
        )
        out['qWmatchedJet'].fill(
            dataset=dataset,
            tag='fail',
            qWmatchedJet=fatjet['particleNet_TvsQCD'][qWmatch&~passdr].flatten()
        )

        out['ttag'].fill(
            dataset=dataset,
            wp='default',
            ttag='pass',
            pt=fatjet[passttag].pt.flatten(),
            abseta=abs(fatjet[passttag].eta.flatten())
 #           TvsQCD=fatjet[passttag].particleNet_TvsQCD.sum()
        )
        out['ttag'].fill(
            dataset=dataset,
            wp='default',
            ttag='fail',
            pt=fatjet[~passttag].pt.flatten(),
            abseta=abs(fatjet[~passttag].eta.flatten())
 #           TvsQCD=fatjet[passttag].particleNet_TvsQCD.sum()
        )
        out['TvsQCD'].fill(
            dataset=dataset,
            TvsQCD=fatjet['particleNet_TvsQCD'].flatten()
        )
        out['fjTvsQCD'].fill(
            dataset=dataset,
            fjTvsQCD=fjtvsqcd.flatten()
        )

        return out

    def postprocess(self, a):
        return a

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    (options, args) = parser.parse_args()


    #with open('metadata/'+options.year+'.json') as fin:
    with open('metadata/20UL18_Mphi-2000.json') as fin:
    #with open('metadata/20UL18_QCDfileList.json') as fin:
        samplefiles = json.load(fin)

    common = load('data/common.coffea')
    processor_instance=TTagEfficiency(year=options.year,wp=common['btagWPs'])

    save(processor_instance, 'data/ttagNew'+options.year+'.processor')

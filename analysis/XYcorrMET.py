#!/usr/bin/env python

import awkward as ak
import numpy as np
import glob as glob
import re
import itertools

def get_data_era(events, year):
    runNMax = max(events.run)
    runNMin = min(events.run)
    if "2018" in year:
        if (runNMin >=315252) and (runNMax <=316995):
            era = "A"
        elif (runNMin >=316998) and (runNMax <=319312):
            era = "B"
        elif (runNMin >=319313) and (runNMax <=320393):
            era = "C"
        elif (runNMin >=320394) and (runNMax <=325273):
            era = "D"
    
    elif "2017" in year:
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

        else: era = "F"
    if "2016" in year:

        if (runNMin >=272007) and (runNMax <=275376):
            era = "BCD"

        elif (runNMin >=275656) and (runNMax <=276283):
            era = "BCD"
        elif (runNMin >=276315) and (runNMax <=276811):
            era = "BCD"

        elif (runNMin >=276831) and (runNMax <=277420) :
            era = "EF"

        elif (runNMin >=277772 and runNMax <=278807) or (runNMin in [278770, 278806, 278807]) or (runNMax in [278770, 278806, 278807] ):
            era = "EF"
        elif (runNMin >=278801 and runNMax <=278808) or (runNMax in [278769, 278806, 278807]) or (runNMax in [278769,278806, 278807]):
            era = "EF"

        elif (runNMin >=278820) and (runNMax <=280385):
            era = "G"
        elif (runNMin >=281613) and (runNMax <=284044):
            era = "H"
    return(era)

def METXYCorr_Met_MetPhi(events, year, vfp, metpt, metphi):
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
        if (year == '2016') and ("pre" in vfp):
            print('do this')
            METxcorr = -(-0.188743*npv +0.136539)
            METycorr = -(0.0127927*npv +0.117747)

        if (year == '2016') and ("post" in vfp):
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

            if (runNMin >= 315252) and (runNMax <= 316995):
                era = "A"

            elif (runNMin >= 316998) and (runNMax <= 319312):
                era = "B"

            elif (runNMin >= 319313) and (runNMax <= 320393):
                era = "C"

            elif (runNMin >= 320394) and (runNMax <= 325273):
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

        if '2016' in year:
            runNMax = max(events.run)
            runNMin = min(events.run)

            if (runNMin >=272007) and (runNMax <=275376):
                era = "B"
            elif (runNMin >=275656) and (runNMax <=276283):
                era = "C"
            elif (runNMin >=276315) and (runNMax <=276811):
                era = "D"
            elif (runNMin >=276831) and (runNMax <=277420):
                era = "E"
            elif (runNMin >=277772 and runNMax <=278807) or (runNMin in [278770, 278806, 278807]) or (runNMax in [278770, 278806, 278807] ):
                era = "F"
#                  https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVDatasetsUL2016
            elif (runNMin >=278801 and runNMax <=278808) or (runNMax in [278769, 278806, 278807]) or (runNMax in [278769,278806, 278807]):
                era = "F_late"
            elif (runNMin >=278820) and (runNMax <=280385):
                era = "G"
            elif (runNMin >=281613) and (runNMax <=284044):
                era = "H"

            if "B" in era:
                METxcorr = -(-0.0214894*npv +-0.188255)
                METycorr = -(0.0876624*npv +0.812885)
            elif "C" in era:
                METxcorr = -(-0.032209*npv +0.067288)
                METycorr = -(0.113917*npv +0.743906)
            elif "D" in era:
                METxcorr = -(-0.0293663*npv +0.21106)
                METycorr = -(0.11331*npv +0.815787)
            elif "E" in era:
                METxcorr = -(-0.0132046*npv +0.20073)
                METycorr = -(0.134809*npv +0.679068)
            elif "F" in era:
                METxcorr = -(-0.0543566*npv +0.816597)
                METycorr = -(0.114225*npv +1.17266)
            elif "F_late" in era:
                METxcorr = -(0.134616*npv +-0.89965)
                METycorr = -(0.0397736*npv +1.0385)
            elif "G" in era:
                METxcorr = -(0.121809*npv +-0.584893)
                METycorr = -(0.0558974*npv +0.891234)
            elif "H" in era:
                METxcorr = -(0.0868828*npv +-0.703489)
                METycorr = -(0.0888774*npv +0.902632)

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


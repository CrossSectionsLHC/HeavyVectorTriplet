#!/usr/bin/env 
import pandas as pd
import numpy as np
import time
from array import array
import seaborn as sns
# based on https://gitlab.cern.ch/cms-b2g/diboson-combination/combination-2016/-/blob/master/hvt.py

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import glob

filter_gH_1 = np.arange(start=-8, stop=8, step=0.5)
filter_gH_2 = np.arange(start=-4, stop=4, step=0.1)
filter_gH_3 = np.arange(start=-1, stop=1, step=0.01)
filter_gH = np.unique(np.concatenate((filter_gH_1,filter_gH_2,filter_gH_3),0))

print(len(filter_gH), np.around(filter_gH,2))
#[-8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
filter_gH = np.around(filter_gH,2)

df_BR_total = pd.DataFrame()
df_mod_C_BR_total = pd.DataFrame()
df_mod_B_BR_total = pd.DataFrame()
df_mod_A_BR_total = pd.DataFrame()
df_mod_gf3o5_BR_total = pd.DataFrame()
df_mod_gf0o1_BR_total = pd.DataFrame()

for particle in ["Z", "W"]: # 
    df_BR = pd.DataFrame()
    name = 'HVT/BRs/BRs_%sprime.csv' % (particle)
    print(name)
    df_local = pd.read_csv(name)
    
    #df_part = df_local[(abs(df_local["gH"]) > 1 )]
    df_local["gH"] = round(df_local["gv"]*df_local["ch"], 3)
    #df_local[(abs(df_local["gH"]) > 0.5)]["gH"] = round(df_local[(abs(df_local["gH"]) > 0.5)]["gH"], 1)

    df_local["gF"] = round(df_local["g"]*df_local["g"]*df_local["cl"]/df_local["gv"], 3)
    # make specific tables to model A / B / C
    #remove
    #Model A: 'gh':-0.556, 'gf':-0.562
    #Model B: 'gh':-2.928, 'gf':0.146
    #Model C: gF = 0 
    df_mod_C_BR = df_local[df_local['gH'].isin([1]) & df_local['gF'].isin([0])]
    df_mod_B_BR = df_local[df_local['gH'].isin([-2.928]) & df_local['gF'].isin([0.146])]
    df_mod_A_BR = df_local[df_local['gH'].isin([-0.556]) & df_local['gF'].isin([-0.562])]
    df_mod_gf3o5_BR = df_local[df_local['gH'].isin([0.5]) & df_local['gF'].isin([0.35])]
    df_mod_gf0o1_BR = df_local[df_local['gH'].isin([0.6]) & df_local['gF'].isin([0.1])]

    # drom model A/B for table for 2D
    df_local = df_local[~df_local['gF'].isin([-0.562, 0.146])]

    df_local = df_local[~df_local['gH'].isin([-2.928, -0.556])]
    #df_local = df_local[~df_local['gH'].isin([-2.93, 2.93,  2.92,  3.0, -0.56, -0.55])]

    # drop glanularity where we do not need
    #df_local = df_local[df_local['gH'].isin(filter_gH)]

    if particle == "Z" :
        df_local["Zp_dijet"] = df_local[['BRuu', 'BRdd', 'BRcc', 'BRss']].sum(axis=1)  
        df_local["ll"] = df_local[['BRee', 'BRmumu']].sum(axis=1)
        df_local['Zp_VVVH'] = df_local[['BRWW', 'BRhZ']].sum(axis=1) 
    else:
        df_local["Wp_dijet"] = df_local[['BRud', 'BRus', 'BRcd', 'BRcs']].sum(axis=1)  
        df_local["lv"] = df_local[['BReve', 'BRmvm']].sum(axis=1) 
        df_local['Wp_VVVH'] = df_local[['BRWZ', 'BRWH']].sum(axis=1)
    
    df_local['Vp_VVVH'] = df_local[['Zp_VVVH', 'Wp_VVVH']].sum(axis=1) / 2 # average of Zp_VVVH and Wp_VVVH 
    #print(df_local.columns)
    # Z' 
    # 'M0', 'g', 'gv', 'ch', 'cl', 'GammaTot', 'BRWW', 'BRhZ', 'BRee', 'BRmumu', 'BRtautau', 'BRnunu', 'BRuu', 'BRdd', 'BRcc', 'BRss', 'BRbb', 'BRtt', 'BRll', 'BRqq', 'BRjets'
    # W' 
    # 'M0', 'g', 'gv', 'ch', 'cl', 'GammaTot', 'BRWH', 'BRWZ', 'BReve', 'BRmvm', 'BRtauvt', 'BRud', 'BRus', 'BRcd', 'BRcs', 'BRtb', 'BRlnu', 'BRqqbar', 'BRjets'
    gF_list_anal = sorted(list(set(list((df_local['gF'].values)))))
    gH_list_anal = sorted(np.around(list(set(list((df_local['gH'].values)))),2))
    #print(gF_list_anal, len(gF_list_anal))
    #print(gH_list_anal, len(gH_list_anal))

    for dataframes in (df_local, df_mod_C_BR, df_mod_B_BR, df_mod_A_BR, df_mod_gf3o5_BR, df_mod_gf0o1_BR) :
        print(particle)

        if particle == "Z" :
            dataframes["Zp_dijet"] = dataframes[['BRuu', 'BRdd', 'BRcc', 'BRss']].sum(axis=1)  
            dataframes["ll"] = dataframes[['BRee', 'BRmumu']].sum(axis=1) 
            dataframes['Zp_VVVH'] = dataframes[['BRWW', 'BRhZ']].sum(axis=1)
        else:
            dataframes["Wp_dijet"] = dataframes[['BRud', 'BRus', 'BRcd', 'BRcs']].sum(axis=1)  
            dataframes["lv"] = dataframes[['BReve', 'BRmvm']].sum(axis=1) 
            dataframes['Wp_VVVH'] = dataframes[['BRWH', 'BRWZ']].sum(axis=1)
        
        dataframes['Vp_VVVH'] = dataframes[['Zp_VVVH', 'Wp_VVVH']].sum(axis=1) / 2 # average of Zp_VVVH and Wp_VVVH

        dataframes.rename({
            'BRhZ': 'ZH', 
            'BRWH': 'WH', 
            'BRWW' : 'WW',
            'BRWZ' : 'WZ',
            'BRtautau' : 'tautau',
            'BRtauvt' : 'tauvt',
            'BRnunu' : "vv",
            'BRtt' : 'tt',
            'BRtb' : "tb",
            "GammaTot" : "GammaTot%sp" % particle,
            }, axis=1, inplace=True)

        
        dataframes["gH_mod_X_sign_gF"] = ((dataframes['gF'].apply(func = lambda x : 1 if not x else -1)))*dataframes["gH"]
        dataframes["gF_mod"] = abs(dataframes['gF'])
        dataframes = dataframes.drop(columns=[ "g", "gv",  "ch", "cl", "BRjets", "gH", "gF"])
        #print(df_local.columns)

        dataframes = dataframes.drop_duplicates(subset=["gH_mod_X_sign_gF", "gF_mod", 'M0'])

        if particle == "Z" :
            dataframes = dataframes.drop(columns=[ 'BRuu', 'BRdd', 'BRcc', 'BRss', 'BRee',  'BRmumu'])
        else:
            dataframes = dataframes.drop(columns=[ 'BReve', 'BRmvm', 'BRlnu', 'BRqqbar', 'BRud', 'BRus', 'BRcd','BRcs'])


    if len(df_BR_total) == 0:
        df_BR_total = df_local
        df_mod_C_BR_total = df_mod_C_BR
        df_mod_B_BR_total = df_mod_B_BR
        df_mod_A_BR_total = df_mod_A_BR
        df_mod_gf3o5_BR_total = df_mod_gf3o5_BR
        df_mod_gf0o1_BR_total = df_mod_gf0o1_BR
    else:
        df_BR_total = df_BR_total.merge(df_local, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')
        df_mod_C_BR_total = df_mod_C_BR_total.merge(df_mod_C_BR, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')
        df_mod_B_BR_total = df_mod_B_BR_total.merge(df_mod_B_BR, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')
        df_mod_A_BR_total = df_mod_A_BR_total.merge(df_mod_A_BR, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')
        df_mod_gf3o5_BR_total = df_mod_gf3o5_BR_total.merge(df_mod_gf3o5_BR, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')
        df_mod_gf0o1_BR_total = df_mod_gf0o1_BR_total.merge(df_mod_gf0o1_BR, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')

for dataframes in [df_BR_total, df_mod_C_BR_total, df_mod_B_BR_total, df_mod_A_BR_total, df_mod_gf3o5_BR_total, df_mod_gf0o1_BR_total]:
    dataframes["mass"] = dataframes["M0"]/1000
    dataframes["GoM"] = dataframes["GammaTotWp"]/dataframes["M0"]
    dataframes["dijet"] = dataframes["Zp_dijet"]

#df_BR_total["GoM"] = df_BR_total["GammaTotWp"]/df_BR_total["M0"]
#df_mod_C_BR_total["GoM"] = df_mod_C_BR_total["GammaTotWp"]/df_mod_C_BR_total["M0"]
#df_mod_B_BR_total["GoM"] = df_mod_B_BR_total["GammaTotWp"]/df_mod_B_BR_total["M0"]
#df_mod_A_BR_total["GoM"] = df_mod_A_BR_total["GammaTotWp"]/df_mod_A_BR_total["M0"]


#print(df_BR_total)
#print(df_BR_total.columns)
df_BR_total.to_csv("HVT_BRs.csv", index=False)

print("Model C")
print(df_mod_C_BR)
df_mod_C_BR_total.to_csv("HVT_modelC_BRs.csv", index=False)
print(df_mod_C_BR_total.columns)

print("Model B")
print(df_mod_B_BR_total)
df_mod_B_BR_total.to_csv("HVT_modelB_BRs.csv", index=False)
print(df_mod_B_BR_total.columns)

print("Model A")
print(df_mod_A_BR_total)
df_mod_A_BR_total.to_csv("HVT_modelA_BRs.csv", index=False)
print(df_mod_A_BR_total.columns)

print("Model gf=3.5")
print(df_mod_gf3o5_BR_total)
df_mod_gf3o5_BR_total.to_csv("HVT_model_gf_3o5_BRs.csv", index=False)
print(df_mod_gf3o5_BR_total.columns)

print("Model gf=3.5")
print(df_mod_gf0o1_BR_total)
df_mod_gf0o1_BR_total.to_csv("HVT_model_gf_0o1_BRs.csv", index=False)
print(df_mod_gf0o1_BR_total.columns)

print(len(df_BR_total[(df_BR_total["M0"] == 2000)]))
print(df_BR_total.columns)
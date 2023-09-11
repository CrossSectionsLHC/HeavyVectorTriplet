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


df_mod_B_BR_total = pd.DataFrame()
for particle in ["Z", "W"]: # 
    df_mod_B_BR = pd.DataFrame()
    name = 'HVT/BRs/BRs_%sprime.csv' % (particle)
    print(name)
    df_local = pd.read_csv(name)
    df_local["gH"] = round(df_local["gv"]*df_local["ch"], 1)
    df_local["gF"] = round(df_local["g"]*df_local["g"]*df_local["cl"]/df_local["gv"], 2)
    if particle == "Z" :
        df_local["Zp_dijet"] = df_local[['BRuu', 'BRdd', 'BRcc', 'BRss']].sum(axis=1)  
        df_local["ll"] = df_local[['BRee', 'BRmumu']].sum(axis=1) 
    else:
        df_local["Wp_dijet"] = df_local[['BRud', 'BRus', 'BRcd', 'BRcs']].sum(axis=1)  
        df_local["lv"] = df_local[['BReve', 'BRmvm']].sum(axis=1) 
    print(df_local.columns)
    # Z' 
    # 'M0', 'g', 'gv', 'ch', 'cl', 'GammaTot', 'BRWW', 'BRhZ', 'BRee', 'BRmumu', 'BRtautau', 'BRnunu', 'BRuu', 'BRdd', 'BRcc', 'BRss', 'BRbb', 'BRtt', 'BRll', 'BRqq', 'BRjets'
    # W' 
    # 'M0', 'g', 'gv', 'ch', 'cl', 'GammaTot', 'BRWH', 'BRWZ', 'BReve', 'BRmvm', 'BRtauvt', 'BRud', 'BRus', 'BRcd', 'BRcs', 'BRtb', 'BRlnu', 'BRqqbar', 'BRjets'
    gF_list_anal = sorted(list(set(list((df_local['gF'].values)))))
    gH_list_anal = sorted(np.around(list(set(list((df_local['gH'].values)))),2))
    print(gF_list_anal, len(gF_list_anal))
    print(gH_list_anal, len(gH_list_anal))

    df_local.rename({
        'BRhZ': 'ZH', 
        'BRWH': 'WH', 
        'BRWW' : 'WW',
        'BRWZ' : 'WZ',
        'BRtautau' : 'tautau',
        'BRnunu' : "vv",
        'BRtt' : 'tt',
        'BRtb' : "tb",
        "GammaTot" : "GammaTot%sp" % particle,
        }, axis=1, inplace=True)
    
    df_local["gH_mod_X_sign_gF"] = ((df_local['gF'].apply(func = lambda x : 1 if not x else -1)))*df_local["gH"]
    df_local["gF_mod"] = abs(df_local['gF'])
    df_local = df_local.drop(columns=[ "g", "gv",  "ch", "cl", "BRjets", "gH", "gF"])
    print(df_local.columns)

    df_local = df_local.drop_duplicates(subset=["gH_mod_X_sign_gF", "gF_mod", 'M0'])

    if len(df_mod_B_BR_total) == 0:
        df_mod_B_BR_total = df_local
    else:
        df_mod_B_BR_total = df_mod_B_BR_total.merge(df_local, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')

df_mod_B_BR_total["GoM"] = df_mod_B_BR_total["GammaTotWp"]/df_mod_B_BR_total["M0"]

df_mod_B_BR_total = df_mod_B_BR_total.drop(columns=[ 'BRuu', 'BRdd', 'BRcc', 'BRss', 'BReve', 'BRmvm', 'BRlnu', 'BRqqbar', 'BRee',  'BRmumu'])

print(df_mod_B_BR_total)
print(df_mod_B_BR_total.columns)
df_mod_B_BR_total.to_csv("HVT_BRs.csv", index=False)

print(len(df_mod_B_BR_total[(df_mod_B_BR_total["M0"] == 2000)]))
#import ROOT
#import sys
#from ROOT import *
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

plt.style.use(hep.style.ROOT)
pd.options.display.float_format = '{:,.5f}'.format

"""import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-i", "--input", action="store", type="string", dest="input", default="")
parser.add_option("-o", "--output", action="store", type="string", dest="output", default="")
(options, args) = parser.parse_args()"""

mass = "1000"
mass_int = 1000
inputName = "13TeV/comb2016/HVTscan_M%s.txt" % mass # options.input
outputName_intermediary = "13TeV/comb2016/HVTscan_M%s.csv" % mass # options.input
outputName = "13TeV/HVT_couplings_scan_M%s_XS.csv" % mass #options.output
#if len(options.output)==0: outputName = inputName.replace('HVTscan', 'scanHVT').split('_20')[0] + '.root'

columns = ['M0',
        #'Mc',
        'g',
        'gv',
        'ch',
        #'cq',
        'cl',
        #'c3',
        #'cvvw',
        #'cvvhh',
        #'cvvv',
        'total_widthV0',
        'BRWW',
        'BRhZ',
        'BRuu',
        'BRdd',
        'BRjets',
        'BRll',
        'BRnunu',
        'BRbb',
        'BRtt',
        'total_widthVc',
        'BRZW',
        'BRWGam',
        'BRWh',
        'BRud',
        'BRus',
        'BRlnu',
        'BRtb',
        #'CXp',
        'CX0',
        #'CXm',
        'CXc',]

if 1 < 0 :
    infile = open(inputName,'r')
    lines = infile.read().splitlines()
    df = pd.DataFrame(columns=columns)

    for mm, line in enumerate(lines):
        line = line.replace('{', '').replace('}', '').replace('*^', 'e')
        l = line.split(', ')
        df.loc[mm] = [
        float(l[0]),
        #float(l[1]),
        float(l[2]),
        float(l[3]),
        float(l[4]),
        #float(l[5]),
        float(l[6]),
        #float(l[7]),
        #float(l[8]),
        #float(l[9]),
        #float(l[10]),
        float(l[11]),
        float(l[12]),
        float(l[13]),
        float(l[14]),
        float(l[15]),
        float(l[14]) + float(l[15]),
        float(l[16]),
        float(l[17]),
        float(l[18]),
        float(l[19]),
        float(l[20]),
        float(l[21]),
        float(l[22]),
        float(l[23]),
        float(l[24]),
        float(l[25]),
        float(l[26]),
        float(l[27]),
        #float(l[28]),
        float(l[29]),
        #float(l[30]),
        float(l[28])+float(l[30])
        ]
    df.to_csv(outputName_intermediary, index=False)
else:
    #df = pd.DataFrame()
    df = pd.read_csv(outputName_intermediary)

"""
1.0TeV 1552 2160 50142
1.5TeV 1907 3267 74725
2.0TeV 2106 3996 77531
2.5TeV 2106 3996 77531
3.0TeV 2104 3996 77531
4.0TeV 2102 3996 77531
5.0TeV 2104 3996 77531
"""
# drop some columns
# 
#df = df.drop()

df["gH"] = round(df["gv"]*df["ch"], 1)
df["gF"] = round(df["g"]*df["g"]*df["cl"]/df["gv"], 1)

df["gFnotround"] = round(df["g"]*df["g"]*df["cl"]/df["gv"],3) 
df["gHnotround"] = round((df["gv"]*df["ch"]), 3)

df["gFnotround_diff"] = abs(round((df["g"]*df["g"]*df["cl"]/df["gv"]) - df["gF"], 3))
df["gHnotround_diff"] = abs(round((df["gv"]*df["ch"]) - df["gH"], 3))

for row in ['total_widthV0', 'total_widthVc']:
    df[row] = round(df[row],2)

for row in ['CX0', 'CXc']:
    df[row] = round(df[row],8)

for row in ['BRZW', 'BRWGam', 'BRWh', 'BRud', 'BRus', 'BRlnu', 'BRtb', 'BRWW', 'BRhZ', 'BRuu', 'BRdd', 'BRjets', 'BRll', 'BRnunu', 'BRbb', 'BRtt',]:
    df[row] = round(df[row],3)

#df_HVTC = df_HVTC.sort_values(by=["mass"])
df = df.sort_values(by=["gF", "gH"])


#That works, and I dont need to provide any table.
#As for sign, again was the numerics making me think that this sensitive to the relative sign.
#gF_round [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
#gH_round [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1] 

#print(df['gH'].nunique(), df['gF'].nunique())

#print("gF", df['gF'].value_counts())
#print("gH", df['gH'].value_counts())

#print("Why ")
#print(df['gH'])
#print(df.loc[(df['gH']== 3.6) & (df['gF']== -0.4)].to_string())
#print("===================== ")
#print(df.loc[(df['gH']== -3.6) & (df['gF']== -0.4)].to_string())


# this removes duplicates
gF_list = sorted(list(set(list((df['gF'].values)))))
gH_list = sorted(np.around(list(set(list((df['gH'].values)))),2))

print("gF", gF_list)
print("gH", gH_list)

#print("gF", max(df['gF'].values), min(df['gF'].values), len(gF_list))
#print("gH", max(df['gH'].values), min(df['gH'].values), len(gH_list))

print(" df_new ===================== ")

df_new = pd.DataFrame(columns=columns + ["n_points_interval","CX_min", "CX_max", "npoint_after"]) #.drop(columns=["gFnotround",  "gHnotround",  "gFnotround_diff",  "gHnotround_diff", "g",    "gv",    "ch",    "cl" ]) # df[(df['gHnotround_diff']== 0.0)]

#print(df_new.loc[(df_new['gH']== 3.0) ].to_string())
#print(df_new.loc[(df_new['gH']== 2.0)].to_string())

# drop if not the minimum
duplicate_sizes = []
zero_size = 0
multiple_size = 0
# better way here, implement after 
# https://stackoverflow.com/questions/54470917/pandas-groupby-and-select-rows-with-the-minimum-value-in-a-specific-column
#avr_CX_o_point_among_duplicates = [] # add on dataframe
#var_CX_among_duplicates = []

for ggH in gH_list:
    for ggF in gF_list:
        df_local = df.loc[(df['gH'] == ggH) & (df['gF'] == ggF)]
        if len(df_local) > 0:
            CX_min = min(df_local["CX0"].values)
            CX_max = max(df_local["CX0"].values)
            n_points_interval = len(df_local)

            ## let only min gH diff
            min_value_gH = df_local.groupby(['gH','gF'])["gHnotround_diff"].min()
            df_local = df_local.merge(min_value_gH, on=['gH','gF'], suffixes=('', '_min'))
            df_local = df_local[df_local["gHnotround_diff"]==df_local["gHnotround_diff_min"]]#.drop('gHnotround_diff_min', axis=1)

            ## let only gF min
            #print("only gH min", df_local)
            min_value_gF = df_local.groupby(['gH','gF'])["gFnotround_diff"].min()
            df_local = df_local.merge(min_value_gF, on=['gH','gF'], suffixes=('', '_min'))
            df_local = df_local[df_local["gFnotround_diff"]==df_local["gFnotround_diff_min"]]# .drop('gFnotround_diff_min', axis=1)
            #print(df_local)

            # That still has duplicates
            if(not len(df_local) == 1):
                #print("only gF min")
                df_local = df_local.drop_duplicates(subset=['gH', 'gF', 'gHnotround_diff_min',  'gFnotround_diff_min'])
                #print(df_local)           
            
            df_local["npoint_after"] = len(df_local)
            df_local["n_points_interval"] = n_points_interval
            df_local["CX_min"] = CX_min
            df_local["CX_max"] = CX_max
            df_new = df_new.append(df_local)
            ## drop the duplicates in gH and gF

print("entries with zero double minima", zero_size)    
print("entries with multiple double minima", multiple_size)   

df_new = df_new.drop(columns=["gFnotround",  "gHnotround" ])
print(df_new)

print(len(df_new), len(gF_list)*len(gH_list), len(df))

for gHtest in [3.6, 1.0, 0.5]:
  
  for gFtest in [0.4, 1]:
    fig = plt.figure(figsize=(8, 8))
    
    plt.plot(
        abs(df['gFnotround'].loc[(df['gH']== gHtest) & (df['gF']== gFtest)]).values, #  & (df["gHnotround_diff"] == 0)
        df['CX0'].loc[(df['gH']== gHtest) & (df['gF']== gFtest)].values, #  & (df["gHnotround_diff"] == 0)
        label='gH around %s, gF around %s' % (str(gHtest), str(gFtest))
    )

    plt.plot(
        abs(df['gFnotround'].loc[(df['gH']== -1*gHtest) & (df['gF']== gFtest) ]).values, # & (df["gHnotround_diff"] == 0)
        df['CX0'].loc[(df['gH']== -1*gHtest) & (df['gF']== gFtest) ].values, # & (df["gHnotround_diff"] == 0)
        label='gH around -%s, gF around %s' % (str(gHtest), str(gFtest))
    )

    plt.plot(
        abs(df['gFnotround'].loc[(df['gH']== gHtest) & (df['gF']== -1*gFtest)]).values, #  & (df["gHnotround_diff"] == 0)
        df['CX0'].loc[(df['gH']== gHtest) & (df['gF']== -1*gFtest)].values, #  & (df["gHnotround_diff"] == 0)
        label='gH around %s, gF around -%s' % (str(gHtest), str(gFtest))
    )

    plt.plot(
        abs(df['gFnotround'].loc[(df['gH']== -1*gHtest) & (df['gF']== -1*gFtest)]).values, #  & (df["gHnotround"] == 0)
        df['CX0'].loc[(df['gH']== -1*gHtest) & (df['gF']== -1*gFtest)].values, #  & (df["gHnotround"] == 0)
        label='gH around -%s, gF around -%s' % (str(gHtest), str(gFtest))
    )
    

    plt.legend(loc='lower right') # , header=
    plt.title("MZ' = %s GeV"  %(mass))
    plt.xlabel("|gF|")
    plt.ylabel("CX0 [pb]")
    for ext in ('.png', '.pdf'):
        plt.savefig("13TeV/comb2016/HVTscan_M%s_CX0_gF_around%s_gH_%s" %(mass, str(gFtest), str(gHtest)) + ext)
#print(df[['gH', 'gH', 'CX0', 'CXc', 'BRhZ', 'BRWh']].to_string()) # 

#print("duplicated")
#print(np.where(df.index.duplicated()))

# plot 
#"""
size_y = df['gv'].nunique()
size_x = df['ch'].nunique()
for mass_loop in [mass_int]:
    
    df_new['gF_sign'] = df_new['gF']/abs(df_new['gF'])
    df['gF_sign'] = df['gF']/abs(df['gF'])
    xlim = 3.9
    ylim = 1.3
    
    plt.xlabel('gH') # lam112
    plt.ylabel('gF') # sinTheta
    df_new["gH_mod"] = df_new['gF_sign']*df_new["gH"]
    df["gH_mod"] = df['gF_sign']*df["gH"]
    df_new["BRWp_jets"] = df_new["BRud"] + df_new["BRus"]
    
    for quantity in ['CX0', 'CXc', 'BRWW', 'BRhZ', 'BRjets', 'BRll', 'BRnunu', 'BRbb', 'BRtt',  'BRZW',  'BRWGam',  'BRWh',  "BRWp_jets",  'BRlnu' , 'BRtb', 'total_widthV0', 'total_widthVc' ]:
        fig = plt.figure(figsize=(8, 8))
        #table = df_new.reset_index().pivot('gF', 'gH', quantity)
        #ax = sns.heatmap(table)
        if not "CX" in quantity:
            xmin = -0.1
        else:
            xmin = -1*xlim
        ymin = 0
        labelZ = quantity
        if quantity == 'BRjets':
            labelZ = "BRZp_jets"

        ax = sns.heatmap(
            df_new[(df_new['gF'] > ymin) & (df_new['gH'] > xmin)].pivot_table(
                #index='y', columns='x', values='z'), 
                index='gF', columns='gH', values=quantity), 
                cbar_kws={'label': quantity}
            )


        ax.invert_yaxis()
        #cbar = sns.heatmap.colorbar(im)
        #cbar.ax.set_ylabel(quantity)
        #print(table)
        for ext in ('.png', '.pdf'):
            plt.savefig("13TeV/comb2016/HVTscan_M%s_2D_" % str(mass_loop) + quantity + ext)
        plt.clf()
        del fig
    #"""
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    #"""
    
    for quantity in ['CX0', 'CXc', 'BRhZ', 'BRWh']:
        for ff, frame in enumerate([df_new, df]):
            fig = plt.figure()
            #ax = fig.add_subplot()
            #ax.set_aspect('equal', adjustable='box')
            #legends = []
            #example_legend = Patch(facecolor=mcolors.to_rgba('black',0.5), edgecolor='black',linestyle="--",label="Exp. Excl. @ 95% CL",hatch="///")   
            #legends.append(example_legend)
            for gg, ggF in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3]):


                plt.plot(
                    frame['gH_mod'][(frame['M0']==mass_loop) & (frame['gF']==ggF)].values,
                    frame[quantity][(frame['M0']==mass_loop) & (frame['gF']==ggF)].values, 
                    #label='gH = %s' %str(ggF),
                    linestyle='None',
                    color=colors[gg],
                    marker="x"
                )

                plt.plot(
                    frame['gH_mod'][(frame['M0']==mass_loop) & (frame['gF']==-1*ggF)].values,
                    frame[quantity][(frame['M0']==mass_loop) & (frame['gF']==-1*ggF)].values, 
                    label='gF = %s' %str(ggF),
                    linestyle='solid',
                    linewidth=3,
                    color=colors[gg]
                )
            
            plt.xlabel("gH * sign(gF)")
            plt.ylabel(quantity)

            #plt.xscale('log')
            plt.yscale('log')

            plt.legend(loc='upper left', bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
            for ext in ('.png', '.pdf'):
                plt.savefig("13TeV/comb2016/HVTscan_M%s_%s_gH_pos_neg_" % (mass, quantity) + str(ff)  + ext)

            plt.clf()
            del fig
    #"""



#"""

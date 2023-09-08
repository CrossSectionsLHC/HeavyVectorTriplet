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
import glob

plt.style.use(hep.style.ROOT)
#pd.options.display.float_format = '{:,.8f}'.format

outputName_clean = "13TeV/comb2016/HVTscan_clean.csv" # options.input
txt_processed = True
csv_intermediary_processed = True
plot_wings = False

columns = ['M0',
        'Mc',
        'g',
        'gv',
        'ch',
        'cq',
        'cl',
        'c3',
        'cvvw',
        'cvvhh',
        'cvvv',
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
df_new_all = pd.DataFrame(columns=columns) 
colors = ['blue', 'orange', 'green',  'red', 'brown', 'pink', 'cyan', 'red',  'gray', 'olive', 'mediumseagreen', 'aquamarine']
masses = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500] # 
unc =    [0.1,   0.1,  0.1, 0.15, 0.15, 0.18,  0.4,  0.6,  0.6,  0.6,  0.6,  0.6  ]

if not csv_intermediary_processed : 
    for mass_int in masses:
        mass = str(mass_int) #"1500"
        print("processing points for mass", mass)
        #mass_int = 1500
        
        outputName_intermediary = "13TeV/comb2016/HVTscan_M%s.csv" % mass # options.input
        outputName = "13TeV/HVT_couplings_scan_M%s_XS.csv" % mass #options.output
        #if len(options.output)==0: outputName = inputName.replace('HVTscan', 'scanHVT').split('_20')[0] + '.root'

        if not txt_processed:
            inputName = "13TeV/comb2016/HVTscan_M%s.txt" % mass # options.input
            # take from the txt only once, that takes time
            infile = open(inputName,'r')
            lines = infile.read().splitlines()
            df = pd.DataFrame(columns=columns)

            for mm, line in enumerate(lines):
                line = line.replace('{', '').replace('}', '').replace('*^', 'e')
                l = line.split(', ')
                df.loc[mm] = [
                float(l[0]),
                float(l[1]),
                float(l[2]),
                float(l[3]),
                float(l[4]),
                float(l[5]),
                float(l[6]),
                float(l[7]),
                float(l[8]),
                float(l[9]),
                float(l[10]),
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

        df["gH"] = round(df["gv"]*df["ch"], 1)
        df["gF"] = round(df["g"]*df["g"]*df["cl"]/df["gv"], 1)

        df["gFnotround"] = round(df["g"]*df["g"]*df["cl"]/df["gv"],4) 
        df["gHnotround"] = round((df["gv"]*df["ch"]), 3)

        df["gFnotround_diff"] = abs(round((df["g"]*df["g"]*df["cl"]/df["gv"]) - df["gF"], 3))
        df["gHnotround_diff"] = abs(round((df["gv"]*df["ch"]) - df["gH"], 3))

        for row in ['total_widthV0', 'total_widthVc']:
            df[row] = round(df[row],2)

        for row in ['M0']:
            df[row] = round(df[row],0)

        for row in ['gv', 'ch', 'cl']:
            df[row] = round(df[row],1)

        for row in ['CX0', 'CXc']:
            df[row] = round(df[row],8)

        for row in ['BRZW', 'BRWGam', 'BRWh', 'BRud', 'BRus', 'BRlnu', 'BRtb', 'BRWW', 'BRhZ', 'BRuu', 'BRdd', 'BRjets', 'BRll', 'BRnunu', 'BRbb', 'BRtt',]:
            df[row] = round(df[row],3)

        #df_HVTC = df_HVTC.sort_values(by=["mass"])
        df = df.sort_values(by=["gF", "gH"])

        #print(df['gH'].nunique(), df['gF'].nunique())
        # this removes duplicates
        gF_list = sorted(list(set(list((df['gF'].values)))))
        gH_list = sorted(np.around(list(set(list((df['gH'].values)))),2))

        print("gF", gF_list)
        print("gH", gH_list)

        #print("gF", max(df['gF'].values), min(df['gF'].values), len(gF_list))
        #print("gH", max(df['gH'].values), min(df['gH'].values), len(gH_list))

        df_new = pd.DataFrame(columns=columns) 

        #df_new = pd.DataFrame(columns=columns + ["n_points_interval","CX_min", "CX_max", "npoint_after"]) 
        #.drop(columns=["gFnotround",  "gHnotround",  "gFnotround_diff",  "gHnotround_diff", "g",    "gv",    "ch",    "cl" ]) # df[(df['gHnotround_diff']== 0.0)]

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
                    
                    #df_local["npoint_after"] = len(df_local)
                    #df_local["n_points_interval"] = n_points_interval
                    #df_local["CX_min"] = CX_min
                    #df_local["CX_max"] = CX_max
                    df_new = df_new.append(df_local)
                    ## drop the duplicates in gH and gF

        print("entries with zero double minima", zero_size)    
        print("entries with multiple double minima", multiple_size)   

        #df_new_all = df_new_all.drop(columns=["gFnotround",  "gHnotround", "gFnotround_diff",  "gHnotround_diff"]) # , "n_points_interval" ,    "CX_min",     "CX_max",  "npoint_after" 
        #df_new.to_csv(outputName, index=False)

        #print(df_new)

        print("cleaning", mass_int, len(df_new), len(gF_list)*len(gH_list), len(df))
        df_new_all = df_new_all.append(df_new) 

        if plot_wings :
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
                for ext in ['.pdf']:
                    plt.savefig("13TeV/comb2016/HVTscan_M%s_CX0_gF_around%s_gH_%s" %(mass, str(gFtest), str(gHtest)) + ext)

        for mass_loop in [mass_int]:
            
            df_new['gF_sign'] = df_new['gF']/abs(df_new['gF'])
            df['gF_sign'] = df['gF']/abs(df['gF'])
            
            plt.xlabel('gH') # lam112
            plt.ylabel('gF') # sinTheta
            df_new["gH_mod"] = df_new['gF_sign']*df_new["gH"]
            df["gH_mod"] = df['gF_sign']*df["gH"]

            if plot_wings :
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
                        for ext in ['.pdf']:
                            plt.savefig("13TeV/comb2016/HVTscan_M%s_%s_gH_pos_neg_" % (mass, quantity) + str(ff)  + ext)

                        plt.clf()
                        del fig

    df_new_all = df_new_all.drop(columns=[ "gFnotround_diff",  "gHnotround_diff"]) # , "n_points_interval" ,    "CX_min",     "CX_max",  "npoint_after" 
    df_new_all.to_csv(outputName_clean, index=False)
else:
    # read the csv
    df_new_all = pd.read_csv(outputName_clean)

df_new_all['CX0_hat'] = df_new_all['CX0']/(df_new_all["gFnotround"]*df_new_all["gFnotround"])
df_new_all['CXc_hat'] = df_new_all['CXc']/(df_new_all["gFnotround"]*df_new_all["gFnotround"])
df_new_all.replace([np.inf, -np.inf], 0, inplace=True)

ext = '.pdf'
# 'cq' == 'cl' == 'c3'
# 'g', 'gv', 'ch', 

# gH = gv*ch
# gF = g*g*cl/gv

print(df_new_all[['M0', 'Mc', 'g', 'gv', 'ch', 'cq', 'cl', 'c3', 'gF', 'gH' , 'CX0_hat', 'CXc_hat']])

plt_BRs = False
if plt_BRs :
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
            #for ext in ['.pdf']:
            plt.savefig("13TeV/comb2016/HVTscan_M%s_2D_" % str(mass_loop) + quantity + ext)
            plt.clf()
            del fig


n_bins = 200
#print(df_new_all['gv'][(df_new_all['M0']==1000)].to_string())


# draw sigHat with gF
cx0_hat = []
cxp_hat = []
for mm, mass_loop in enumerate(masses): # 
    fig = plt.figure()
    for gg, quantity in enumerate(['CX0_hat', 'CXc_hat', 'CX0', 'CXc']):
        plt.plot(
            abs(df_new_all["gFnotround"][(df_new_all['M0']==mass_loop) & (df_new_all['gH']==1.0) & (abs(df_new_all['gF']) > 0)].values),
            df_new_all[quantity][(df_new_all['M0']==mass_loop) & (df_new_all['gH']==1.0) & (abs(df_new_all['gF']) > 0)].values, 
            label=quantity,
            linestyle='solid',
            color=colors[gg],
            marker="x"
        )

        if "hat" in quantity:
            truthC_hat = df_new_all[quantity][(df_new_all['M0']==mass_loop) & (df_new_all['gH']==1.0) & (df_new_all['gF'] == 1.0)].values

            if 'CX0' in quantity:
                cx0_hat += [truthC_hat]
            else:
                cxp_hat += [truthC_hat]

            #print(truthC_hat[0])
            plt.plot(
                [0, max(abs(df_new_all["gFnotround"][(df_new_all['M0']==mass_loop) & (df_new_all['gH']==1.0) & (abs(df_new_all['gF']) > 0)].values))],
                [truthC_hat, truthC_hat], 
                linestyle='dotted',
                color=colors[gg],
            )

            plt.plot(
                [0, max(abs(df_new_all["gFnotround"][(df_new_all['M0']==mass_loop) & (df_new_all['gH']==1.0) & (abs(df_new_all['gF']) > 0)].values))],
                [truthC_hat*(1+unc[mm]), truthC_hat*(1+unc[mm])], 
                linestyle='dashed',
                color=colors[gg],
            )

            plt.plot(
                [0, max(abs(df_new_all["gFnotround"][(df_new_all['M0']==mass_loop) & (df_new_all['gH']==1.0) & (abs(df_new_all['gF']) > 0)].values))],
                [truthC_hat*(1-unc[mm]), truthC_hat*(1-unc[mm])], 
                linestyle='dashed',
                color=colors[gg],
            )

    plt.xlabel("|gF|")
    plt.ylabel("Cross section [pb]")

    #plt.xscale('log')
    #plt.yscale('log')

    plt.legend(loc='upper left', title="MZp = %s GeV" % str(mass_loop), ncol=2) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
    #for ext in ['.pdf']:
    plt.savefig("13TeV/comb2016/HVTscan_sigHat_%s_gF_mZp%s" % (quantity, str(mass_loop)) + ext)

    plt.clf()
    del fig

### now take the XS of model B, turn to CX_hat
gF_modelB = 0.145725972
df_mod_B = pd.read_csv("13TeV/HVTB_XS.csv") # model B Xsec
df_mod_C = pd.read_csv("13TeV/HVTC_XS.csv") 

###############
# now reformat Andrea's BR's
# for W' it is missing tb, take from jets
# how many families are in Zp > jets?
# I try to get by df_local["BRjets"]/ df_local["BRtt"] and do not get an exact number, I get 6.047057, 6.030542, ...
# why BR ll and nunu are different? isn't 3 families in each?
do_plots = False
df_mod_B_BR_total_all_masses = pd.DataFrame()
for massVp in [1000, 2000, 3000, 4000] : # 
    df_mod_B_BR_total = pd.DataFrame()
    for particle in ["Z", "W"]: # 
        df_mod_B_BR = pd.DataFrame()
        #for name in glob.glob('HVT/%sprime/BRs_%sprime_M%s_*.csv' % (particle, particle, str(massVp))):
        for name in glob.glob('HVT/%sprime/BRs_%sprime_M%s_gv1_*.csv' % (particle, particle, str(massVp))):
            #brjets = "Zp_BRjets" if "Zprime" in name else "Wp_BRjets" 
            #print(name)
            df_local = pd.read_csv(name)
            df_local["gH"] = round(df_local["gv"]*df_local["ch"], 1)
            df_local["gF"] = round(df_local["g"]*df_local["g"]*df_local["cl"]/df_local["gv"], 1)
            if particle == "Z" :
                df_local["ZpBRjets"] = 4*df_local["BRtt"]
                df_local["BRbb"] = df_local["BRtt"]
            else:
                df_local["WpBRjets"] = df_local["BRjets"]

            
            df_local.rename({'BRWZ': 'BRZW', 'BRWH': 'BRWh', "GammaTot" : "GammaTot%sp" % particle}, axis=1, inplace=True)
            #print(df_local)
            #df_mod_B_BR = df_mod_B_BR.merge(df_local, on=["gH", "gF"]).fillna(method='ffill') # model B BR
            df_mod_B_BR = df_mod_B_BR.append(df_local)

        #print(df_mod_B_BR)
        #print(df_mod_B_BR.columns)
        #print(df_new_all.columns)

        if particle == "Z" :
            df_mod_B_BR["Zp_BRsum"] = df_mod_B_BR["BRWW"] + df_mod_B_BR["BRhZ"] + df_mod_B_BR["BRll"] + df_mod_B_BR["BRnunu"] + df_mod_B_BR["BRtt"] + df_mod_B_BR["BRjets"]
            df_mod_B_BR["Zp_BRsum_tweak"] = df_mod_B_BR["BRWW"] + df_mod_B_BR["BRhZ"] + df_mod_B_BR["BRnunu"] + df_mod_B_BR["BRnunu"] + df_mod_B_BR["BRtt"] + df_mod_B_BR["ZpBRjets"] + df_mod_B_BR["BRbb"]

            df_new_all["Zp_BRsum"] = df_new_all["BRWW"] + df_new_all["BRhZ"] + df_new_all["BRll"] + df_new_all["BRnunu"] + df_new_all["BRtt"] + 4*df_new_all["BRuu"] +df_new_all['BRbb']
            df_new_all["ZpBRjets"] = 4*df_new_all["BRtt"]
        else :
            df_new_all["WpBRjets"] = 2*df_new_all["BRud"]
            df_new_all["Wp_BRsum"] = df_new_all["BRZW"] + df_new_all["BRWh"] + df_new_all["WpBRjets"] + df_new_all["BRlnu"] + df_new_all["BRtb"]

            df_mod_B_BR["Wp_BRsum"] = df_mod_B_BR["BRZW"] + df_mod_B_BR["BRWh"] + df_mod_B_BR["WpBRjets"] + df_mod_B_BR["BRlnu"]


        gF_list_anal = sorted(list(set(list((df_mod_B_BR['gF'].values)))))
        gH_list_anal = sorted(np.around(list(set(list((df_mod_B_BR['gH'].values)))),2))

        gF_list = sorted(list(set(list((df_new_all['gF'][(df_new_all['M0']==4000)].values)))))

        #print("gF_anal", gF_list_anal)
        #print("gH_anal", gH_list_anal)
        #print("gF", gF_list_anal)
        # remove 
        df_mod_B_BR["gH_mod_X_sign_gF"] = ((df_mod_B_BR['gF'].apply(func = lambda x : 1 if not x else -1)))*df_mod_B_BR["gH"]
        df_mod_B_BR["gF_mod"] = abs(df_mod_B_BR['gF'])
        print(df_mod_B_BR[(df_mod_B_BR['gF'] ==0 )] )
        df_mod_B_BR = df_mod_B_BR.drop(columns=[ "g", "gv",  "ch", "cl", "BRjets", "gH", "gF"])
        df_mod_B_BR = df_mod_B_BR.drop_duplicates(subset=["gH_mod_X_sign_gF", "gF_mod", 'M0'])
        #df_new_all.replace([np.nan, -np.nan], 0, inplace=True)
        #df_mod_B_BR = df_mod_B_BR.fillna(0)

        #print(df_mod_B_BR)

        if len(df_mod_B_BR_total) == 0:
            df_mod_B_BR_total = df_mod_B_BR
        else:
            df_mod_B_BR_total = df_mod_B_BR_total.merge(df_mod_B_BR, on=["gH_mod_X_sign_gF", "gF_mod", 'M0']).fillna(method='ffill')

        if particle == "Z" :
            brs_loop = ["BRhZ", "BRWW", "BRll", "Zp_BRsum", "BRjets", "BRtt", 'BRnunu', "ZpBRjets"]
        else:
            brs_loop = ["BRWh", "BRZW", 'WpBRjets', "BRlnu", "Wp_BRsum"]

        # do anal X numeric
        if do_plots:
            for  quantity in brs_loop:
                fig = plt.figure()
                for gg, ggF in enumerate([0.2, 0.4, 0.8, 1.2]) :

                    if 1 > 0 : # 
                        plt.plot(
                            df_new_all['gH'][(df_new_all['gF']==ggF) & (df_new_all['M0']==massVp)].values, 
                            df_new_all[quantity][(df_new_all['gF']==ggF) & (df_new_all['M0']==massVp)].values,
                            label="gF = %s (num.)" % str(ggF),
                            linestyle='dashed',
                            color=colors[gg],
                            marker="x"
                        )

                    if quantity == "Zp_BRsum":
                        plt.plot(
                            df_mod_B_BR['gH'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values, 
                            df_mod_B_BR["Zp_BRsum_tweak"][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values,
                            label="gF = %s (analy.)" % str(ggF),
                            linestyle='solid', # dotted
                            color=colors[gg]
                        )       
                    else :
                        plt.plot(
                            df_mod_B_BR['gH'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values, 
                            df_mod_B_BR[quantity][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values,
                            label="gF = %s (analy.)" % str(ggF),
                            linestyle='solid',
                            color=colors[gg],
                            marker="o"
                        )
            
                plt.xlabel("gH")
                plt.ylabel(quantity)

                if not quantity in ["Zp_BRsum", "Wp_BRsum"]:
                    plt.yscale('log')

                if massVp == 1000:
                    plt.xlim(-4, 4)

                plt.legend(loc='lower left', ncol=1, title="MV' = %s GeV" % str(massVp)) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
                plt.tight_layout()
                #for ext in ['.pdf']:
                plt.savefig("13TeV/comb2016/HVTscan_BR%s_%s_MV%s_check" % (particle, quantity, str(massVp))  + ext)

            #Do quadrants test
            df_mod_B_BR['gH_mod'] = (df_mod_B_BR['gF']/abs(df_mod_B_BR['gF']))*df_mod_B_BR['gH']
            for  quantity in brs_loop:
                fig = plt.figure()
                for gg, ggF in enumerate([0.2, 0.4, 0.8, 1.2]) :
                    plt.plot(
                        #df_mod_B_BR['gH_mod'][(df_mod_B_BR['gF']==ggF)].values, 
                        df_mod_B_BR['gH'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values, 
                        df_mod_B_BR[quantity][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values,
                        label="gF = %s (pos.)" % str(ggF),
                        linestyle='solid',
                        color=colors[gg],
                        marker="o"
                    )
    
                    plt.plot(
                        #df_mod_B_BR['gH_mod'][(df_mod_B_BR['gF']==-1*ggF)].values, 
                        df_mod_B_BR['gH'][(df_mod_B_BR['gF']==-1*ggF) & (df_mod_B_BR['gv']==3)].values, 
                        df_mod_B_BR[quantity][(df_mod_B_BR['gF']==-1*ggF) & (df_mod_B_BR['gv']==3)].values,
                        label="gF = %s (neg.)" % str(ggF),
                        linestyle='dashed',
                        color=colors[gg],
                        marker="x"
                    )
            
                #plt.xlabel("sign(gF)*gH")
                plt.xlabel("gH")
                plt.ylabel(quantity)

                if not quantity in ["Zp_BRsum", "Wp_BRsum"]:
                    plt.yscale('log')

                if massVp == 1000:
                    plt.xlim(-4, 4)

                plt.legend(loc='lower left', ncol=1, title="MV' = %s GeV" % str(massVp)) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
                plt.tight_layout()
                #for ext in ['.pdf']:
                plt.savefig("13TeV/comb2016/HVTscan_BR%s_%s_MV%s_quadrants" % (particle, quantity, str(massVp))  + ext)

            #Do gV test
            for  quantity in brs_loop:
                fig = plt.figure()
                for gg, ggF in enumerate([0.2, 0.4, 0.8, 1.2]) :
                    plt.plot(
                        #df_mod_B_BR['gH_mod'][(df_mod_B_BR['gF']==ggF)].values, 
                        df_mod_B_BR['gH'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values, 
                        df_mod_B_BR[quantity][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values,
                        label="gF = %s (gV = 3.)" % str(ggF),
                        linestyle='solid',
                        color=colors[gg],
                        marker="o"
                    )
    
                    plt.plot(
                        #df_mod_B_BR['gH_mod'][(df_mod_B_BR['gF']==-1*ggF)].values, 
                        df_mod_B_BR['gH'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==1)].values, 
                        df_mod_B_BR[quantity][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==1)].values,
                        label="gF = %s (gV = 1.)" % str(ggF),
                        linestyle='dashed',
                        color=colors[gg],
                        marker="x"
                    )
            
                #plt.xlabel("sign(gF)*gH")
                plt.xlabel("gH")
                plt.ylabel(quantity)

                if not quantity in ["Zp_BRsum", "Wp_BRsum"]:
                    plt.yscale('log')

                if massVp == 1000:
                    plt.xlim(-4, 4)

                plt.legend(loc='lower left', ncol=1, title="MV' = %s GeV" % str(massVp)) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
                plt.tight_layout()
                #for ext in ['.pdf']:
                plt.savefig("13TeV/comb2016/HVTscan_BR%s_%s_MV%s_gVtest" % (particle, quantity, str(massVp))  + ext)

            # suggestion for paper
            fig = plt.figure()
            linestyles = ["solid", "dotted", 'dashed']
            for  qq, quantity in enumerate(brs_loop):
                if "sum" in quantity or quantity ==  "BRjets":
                    continue
                for gg, ggF in enumerate([0.2, 1.2]) : # , 0.4 0.8, 
                    if gg == 0 :
                        plt.plot(
                            df_mod_B_BR['gH_mod'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values, 
                            df_mod_B_BR[quantity][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values,
                            label=quantity,
                            linestyle=linestyles[gg],
                            linewidth=2.0,
                            color=colors[qq]
                        )
                    else:
                        plt.plot(
                            df_mod_B_BR['gH_mod'][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values, 
                            df_mod_B_BR[quantity][(df_mod_B_BR['gF']==ggF) & (df_mod_B_BR['gv']==3)].values,
                            linestyle=linestyles[gg],
                            linewidth=2.0,
                            color=colors[qq]
                        )
            
                plt.xlabel("sign(gF)*gH")
                plt.ylabel("BR(Z' -> XY)")

                if not quantity in ["Zp_BRsum", "Wp_BRsum"]:
                    plt.yscale('log')

                if massVp == 1000:
                    plt.xlim(-4, 4)
                #else:
                #    plt.xlim(0, 8)
                plt.ylim(0.001, 1.4)

            #
            plt.legend(loc='upper center',ncol=3) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
            plt.tight_layout()
            plt.title("MV' = %s GeV" % str(massVp))
            # title=, 
            #for ext in ['.pdf']:
            plt.savefig("13TeV/comb2016/HVTscan_BR%s_MV%s_paper" % (particle, str(massVp))  + ext)

    df_mod_B_BR_total_all_masses = df_mod_B_BR_total_all_masses.append(df_mod_B_BR_total)

df_mod_B_BR_total_all_masses.rename({
    'BRWW' : "WW", 
    'BRhZ' : "ZH", 
    'BRll': "ll", 
    'BRnunu' : "vv", 
    'BRtt' : "tt",
    'ZpBRjets' : "Zp_dijet", 
    'BRbb' : "bb", 
    'BRWh': "WH", 
    'BRZW': "WZ", 
    'BRlnu' : "lv", 
    'WpBRjets' : "Wp_dijet"
    }, axis=1, inplace=True)
#print(df_mod_B_BR_total_all_masses)
#print(df_mod_B_BR_total_all_masses.columns)
df_mod_B_BR_total_all_masses.to_csv("HVT_BRs.csv", index=False)
print(df_mod_B_BR_total_all_masses[(df_mod_B_BR_total_all_masses["gF_mod"] == 0)])
#
###############################


if do_plots:
    ###############################

    fig = plt.figure()
    for gg, quantity in enumerate(['CX0_hat', 'CXc_hat']):
        if 'CX0' in quantity:
            CX = cx0_hat
        else:
            CX = cxp_hat 

        plt.plot(
            masses,         
            CX,
            label=quantity,
            linestyle='dashed',
            color=colors[gg],
            marker="x"
        )

    for gg, quantity in enumerate(['CX0_hat_fromB', 'CXc_hat_fromB']):

        if 'CX0' in quantity:
            CX ='CX0(pb)'
            CX_up = "CX0_up"
            CX_do = "CX0_down"
            quantity_label = "CX0_hat_DY(pb)"
            quantity_label_up = "CX0_hat_DY_up(pb)"
            quantity_label_do = "CX0_hat_DY_down(pb)"
        else:
            df_mod_B['CXc'] = df_mod_B['CX-(pb)'] +  df_mod_B['CX+(pb)']
            CX ='CXc'
            CX_up = "CX+-_up"
            CX_do = "CX+-_down"
            quantity_label = "CXc_hat_DY(pb)"
            quantity_label_up = "CXc_hat_DY_up(pb)"
            quantity_label_do = "CXc_hat_DY_down(pb)"

        df_mod_B[quantity_label] = df_mod_B[CX]/(gF_modelB*gF_modelB)
        df_mod_B[quantity_label_up] = df_mod_B[CX_up]/(gF_modelB*gF_modelB)
        df_mod_B[quantity_label_do] = df_mod_B[CX_do]/(gF_modelB*gF_modelB)

        plt.plot(
            df_mod_B['mass'], 
            df_mod_B[quantity_label],
            label=quantity,
            linestyle='solid',
            color=colors[gg],
            marker="o"
        )

        plt.plot(
            df_mod_B['mass'], 
            df_mod_B[quantity_label_up],
            linestyle='dotted',
            color=colors[gg]
        )

        plt.plot(
            df_mod_B['mass'], 
            df_mod_B[quantity_label_do],
            linestyle='dotted',
            color=colors[gg]
        )

    plt.xlabel("MZp")
    plt.ylabel("#sigma hat DY [pb]")

    plt.yscale('log')
    plt.legend(loc='upper right', ncol=1) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
    #for ext in ['.pdf']:
    plt.savefig("13TeV/comb2016/HVTscan_sigHat_modelB_check"  + ext)

    plt.clf()
    del fig

    ##################################################
    fig = plt.figure()
    for gg, quantity in enumerate(['Zprime_cH1', 'Wprime_cH1']):
        CX_up = quantity + "_Up"
        CX_do = quantity + "_Down"
        if "Zprime" in quantity:
            quantity_label = "CX0_hat_VBF(pb)"
            quantity_label_up = "CX0_hat_VBF_up(pb)"
            quantity_label_do = "CX0_hat_VBF_down(pb)"
        else:
            quantity_label = "CXc_hat_VBF(pb)"
            quantity_label_up = "CXc_hat_VBF_up(pb)"
            quantity_label_do = "CXc_hat_VBF_down(pb)"

        df_mod_C[quantity_label] = df_mod_C[quantity]
        df_mod_C[quantity_label_up] = df_mod_C[CX_up]
        df_mod_C[quantity_label_do] = df_mod_C[CX_do]

        plt.plot(
            df_mod_C['mass'], 
            df_mod_C[quantity],
            label=quantity_label,
            linestyle='solid',
            color=colors[gg+2],
            marker="o"
        )

        plt.plot(
            df_mod_C['mass'], 
            df_mod_C[CX_up],
            linestyle='dotted',
            color=colors[gg+2]
        )

        plt.plot(
            df_mod_C['mass'], 
            df_mod_C[CX_do],
            linestyle='dotted',
            color=colors[gg+2]
        )

    plt.xlabel("MZp")
    plt.ylabel("#sigma hat VBF [pb]")

    plt.yscale('log')
    plt.legend(loc='upper right', ncol=1) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
    #for ext in ['.pdf']:
    plt.savefig("13TeV/comb2016/HVTscan_sigHat_modelC_check"  + ext)
    plt.clf()
    del fig
    ###################################################

    fig = plt.figure()
    for gg, quantity in enumerate(['Zprime_cH3', 'Wprime_cH3']):

        plt.plot(
            df_mod_C['mass'], 
            df_mod_C[quantity],
            linestyle='solid',
            color=colors[gg+2],
            label=quantity,
            marker="o"
        )

    for gg, quantity in enumerate(['Zprime_cH1', 'Wprime_cH1']):

        plt.plot(
            df_mod_C['mass'], 
            9*df_mod_C[quantity],
            linestyle='solid',
            color=colors[gg+2],
            label="sig_hat * (gV * cH)^2, gV =1, cH = 3",
            marker="x"
        )

    plt.xlabel("MZp")
    plt.ylabel("\sigma VBF [pb]")

    plt.yscale('log')
    plt.legend(loc='upper right', ncol=1) #, bbox_to_anchor=(-0.1,1.15),  frameon=True, edgecolor='black',framealpha=1,fancybox=False, ncol=4) 
    #for ext in ['.pdf']:
    plt.savefig("13TeV/comb2016/HVTscan_sigHat_modelC_scaling_check"  + ext)
    plt.clf()
    del fig
    ###################################################

    output_to_sig_hat = "13TeV/HVT_XS_sigma_hat.csv"
    df_hat = df_mod_B.merge(df_mod_C, on="mass").fillna(method='ffill')
    # merge the datasets
    df_hat = df_hat.drop(columns=["CX-(pb)", "CX+(pb)",  "CX+-_PDF-", "CX+-_PDF+",  "CX+-_QCD+",   "CX+-_QCD-", "CX0(pb)",   "CX0_PDF+",   "CX0_PDF-",   "CX0_QCD+",   "CX0_QCD-",	 "CX0_up",	"CX0_down",	"CX+-_up",	"CX+-_down", "Wprime_cH3",	"Wprime_cH3_Up",	"Wprime_cH3_Down",	"Zprime_cH3",	"Zprime_cH3_Up",	"Zprime_cH3_Down",	"Zprime_cH1", 	"Zprime_cH1_Up",	"Zprime_cH1_Down",	"Wprime_cH1",	"Wprime_cH1_Up",	"Wprime_cH1_Down", "WH_x",      "WW_x",      "WZ_x",      "ZH_x",  'WH_y', 'WW_y', 'ZH_y', 'WZ_y', 'CXc',])

    print(df_hat.columns)
    df_hat.to_csv(output_to_sig_hat, index=False)

    ###############
    for quantity in ["gv", "g", 'CX0_hat', 'CXc_hat']: 
        fig = plt.figure(figsize=(8, 8))
        for gg, mass_loop in enumerate([1000, 2000, 3000, 4000]):
            plt.hist(
                df_new_all[quantity][(df_new_all['M0']==mass_loop)].values,
                bins=n_bins,
                label='MZp = %s' % str(mass_loop),
                color=colors[gg]
            )
        if quantity == 'gv':
            plt.yscale('log')
        plt.ylabel("#occurencies")
        plt.xlabel(quantity)
        plt.legend(loc='upper right')
        #for ext in ( '.pdf'):
        plt.savefig("13TeV/comb2016/HVTscan_%s_by_MZp_" % quantity + ext)

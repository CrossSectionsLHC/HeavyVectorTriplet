#!/usr/bin/env 
import pandas as pd
import numpy as np

# uncertainties in production XS, taken from https://gitlab.cern.ch/cms-b2g/diboson-combination/combination-2016/-/blob/master/theory.py#L27-32
THEORY = {}
THEORY['W'] = {}
THEORY['W']['central'] = {"2900": {"BRWh": 0.475608, "BRWW": 0.471506, "CX-(pb)": 0.00147046, "CX+(pb)": 0.00562985, "CX0(pb)": 0.00323048, "BRZh": 0.476804, "BRWZ": 0.472703}, "1300": {"BRWh": 0.489978, "BRWW": 0.463352, "CX-(pb)": 0.110992, "CX+(pb)": 0.285322, "CX0(pb)": 0.198844, "BRZh": 0.495803, "BRWZ": 0.469214}, "5700": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 2.54763e-06, "CX+(pb)": 1.01644e-05, "CX0(pb)": 7.1896e-06, "BRZh": 0.474847, "BRWZ": 0.472759}, "1200": {"BRWh": 0.493896, "BRWW": 0.460749, "CX-(pb)": 0.153962, "CX+(pb)": 0.381432, "CX0(pb)": 0.271104, "BRZh": 0.500699, "BRWZ": 0.467605}, "4100": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 9.8425e-05, "CX+(pb)": 0.000413377, "CX0(pb)": 0.000236723, "BRZh": 0.474847, "BRWZ": 0.472759}, "4300": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 6.29979e-05, "CX+(pb)": 0.00026521, "CX0(pb)": 0.000153786, "BRZh": 0.474847, "BRWZ": 0.472759}, "3000": {"BRWh": 0.475412, "BRWW": 0.471596, "CX-(pb)": 0.00116241, "CX+(pb)": 0.00451288, "CX0(pb)": 0.00257265, "BRZh": 0.47653, "BRWZ": 0.472715}, "3900": {"BRWh": 0.474296, "BRWW": 0.472095, "CX-(pb)": 0.000150705, "CX+(pb)": 0.000629536, "CX0(pb)": 0.000357546, "BRZh": 0.474958, "BRWZ": 0.472758}, "2800": {"BRWh": 0.475827, "BRWW": 0.471405, "CX-(pb)": 0.00186407, "CX+(pb)": 0.00703223, "CX0(pb)": 0.00405245, "BRZh": 0.477109, "BRWZ": 0.472688}, "4200": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 7.87411e-05, "CX+(pb)": 0.000331213, "CX0(pb)": 0.000190806, "BRZh": 0.474847, "BRWZ": 0.472759}, "1900": {"BRWh": 0.479838, "BRWW": 0.469395, "CX-(pb)": 0.0184817, "CX+(pb)": 0.0575243, "CX0(pb)": 0.036303, "BRZh": 0.482604, "BRWZ": 0.472168}, "2200": {"BRWh": 0.477892, "BRWW": 0.470404, "CX-(pb)": 0.0082689, "CX+(pb)": 0.0277791, "CX0(pb)": 0.0168765, "BRZh": 0.479961, "BRWZ": 0.472477}, "5800": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 2.00722e-06, "CX+(pb)": 7.9506e-06, "CX0(pb)": 5.7453e-06, "BRZh": 0.474847, "BRWZ": 0.472759}, "6000": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 1.2378e-06, "CX+(pb)": 4.8299e-06, "CX0(pb)": 3.6528e-06, "BRZh": 0.474847, "BRWZ": 0.472759}, "4700": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 2.57365e-05, "CX+(pb)": 0.000107895, "CX0(pb)": 6.49492e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "4000": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 0.000123035, "CX+(pb)": 0.000515559, "CX0(pb)": 0.000293747, "BRZh": 0.474847, "BRWZ": 0.472759}, "2300": {"BRWh": 0.477422, "BRWW": 0.470638, "CX-(pb)": 0.00638847, "CX+(pb)": 0.021956, "CX0(pb)": 0.0131933, "BRZh": 0.479317, "BRWZ": 0.472536}, "5500": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 4.08137e-06, "CX+(pb)": 1.65057e-05, "CX0(pb)": 1.12173e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "3100": {"BRWh": 0.475236, "BRWW": 0.471677, "CX-(pb)": 0.000921187, "CX+(pb)": 0.00362067, "CX0(pb)": 0.00205583, "BRZh": 0.476283, "BRWZ": 0.472724}, "5900": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 1.5779e-06, "CX+(pb)": 6.2037e-06, "CX0(pb)": 4.5843e-06, "BRZh": 0.474847, "BRWZ": 0.472759}, "5600": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 3.22782e-06, "CX+(pb)": 1.2968e-05, "CX0(pb)": 8.9865e-06, "BRZh": 0.474847, "BRWZ": 0.472759}, "1500": {"BRWh": 0.484915, "BRWW": 0.466514, "CX-(pb)": 0.0591312, "CX+(pb)": 0.162942, "CX0(pb)": 0.109482, "BRZh": 0.489321, "BRWZ": 0.470938}, "3300": {"BRWh": 0.474933, "BRWW": 0.471813, "CX-(pb)": 0.000581736, "CX+(pb)": 0.0023356, "CX0(pb)": 0.00131918, "BRZh": 0.475857, "BRWZ": 0.472738}, "2400": {"BRWh": 0.477015, "BRWW": 0.470839, "CX-(pb)": 0.00495788, "CX+(pb)": 0.0174056, "CX0(pb)": 0.0103538, "BRZh": 0.478756, "BRWZ": 0.472582}, "1400": {"BRWh": 0.487101, "BRWW": 0.46518, "CX-(pb)": 0.0806679, "CX+(pb)": 0.214856, "CX0(pb)": 0.146961, "BRZh": 0.492143, "BRWZ": 0.470248}, "2500": {"BRWh": 0.476659, "BRWW": 0.471011, "CX-(pb)": 0.00387631, "CX+(pb)": 0.013834, "CX0(pb)": 0.00815289, "BRZh": 0.478265, "BRWZ": 0.472619}, "5200": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 8.17488e-06, "CX+(pb)": 3.36384e-05, "CX0(pb)": 2.16883e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "4800": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 2.05462e-05, "CX+(pb)": 8.59149e-05, "CX0(pb)": 5.23231e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "3600": {"BRWh": 0.474573, "BRWW": 0.471973, "CX-(pb)": 0.000295015, "CX+(pb)": 0.00121305, "CX0(pb)": 0.000684318, "BRZh": 0.47535, "BRWZ": 0.472751}, "2600": {"BRWh": 0.476347, "BRWW": 0.47116, "CX-(pb)": 0.00302043, "CX+(pb)": 0.01102, "CX0(pb)": 0.00644032, "BRZh": 0.477833, "BRWZ": 0.472647}, "5100": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 1.02747e-05, "CX+(pb)": 4.24846e-05, "CX0(pb)": 2.69688e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "2700": {"BRWh": 0.476071, "BRWW": 0.47129, "CX-(pb)": 0.00236942, "CX+(pb)": 0.00879738, "CX0(pb)": 0.0051021, "BRZh": 0.477449, "BRWZ": 0.47267}, "3800": {"BRWh": 0.474381, "BRWW": 0.472058, "CX-(pb)": 0.000188334, "CX+(pb)": 0.000783596, "CX0(pb)": 0.000443641, "BRZh": 0.475078, "BRWZ": 0.472756}, "3400": {"BRWh": 0.474802, "BRWW": 0.471872, "CX-(pb)": 0.000463361, "CX+(pb)": 0.00187714, "CX0(pb)": 0.00105885, "BRZh": 0.475673, "BRWZ": 0.472743}, "4500": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 4.0296e-05, "CX+(pb)": 0.000169477, "CX0(pb)": 9.99631e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "3700": {"BRWh": 0.474473, "BRWW": 0.472017, "CX-(pb)": 0.000235716, "CX+(pb)": 0.000975141, "CX0(pb)": 0.000550813, "BRZh": 0.475209, "BRWZ": 0.472754}, "4600": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 3.22129e-05, "CX+(pb)": 0.000135309, "CX0(pb)": 8.05872e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "1700": {"BRWh": 0.481851, "BRWW": 0.468293, "CX-(pb)": 0.0325698, "CX+(pb)": 0.0956578, "CX0(pb)": 0.0622015, "BRZh": 0.485295, "BRWZ": 0.471749}, "900": {"BRWh": 0.522327, "BRWW": 0.439352, "CX-(pb)": 0.409766, "CX+(pb)": 0.901251, "CX0(pb)": 0.687575, "BRZh": 0.534051, "BRWZ": 0.451309}, "800": {"BRWh": 0.552862, "BRWW": 0.413867, "CX-(pb)": 0.511085, "CX+(pb)": 1.0768, "CX0(pb)": 0.855309, "BRZh": 0.567236, "BRWZ": 0.428731}, "5000": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 1.30628e-05, "CX+(pb)": 5.42478e-05, "CX0(pb)": 3.39139e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "1800": {"BRWh": 0.480747, "BRWW": 0.468904, "CX-(pb)": 0.0244536, "CX+(pb)": 0.0739789, "CX0(pb)": 0.0473673, "BRZh": 0.483825, "BRWZ": 0.471991}, "3500": {"BRWh": 0.474682, "BRWW": 0.471925, "CX-(pb)": 0.000369561, "CX+(pb)": 0.00150893, "CX0(pb)": 0.000850838, "BRZh": 0.475504, "BRWZ": 0.472748}, "1600": {"BRWh": 0.48321, "BRWW": 0.467518, "CX-(pb)": 0.0437098, "CX+(pb)": 0.124438, "CX0(pb)": 0.0822156, "BRZh": 0.487091, "BRWZ": 0.471413}, "3200": {"BRWh": 0.475077, "BRWW": 0.471748, "CX-(pb)": 0.000731461, "CX+(pb)": 0.00290702, "CX0(pb)": 0.00164562, "BRZh": 0.47606, "BRWZ": 0.472732}, "4400": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 5.0388e-05, "CX+(pb)": 0.000212136, "CX0(pb)": 0.000123981, "BRZh": 0.474847, "BRWZ": 0.472759}, "5400": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 5.15289e-06, "CX+(pb)": 2.09687e-05, "CX0(pb)": 1.39886e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "5300": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 6.49458e-06, "CX+(pb)": 2.65819e-05, "CX0(pb)": 1.74251e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "2000": {"BRWh": 0.479079, "BRWW": 0.469796, "CX-(pb)": 0.0140562, "CX+(pb)": 0.0449418, "CX0(pb)": 0.0279823, "BRZh": 0.481578, "BRWZ": 0.472301}, "1100": {"BRWh": 0.49948, "BRWW": 0.45685, "CX-(pb)": 0.214767, "CX+(pb)": 0.512156, "CX0(pb)": 0.37177, "BRZh": 0.507523, "BRWZ": 0.464974}, "4900": {"BRWh": 0.474217, "BRWW": 0.472129, "CX-(pb)": 1.63891e-05, "CX+(pb)": 6.83168e-05, "CX0(pb)": 4.21333e-05, "BRZh": 0.474847, "BRWZ": 0.472759}, "2100": {"BRWh": 0.478438, "BRWW": 0.470127, "CX-(pb)": 0.0107528, "CX+(pb)": 0.035265, "CX0(pb)": 0.02168, "BRZh": 0.480707, "BRWZ": 0.472401}, "1000": {"BRWh": 0.507973, "BRWW": 0.450586, "CX-(pb)": 0.299486, "CX+(pb)": 0.687047, "CX0(pb)": 0.509804, "BRZh": 0.517614, "BRWZ": 0.460359}}
THEORY['W']['QCD'] = {800: [0.026, -0.026], 900: [0.033, -0.031], 1000: [0.039, -0.037], 1100: [0.044, -0.041], 1200: [0.049, -0.045], 1300: [0.054, -0.049], 1400: [0.058, -0.053], 1500: [0.062, -0.056], 1600: [0.066, -0.059], 1700: [0.070, -0.062], 1800: [0.074, -0.065], 1900: [0.077, -0.068], 2000: [0.080, -0.071], 2100: [0.084, -0.073], 2200: [0.087, -0.076], 2300: [0.090, -0.078], 2400: [0.093, -0.081], 2500: [0.097, -0.083], 2600: [0.100, -0.086], 2700: [0.103, -0.088], 2800: [0.106, -0.090], 2900: [0.109, -0.093], 3000: [0.112, -0.095], 3100: [0.115, -0.097], 3200: [0.117, -0.099], 3300: [0.120, -0.101], 3400: [0.123, -0.103], 3500: [0.126, -0.105], 3600: [0.128, -0.107], 3700: [0.130, -0.109], 3800: [0.133, -0.110], 3900: [0.135, -0.112], 4000: [0.137, -0.114], 4100: [0.139, -0.115], 4200: [0.140, -0.116], 4300: [0.142, -0.117], 4400: [0.144, -0.118], 4500: [0.145, -0.119], 4600: [0.145, -0.119], 4700: [0.145, -0.119], 4800: [0.145, -0.119], 4900: [0.145, -0.119], 5000: [0.145, -0.119], 5100: [0.145, -0.119], 5200: [0.145, -0.119], 5300: [0.145, -0.119], 5400: [0.145, -0.119], 5500: [0.145, -0.119], 5600: [0.145, -0.119], 5700: [0.145, -0.119], 5800: [0.145, -0.119], 5900: [0.145, -0.119], 6000: [0.145, -0.119]}
THEORY['W']['PDF'] = {800: [0.062, -0.062], 900: [0.064, -0.064], 1000: [0.066, -0.066], 1100: [0.068, -0.068], 1200: [0.070, -0.070], 1300: [0.072, -0.072], 1400: [0.074, -0.074], 1500: [0.077, -0.077], 1600: [0.081, -0.081], 1700: [0.083, -0.083], 1800: [0.085, -0.085], 1900: [0.089, -0.089], 2000: [0.093, -0.093], 2100: [0.098, -0.098], 2200: [0.104, -0.104], 2300: [0.109, -0.109], 2400: [0.114, -0.114], 2500: [0.120, -0.120], 2600: [0.128, -0.128], 2700: [0.136, -0.136], 2800: [0.144, -0.144], 2900: [0.152, -0.152], 3000: [0.160, -0.160], 3100: [0.174, -0.174], 3200: [0.188, -0.188], 3300: [0.202, -0.202], 3400: [0.216, -0.216], 3500: [0.230, -0.230], 3600: [0.258, -0.258], 3700: [0.285, -0.285], 3800: [0.313, -0.313], 3900: [0.340, -0.340], 4000: [0.368, -0.368], 4100: [0.408, -0.408], 4200: [0.448, -0.448], 4300: [0.488, -0.488], 4400: [0.529, -0.529], 4500: [0.569, -0.569], 4600: [0.569, -0.569], 4700: [0.569, -0.569], 4800: [0.569, -0.569], 4900: [0.569, -0.569], 5000: [0.569, -0.569], 5100: [0.569, -0.569], 5200: [0.569, -0.569], 5300: [0.569, -0.569], 5400: [0.569, -0.569], 5500: [0.569, -0.569], 5600: [0.569, -0.569], 5700: [0.569, -0.569], 5800: [0.569, -0.569], 5900: [0.569, -0.569], 6000: [0.569, -0.569]}
THEORY['Z'] = {}
THEORY['Z']['QCD'] = {800: [0.027, -0.026], 900: [0.033, -0.032], 1000: [0.040, -0.037], 1100: [0.045, -0.042], 1200: [0.050, -0.046], 1300: [0.054, -0.050], 1400: [0.059, -0.053], 1500: [0.063, -0.056], 1600: [0.067, -0.060], 1700: [0.070, -0.062], 1800: [0.074, -0.065], 1900: [0.077, -0.068], 2000: [0.080, -0.070], 2100: [0.083, -0.072], 2200: [0.086, -0.075], 2300: [0.089, -0.077], 2400: [0.091, -0.079], 2500: [0.094, -0.082], 2600: [0.097, -0.084], 2700: [0.099, -0.085], 2800: [0.102, -0.087], 2900: [0.104, -0.089], 3000: [0.107, -0.091], 3100: [0.109, -0.093], 3200: [0.112, -0.095], 3300: [0.114, -0.097], 3400: [0.116, -0.098], 3500: [0.119, -0.100], 3600: [0.121, -0.102], 3700: [0.123, -0.103], 3800: [0.125, -0.105], 3900: [0.127, -0.107], 4000: [0.130, -0.108], 4100: [0.131, -0.109], 4200: [0.133, -0.111], 4300: [0.135, -0.112], 4400: [0.137, -0.113], 4500: [0.138, -0.115], 4600: [0.138, -0.115], 4700: [0.138, -0.115], 4800: [0.138, -0.115], 4900: [0.138, -0.115], 5000: [0.138, -0.115], 5100: [0.138, -0.115], 5200: [0.138, -0.115], 5300: [0.138, -0.115], 5400: [0.138, -0.115], 5500: [0.138, -0.115], 5600: [0.138, -0.115], 5700: [0.138, -0.115], 5800: [0.138, -0.115], 5900: [0.138, -0.115], 6000: [0.138, -0.115]}
THEORY['Z']['PDF'] = {800: [0.062, -0.062], 900: [0.065, -0.065], 1000: [0.067, -0.067], 1100: [0.068, -0.068], 1200: [0.069, -0.069], 1300: [0.073, -0.073], 1400: [0.077, -0.077], 1500: [0.079, -0.079], 1600: [0.081, -0.081], 1700: [0.085, -0.085], 1800: [0.089, -0.089], 1900: [0.092, -0.092], 2000: [0.095, -0.095], 2100: [0.100, -0.100], 2200: [0.105, -0.105], 2300: [0.110, -0.110], 2400: [0.115, -0.115], 2500: [0.120, -0.120], 2600: [0.128, -0.128], 2700: [0.135, -0.135], 2800: [0.143, -0.143], 2900: [0.150, -0.150], 3000: [0.157, -0.157], 3100: [0.169, -0.169], 3200: [0.181, -0.181], 3300: [0.192, -0.192], 3400: [0.204, -0.204], 3500: [0.215, -0.215], 3600: [0.230, -0.230], 3700: [0.246, -0.246], 3800: [0.261, -0.261], 3900: [0.276, -0.276], 4000: [0.291, -0.291], 4100: [0.314, -0.314], 4200: [0.337, -0.337], 4300: [0.360, -0.360], 4400: [0.383, -0.383], 4500: [0.406, -0.406], 4600: [0.406, -0.406], 4700: [0.406, -0.406], 4800: [0.406, -0.406], 4900: [0.406, -0.406], 5000: [0.406, -0.406], 5100: [0.406, -0.406], 5200: [0.406, -0.406], 5300: [0.406, -0.406], 5400: [0.406, -0.406], 5500: [0.406, -0.406], 5600: [0.406, -0.406], 5700: [0.406, -0.406], 5800: [0.406, -0.406], 5900: [0.406, -0.406], 6000: [0.406, -0.406]}

#import pandas as pd
df = pd.DataFrame(
    columns=(
        'mass', 
        'CX-(pb)', 
        'CX+(pb)', 
        'CX+-_PDF+', 
        'CX+-_PDF-', 
        'CX+-_QCD+', 
        'CX+-_QCD-', 
        'CX0(pb)', 
        'CX0_PDF+', 
        'CX0_PDF-', 
        'CX0_QCD+', 
        'CX0_QCD-', 
        "WH", 
        "WW", 
        "WZ", 
        "ZH"
        )
    )

for mm, mass in enumerate(THEORY['W']['central'].keys()) :
   df.loc[mm] = [
    int(mass), 
    THEORY['W']['central'][mass]['CX-(pb)'], 
    THEORY['W']['central'][mass]['CX+(pb)'], 
    THEORY['W']['QCD'][int(mass)][0], 
    THEORY['W']['QCD'][int(mass)][1], 
    THEORY['W']['PDF'][int(mass)][0], 
    THEORY['W']['PDF'][int(mass)][1], 
    THEORY['W']['central'][mass]['CX0(pb)'], 
    THEORY['Z']['QCD'][int(mass)][0], 
    THEORY['Z']['QCD'][int(mass)][1], 
    THEORY['Z']['PDF'][int(mass)][0], 
    THEORY['Z']['PDF'][int(mass)][1], 
    THEORY['W']['central'][mass]['BRWh'], 
    THEORY['W']['central'][mass]['BRWW'], 
    THEORY['W']['central'][mass]['BRWZ'], 
    THEORY['W']['central'][mass]['BRZh']
    ]

df = df.sort_values(by=["mass"])   

df['CX0_up'] =   (1 + np.sqrt(df['CX0_PDF+']*df['CX0_PDF+'] + df['CX0_QCD+']*df['CX0_QCD+']) )*df['CX0(pb)']
df['CX0_down'] = (1- np.sqrt(df['CX0_PDF-']*df['CX0_PDF-'] + df['CX0_QCD-']*df['CX0_QCD-']))*df['CX0(pb)']

df['CX+-_up'] = (1 + np.sqrt(df['CX+-_PDF+']*df['CX+-_PDF+'] + df['CX+-_QCD+']*df['CX+-_QCD+']) )*(df['CX+(pb)'] + df['CX-(pb)'])
df['CX+-_down'] = (1 - np.sqrt(df['CX+-_PDF-']*df['CX+-_PDF-'] + df['CX+-_QCD-']*df['CX+-_QCD-']))*(df['CX+(pb)'] + df['CX-(pb)'])

print(df[['CX0(pb)', 'CX0_down', 'CX0_up']])

#'CX0_up', 'CX0_down'
#'CX+-_up' 'CX+-_down'
print("File to HVTB has keys", df.keys())
print("File to HVTB has masses", df["mass"].values)

df.to_csv("13TeV/HVTB_XS.csv", index=False)

df['CX+-(pb)'] = df['CX+(pb)'] + df['CX-(pb)'] 
print(df[['CX+-(pb)', 'CX+-_down', 'CX+-_up']])
#########################
#########################

# taken from: https://github.com/IreneZoi/cmgtools-lite/blob/VV_VH_ttbarFullRun2/VVResonances/scripts/theoryXsec/HVTC.json
THEORY_HVTC = pd.read_json("13TeV/format_inputs/HVTC.json")

df_HVTC = pd.DataFrame(
    columns=(
        'mass',
        'Wprime_cH3', 
        'Wprime_cH3_Up', 
        'Wprime_cH3_Down', 
        'Zprime_cH3', 
        'Zprime_cH3_Up', 
        'Zprime_cH3_Down', 
        'Zprime_cH1', 
        'Zprime_cH1_Up', 
        'Zprime_cH1_Down', 
        'Wprime_cH1', 
        'Wprime_cH1_Up', 
        'Wprime_cH1_Down', 
        "WH", 
        "WW", 
        "ZH", 
        "WZ",
        )
    )

for mm, mass in enumerate(THEORY_HVTC.keys()) :
   df_HVTC.loc[mm] = [
    int(mass), 
    THEORY_HVTC[mass]['Wprime_cH3'], 
    THEORY_HVTC[mass]['Wprime_cH3_Up'], 
    THEORY_HVTC[mass]['Wprime_cH3_Down'], 
    THEORY_HVTC[mass]['Zprime_cH3'], 
    THEORY_HVTC[mass]['Zprime_cH3_Up'], 
    THEORY_HVTC[mass]['Zprime_cH3_Down'], 
    THEORY_HVTC[mass]['Zprime_cH1'], 
    THEORY_HVTC[mass]['Zprime_cH1_Up'], 
    THEORY_HVTC[mass]['Zprime_cH1_Down'], 
    THEORY_HVTC[mass]['Wprime_cH1'], 
    THEORY_HVTC[mass]['Wprime_cH1_Up'], 
    THEORY_HVTC[mass]['Wprime_cH1_Down'], 
    THEORY_HVTC[mass]['BRWh'], 
    THEORY_HVTC[mass]['BRWW'], 
    THEORY_HVTC[mass]['BRZh'], 
    THEORY_HVTC[mass]['BRWZ'],
    ]

df_HVTC = df_HVTC.sort_values(by=["mass"])


print(df_HVTC[['Wprime_cH3', 'Wprime_cH3_Down', 'Wprime_cH3_Up']])
print("File to HVTC has keys", df_HVTC.keys())   
print("File to HVTC has masses", df_HVTC["mass"].values)
df_HVTC.to_csv("13TeV/HVTC_XS.csv", index=False)
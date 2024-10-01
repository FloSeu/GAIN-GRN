## utils/bb_angle_tools.py
# Contains functions for handling backbone-angle related operations, which are used to find and denote backbone angle outliers in the STRIDE files.

from cmath import phase, rect
from math import atan2, cos, degrees, radians, sin

import numpy as np


def detect_outliers(stride_file, outfile, sigmas=2):
    # Absolute calculated limits from all stride files.
    limits = [
            [294.95423509417174, 6.764252033318323],  # hPHI
            [321.2981172211187, 7.981525267998551], # hPSI
            [245.24767527534797, 30.12930372850216], # sPHI
            [136.614974076406, 33.94985372089623]   # sPSI
              ]

    d = open(stride_file).readlines()
    lim_dict = {"H": (limits[0],limits[1]), "E":(limits[2],limits[3])}
    newdata = []
    for l in d:
        if not l.startswith("ASG") or l[24] not in  "HE":
            newdata.append(l)
            continue

        phi_lim, psi_lim = lim_dict[l[24]] # Depending on whether its strand or helix.

        i = l.split()
        angles = [float(i[7]), float(i[8])]

        adj_angles = [a+360 if a<0 else a for a in angles]

        deviations = [abs(adj_angles[0]-phi_lim[0])/phi_lim[1], abs(adj_angles[1]-psi_lim[0])/psi_lim[1]]
        if deviations[0] > sigmas or deviations[1] > sigmas:
            # print("outlier found.", l, sep="\n")
            k = l[:24]+l[24].lower()+l[25:76]+"{:>4.2f}\n".format(max(deviations))
            newdata.append(k)
            continue

        newdata.append(l[:76]+"~~~~\n")

    with open(outfile, 'w') as out:
        out.write("".join(newdata))
    print(f"Modified STRIDE file {stride_file} into {outfile} to include outliers and the last column items[10] col 77-80 (1-indexed)")


def grab_sse_bb(stride_file):
    data = [l for l in open(stride_file).readlines() if l.startswith("ASG")]

    helix_dict = {}
    strand_dict = {}

    for l in data:
        items = l.split()
        if items[5] == "E":
            strand_dict[int(items[3])] = [float(items[7]), float(items[8])]
        if items[5] == "H":
            helix_dict[int(items[3])] = [float(items[7]), float(items[8])]
    return helix_dict, strand_dict


def mean_angle(deg):
    return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))


def angle_diff(a,b):
    ra, rb = radians(a), radians(b)
    return degrees(atan2(sin(ra-rb), cos(ra-rb)))


def angle_stdev(deg, mean):
    return np.sqrt( np.sum([(angle_diff(d,mean))**2 for d in deg])/len(deg) )


def get_bb_distribution(stride_set):
    all_angles = [[],[],[],[]]
    for stride_file in stride_set:
        helix_dict, strand_dict = grab_sse_bb(stride_file)
        [all_angles[0].append(v[0]) for v in helix_dict.values()]
        [all_angles[1].append(v[1]) for v in helix_dict.values()]
        [all_angles[2].append(v[0]) for v in strand_dict.values()]
        [all_angles[3].append(v[1]) for v in strand_dict.values()]
    hphi = [mean_angle(all_angles[0]), angle_stdev(all_angles[0], mean_angle(all_angles[0]))]
    hpsi = [mean_angle(all_angles[1]), angle_stdev(all_angles[1], mean_angle(all_angles[1]))]
    sphi = [mean_angle(all_angles[2]), angle_stdev(all_angles[2], mean_angle(all_angles[2]))]
    spsi = [mean_angle(all_angles[3]), angle_stdev(all_angles[3], mean_angle(all_angles[3]))]
    return hphi, hpsi, sphi, spsi


def modify_stride(stride_file, outfolder, phi_lim, psi_lim, n_sigma=2.0):
    # modify a stride file where the outliers of backbone angles are marked with loer case letters
    # in line[75:79], the multiple of the standard deviation is noted for manually adjusting the respective cutoffs.
    outliers = []
    # also add the max float mult of sigma into the "~~~~" (line[75:79])
    # "{:.2f}".format(maxsigma)
    with open(stride_file) as stride:
        d = stride.readlines()
    newdata = []
    for l in d:
        if not l.startswith("ASG") or l[24] != "E":
            newdata.append(l)
            continue

        i = l.split()
        angles = [float(i[7]), float(i[8])]
        adj_angles = [a+360 if a<0 else a for a in angles]
        if abs(angle_diff(adj_angles[0], phi_lim[0])) > n_sigma*phi_lim[1] or abs(angle_diff( adj_angles[1], psi_lim[0])) > n_sigma*psi_lim[1]:
            # print("outlier found.", l, sep="\n")
            maxsigma = max([ abs(angle_diff(adj_angles[0], phi_lim[0]) / phi_lim[1]) , 
                             abs(angle_diff(adj_angles[1],psi_lim[0]) / psi_lim[1])    
                           ])
            k = l[:24]+"e"+l[25:75]+"{:.2f}".format(maxsigma)+"\n"
            #print("DEBUG:", k)
            newdata.append(k)
            outliers.append(round(maxsigma, 2))
            continue
        
        newdata.append(l)
    
    open(f"{outfolder}/{stride_file.split('/')[-1]}", 'w').write("".join(newdata))
    
    return outliers

def stride_file_processing(stride_files, outfolder):

    phi_lim = [-113.01754866504291, 29.968104201971208] # This is mean and SD of the Angle PHI
    psi_lim = [132.75257372738366, 31.172184167730734]  #                                  PSI
    outliers = []
    for i,stride_file in enumerate(stride_files):
        outliers += modify_stride(stride_file, outfolder, phi_lim, psi_lim)
    print(f"Modified {i} stride files into the directory {outfolder}.")

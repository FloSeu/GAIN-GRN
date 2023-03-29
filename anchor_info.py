# INPUT: a collection of GAIN domain PDBs, their sequences as one large ".fa" file
from gain_classes import GainDomain, GainCollection, Anchors, GPS
import sse_func
import execute
import numpy as np
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from shutil import copyfile
import math
import re

valid_seqs = sse_func.read_multi_seq("/home/hildilab/projects/agpcr_nom/app_gain_gain.fa")
quality_file = "/home/hildilab/projects/agpcr_nom/app_gain_gain.mafft.jal"
alignment_file = "/home/hildilab/projects/agpcr_nom/app_gain_gain.mafft.fa"
stride_files = glob.glob("/home/hildilab/projects/agpcr_nom/sigmas/sigma_2/*")
# This only contains the sigma files for truncated (?) PDBs.
quality = sse_func.read_quality(quality_file)
gps_minus_one = 6781 
aln_cutoff = 6826 
alignment_dict = sse_func.read_alignment(alignment_file, aln_cutoff)

# Pre-calculated Anchor data.
anchors = [ 662, 1194, 1912, 2490, 2848, 3011, 3073, 3260, 3455, 3607, 3998, 4279, 4850, 5339, 5341, 5413, 5813, 6337, 6659, 6696, 6765, 6808 ]

anchor_occupation = [ 4594.0,  6539.0, 11392.0, 13658.0,  8862.0,  5092.0,  3228.0, 14189.0,  
					  9413.0, 12760.0, 9420.0, 11201.0, 12283.0,  3676.0,  4562.0, 13992.0, 
					  12575.0, 13999.0, 14051.0, 14353.0, 9760.0, 14215.0]

anchor_dict = sse_func.make_anchor_dict(anchors, 3425) # 3425 is the subdomain boundary in the GAIN alignment
print(anchor_dict)

def alignment_to_array(alnfile, aln_cutoff):
    with open(alnfile) as a:

        data = a.read()
        seqs = data.split(">")
        n_seqs = len(seqs)-1
        alignment_matrix = np.empty((n_seqs, aln_cutoff), dtype='U1')

        for i,seq in enumerate(seqs[1:]): # First item is empty

            sd = seq.splitlines()
            try: 
                sd[0] = sd[0].split("/")[0]
            except: 
                pass

            alignment_matrix[i,:] = list("".join(sd[1:])[:aln_cutoff]) 

    return alignment_matrix, n_seqs

def anchor_stats(alignment_matrix, col_idx, n_seqs):
    col = alignment_matrix[:, col_idx]
    letters, counts = np.unique(col, return_counts=True)

    resid_counts = {}
    for i, res in enumerate(letters):
        if res == "-":
            empty_seqs = int(counts[i])
        else:
            resid_counts[int(counts[i])] = res
    
    sorted_counts = sorted(resid_counts.keys())[::-1]

    occupancy = int( (n_seqs-empty_seqs)*100 / n_seqs )
    conserv_string = []
    residue_occupancies = [ int( x*100 / n_seqs ) for x in sorted_counts]
    for idx, occ in enumerate(residue_occupancies):
        if occ >= 5: conserv_string.append(f"{resid_counts[sorted_counts[idx]]}:{occ}%")

    return occupancy, ", ".join(conserv_string)


alignment_matrix, n_seqs = alignment_to_array(alignment_file, aln_cutoff)

with open("anchor_stats.tsv", "w") as stats:
    stats.write("Element\tOccupation\tConsensus\n")
    for anchor in anchors:
        occ, residues = anchor_stats(alignment_matrix, anchor, n_seqs)
        stats.write(f"{anchor_dict[anchor]}\t{occ}%\t{residues}\n")
#print(letters,"\n", counts, np.sum(counts), "\nTOTAL SEQS:", n_seqs)

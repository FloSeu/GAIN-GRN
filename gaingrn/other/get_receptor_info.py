## get_receptor_info.py
# 
# This is a script to extract information about a specific receptor from the GAIN-GRN indexing and the set of GainDomain objects.

import pandas as pd
import numpy as np

if __name__ == "main":

    receptor_of_interest = "E4"

    valid_collection = np.load("../data/valid_collection.pkl", allow_pickle=True)
    stal_indexing = pd.read_pickle("../data/stal_indexing.pkl")
    print(dir(stal_indexing))
    print(stal_indexing.accessions[0])
    print(stal_indexing.indexing_dirs[0])
    print(stal_indexing.receptor_types[0])

    roi_species = []

    for i, gain in enumerate(valid_collection.collection):

        if stal_indexing.receptor_types[i] == receptor_of_interest:

            roi_species.append(gain.name.split("-")[-1])
            
    print(len(roi_species), np.unique(roi_species), len(np.unique(roi_species)))
    print("\n".join(np.unique(roi_species)))
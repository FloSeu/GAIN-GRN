# This is a script for the creation of an indexing class instance
# Standalone version for multithreaded class creation
# due to jupyter notebooks failure to successfully run mp.Pool

import glob, pickle, os
import pandas as pd

from gaingrn.utils.indexing_classes import StAlIndexing
from gaingrn.utils.gain_classes import *
try: 
    GESAMT_BIN = os.environ.get('GESAMT_BIN')
except:
    GESAMT_BIN = "/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt"

def find_pdb(name, pdb_folder):
    identifier = name.split("-")[0]
    target_pdb = glob.glob(f"{pdb_folder}/*{identifier}*.pdb")[0]
    return target_pdb

if __name__ == '__main__':
    valid_collection = np.load("../data/valid_collection.pkl", allow_pickle=True)

    stal_indexing = StAlIndexing(valid_collection.collection, 
                                prefix="../../test_stal_indexing/stal_", 
                                pdb_dir='../../all_pdbs',  
                                template_dir='../data/template_pdbs', 
                                #fasta_offsets=fasta_offsets,
                                n_threads=3,
                                template_json='../data/template_data.json',
                                gesamt_bin=GESAMT_BIN,
                                debug=False
                               )

    header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=False)
    stal_indexing.data2csv(header, matrix, "../data/gain_stal_indexing.NEW.csv")
    #header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=True)
    #stal_indexing.data2csv(header, matrix, "../data/gain_stal_indexing.unique.csv")

    with open("../data/stal_indexing.NEW.pkl","wb") as save:
        pickle.dump(stal_indexing, save)

    print("Done creating and saving stal_indexing.NEW.pkl")

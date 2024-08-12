# This is a script for the creation of an indexing class instance
# Standalone version for multithreaded class creation
# due to jupyter notebooks failure to successfully run mp.Pool

import glob, pickle, os
import pandas as pd
import numpy as np
import sse_func
import gaingrn.scripts.alignment_utils
from gaingrn.scripts.indexing_classes import StAlIndexing

try: 
    GESAMT_BIN = os.environ.get('GESAMT_BIN')
except:
    GESAMT_BIN = "/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt"

def find_pdb(name, pdb_folder):
    identifier = name.split("-")[0]
    target_pdb = glob.glob(f"{pdb_folder}/*{identifier}*.pdb")[0]
    return target_pdb

if __name__ == '__main__':
    valid_collection = pd.read_pickle("../valid_collection.q.pkl")
    #all_accessions = [gain.name.split("-")[0].split("_")[0] for gain in valid_collection.collection]
    #all_sequences = ["".join(gain.sequence) for gain in valid_collection.collection]
    #fasta_offsets = gaingrn.scripts.alignment_utils.find_offsets("/home/hildilab/projects/GPS_massif/uniprot_query/agpcr_celsr.fasta", 
    #                                all_accessions, 
    #                                all_sequences)
    # DEBUG:
    #for i in range(20):
        #print(valid_collection.collection[i].name, fasta_offsets[i], valid_collection.collection[i].start, sep="\n")
    print(glob.glob('../data/template_pdbs/*pdb'))
    stal_indexing = StAlIndexing(valid_collection.collection, 
                                prefix="/home/hildilab/agpcr_nom/test_stal_indexing/stal_", 
                                pdb_dir='/home/hildilab/agpcr_nom/all_pdbs/',  
                                template_dir='/home/hildilab/agpcr_nom/r4_template_pdbs', 
                                #fasta_offsets=fasta_offsets,
                                n_threads=6,
                                template_json='template_data_s4.json',
                                gesamt_bin=GESAMT_BIN,
                                debug=False
                               )

    header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=False)
    stal_indexing.data2csv(header, matrix, "../s4_test_indexing.csv")
    #header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=True)
    #stal_indexing.data2csv(header, matrix, "../pkd_indexing.csv")

    with open("../data/s4_test_indexing.pkl","wb") as save:
        pickle.dump(stal_indexing, save)

    print("Done creating and saving s4_test_indexing.pkl")
# This is a script for the creation of an indexing class instance
# Standalone version for multithreaded class creation
# due to jupyter notebooks failure to successfully run mp.Pool

# DEPENDENCIES
import glob, pickle, os, sys
import pandas as pd
# LOCAL IMPORTS
import gaingrn.utils.gain_classes
from gaingrn.utils.indexing_classes import StAlIndexing
# FIX FOR CHANGED PICKLE PATHING
sys.modules['gaingrn.scripts.gain_classes'] = gaingrn.utils.gain_classes

try: 
    GESAMT_BIN = os.environ.get('GESAMT_BIN')
except:
    GESAMT_BIN = "/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt"

if GESAMT_BIN is None:
    GESAMT_BIN = "/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt"

pkd_folder = "../../data/pkd/"

if __name__ == '__main__':
    valid_collection = pd.read_pickle(f"{pkd_folder}/pkd_collection.pkl")
    #all_accessions = [gain.name.split("-")[0].split("_")[0] for gain in valid_collection.collection]
    #all_sequences = ["".join(gain.sequence) for gain in valid_collection.collection]
    #fasta_offsets = gaingrn.utils.alignment_utils.find_offsets("/home/hildilab/projects/GPS_massif/uniprot_query/agpcr_celsr.fasta",
    #                                all_accessions, 
    #                                all_sequences)
    # DEBUG:
    #for i in range(20):
        #print(valid_collection.collection[i].name, fasta_offsets[i], valid_collection.collection[i].start, sep="\n")
    print(glob.glob('../../data/template_pdbs/*pdb'))
    stal_indexing = StAlIndexing(valid_collection.collection, 
                                prefix="../../TESTING/pkds_", 
                                pdb_dir='../../../pkd_pdbs/',  
                                template_dir='../../data/template_pdbs', 
                                #fasta_offsets=fasta_offsets,
                                n_threads=1,
                                template_json='../../data/template_data.json',
                                gesamt_bin=GESAMT_BIN,
                                debug=False
                               )

    header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=False)
    stal_indexing.data2csv(header, matrix, "../../data/pkd/pkd_indexing.NEW.csv")

    with open("../../data/pkd/pkd_indexing.pkl","wb") as save:
        pickle.dump(stal_indexing, save)

    print("Done creating and saving pkd_indexing.pkl")
# This is a script for the creation of an indexing class instance
# Standalone version for multithreaded class creation
# due to jupyter notebooks failure to successfully run mp.Pool

# DEPENDENCIES
import glob, pickle, os
import pandas as pd
import numpy as np
import sse_func
# LOCAL IMPORTS
os.chdir("/home/hildilab/agpcr_nom/repo")
#os.getcwd()

from indexing_classes import StAlIndexing

def find_pdb(name, pdb_folder):
    identifier = name.split("-")[0]
    target_pdb = glob.glob(f"{pdb_folder}/*{identifier}*.pdb")[0]
    return target_pdb

#offset information
#offset information
def find_offsets(fasta_file, accessions, sequences):
    # searches through the accessions in the big sequence file,
    # finds the start for the provided sequence
    with open(fasta_file,"r") as fa:
        fa_data = fa.read()
        fasta_entries = fa_data.split(">")
    seqs = []
    headers = []
    offsets = []
    for seq in fasta_entries:
        # Fallback for too short sequences
        if len(seq) < 10: 
            continue
        data = seq.strip().split("\n")
        headers.append(data[0].split("|")[1]) # This is only the UniProtKB Accession Number and will be matched EXACTLY
        seqs.append("".join(data[1:]))
    
    heads = np.array(headers)

    for idx, accession in enumerate(accessions):
        #print(np.where(heads == accession), accession, idx)
        seq_idx = np.where(heads == accession)[0][0]
        offset = sse_func.find_the_start(seqs[seq_idx], sequences[idx])+1 # we are in one_indexed lists here, meaning that the first residue is 1, hence +1.
        #print(offset)
        offsets.append(offset)
    
    return offsets

if __name__ == '__main__':
    valid_collection = pd.read_pickle("../valid_collection.q.pkl")
    #all_accessions = [gain.name.split("-")[0].split("_")[0] for gain in valid_collection.collection]
    #all_sequences = ["".join(gain.sequence) for gain in valid_collection.collection]
    #fasta_offsets = find_offsets("/home/hildilab/projects/GPS_massif/uniprot_query/agpcr_celsr.fasta", 
    #                                all_accessions, 
    #                                all_sequences)
    # DEBUG:
    #for i in range(20):
        #print(valid_collection.collection[i].name, fasta_offsets[i], valid_collection.collection[i].start, sep="\n")
    print(glob.glob('../r4_template_pdbs/*pdb'))
    stal_indexing = StAlIndexing(valid_collection.collection, 
                                prefix="/home/hildilab/agpcr_nom/test_stal_indexing/stal_", 
                                pdb_dir='/home/hildilab/agpcr_nom/all_pdbs/',  
                                template_dir='/home/hildilab/agpcr_nom/r4_template_pdbs', 
                                #fasta_offsets=fasta_offsets,
                                n_threads=6,
                                template_json='template_data_s4.json',
                                gesamt_bin="/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt",
                                debug=False
                               )

    header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=False)
    stal_indexing.data2csv(header, matrix, "../s4_test_indexing.csv")
    #header, matrix = stal_indexing.construct_data_matrix(overwrite_gps=True, unique_sse=True)
    #stal_indexing.data2csv(header, matrix, "../pkd_indexing.csv")

    with open("../s4_test_indexing.pkl","wb") as save:
        pickle.dump(stal_indexing, save)

    print("Done creating and saving s4_test_indexing.pkl")
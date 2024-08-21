## scritps/alignment_utils.py
# Contains functions for handling alignment-realted operations.

from gaingrn.scripts.structure_utils import find_the_start
import gaingrn.scripts.alignment_utils
import gaingrn.scripts.io

import numpy as np

def get_indices(name, sequence, alignment_file, aln_cutoff, alignment_dict=None, truncation_map=None, aln_start_res=None, debug=False):
    '''
    Find a sequence in the alignment file, output a number of corresponding alignment indices for each residue in sequence

    Parameters:
        name : str, required
            The sequence name, must correspond to name in the alignment file
        sequence : numpy array, required
            The one-letter coded amino acid sequence
        alignment_file : str, required
            The alignment file containing all the information and name
        aln_cutoff : int, required
            The integer value of the last alignment residue column to be parsed
        alignment_dict : dict, optional
            If specified, skips loading the alignment file and directly looks up the sequences in the dictionary. Improves Performance
        aln_start_res : int, optional
            If known, specify the start column of the first residue in the Alignment.
        debug : bool, optional
            Specifies additional output for debugging
    Returns:
        mapper : list
            A list of alignment indices for each residue index of the sequence
    '''
    #print(f"[DEBUG] gaingrn.scripts.alignment_utils.get_indices : {sequence.shape = }")
    mapper = np.zeros([sequence.shape[0]],dtype=int) # Initialize the mapper for output
    #print(f"[DEBUG] gaingrn.scripts.alignment_utils.get_indices : {mapper.shape = }")
    if not alignment_dict:
        alignment_dict = gaingrn.scripts.io.read_alignment(alignment_file, aln_cutoff)

    # PATCH: If the name ends on ".fa", eliminate that.
    try: nam = name.split(".fa")[0]
    except: nam = name

    #print(f"[DEBUG] gaingrn.scripts.alignment_utils.get_indices : \n\t{nam = }, is it in the dict? {nam in alignment_dict.keys()}")
    try:
        align_seq = alignment_dict[nam]#[::-1] # make the reference alignment reverse to compare rev 2 rev
    except KeyError:
        print("[WARNING]: Sequence not found. If this is unintended, check the Alignment file!\n", nam)
        return None

    if aln_start_res is None:
        try:
            aln_start_res = find_the_start(alignment_dict[nam], sequence)
            # Finds the first index matching the sequence end and outputs the index
            #print(f"Found the start! {aln_start_res = }")#\n {align_seq = }")
        except:
            print("[ERROR]: Did not find the Sequence! - No start.")
            return None

    align_index = aln_start_res
    #print("".join(sequence), len(sequence))
    for i,residue in enumerate(sequence):   # For each residue, enter the While loop

        # If the current residue is truncated, skip to the next one.
        if truncation_map is not None and truncation_map[i]:
            mapper[i] = -1 #(integer for now! DEBUGGABLE!)
            if debug:
                print(f"[NOTE]: Skipping truncated residue @ {residue}{i+1}")
            continue

        while align_index < len(align_seq):                         # True ; To find the residue
            if residue == align_seq[align_index]:
                mapper[i] = align_index     # If it is found, note the index
                align_index += 1            # advance the index to avoid double counted identical resiudes (i.e. "EEE")
                break
            elif align_seq[align_index] != "-":
                if debug:
                    print("[DEBUG] WARNING! OUT OF PLACE RESIDUE FOUND:", align_seq[align_index], "@", align_index, "while searching for", residue, i)
            align_index += 1
    # return the matching list of alignment indices for each Sequence residue
    #print(f"[DEBUG] gaingrn.scripts.alignment_utils.get_indices : mapper constructed successfully.")
    #print(f"{mapper}")
    return mapper

def get_quality(alignment_indices, quality_arr):
    '''
    Parses through the quality array and extracts the matching columns of alignment indices to assign each residue a quality value.

    Parameters:
        alignment_indices : list, required
            A list of alignment indices that will be read from
        quality_arr : array, (1D), required
            An array containing the quality value for each column in the underlying alignment,
            can be substituted for any kind of signal used for assigning anchor residues

    Returns:
        index_qualities : list
            A list of values matching each index in alignment_indices
    '''
    index_qualities = np.zeros([len(alignment_indices)])

    for i,position in enumerate(alignment_indices):
        index_qualities[i] = quality_arr[position]

    return index_qualities

def make_anchor_dict(fixed_anchors, sd_boundary):
    '''
    Enumerate the secondary structures according to the nomenclature conventions, st
    - Separate both Subdomains A and B
    - Only label Helices in Subdomain A, Sheets in Subdomain B
    - The most C-terminal anchor will be alpha and increased from there in N->C direction

    Parameters : 
        fixed_anchors : list, required
            A list of anchor residues as alignment indices (integer)
        sd_boundary :   int,  required
            The integer value of the subdomain boundary from the alignment

    Return : 
        anchor_dict : dict
            the enumerated adressing of each anchor residue with greek letters
    '''
    anchor_dict = {}
    helices = [a for a in fixed_anchors if a < sd_boundary]
    sheets = [a for a in fixed_anchors if a > sd_boundary]
    for idx, h in enumerate(helices):
        anchor_dict[h] = "H"+str(idx+1)
    for idx, s in enumerate(sheets):
        anchor_dict[s] = "S"+str(idx+1)
    return anchor_dict

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
        seq_idx = np.where(heads == accession)[0][0]
        offset = gaingrn.scripts.alignment_utils.find_the_start(seqs[seq_idx], sequences[idx])
        #print(offset)
        offsets.append(offset)
    
    return offsets

def calc_identity(aln_matrix):
    # This takes an alignment matrix with shape=(n_columns, n_sequences) and generates counts based on the identity matrix.
    # Returns the highest non "-" residue count as the most conserved residue and its occupancy based on count("-") - n_struc
    n_struc = aln_matrix.shape[0]
    quality = []
    occ = []
    for col in range(aln_matrix.shape[1]):
        chars, count = np.unique(aln_matrix[:,col], return_counts=True)
        dtype = [('aa', 'S1'), ('counts', int)]
        values = np.array(list(zip(chars,count)), dtype=dtype)
        s_values = np.sort(values, order='counts')

        if s_values[-1][0] == b'-':
            q = s_values[-2][1]
        else:
            q = s_values[-1][1]
        x = np.where(chars == '-')[0][0]
        occ.append(n_struc - count[x])
        quality.append(q)
    return quality, occ

def offset_sequences(full_seqs, short_seqs, debug=False):
    #re-indexes short sequences (i.e. from models) to have their start corresponding to the FASTA sequence contained in full seqs.
    # short_seqs should be a read_multi_seq() object:
    #           [(name, sequence), ()]
    # full_seqs is a read_alignment() object:
    #           a dictionary with {sequence_name}:{sequence} as items
    # Returns total number of sequences, multi_seq object [(name, sequence), (), ...]
    adjusted_seqs = []
    for tup in short_seqs:
        x = find_the_start(longseq=full_seqs[tup[0]], shortseq=tup[1])
        if debug:
            print(f"DEBUG: {tup[0]},{x}")
        if x == 0:
            # In this case, no offset is needed.
            adjusted_seqs.append( (tup[0], full_seqs[tup[0]][:len(tup[1])]) )
        else:
            adjusted_seqs.append( (tup[0], full_seqs[tup[0]][x:x+len(tup[1])]) )
    return adjusted_seqs

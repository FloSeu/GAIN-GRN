from gaingrn.scripts.structure_utils import find_the_start
import gaingrn.scripts.alignment_utils
import gaingrn.scripts.io

import numpy as np

from gaingrn.scripts.template_utils import match_gain2subdomain_template


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

def gain_set_to_template(list_of_gains, index_list, template_anchors, gesamt_folder, penalty=None, subdomain='sda', return_unmatched_mode='quality', threshold=3, debug=False):
    '''This matches a subselection of GAIN domains (i.e. all ADGRB1 proteins) against a given template with which GESAMT calculation have already been run.
    list_of_gains :         list of GainDomain objects
    index_list :            list of indiced for each GainDomain object according to its index in original collection
    template_anchors :      dict of template anchor residues (with actual PDB-matching residue number, since this is was GESAMT will be taking)
    gesamt_folder :         string of the dictionary where the calculation files of GESAMT will be found. They should have indices ("123.out") according to "index_list"
    penalty:                int, For qunatitative plotting, use the penalty value for unmatched elements.
    subdomain :             string, ["sda","sdb"] to retrieve the corresponding set of elements.
    return_unmatched_mode:  str, ['index', 'aln', 'quality'], specifies the residues returned:
                                    'index'     just sequence index of first and last element residue
                                    'aln'       alignment column of the first and last element residue
                                    'quality'   aln_column and quality of best quality element residue
                                    'full'      return all of the above
    threshold:              int, specifies the lower boundary where an element is considered. default=3
    debug:                  bool, specifies if DEBUG messages will be printed. Take care, this can easily be >100k lines.

    RETURNS
    distances:              np.array of shape (len(list_of_gains), #template_anchors) filled with distances in A for each anchor, penalty if unmatched.
    all_matched_anchors:    list,  with dict as entries, corresponding to every GainDomain with Anchor label, the residue and the distance : {'H1': (653, 1.04) , ...}
    unindexed_elements:     dict,  gain.name:[[elementstart/start_alignment_column, elementend/end_alignment_column, elementlength], [e2], ...] for the unmatched elements
    unindexed_counter:      int, a total counter of unindexed elements. This can also include non-conserved initital SSE that have no anchor.
    '''
    def compose_element(element, gain, mode=return_unmatched_mode):
        # no match case since python < 3.10
        element_length = element[1] - element[0] +1
        if mode == 'index':
            return [element[0], element[1], element_length]
        elif mode == 'aln':
            return gain.alignment_indices[element[0]], gain.alignment_indices[element[1], element_length]
        elif mode == 'quality':
            qual = [gain.residue_quality[i] for i in range(element[0], element[1]+1)]
            return [gain.alignment_indices[element[0]+np.argmax(qual)], np.max(qual), element_length]
        elif mode == 'all':
            qual = [gain.residue_quality[i] for i in range(element[0], element[1]+1)]
            return [
                    gain.alignment_indices[element[0]+np.argmax(qual)], np.max(qual), element_length,
                    element[0], element[1], gain.alignment_indices[element[0]], gain.alignment_indices[element[1]]
                    ]
        else:
            return [element[0], element[1], element_length]

    n_anch = len(template_anchors.keys())
    distances = np.full(shape=(len(list_of_gains), n_anch), fill_value=penalty)
    all_matched_anchors = []
    unindexed_elements = {}
    unindexed_counter = 0
    #for gain_idx in range(len(list_of_gains)):
    for gain_idx, gain in enumerate(list_of_gains):
        gain_distances, matched_anchors = match_gain2subdomain_template(index_list[gain_idx],
                                                                            template_anchors=template_anchors,
                                                                            gesamt_folder=gesamt_folder,
                                                                            penalty=penalty,
                                                                            debug=debug)
        # check if there are unindexed elements in the GAIN larger than 3 residues
        anchor_residues = [v[0] for v in matched_anchors.values()]
        if debug: print(f"[DEBUG]: gain_set_to_template {anchor_residues = }")

        if subdomain == 'sda':
            sse = [element for element in gain.sda_helices if element[0] < gain.subdomain_boundary-gain.start]
        elif subdomain == 'sdb':
            sse = gain.sdb_sheets
        else: raise ValueError("NO SUBDOMAIN DETECTED")

        if debug: print(f"[DEBUG]: gain_set_to_template {sse = }")

        for element in sse:
            if element[1]-element[0] < threshold-1:
                continue
            if debug:
                print(f"[DEBUG]: gain_set_to_template {element = } {gain.start = } with range\n\t{range(element[0]+gain.start-1, element[1]+gain.start+2) = }",
                      f'and {anchor_residues = }')
        # Introduce a wider detection range for the helical match by widening the SSE interval edge by 1. 
        # This has no relevance to the actual indexing and it just for Detection purposes.
            isMatch = [a in range(element[0]+gain.start-1, element[1]+gain.start+2) for a in anchor_residues] # <- -1 and +2 for widened edge
            if not np.any(isMatch):
                composed_element = compose_element(element=element, gain=gain, mode=return_unmatched_mode)
                if gain.name not in unindexed_elements:
                    unindexed_elements[gain.name]=[composed_element]
                else:
                    unindexed_elements[gain.name].append(composed_element)

        if debug: print(f"[DEBUG]: gain_set_to_template : {gain_distances = }")
        distances[gain_idx,:] = gain_distances
        all_matched_anchors.append(matched_anchors)
    return distances, all_matched_anchors, unindexed_elements, unindexed_counter

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
## scripts/structure_utils.py
# Functions for extracting, parsing and handling secondary strcutural data

import re
import numpy as np

def detect_GPS(alignment_indices, gps_minus_one, debug=False):
    '''
    Detects the GPS residue at the specified index of the Alignment
    This is not very robust - the quality depends on the respecitve MSA

    Parameters:
        alignment_indices : list, required
            A list of the alignment indices contained in a target protein (all integer)
        gps_minus_one : int, required
            The alignment index of the (most conserved) GPS-1 residue right C-terminal of the cleavage site

    Returns:
        gps_center : int
            The residue index of the GAIN domain residue matching the specified alignment index
    '''
    try: 
        gps_center = np.where(alignment_indices == gps_minus_one)[0][0]
        return gps_center
    except IndexError:
        if debug:   
            print("[WARNING] sse_func.detect_GPS: GPS-1 column is empty. Returning empty for alternative Detection.")
            print(f"\t{gps_minus_one  = }\n\t{alignment_indices[-15:] = }")
        return None

def detect_signchange(signal_array, exclude_zero=False, check=False):
    ''' 
    Detect signchanges in a smoothened numerical signal array
    Can be adjusted to view "0" as separate sign logic or not.

    Parameters:
        signal_array : np.array, required
            A 1D-array containing the signal values. This should already be smoothened by np.concolve
        exclude_zero : bool, optional
            Specifies whether to view 0 as a separate entity for a sign change. Defalt: False
        check :        bool, optional
            Specifies whether to check the occurrence of a signchange along the wrap of last and first array value.
            Deault: False

    Returns:
        boundaries :   np.array
            An array with the indices of the detected sign changes
    '''
    asign = np.sign(signal_array)   # -1 where val is negative, 0 where val is zero, 1 where val is positive.
    sz = asign == 0                 # A boolean list where True means that asign is zero

    if exclude_zero == True:        #  Exclude where the value is EXACTLY zero, 0 has an unique sign [-x,0,x]
        while sz.any(): 
            asign[sz] = np.roll(asign, 1)[sz]
            sz = asign == 0
    
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    boundaries = np.asarray(signchange != 0).nonzero()[0]
    if boundaries.shape[0] == 0:
        print("WARNING: No boundaries detected!")
        return None
    # CHECK:
    # Special case. If the first and last value of the array have a different sign,
    # boundaries[0] = 0, which should be discarded depending on the signal. 
    # Keep only the value where val[i] != 0
    if boundaries[0] == 0 and check == True:
        #print("NOTE: array wrap signchange detected. Checking with Flag check = True.")
        if signal_array[0] == 0:       # There is no SSE begin @ residue 0, otherwise its ok
            boundaries = np.delete(boundaries, 0)
        if boundaries.shape[0]%2 != 0: # If the length is now uneven, means that last res. is SSE
            boundaries = np.append(boundaries, len(signal_array))
    
    return boundaries

def find_boundaries(sse_dict, seq_len, bracket_size=50, domain_threshold=50, coil_weight=0, truncate_N=None):
    '''
    make an array with the length of the sequence
    via np.convolve generate the freq of alpha-helix vs beta-sheets, 
    assigning 1 to Helices, -1 to sheets
    The boundaries are the two most C-terminal crossing point 
    of the convolve function of x > 0 to x < 0, respectively

    Parameters:
        sse_dict :      dict, required
            A dictionary with the SSE of the specified region
        seq_len :       int, required
            The total length of the target sequence
        bracket_size :  int, optional
            The size of the bracket used for convolving the signal. default = 50
        domain_threshold : int, optional
            Minimum size of a helical block to be considered Subdomain A candidate. default = 50
        coil_weight :   float, optional
            The numerical value assigned to unordered residues, used for faster "decay" of helical blocks if <0
            values of 0 to +0.2 may be tried
        truncate_N:     int, optional, default: None
            If set to True, the Domain will be immedately truncated $truncate_N residues N-terminally of the most N-terminal helix. 

    Returns:
        gain_start : int
            The most N-terminal residue index of the GAIN domain
        subdomain_boundary : int
            The residues index of the Subdomain boundary between A (helical) and B (sheet)
    '''
    # Check if the dictionary even contains AlphaHelices, and also check for 310Helix
    helices = []
    sheets = []
    if 'AlphaHelix' not in sse_dict.keys() or 'Strand' not in sse_dict.keys():
        print("This is not a GAIN domain.")
        return None, None

    [helices.append(h) for h in sse_dict['AlphaHelix']]

    if '310Helix' in sse_dict.keys(): 
        [helices.append(h) for h in sse_dict['310Helix']]

    [sheets.append(s) for s in sse_dict['Strand']]
    
    # coil_weight can be used to add a "decay" of unstructured residues into the signal
    # in that case, a value like +0.1 is alright, this will sharpen helical regions
    scored_seq = np.full([seq_len], fill_value=coil_weight)

    for element_tuple in helices:
        scored_seq[element_tuple[0]:element_tuple[1]] = -1
    for element_tuple in sheets:
        scored_seq[element_tuple[0]:element_tuple[1]] = 1

    # Smooth the SSE signal with np.convolve 
    signal = np.convolve(scored_seq, np.ones([bracket_size]), mode='same')
    boundaries = detect_signchange(signal, exclude_zero=True)

    ### Find the interval with most negative values
    # First, check if there are valid boundaries!
    if boundaries is None:
        print('No boundaries detected. Returning empty.')
        return None, None
    
    helical_counts = np.zeros([len(boundaries)-1],dtype=int)

    for k in range(len(boundaries)-1):
        start, end = boundaries[k], boundaries[k+1]
        sliced_array = scored_seq[start:end]
        helical_counts[k] = np.count_nonzero(sliced_array == -1)

    # Find the "first" (from C-terminal) helical section larger than the threshold
    # maxk is the size of the helical region
    maxk = None
    # reverse the helical array
    rev_helical = helical_counts[::-1]

    for rev_count, hel_len in enumerate(rev_helical):
        if hel_len >= domain_threshold:
            maxk = len(helical_counts)-(rev_count+1)
            break

    #print(f"[DEBUG] sse_func.find_boundaries : \n\tFound Helix boundary with the following characteristics: {maxk = } {helical_counts[maxk] = }(new variant)")
    # if it finds maxk, store the valies denoting the helical block for further refinement
    if maxk is None:
        print("No Helical segment satisfying the domain_size found: Maximum helical segment =", max(helical_counts))
        return None, None
    
    gain_start, initial_boundary = boundaries[maxk], boundaries[maxk+1]

    if truncate_N is not None:
        for i, res in enumerate(scored_seq[gain_start:]):
            if res == -1:
                print(f"[NOTE] Overwriting initial {gain_start = } with {gain_start+i-truncate_N}.")
                gain_start = gain_start+i-truncate_N
                break
    # After it found the most likely helical block, adjust the edge of that, designate as Subdomain A
    # adjust the subdomain boundary to be in the middle of the loop between Helix and Sheet

    helix_end = initial_boundary
    sheet_start = initial_boundary
    
    if scored_seq[sheet_start] == 1:
        while True:
            sheet_start -= 1 # go left to find the end of the current sheet
            if scored_seq[sheet_start] != 1:
                break
    else:
        while True:
            sheet_start  += 1 # go right to find the start of the first Subdomain B sheet
            if scored_seq[sheet_start] == 1:
                break

    if scored_seq[helix_end] != -1: # go left to find the end of the last Subdomain A Helix
        while True:
            helix_end -= 1
            if scored_seq[helix_end] == -1:
                break
    else:   
        while True:
            helix_end += 1 # go right to find the end of the last Subdomain A Helix
            if scored_seq[helix_end] != -1:
                break
    nocoil = True
    if coil_weight != 0:
        nocoil = False
    # Final sanity check to see if there are any SSE left in the interval between helix_end and sheet_start
    #if np.count_nonzero(scored_seq[helix_end+1:sheet_start]) > 1 : # This threw a warning when coil_weight != 0
    if not nocoil and 1 in scored_seq[helix_end+1:sheet_start]:
        print("[WARNING] sse_func.find_boundaries : "
            "There are still secondary-structure associated residues within the Subdomain connecting loop. Please check if this is within limits:\n"
            f"{scored_seq[helix_end+1:sheet_start]}")

    subdomain_boundary = (helix_end+sheet_start) // 2  # The subdomain bondary is the middle of the two SSEs limiting the sections
    #if subdomain_boundary - gain_start >= domain_threshold:
    #print(f"DEBUG: find_boundaries returning {boundaries[maxk] = }, {boundaries[maxk+1] = }")
    return gain_start, subdomain_boundary

def sse_sequence2bools(sse_dict:dict):
    '''
    Create a dictionary containing the actually detected alpha helices and beta sheets for all residues in sse_dict, 
    using lower_case mode detection and the spacing variable
    '''
    #print(f"[DEBUG] sse_func.sse_sequence2bools:\n\t{sse_dict = }")
    len = max(sse_dict.keys())
    hel_signal = np.zeros(shape=[len], dtype=int)
    she_signal = np.zeros(shape=[len], dtype=int)
    
    for res, assigned_sse in sse_dict.items():
        if assigned_sse.upper() == "H" or assigned_sse.upper() == "G":
            hel_signal[res] = 1
        elif assigned_sse.upper() == "E":
            she_signal[res] = 1

    return hel_signal, she_signal

def count_domain_sses(domain_start, domain_end, tuple_list=None, spacing=1, minimum_length=3, sse_bool=None, debug=False):
    if debug:
        print(f"[DEBUG] sse_func.count_domain_sses CALLED WITH: \n\t tuple list",
              tuple_list, "sse_bool", sse_bool)
    # provide either sse_bool or tuple_list
    if sse_bool is None:
        parsed_sses = []
        if debug: 
            print(f"[DEBUG] sse_func.count_domain_sses : No sse_bool specified. Constructing from:\n\t{tuple_list = } \n\t{domain_start = } {domain_end = }")
        # First, check if the SSE limits exceed the provided Domain boundary (sort of a sanity check)
        for sse in tuple_list:
            if sse[0] >= domain_start and sse[1] <= domain_end:
                #print(f"[DEBUG] sse_func.count_domain_sses : {sse = }")
                parsed_sses.append(sse)
            
        sse_bool = np.zeros(shape=[domain_end], dtype=bool)
        for element_tuple in parsed_sses:
            if debug: 
                print(f"{element_tuple = }")
            sse_bool[element_tuple[0]:element_tuple[1]+1] = 1 # THE LAST NUMBER OF THE TUPLE IN LIST IS INCLUDED!
                                                              #        0  1  2  3  4  5  6  7  8  9 10 11  ...
                                                              # (1,9) [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
            #print(f"[DEBUG] sse_func.count_domain_sses : {element_tuple = }")
    
    # Otherwise SSE_BOOL has been passed and is used directly
    # Truncate the SSE BOOL by setting everything outside the boundaries to zero.
    sse_bool[:domain_start] = 0
    up_edges = []
    down_edges = []
    #break_residues = []

    for i in range(len(sse_bool)-1):
        if sse_bool[i] and not sse_bool[i+1]: # [.. 1 0 ..]
            down_edges.append(i+1)
        if not sse_bool[i] and sse_bool[i+1]: # [.. 0 1 ..]
            up_edges.append(i+1)
    down_edges.append(len(sse_bool))

    if debug:
        print(f"[DEBUG] sse_func.count_domain_sses :\n\t{up_edges = }\n\t{down_edges = }")
    # Remove all segments between down-edge and up-edge where count(0) <= spacing
    # remove zero-segments whose length is smaller than $spacing
    i = 0
    n_elements = len(up_edges)
    while i < n_elements-1:
        if debug: print(f"{i = } {n_elements = }\n\t{up_edges = }\n\t{down_edges = }")
        unordered_length = up_edges[i+1] - down_edges[i]

        if unordered_length <= spacing:  
            if debug:
                print("[DEBUG] sse_func.count_domain_sses: Found break within specified spacing. Fusing two elements", up_edges[i:i+2], down_edges[i:i+2])
            del up_edges[i+1]
            del down_edges[i]
            n_elements -= 1
            continue
        # If this is a unique element, append empty breaks.

        i += 1
    
    # With the cleaned up lists of up_edges and down_edges, get all elements satisfying minium_length and within boundaries.
    intervals = []
    for i in range(n_elements):
        element_length = down_edges[i] - up_edges[i]
        if element_length < minimum_length:
            continue
        if up_edges[i] < domain_end:
            intervals.append([up_edges[i], down_edges[i]-1])
    if debug:
        print(f"[DEBUG] sse_func.count_domain_sses : RETURNING \n\t{intervals = }")#\n\t{break_residues = }")

    return np.asarray(intervals)#, break_residues

def get_sse_type(sse_types, sse_dict):
    '''
    Get a list of all SSEs of a type (type/s as str / list) in sse_dict within
    a given Interval. Returns [] if the given types are not contained in sse_dict.keys()

    Parameters:
        sse_types : (list of str) or str, required
            Specifies the key(s) in the dict to be looked up
        sse_dict : dict, required
            The dectionary to be parsed

    Returns:
        sse_tuples : list
            A list of tuples containing all residue indices with start and end
            of each SSE corresponding to the specified types
    '''
    sse_tuples = []
    
    if type(sse_types) == list:
        for sse in sse_types:
            if sse not in sse_dict.keys(): 
                if sse != "310Helix":
                    print(f"KeyNotFound: {sse}") # Is is frequently the case that there are no 310Helices, nothing to worry there. print no Note.
                continue
            sse_tuples = sse_tuples + sse_dict[sse]
        return sse_tuples
    
    elif type(sse_types) == str:
        if sse_types not in sse_dict.keys():
            print(f"KeyError: no {sse_types} in dict.keys()")
            return []
        return sse_dict[sse_types]
    
    print(f"Error: Key(s) not found {sse_types} (get_sse_type)")
    return []

def find_the_start(longseq, shortseq): 
    '''
    Find the start of a short sequence in a long sequence.
    Pattern matching while the longseq from an MSA may also contain "-" as in Alignments
    This enables to extract the sequence from an alignment where is might be spaced out

    Parameters : 
        longseq : str, required
            A sequence string of one-letter amino acids and the "-" character, as in parsed from a FASTA alignment
        shortseq : str, required
            The short sequence being an exact slice of the long sequence, without "-"

    Returns :
        start : int
            The index of the first shortseq residue in longseq. Used for parsing alingment indices. 
    '''
    cat_seq = "".join(shortseq)
    findstring = cat_seq[:15] # The First 15 residues of N-->C direction as matching string
    id_array = np.arange(len(longseq)) # assign column indices to longseq, starting @ zero
    map_matrix = np.ones([len(longseq)], dtype=bool)

    for i, char in enumerate(longseq):
        if char == "-":
            map_matrix[i]=False 
    
    longseq_array = np.array([char for char in longseq]) #array of the FULL Seq
    
    filter_seq = longseq_array[map_matrix == True] # longseq without all "-" character
    filter_id = id_array[map_matrix == True] #  The indices of non- "-" characters are preserved here
    
    locator = "".join(filter_seq).find(findstring) #print(f"DEBUG: {''.join(filter_seq) = } {locator = }")
    start = filter_id[locator] #print("Found the Start @", start)
    
    return start

def get_subdomain_sse(sse_dict:dict, subdomain_boundary:int, start:int, end:int, residue_sse:dict, stride_outlier_mode=False, debug=False):
    '''
    Fuzzy detection of Helices in Subdomain A and Sheets in Subdomain B
    Count them and return their number + individual boundaries

    Parameters:
        sse_dict : dict, required
            Dict containing all SSE information about the GAIN domain
        subdomain_boundary : int, required
            The residue index of the residue denoting the subdomain boundary
        start : int, required
            Residue index of the most N-terminal domain residue
        end : int, required
            Residue index of the most C-terminal domain residue
        stride_outlier_mode : bool, optional

    Returns:
        alpha : 
            2D array of dimension ((number of sse), (start, end)) for alpha helices in Subdomain A
        beta : 
            2D array of dimension ((number of sse), (start, end)) for beta sheets in Subdomain B
        alpha_breaks:
            array of breaking points in the SSE definiton. Used for disambiguating close SSE in Subdomain A
        beta_breaks:
            array of breaking points in the SSE definiton. Used for disambiguating close SSE in Subdomain B
    '''
    helices = get_sse_type(["AlphaHelix", "310Helix"], sse_dict)
    sheets = get_sse_type("Strand", sse_dict)
    if debug:
        print(f"[DEBUG] sse_func.get_subdomain_sse : \n\t {helices = } \n\t {sheets = } \n\t {subdomain_boundary = }")

    # Parse boundaries, relevant for other uses of this function like enumerating other sections
    if subdomain_boundary == None:
        helix_upperbound = end
        sheet_lowerbound = start
    else:
        helix_upperbound = subdomain_boundary
        sheet_lowerbound = subdomain_boundary

    if stride_outlier_mode == False:    
        alpha = count_domain_sses(start,helix_upperbound, helices, spacing=0, minimum_length=3, debug=debug) # PARSING BY SSE DICTIONARY
        beta = count_domain_sses(sheet_lowerbound, end, sheets, spacing=0, minimum_length=2, debug=debug) # 
    
    if stride_outlier_mode == True:
        # This version should be generall the case for GAIN domains evaluated in GainCollection.__init__()
        hel_bool, she_bool = sse_sequence2bools(residue_sse)
        alpha = count_domain_sses(start, helix_upperbound, helices, spacing=0, minimum_length=3, sse_bool=hel_bool, debug=debug) # PARSING BY SSE-SEQUENCE
        beta = count_domain_sses(sheet_lowerbound, end, sheets, spacing=0, minimum_length=2, sse_bool=she_bool, debug=debug) # 
    if debug:
        print(f"[DEBUG] sse_func.get_subdomain_sse : \n\t{stride_outlier_mode = }\n\t {alpha = } \n\t {beta = }")
    return alpha, beta

def cut_sse_dict(start, end, sse_dict):
    '''
    Truncate all SSE in the complete dictionary read from STRIDE to SSE only within the GAIN domain
    
    Parameters:
        start : int, required
            Residue index of the most N-terminal domain residue
        end : int, required
            Residue index of the most C-terminal domain residue
        sse_dict : dict, required
            The full SSE dictionary containing all SSE information

    Returns:
        new_dict : dict
            The dictionary containing only SSE within the domain boundaries.
    '''
    new_dict = {}
    
    for key in sse_dict.keys():

        new_sse_list = []

        for item in sse_dict[key]:

            if item[0] > end or item[1] < start:
                continue
            # Should not happen, but just in case, truncate SSE to the GAIN boundary 
            # if they exceed them
            elif item[0] < start: 
                item = (start, item[1])
            elif item[1] > end:
                item = (item[0], end)
                
            new_sse_list.append(item)
        
        # Some structures (like specific Turns) may not be within the GAIN. Skip that.
        if len(new_sse_list) == 0:
            continue
        
        # Read the list into the dictionary    
        new_dict[key] = new_sse_list     

    return new_dict

def get_pdb_extents(pdb, subdomain_boundary = None):
    with open(pdb) as p:
        data = [l for l in p.readlines() if l.startswith("ATOM")]
    first = data[0][22:26]
    last = data[-1][22:26]
    if subdomain_boundary is not None:
        all_res = sorted(np.unique([int(l[22:26]) for l in data]))
        if subdomain_boundary not in all_res:
            # Find the two closest residues to the subdomain boundary (N- and C-terminal)
            sda_boundary = np.max([res for res in all_res if res < subdomain_boundary])
            sdb_boundary = np.min([res for res in all_res if res > subdomain_boundary])
        else:
            sda_boundary, sdb_boundary = subdomain_boundary, subdomain_boundary
        return int(first), sda_boundary, sdb_boundary, int(last)
    print("NOTE: Subdomain boundary not specified. Returning [start, None, None, end]")
    return int(first), None, None, int(last)

def truncate_pdb(pdbfile:str, start:int, end:int):
    # truncates pdbfile at the start and end residue, including them.
    pdblines = open(pdbfile).readlines()
    newlines = []
    for line in pdblines:
        if line.startswith("ATOM") and ( int(line[22:26]) < start or int(line[22:26]) > end ):
            continue
        if line.startswith("TER"):
            prev_info = newlines[-1]
            new_TER_line = f"TER   {str(int(prev_info[7:12])+1).rjust(5)}      {prev_info[17:26]}                                                      \n"
            newlines.append(new_TER_line)
            continue
        newlines.append(line)

    open(f'{pdbfile.replace(".pdb","_trunc.pdb")}', 'w').write("".join(newlines))
    print(f"[NOTE] Truncated PDB to residues {start }-{end}.")
    return pdbfile.replace(".pdb","_trunc.pdb")

def truncate_stride_dict(stride_dict:dict, start:int, end:int):
    # removes entries from the dictionary where the ssestart is behind the end or the sseend is before the start.
    truncated_dict = {}
    for key, list in stride_dict.items():
        trunc_list = [tup for tup in list if (tup[0] < end and tup[1] > start)]
        truncated_dict[key] = trunc_list
    return truncated_dict


def get_ca_indices(pdbfile, offset=0):
    atoms = [l for l in open(pdbfile).readlines() if l[13:15] == "CA" and l.startswith("ATOM")]
    ca_indices = {int(l[22:26]):int(l[6:11])-offset for l in atoms}
    return ca_indices


def get_pdb_offset(pdbfile):
    match = re.findall("ATOM\s+\d+", open(pdbfile).read())
    offset = int(match[0].split()[-1])
    return offset
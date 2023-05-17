import glob,shutil
import matplotlib.pyplot as plt
import sse_func
x =  [('A0A4W2FKG7-A0A4W2FKG7_BOBOX-AGRE1-Bos_indicus_x_Bos_taurus', 0, 45), 
 ('A0A3P8S994-A0A3P8S994_AMPPE-AGRE5b,duplicate2-Amphiprion_percula', 1, 39), 
 ('A0A7J7T9K9-A0A7J7T9K9_PIPKU-AGRE1-Pipistrellus_kuhlii', 2, 20), 
 ('A0A3Q7VM07-A0A3Q7VM07_URSAR-AGRE2-like-Ursus_arctos_horribilis.', 3, 13), 
 ('A0A6P5C2Y5-A0A6P5C2Y5_BOSIN-AGRE1isoformX2-Bos_indicus', 4, 20), 
 ('F1SD42-F1SD42_PIG-AGRE5-Sus_scrofa', 5, 104), 
 ('F6YLD2-F6YLD2_MACMU-AGRE2-Macaca_mulatta', 6, 12), 
 ('A0A452F289-A0A452F289_CAPHI-AGRE2-Capra_hircus', 7, 108), 
 ('A0A6P3R5Z9-A0A6P3R5Z9_PTEVA-AGRE4-like-Pteropus_vampyrus', 8, 39),
 ('A0A3P8S994-A0A3P8S994_AMPPE-AGRE5b,duplicate2-Amphiprion_percula', 0 ,0)]
allpdbs = glob.glob('../all_pdbs/*.pdb')
for i,t in enumerate(x): 
    ident = t[0].split("-")[0]
    pdb = [x for x in allpdbs if ident in x][0]
    shutil.copyfile(pdb, f'../sda_templates/cl_E{i}_sda_{pdb.split("/")[-1]}')


class Template:
    def __init__(self, subdomain, name, pdb_file):
        self.subdomain = subdomain
        self.anchor_dict = {}
        self.pdb_file = pdb_file
        self.identifier = name.split("-")[0]
        self.anchor_coords = []

import numpy as np

def count_domain_sses(domain_start, domain_end, tuple_list=None, spacing=1, minimum_length=3, sse_bool=None, debug=False):
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
            sse_bool[element_tuple[0]:element_tuple[1]+1] = 1 # THE LAST NUMBER OF THE TUPLE IN LIST IS INCLUDED!
                                                              #        0  1  2  3  4  5  6  7  8  9 10 11  ...
                                                              # (1,9) [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
            #print(f"[DEBUG] sse_func.count_domain_sses : {element_tuple = }")
    
    #print(f"[DEBUG] sse_func.count_domain_sses : {sse_bool = }")
    # Otherwise SSE_BOOL has been passed and is used directly
    # Truncate the SSE BOOL by setting everything outside the boundaries to zero.
    sse_bool[:domain_start] = 0
    up_edges = []
    down_edges = []
    break_residues = []

    for i in range(len(sse_bool)-1):
        if sse_bool[i] and not sse_bool[i+1]: # [.. 1 0 ..]
            down_edges.append(i+1)
        if not sse_bool[i] and sse_bool[i+1]: # [.. 0 1 ..]
            up_edges.append(i+1)
    down_edges.append(len(sse_bool))

    # Remove all segments between down-edge and up-edge where count(0) <= spacing
    # remove zero-segments whose length is smaller than $spacing
    i = 1
    n_elements = len(up_edges)
    while i < n_elements:
        unordered_length = up_edges[i+1] - down_edges[i]
        if unordered_length <= spacing:
            breaks = list(range(down_edges[i]+1, up_edges[i+1]))
            break_residues.append(breaks)
            del up_edges[i+1]
            del down_edges[i]
            n_elements -= 1
            continue
        # If this is a unique element, append empty breaks.
        break_residues.append([])
        i += 1
    
    # With the cleaned up lists of up_edges and down_edges, get all elements satisfying minium_length.
    intervals = []
    for i in range(n_elements):
        element_length = down_edges[i] - up_edges[i]
        if element_length < minimum_length:
            continue
        intervals.append([up_edges[i], down_edges[i]-1])
    if debug:
        print(f"[DEBUG] sse_func.count_domain_sses : RETURNING \n\t{intervals = }\n\t{break_residues = }")
    return np.asarray(intervals), break_residues

def create_indexing(gain_domain, anchors:dict, anchor_occupation:dict, anchor_dict:dict, outdir=None, offset=0, silent=False, split_mode='single',debug=False):
    ''' 
    Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
    Prints out the final list and writes it to file if outdir is specified
    
    Parameters
    ----------
    anchors : list, required
        List of anchors with each corresponding to an alignment index
    anchor_occupation : list, required
        List of Occupation values corresponding to each anchor for resolving ambiguity conflicts
    anchor_dict : dict, required
        Dictionary where each anchor index is assigned a name (H or S followed by a greek letter for enumeration)
    offfset : int,  optional (default = 0)
        An offsset to be incorporated (i.e. for offsetting model PDBs against UniProt entries)
    silent : bool, optional (default = False)
        opt in to run wihtout so much info.
    outdir : str, optional
        Output directory where the output TXT is going to be written as {gain_domain.name}.txt

    Returns
    ---------
    indexing_dir : dict
        A dictionary containing the residue indices for each assigned SSE and the GPS
    indexing_centers : dict
        A dictionary containing the XX.50 Residue for each assigned SSE
    named_residue_dir : dict
        A dictionary mapping individual consensus labels to their respective position.
    unindexed : list
        A list of detected SSE that remain unindexed.
    '''
    # recurring functional blocks used within this function

    def create_name_list(sse, ref_res, sse_name):
        ''' Creates a indexing list for an identified SSE with reference index as X.50,
            Also returns a tuple containing the arguments, enabling dict casting for parsing '''
        number_range = range(50+sse[0]-ref_res, 51+sse[1]-ref_res)
        name_list = [f"{sse_name}.{num}" for num in number_range]
        cast_values = (sse, ref_res, sse_name)
        return name_list, cast_values

    def indexing2longfile(gain_domain, nom_list, outfile, offset):
        ''' Creates a file where the indexing will be denoted line-wise per residue'''
        with open(outfile, "w") as file:
            file.write(gain_domain.name+"\nSequence length: "+str(len(gain_domain.sequence))+"\n\n")
            print("[DEBUG] GainDomain.create_indexing.indexing2longfile\n", nom_list, "\n", outfile)
            for j, name in enumerate(nom_list[gain_domain.start:]):
                file.write(f"{gain_domain.sequence[j]}  {str(j+gain_domain.start+offset).rjust(4)}  {name.rjust(7)}\n")

    def disambiguate_anchors(gain_domain, stored_anchor_weight, stored_res, new_anchor_weight, new_res, sse, coiled_residues, outlier_residues, mode='single'):
        #Outlier and Coiled residues are indicated as relative indices (same as sse, stored_res, new_res etc.)
        #print(f"[DEBUG] disambiguate_anchors: {stored_anchor_weight = }, {stored_res = }, {new_anchor_weight = }, {new_res = }, {sse = }")
        def terminate(sse, center_res, break_residues, include=False):
            #print(f"DEBUG terminate : {sse = }, {sse[0] = }, {sse[1] = }, {center_res = }, {adj_break_res = }")           
            # look from the center res to the N and C direction, find the first break residue
            # Return the new boundaries
            if include:
                terminal = 0
            if not include:
                terminal = -1

            n_breaks = [r for r in break_residues if r < center_res and r >= sse[0]]
            c_breaks = [r for r in break_residues if r > center_res and r <= sse[1]]
            
            if n_breaks != []:
                N_boundary = max(n_breaks) - terminal
            else:
                N_boundary = sse[0]
            
            if c_breaks != []:
                C_boundary = min(c_breaks) + terminal
            else:
                C_boundary = sse[1]
            
            return [N_boundary, C_boundary]

        # this mode indicates if the anchor will just be overwritten instead of split,
        #   this will be set to OFF if there are breaks between both anchors
        hasCoiled = False
        hasOutlier = False

        # GO THROUGH CASES DEPENDING ON PRESENT BREAKS

        # a) there are no break residues
        if coiled_residues == [] and outlier_residues == []:
            if new_anchor_weight > stored_anchor_weight:
                name_list, cast_values = create_name_list(sse, new_res, anchor_dict[gain_domain.alignment_indices[new_res]])
            else: 
                name_list, cast_values = create_name_list(sse, stored_res, anchor_dict[gain_domain.alignment_indices[stored_res]])
            
            return [(sse, name_list, cast_values)], False

        # b) check first if there is a Coiled residue in between the two conflicting anchors
        for coiled_res in coiled_residues:
            if stored_res < coiled_res and coiled_res < new_res:
                hasCoiled = True 
            if new_res < coiled_res and coiled_res < stored_res:
                hasCoiled = True

        # c) check if there is an outlier residue in between the two conflicting anchors
        for outlier_res in outlier_residues:
            if stored_res < outlier_res and outlier_res < new_res:
                hasOutlier = True 
            if new_res < outlier_res and outlier_res < stored_res:
                hasOutlier = True

        # If there are no breaks, take the anchor with higher occupation, discard the other.
        if hasCoiled == False and hasOutlier == False:
            if not silent: 
                print(f"DEBUG gain_classes.disambiguate_anchors : no break found between anchors, will just overwrite.")
            if new_anchor_weight > stored_anchor_weight:
                name_list, cast_values = create_name_list(sse, new_res, anchor_dict[gain_domain.alignment_indices[new_res]])
            else: 
                name_list, cast_values = create_name_list(sse, stored_res, anchor_dict[gain_domain.alignment_indices[stored_res]])
            
            return [(sse, name_list, cast_values)], False

        # If breaks are present, find the closest break_residue to the lower weight anchor
        if mode == 'single':
            # "SINGLE" mode: Go and find the split closest to the low-priority anchor. 
            #   Split there and include everything else (even other break residues) in the higher priority anchor segment
            if stored_anchor_weight > new_anchor_weight:
                lower_res = new_res
            else:
                lower_res = stored_res

            if hasCoiled:
                breaker_residues = coiled_residues
            if not hasCoiled:
                breaker_residues = outlier_residues
            print(f"{breaker_residues = }, {lower_res = }")

            lower_seg = terminate(sse, lower_res, breaker_residues, include= not hasCoiled)
            # The rest will be assigned the other SSE, including the breaker residue if it is not coiled:
            lower_n, lower_c = lower_seg[0], lower_seg[1]
            if hasCoiled:
                terminal = -1
            else:
                terminal = 0
            print(f"{sse = }")
            print(f"{sse[0]-lower_n = }, {sse[1]-lower_c = }")
            if sse[0]-lower_n > sse[1]-lower_c: # either should be 0 or positive. This case: lower_seg is C-terminal
                upper_seg = [sse[0], lower_n+terminal]
            else: # This case: lower_seg is N-terminal
                upper_seg = [lower_c-terminal, sse[1]]

            # Go in left and right direction, check where there is the first break
            breaker = None
            offset = 1
            while offset < 20:
                if lower_res + offset in breaker_residues:
                    offset_idx = breaker_residues.index(lower_res + offset)
                    breaker = lower_res + offset
                    break
                if lower_res - offset in breaker_residues:
                    offset_idx = breaker_residues.index(lower_res - offset)
                    breaker = lower_res - offset
                    break

                offset += 1 

            if breaker is None: 
                print("[ERROR] BREAKER residue not found.") 
                return None, None

            # Divide SSE via BREAKER into two segments, create two separate name_list instances for them
            # If the breaker was a coil, set terminal to 1 to exclude this residue; otherwise, include it.
            if hasCoiled:
                terminal = 1
            if not hasCoiled:
                terminal = 0
            seg_N = [sse[0],breaker_residues[offset_idx]-terminal]
            seg_C = [breaker_residues[offset_idx]+terminal, sse[1]]

            if stored_res < breaker:
                seg_stored = seg_N
                seg_new = seg_C
            else:
                seg_stored = seg_C
                seg_new = seg_N
            
            print(f"[TESTING] terminate :{upper_seg = } {lower_seg = }")
            print(f"[TESTING] terminate :{seg_stored = } {seg_new = } {breaker = }")

        if mode == 'double':
            # "DOUBLE" mode: from both sides of the anchor, find the closest breaker residue and terminate each of the new segments
            seg_stored = terminate(sse, stored_res, breaker_residues, include= not hasCoiled)
            seg_new = terminate(sse, new_res, breaker_residues, include= not hasCoiled)

        if not silent: 
            print(f"[NOTE] disambiguate_anchors: Split the segment into: {seg_stored = }, {seg_new = }")

        stored_name_list, stored_cast_values = create_name_list(seg_stored, stored_res, anchor_dict[gain_domain.alignment_indices[stored_res]])
        new_name_list, new_cast_values = create_name_list(seg_new, new_res, anchor_dict[gain_domain.alignment_indices[new_res]])

        #print(f"[NOTE] disambiguate_anchors: Successful SSE split via BREAKER residue @ {breaker}")

        return [(seg_stored, stored_name_list, stored_cast_values),(seg_new, new_name_list, new_cast_values)], True # True indicates if lists were split or not!

    def cast(nom_list, indexing_dir, indexing_centers, sse_x, name_list, cast_values):
        #print(f"DEBUG CAST",sse_x, name_list, cast_values)
        nom_list[sse_x[0]+gain_domain.start : sse_x[1]+1+gain_domain.start] = name_list
        indexing_dir[cast_values[2]] = cast_values[0] # all to sse 
        indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the anchor is located
        return nom_list, indexing_dir, indexing_centers

### END OF FUNCTION BLOCK

    # Initialize Dictionaries
    indexing_dir = {}
    indexing_centers = {}
    named_residue_dir = {}
    unindexed = []
    # One-indexed Indexing list for each residue, mapping for the actual residue index
    nom_list = np.full([gain_domain.end+1], fill_value='      ', dtype='<U7')

    for i,typus in enumerate([gain_domain.sda_helices, gain_domain.sdb_sheets]): # Both types will be indexed separately
        # Go through each individual SSE in the GAIN SSE dictionary
        for idx, sse in enumerate(typus):
            # Get first and last residue of this SSE
            first_col = gain_domain.alignment_indices[sse[0]]

            if debug: print(f"DEBUG {sse[1] = }; {gain_domain.end-gain_domain.start = }, {len(gain_domain.alignment_indices) = }")
            if sse[1] > gain_domain.end-gain_domain.start-1:
                last_col = gain_domain.alignment_indices[-1]
                sse_end = sse[1]-1
            else:
                last_col = gain_domain.alignment_indices[sse[1]]
                sse_end = sse[1]
            exact_match = False                             # This is set to True, otherwise continue to Interval search
            fuzzy_match = False                             # Flag for successful Interval search detection
            ambiguous = False                               # Flag for ambiguity
            if debug:print(f"[DEBUG] GainDomain.create_indexing : \n{typus} No. {idx+1}: {sse}")
            if debug:print(f"[DEBUG] GainDomain.create_indexing : \n{first_col = }, {last_col = }")
            
            for sse_res in range(sse[0],sse_end+1):
                # Find the corresponding alignment index for that residue:
                if sse_res < len(gain_domain.alignment_indices):
                    sse_idx = gain_domain.alignment_indices[sse_res]
                else:
                    continue
                
                if sse_idx in anchors:

                    if exact_match == False:  
                        if debug: print(f"PEAK FOUND: @ {sse_res = }, {sse_idx = }, {anchor_dict[sse_idx]}")
                        #sse_name = anchor_dict[sse_idx]
                        anchor_idx = np.where(anchors == sse_idx)[0][0]
                        if debug: print(f"{np.where(anchors == sse_idx)[0] = }, {sse_idx = }, {anchor_idx = }")
                        stored_anchor_weight = anchor_occupation[anchor_idx]
                        #stored_anchor = anchors[anchor_idx]
                        stored_res = sse_res
                        name_list, cast_values = create_name_list(sse, sse_res, anchor_dict[sse_idx])
                        # name_list has the assignment for the SSE, cast_values contains the passed values for dict casting
                        exact_match = True
                        continue
                    ''' HERE is an ANCHOR AMBIGUITY CASE
                            There might occur the case where two anchors are within one SSE, 
                            check for present break residues in between the two anchors, 
                            > If there are some, eliminate that residue and break the SSE it into two.
                                >   If there are multiple break residues, use the one closest to the lower occupancy anchor
                            > If there is no break residue the anchor with highest occupation wins. '''
                    ambiguous = True
                    if not silent: 
                        print(f"[NOTE] GainDomain.create_indexing : ANCHOR AMBIGUITY in this SSE:")
                        print(f"\n\t {sse_idx = },")
                        print(f"\n\t {anchor_dict[sse_idx] = },")
                    # if the new anchor is scored better than the first, replace!

                    # Check for residues that have assigned "C" or "h" in GainDomain.sse_sequence
                    coiled_residues = []
                    outlier_residues = []
                    max_key = max(gain_domain.sse_sequence.keys())
                    for i in range(sse[0]+gain_domain.start, sse[1]+gain_domain.start+1):
                        if i > max_key:
                            if debug:
                                print("[DEBUG]: GainDomain.create_indexing. {i = } exceeded {max_key = }")
                            break
                        if gain_domain.sse_sequence[i] ==  "C":
                            coiled_residues.append(i-gain_domain.start)
                        if gain_domain.sse_sequence[i] == 'h':
                            outlier_residues.append(i-gain_domain.start)
                    if debug:
                        print(f"[DEBUG] GainDomain.create_indexing :\n\t{coiled_residues  = }\n\t{outlier_residues = }")
                    disambiguated_lists, isSplit = disambiguate_anchors(gain_domain,
                                                                        stored_anchor_weight=stored_anchor_weight,
                                                                        stored_res=stored_res,
                                                                        new_anchor_weight=anchor_occupation[np.where(anchors == sse_idx)[0][0]],
                                                                        new_res=sse_res,
                                                                        sse=sse,
                                                                        coiled_residues=coiled_residues,
                                                                        outlier_residues=outlier_residues,
                                                                        mode=split_mode)
                    if not silent: print(disambiguated_lists)
                    sse_adj, name_list, cast_values = disambiguated_lists[0]
                    if isSplit:
                        sse_adj_2, name_2, cast_2 = disambiguated_lists[1]
                        if not silent: print(f"[DEBUG] GainDomain.create_indexing : Found a split list:\n"
                            f"{sse_adj_2  = },\t{name_2 = },\t{cast_2 =  }")
                        nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj_2, name_2, cast_2)
                        # Also write split stuff to the new dictionary
                        for entryidx, entry in enumerate(name_2): 
                            named_residue_dir[entry] = entryidx+sse_adj[0]+gain_domain.start
                    # if anchor_occupation[np.where(anchors == sse_idx)[0]] > stored_anchor_weight:
                    #    name_list, cast_values = create_name_list(sse, sse_res, anchor_dict[sse_idx])
            # if no exact match is found, continue to Interval search and assignment.
            if exact_match == False:
                # expand anchor detection to +1 and -1 of SSE interval
                ex_first_col = gain_domain.alignment_indices[sse[0]-1]
                try:
                    ex_last_col = gain_domain.alignment_indices[sse[1]+1]
                except:
                    ex_last_col = last_col
                # Construct an Interval of alignment columns corresp. to the SSE residues
                if debug:print(f"[DEBUG] GainDomain.create_indexing : \nNo exact match found: extindeing search.\n{typus} No. {idx+1}: {sse}")
                for peak in anchors: # Look if any peak is contained here
                    
                    if ex_first_col <= peak and ex_last_col >= peak:
                        fuzzy_match = True
                        if not silent: print(f"[DEBUG] GainDomain.create_indexing : Interval search found anchor @ Column {peak}")
                        # Find the closest residue to the anchor column index. N-terminal wins if two residues tie.
                        peak_dists = [abs(gain_domain.alignment_indices[res]-peak) \
                                                for res in range(sse[0], sse_end+1)]

                        ref_idx = peak_dists.index(min(peak_dists))
                        ref_res = range(sse[0], sse_end+1)[ref_idx]

                        if not silent: print(f"NOTE: GainDomain.create_indexing : Interval search found SSE:"
                                                f"{peak = }, {peak_dists = }, {ref_res = }. \n"
                                                f"NOTE: GainDomain.create_indexing : This will be named {anchor_dict[peak]}")

                        name_list, cast_values = create_name_list(sse, ref_res, anchor_dict[peak])

            # Finally, if matched, write the assigned nomeclature segment to the array
            
            if ambiguous == False and exact_match == True or fuzzy_match == True:
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse, name_list, cast_values)
                # Also cast to general indexing dictionary
                for namidx, entry in enumerate(name_list):
                    named_residue_dir[entry] = namidx+sse[0]+gain_domain.start
            elif ambiguous == True:
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj, name_list, cast_values)
                # Also cast to general indexing dictionary
                for namidx, entry in enumerate(name_list): 
                    named_residue_dir[entry] = namidx+sse_adj[0]+gain_domain.start
            else: # If there is an unadressed SSE with length 3 or more, then add this to unindexed.
                if sse[1]-sse[0] > 3:
                    if debug: print(f"[DEBUG] GainDomain.create_indexing : No anchor found! \n {gain_domain.alignment_indices[sse[0]] = } \ns{gain_domain.alignment_indices[sse_end] = }")
                    unindexed.append(gain_domain.alignment_indices[sse[0]])
    # Patch the GPS into the nom_list
    labels = ["GPS-2","GPS-1","GPS+1"]
    for i, residue in enumerate(gain_domain.GPS.residue_numbers[:3]):
        #print(residue)
        nom_list[residue] = labels[i]
        indexing_dir["GPS"] = gain_domain.GPS.residue_numbers
        # Also cast this to the general indexing dictionary
        named_residue_dir[labels[i]] = gain_domain.GPS.residue_numbers[i]
    # FUTURE CHANGE : GPS assignment maybe needs to be more fuzzy -> play with the interval of the SSE 
    #       and not the explicit anchor. When anchor col is missing, the whole SSE wont be adressed
    # print([DEBUG] : GainDomain.create_indexing : ", nom_list)

    # Create a indexing File if specified
    if outdir is not None:
        indexing2longfile(gain_domain, nom_list, f"{outdir}/{gain_domain.name}.txt", offset=offset)

    return indexing_dir, indexing_centers, named_residue_dir, unindexed





def create_subdomain_indexing(self, subdomain, actual_anchors, anchors, anchor_occupation, anchor_dict, silent=False, split_mode='single',debug=False):
        ''' 
        Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
        Prints out the final list and writes it to file if outdir is specified
        
        Parameters
        ----------
        anchors : list, required
            List of anchors with each corresponding to an alignment index
        anchor_occupation : list, required
            List of Occupation values corresponding to each anchor for resolving ambiguity conflicts
        anchor_dict : dict, required
            Dictionary where each anchor index is assigned a name (H or S followed by a greek letter for enumeration)
        offset : int,  optional (default = 0)
            An offsset to be incorporated (i.e. for offsetting model PDBs against UniProt entries)
        silent : bool, optional (default = False)
            opt in to run wihtout so much info.
        outdir : str, optional
            Output directory where the output TXT is going to be written as {self.name}.txt

        Returns
        ---------
        indexing_dir : dict
            A dictionary containing the residue indices for each assigned SSE and the GPS
        indexing_centers : dict
            A dictionary containing the XX.50 Residue for each assigned SSE
        named_residue_dir : dict
            A dictionary mapping individual consensus labels to their respective position.
        unindexed : list
            A list of detected SSE that remain unindexed.
        '''
        # recurring functional blocks used within this function

        def create_name_list(sse, ref_res, sse_name):
            ''' Creates a indexing list for an identified SSE with reference index as X.50,
                Also returns a tuple containing the arguments, enabling dict casting for parsing '''
            number_range = range(50+sse[0]-ref_res, 51+sse[1]-ref_res)
            name_list = [f"{sse_name}.{num}" for num in number_range]
            cast_values = (sse, ref_res, sse_name)
            return name_list, cast_values

        def disambiguate_anchors(self, stored_anchor_weight, stored_res, new_anchor_weight, new_res, sse, break_residues, mode='single'):
            #print(f"[DEBUG] disambiguate_anchors: {stored_anchor_weight = }, {stored_res = }, {new_anchor_weight = }, {new_res = }, {sse = }")
            def terminate(sse, center_res, adj_break_res):
                #print(f"DEBUG terminate : {sse = }, {sse[0] = }, {sse[1] = }, {center_res = }, {adj_break_res = }")
                # neg bound
                offset = 0
                while center_res-offset >= sse[0]:
                    if center_res-offset == sse[0]: 
                        N_boundary = center_res - offset
                        break
                    if center_res-offset in adj_break_res:
                        N_boundary = center_res - offset + 1   # Do not include the BREAK residue itself
                        break
                    offset+=1
                
                offset = 0
                while center_res+offset <= sse[1]:
                    if center_res+offset == sse[1]: 
                        C_boundary = center_res + offset
                        break
                    if center_res+offset in adj_break_res:
                        C_boundary = center_res + offset - 1   # Do not include the BREAK residue itself
                        break
                    offset+=1

                return [N_boundary, C_boundary]

            # this mode indicates if the anchor will just be overwritten instead of split,
            #   this will be set to OFF if there are breaks between both anchors
            priority_mode = True

            # First case, there are no break residues
            if break_residues == []:
                if new_anchor_weight > stored_anchor_weight:
                    name_list, cast_values = create_name_list(sse, new_res, anchor_dict[self.alignment_indices[new_res]])
                else: 
                    name_list, cast_values = create_name_list(sse, stored_res, anchor_dict[self.alignment_indices[stored_res]])
                
                return [(sse, name_list, cast_values)], False

            adjusted_break_residues = [res+sse[0] for res in break_residues]

            # Check if there is a break residue in between the two conflicting anchors
            for break_res in adjusted_break_residues:
                if stored_res < break_res and break_res < new_res:
                    priority_mode = False 
                if new_res < break_res and break_res < stored_res:
                    prority_mode = False

            if priority_mode == True:
                if not silent: 
                    print(f"DEBUG gain_classes.disambiguate_anchors : no break found between anchors, will just overwrite.")
                if new_anchor_weight > stored_anchor_weight:
                    name_list, cast_values = create_name_list(sse, new_res, anchor_dict[self.alignment_indices[new_res]])
                else: 
                    name_list, cast_values = create_name_list(sse, stored_res, anchor_dict[self.alignment_indices[stored_res]])
                
                return [(sse, name_list, cast_values)], False

            # Find the closest break_residue to the lower weight anchor
            if mode == 'single':
                if stored_anchor_weight > new_anchor_weight:
                    lower_res = new_res
                else:
                    lower_res = stored_res

                print(f"{adjusted_break_residues = }, {lower_res = }")
                # Go in left and right direction, check where there is the first break
                breaker = None
                offset = 1
                while offset < 20:
                    if lower_res + offset in adjusted_break_residues:
                        offset_idx = adjusted_break_residues.index(lower_res + offset)
                        breaker = lower_res + offset
                        break
                    if lower_res - offset in adjusted_break_residues:
                        offset_idx = adjusted_break_residues.index(lower_res - offset)
                        breaker = lower_res - offset
                        break

                    offset += 1 

                if breaker is None: 
                    print("[ERROR] BREAKER residue not found.") 
                    return None, None

                # Divide SSE via BREAKER into two segments, create two separate name_list instances for them
                seg_N = [sse[0],adjusted_break_residues[offset_idx]-1]
                seg_C = [adjusted_break_residues[offset_idx]+1, sse[1]]

                if stored_res < breaker:
                    seg_stored = seg_N
                    seg_new = seg_C
                else:
                    seg_stored = seg_C
                    seg_new = seg_N

            if mode == 'double':
                seg_stored = terminate(sse, stored_res, adjusted_break_residues)
                seg_new = terminate(sse, new_res, adjusted_break_residues)

            if not silent: print(f"[NOTE] disambiguate_anchors: Split the segment into: {seg_stored = }, {seg_new = }")
            stored_name_list, stored_cast_values = create_name_list(seg_stored, stored_res, anchor_dict[stored_res])
            new_name_list, new_cast_values = create_name_list(seg_new, new_res, anchor_dict[new_res])

            #print(f"[NOTE] disambiguate_anchors: Successful SSE split via BREAKER residue @ {breaker}")

            return [(seg_stored, stored_name_list, stored_cast_values),(seg_new, new_name_list, new_cast_values)], True # True indicates if lists were split or not!

        def cast(nom_list, indexing_dir, indexing_centers, sse_x, name_list, cast_values):
            #print(f"DEBUG CAST",sse_x, name_list, cast_values)
            nom_list[sse_x[0]+self.start : sse_x[1]+1+self.start] = name_list
            indexing_dir[cast_values[2]] = cast_values[0] # all to sse 
            indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the anchor is located
            return nom_list, indexing_dir, indexing_centers

    ### END OF FUNCTION BLOCK

        # Initialize Dictionaries
        indexing_dir = {}
        indexing_centers = {}
        named_residue_dir = {}
        unindexed = []

        # Invert the actual anchors dict to match the GAIN residue to the named anchor
        res2anchor = {v:k for k,v in actual_anchors.items()}
        # One-indexed Indexing list for each residue, mapping for the actual residue index
        nom_list = np.full([self.end+1], fill_value='      ', dtype='<U7')
        breaks = [self.a_breaks, self.b_breaks]
        if not silent: print(f"{breaks = }")
        if subdomain.lower() == 'a':
            sses = self.sda_helices
            type_breaks = breaks[0]
        elif subdomain.lower() == 'b':
            sses = self.sdb_sheets
            type_breaks = breaks[1]

        if not silent:
            print(type_breaks)

        # Go through each individual SSE in the GAIN SSE dictionary
        for idx, sse in enumerate(sses):
            # Get first and last residue of this SSE
            try:first_col = self.alignment_indices[sse[0]]
            except: continue
            # Error correction. Sometimes the detected last Strand exceeds the GAIN boundary.
            if debug: print(f"DEBUG {sse[1] = }; {self.end-self.start = }, {len(self.alignment_indices) = }")
            if sse[1] > self.end-self.start-1:
                last_col = self.alignment_indices[-1]
                sse_end = sse[1]-1
            else:
                last_col = self.alignment_indices[sse[1]]
                sse_end = sse[1]

            exact_match = False                             # This is set to True, otherwise continue to Interval search
            fuzzy_match = False                             # Flag for successful Interval search detection
            ambiguous = False                               # Flag for ambiguity

            if debug:print(f"[DEBUG] GainDomain.create_indexing : \nNo. {idx+1}: {sse}\n{first_col = }, {last_col = }")
                
            for sse_res in range(sse[0],sse_end+1):
                # Find the anchor within, since the anchors here are already aligned!
                if sse_res < len(self.alignment_indices):
                    sse_idx = self.alignment_indices[sse_res]
                else:
                    continue
                    
                if sse_res in res2anchor.keys():

                    if exact_match == False:  
                            if debug: print(f"ANCHOR FOUND: @ {sse_res = }, {res2anchor[sse_res]}")
                            sse_name = res2anchor[sse_res]
                            stored_anchor_weight = anchor_occupation[sse_name]
                            #stored_anchor = sse_name
                            stored_res = sse_res
                            name_list, cast_values = create_name_list(sse, sse_res, sse_name)
                            # name_list has the assignment for the SSE, cast_values contains the passed values for dict casting
                            exact_match = True
                            continue
                    ''' HERE is an ANCHOR AMBIGUITY CASE
                       There might occur the case where two anchors are within one SSE, 
                       check for present break residues in between the two anchors, 
                        > If there are some, eliminate that residue and break the SSE it into two.
                            >   If there are multiple break residues, use the one closest to the lower occupancy anchor
                        > If there is no break residue the anchor with highest occupation wins. '''
                    ambiguous = True
                    if not silent: 
                            print(f"[NOTE] GainDomain.create_indexing : ANCHOR AMBIGUITY in this SSE:")
                            print(f"\n\t {sse_res = },")
                            print(f"\n\t {res2anchor[sse_res] = },")
                            print(f"{type_breaks = } \t {idx = }")
                            print(f"\n\t {type_breaks[idx] = }")
                        # if the new anchor is scored better than the first, replace!
                    disambiguated_lists, isSplit = disambiguate_anchors(self,
                                                                        stored_anchor_weight=stored_anchor_weight,
                                                                        stored_res=stored_res,
                                                                        new_anchor_weight=anchor_occupation[res2anchor[sse_res]],
                                                                        new_res=sse_res,
                                                                        sse=sse,
                                                                        break_residues=type_breaks[idx],
                                                                        mode=split_mode)
                    if not silent: print(disambiguated_lists)
                    sse_adj, name_list, cast_values = disambiguated_lists[0]
                    if isSplit:
                        sse_adj_2, name_2, cast_2 = disambiguated_lists[1]
                        if not silent: print(f"[DEBUG] GainDomain.create_indexing : Found a split list:\n"
                            f"{sse_adj_2  = },\t{name_2 = },\t{cast_2 =  }")
                        nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj_2, name_2, cast_2)
                        # Also write split stuff to the new dictionary
                        for entryidx, entry in enumerate(name_2): 
                            named_residue_dir[entry] = entryidx+sse_adj[0]+self.start
                    # if anchor_occupation[np.where(anchors == sse_idx)[0]] > stored_anchor_weight:
                    #    name_list, cast_values = create_name_list(sse, sse_res, anchor_dict[sse_idx])
            # if no exact match is found, continue to Interval search and assignment.
            if exact_match == False:
                # expand anchor detection to +1 and -1 of SSE interval
                ex_first_col = self.alignment_indices[sse[0]-1]
                try:
                    ex_last_col = self.alignment_indices[sse[1]+1]
                except:
                    ex_last_col = last_col
                # Construct an Interval of alignment columns corresp. to the SSE residues
                if debug:print(f"[DEBUG] GainDomain.create_indexing : \nNo exact match found: extindeing search.\n No. {idx+1}: {sse}")
                for peak in anchors: # Look if any peak is contained here
                        
                    if ex_first_col <= peak and ex_last_col >= peak:
                        fuzzy_match = True
                        if not silent: print(f"[DEBUG] GainDomain.create_indexing : Interval search found anchor @ Column {peak}")
                        # Find the closest residue to the anchor column index. N-terminal wins if two residues tie.
                        peak_dists = [abs(self.alignment_indices[res]-peak) \
                                               for res in range(sse[0], sse_end+1)]

                        ref_idx = peak_dists.index(min(peak_dists))
                        ref_res = range(sse[0], sse_end+1)[ref_idx]

                        if not silent: print(f"NOTE: GainDomain.create_indexing : Interval search found SSE:"
                                                f"{peak = }, {peak_dists = }, {ref_res = }. \n"
                                                f"NOTE: GainDomain.create_indexing : This will be named {anchor_dict[peak]}")

                        name_list, cast_values = create_name_list(sse, ref_res, anchor_dict[peak])

            # Finally, if matched, write the assigned nomeclature segment to the array
                
            if ambiguous == False and exact_match == True or fuzzy_match == True:
                    nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse, name_list, cast_values)
                    # Also cast to general indexing dictionary
                    for namidx, entry in enumerate(name_list):
                        named_residue_dir[entry] = namidx+sse[0]+self.start
            elif ambiguous == True:
                    nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj, name_list, cast_values)
                    # Also cast to general indexing dictionary
                    for namidx, entry in enumerate(name_list): 
                        named_residue_dir[entry] = namidx+sse_adj[0]+self.start
            else: # If there is an unadressed SSE with length 3 or more, then add this to unindexed.
                    if sse[1]-sse[0] > 3:
                        if debug: print(f"[DEBUG] GainDomain.create_indexing : No anchor found! \n {self.alignment_indices[sse[0]] = } \ns{self.alignment_indices[sse_end] = }")
                        unindexed.append(self.alignment_indices[sse[0]])
        # Patch the GPS into the nom_list
        """        labels = ["GPS-2","GPS-1","GPS+1"]
        for i, residue in enumerate(self.GPS.residue_numbers[:3]):
            #print(residue)
            nom_list[residue] = labels[i]
            indexing_dir["GPS"] = self.GPS.residue_numbers
            # Also cast this to the general indexing dictionary
            named_residue_dir[labels[i]] = self.GPS.residue_numbers[i]"""
        # FUTURE CHANGE : GPS assignment maybe needs to be more fuzzy -> play with the interval of the SSE 
        #       and not the explicit anchor. When anchor col is missing, the whole SSE wont be adressed
        # print([DEBUG] : GainDomain.create_indexing : ", nom_list)
            
        return indexing_dir, indexing_centers, named_residue_dir, unindexed



def find_boundaries(sse_dict, seq_len, bracket_size=50, domain_threshold=50, coil_weight=0):
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
    boundaries = sse_func.detect_signchange(signal, exclude_zero=True)
    plt.plot(scored_seq)
    print(boundaries)
    ### Find the interval with most negative values
    # First, check if there are valid boundaries!
    try:
        len(boundaries)
    except TypeError:
        print('No boundaries detected.')
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
    if maxk != None:
        gain_start, initial_boundary = boundaries[maxk], boundaries[maxk+1]
    
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

    # If you did not find the maxk previously:
    return None, None
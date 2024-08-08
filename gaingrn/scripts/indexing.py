import numpy as np

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
    # FUNCTION BLOCK FOR INTERNAL FUNCTIONS
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
            #try:
            first_col = gain_domain.alignment_indices[sse[0]]
            #except: continue
            # Error correction. Sometimes the detected last Strand exceeds the GAIN boundary.
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
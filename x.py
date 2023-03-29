import glob,shutil
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
## scritps/assign.py
# contains the main function for assigning the GAIN-GRN indexing on single GAIN domains or whole sets of GAIN domains

import glob
import json
import os
from types import SimpleNamespace
import numpy as np

from gaingrn.utils.io import get_agpcr_type, read_gesamt_pairs, run_logged_command
from gaingrn.utils.structure_utils import get_pdb_extents
from gaingrn.utils.template_utils import find_center_matches, find_best_templates


def create_subdomain_indexing(gain_obj, subdomain, actual_centers, threshold=3, padding=1, split_mode='single', silent=False,  debug=False):
    ''' 
    Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
    Prints out the final list and writes it to file if outdir is specified

    Parameters
    ----------
    gain_obj: GainDomain object, requred
        GainDomain object with corresponding parameters
    actual_centers : dict, required
        Dictionary of centers with each corresponding to the matched template center. This should already be the residue of this GAIN domain
    center_dict : dict, required
        Dictionary where each center index is assigned a name (H or S followed by a greek letter for enumeration)
    offset : int,  optional (default = 0)
        An offsset to be incorporated (i.e. for offsetting model PDBs against UniProt entries)
    silent : bool, optional (default = False)
        opt in to run wihtout so much info.
    outdir : str, optional
        Output directory where the output TXT is going to be written as {gain_obj.name}.txt

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
            return name_list, (sse, ref_res, sse_name)

    def disambiguate_segments(stored_res, new_res, sse, break_residues):

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

        if not silent: print(f"[DEBUG] disambiguate_segments: {stored_res = }\n\t{new_res = }\n\t{sse = }")
        # First case, there are no break residues. Raise exception, we do not want this, there should always be a break here.
        # GO THROUGH CASES DEPENDING ON PRESENT BREAKS

        # a) there are no break residues
        if coiled_residues == [] and outlier_residues == []:
            raise IndexError(f"No Break residues in between to Anchors:\n\t{stored_res = }\n\t{new_res = }\n\t{sse = }")

        # b) check first if there is a Coiled residue in between the two conflicting centers
        for coiled_res in coiled_residues:
            if stored_res < coiled_res and coiled_res < new_res:
                hasCoiled = True
            if new_res < coiled_res and coiled_res < stored_res:
                hasCoiled = True

        # c) check if there is an outlier residue in between the two conflicting centers
        for outlier_res in outlier_residues:
            if stored_res < outlier_res and outlier_res < new_res:
                hasOutlier = True
            if new_res < outlier_res and outlier_res < stored_res:
                hasOutlier = True

        # If there are no breaks, take the center with higher occupation, discard the other.
        if hasCoiled == False and hasOutlier == False:
            raise IndexError(f"No Break residues in between to Anchors:\n\t{stored_res = }\n\t{new_res = }\n\t{sse = }\n\t{coiled_res = }\n\t{outlier_res = }")

        # Check if there is a break residue in between the two conflicting centers
        adjusted_break_residues = [res+sse[0] for res in break_residues]

        seg_stored = terminate(sse, stored_res, adjusted_break_residues)
        seg_new = terminate(sse, new_res, adjusted_break_residues)

        if not silent: print(f"[NOTE] disambiguate_segments: Split the segment into: {seg_stored = }, {seg_new = }")

        stored_name_list, stored_cast_values = create_name_list(seg_stored, stored_res, res2center[stored_res])
        new_name_list, new_cast_values = create_name_list(seg_new, new_res, res2center[new_res])

        if not silent: print(f"[NOTE] disambiguate_segments: Successful SSE split via BREAK.")
        return [(seg_stored, stored_name_list, stored_cast_values),(seg_new, new_name_list, new_cast_values)]

    def cast(nom_list, indexing_dir, indexing_centers, sse_x, name_list, cast_values):
        # Inserts the names of an SSE into the nom_list list of strings, returns the modified version
        if debug:
            print(f"DEBUG CAST",sse_x, name_list, cast_values)

        nom_list[sse_x[0] : sse_x[1]+1] = name_list
        indexing_dir[cast_values[2]] = cast_values[0] # all to sse 
        indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the center is located

        return nom_list, indexing_dir, indexing_centers

    if debug: print(f"[DEBUG] create_subdomain_indexing: passed arguments:\n\t{gain_obj.name = }\n\t{subdomain = }\n\t{actual_centers = }")
    ### END OF FUNCTION BLOCK
    # Initialize Dictionaries
    indexing_dir = {}
    indexing_centers = {}
    named_residue_dir = {}
    unindexed = []

    # Invert the actual centers dict to match the GAIN residue to the named center
    res2center = {v[0]:k for k,v in actual_centers.items()} # v is a tuple  (residue, distance_to_template_center_residue)
    if debug: print(f"[DEBUG] create_subdomain_indexing: {res2center = }")
    # One-indexed Indexing list for each residue, mapping for the actual residue index
    nom_list = np.full(shape=[gain_obj.end+1], fill_value='      ', dtype='<U7')

    if subdomain.lower() == 'a':
        sses = gain_obj.sda_helices
    elif subdomain.lower() == 'b':
        sses = gain_obj.sdb_sheets

    # Go through each individual SSE in the GAIN SSE dictionary
    for idx, sse in enumerate(sses):
        first_res, last_res = sse[0]+gain_obj.start, sse[1]+gain_obj.start

        exact_match = False                             # This is set to True, otherwise continue to Interval search
        fuzzy_match = False                             # Flag for successful Interval search detection
        ambiguous = False                               # Flag for ambiguity

        if debug:print(f"[DEBUG] create_subdomain_indexing : \nNo. {idx+1}: {sse}\n{first_res = }, {last_res = }")

        for sse_res in range(first_res, last_res+1):
            # Find the center within, since the centers here are already aligned!

            if sse_res in res2center.keys():

                if exact_match == False:
                    if debug: print(f"ANCHOR FOUND: @ {sse_res = }, {res2center[sse_res]}")
                    sse_name = res2center[sse_res]
                    #stored_center = sse_name
                    stored_res = sse_res
                    name_list, cast_values = create_name_list([first_res, last_res], sse_res, sse_name)
                    # name_list has the assignment for the SSE, cast_values contains the passed values for dict casting
                    exact_match = True
                    continue
                ''' HERE is an ANCHOR AMBIGUITY CASE
                   There might occur the case where two centers are within one SSE, 
                   check for present break residues in between the two centers, 
                    > If there are some, eliminate that residue and break the SSE it into two.
                        >   If there are multiple break residues, use the one closest to the lower occupancy center
                    > If there is no break residue the center with highest occupation wins. '''
                ambiguous = True
                if not silent:
                            print(f"[NOTE] GainDomain.create_indexing : ANCHOR AMBIGUITY in this SSE:")
                            print(f"\n\t {sse_res = },")
                            print(f"\n\t {res2center[sse_res] = },")

                coiled_residues = []
                outlier_residues = []
                for i in range(sse[0]+gain_obj.start, sse[1]+1+gain_obj.start):
                    if gain_obj.sse_sequence[i] ==  "C":
                        coiled_residues.append(i-gain_obj.start)
                    if gain_obj.sse_sequence[i] in ['h','e']:
                        outlier_residues.append(i-gain_obj.start)

                if debug:
                    print(f"[DEBUG] GainDomain.create_indexing :\n\t{coiled_residues  = }\n\t{outlier_residues = }")

                disambiguated_list = disambiguate_segments(stored_res=stored_res,
                                                            new_res=sse_res,
                                                            sse=[first_res, last_res],
                                                            coiled_residue=coiled_residues,
                                                            outlier_residues=outlier_residues,
                                                            mode=split_mode
                                                            )
                if not silent:
                    print(f"[NOTE] {disambiguated_list = }")
                sse_adj, name_list, cast_values = disambiguated_list[0]
                sse_adj_2, name_2, cast_2 = disambiguated_list[1]

                if not silent: print(f"[DEBUG] GainDomain.create_indexing : Found a split list:\n"
                    f"{sse_adj_2  = },\t{name_2 = },\t{cast_2 =  }")
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj_2, name_2, cast_2)
                # Also write split stuff to the new dictionary
                for entryidx, entry in enumerate(name_2):
                    named_residue_dir[entry] = entryidx+sse_adj[0]

        # if no exact match is found, the center does not exist in this domain
        if exact_match == False:
            if debug: print("Anchors not found in this sse", idx, sse)
            # expand center detection to +1 and -1 of SSE interval
            for res in res2center:
                if res == sse[0]-padding or res == sse[1]+padding:
                    fuzzy_match = True
                    if not silent: print(f"[DEBUG] GainDomain.create_indexing : Interval search found center @ Residue {res}")
                    # Find the closest residue to the center column index. N-terminal wins if two residues tie.
                    name_list, cast_values = create_name_list([first_res, last_res], res, res2center[res])

        # Finally, if matched, write the assigned indexed segment to the array

        if ambiguous == False and exact_match == True or fuzzy_match == True:
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, [first_res, last_res], name_list, cast_values)
                # Also cast to general indexing dictionary
                for namidx, entry in enumerate(name_list):
                    named_residue_dir[entry] = namidx+first_res
        elif ambiguous == True:
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj, name_list, cast_values)
                # Also cast to general indexing dictionary
                for namidx, entry in enumerate(name_list):
                    named_residue_dir[entry] = namidx+sse_adj[0]
        else: # If there is an unadressed SSE with length 3 or more, then add this to unindexed.
                if sse[1]-sse[0] >= threshold:
                    if debug: print(f"[DEBUG] GainDomain.create_indexing : No center found! \n {first_res = } \ns{last_res = }")
                    unindexed.append(first_res)
    # [NOTE] GPS patching moved to central function -> assign_indexing      
    return indexing_dir, indexing_centers, named_residue_dir, unindexed


def assign_indexing(gain_obj:object, file_prefix: str, gain_pdb: str, template_dir: str, gesamt_bin:str,
                    template_json="tdata.json", outlier_cutoff=10.0,
                    hard_cut=None, debug=False, create_pdb=False, patch_gps=False, pseudocenters=None,
                    template_mode='extent', sda_mode='rmsd'):
    if debug:
        print(f"[DEBUG] assign_indexing: {gain_obj.start = }\n\t{gain_obj.end = }\n\t{gain_obj.subdomain_boundary = }\n\t{gain_pdb = }\n\t{template_mode = }")
    # Arbitrarily defined data for templates. Receptor type -> template ID
    # Load the data associated with the templates from the JSON file.
    with open(template_json) as jj:
        tdata = SimpleNamespace(**json.load(jj))
    # evaluate the template dir and find sda and sdb templates:
    sda_templates = {}
    sdb_templates = {}
    pdbs = glob.glob(f"{template_dir}/*pdb")

    for pdb in pdbs:
        p_name = pdb.split("/")[-1].split("_")[0]
        if "b" in p_name:
            sdb_templates[p_name] = pdb
        else:
            sda_templates[p_name] = pdb

    if debug:
        print(f"{sda_templates = }\n{sdb_templates = }")

    # In case the file_prefix specifies a folder, check if the folder exists. If not, create it.
    if "/" in file_prefix:
        target_path = "/".join(file_prefix.split("/")[:-1])
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

    # Get the agpcr-type and the corresponding templates to be matched.
    agpcr_type = get_agpcr_type(gain_obj.name)
    if debug: print(f"[DEBUG] assign_indexing: {agpcr_type = }")
    # If the type is unknown, get the best matching templates by performing RMSD after alignment via GESAMT
    if agpcr_type not in tdata.type_2_sda_template.keys():
        if agpcr_type[0] not in tdata.type_2_sda_template.keys():
            best_a, best_b = find_best_templates(gain_obj, gesamt_bin, gain_pdb, sda_templates, sdb_templates, template_mode=template_mode, debug=debug, sda_mode=sda_mode)
            if debug:
                print(f"[DEBUG] assign_indexing: running template search with unknown receptor.\n{best_a = } {best_b = }")
        else:
            best_a = tdata.type_2_sda_template[agpcr_type[0]]
            best_b = tdata.type_2_sdb_template[agpcr_type[0]]
    else:
        best_a = tdata.type_2_sda_template[agpcr_type] # This is the best template ID, i.e. "A"
        best_b = tdata.type_2_sdb_template[agpcr_type] #              -"-                   "E5b"
        if debug:
            print(f"Looking up best template for {agpcr_type = }: {best_a = } {best_b = }")

    a_centers = tdata.sda_centers[best_a]
    b_centers = tdata.sdb_centers[best_b]

    if debug:
        print(f"[DEBUG] assign_indexing: Found receptor type and best corresponding templates:\n\t{agpcr_type = }\n\t{best_a = }\n\t{best_b = }\n\t{a_centers = }\n\t{b_centers = }")

    pdb_start, sda_boundary, sdb_boundary, pdb_end = get_pdb_extents(gain_pdb, gain_obj.subdomain_boundary)
    a_cmd_string = f'{gesamt_bin} {sda_templates[best_a]} {gain_pdb} -s A/{pdb_start}-{sda_boundary}'
    b_cmd_string = f'{gesamt_bin} {sdb_templates[best_b]} {gain_pdb} -s A/{sdb_boundary}-{pdb_end}'

    if create_pdb:
        a_cmd_string = a_cmd_string + f" -o {file_prefix}_sda.pdb"
        b_cmd_string = b_cmd_string + f" -o {file_prefix}_sdb.pdb"
    # run GESAMT, keep the file to read later and documentation
    run_logged_command(a_cmd_string, logfile=None, outfile=f'{file_prefix}_sda.out')
    run_logged_command(b_cmd_string, logfile=None, outfile=f'{file_prefix}_sdb.out')

    target_a_centers = find_center_matches(f'{file_prefix}_sda.out', a_centers, isTarget=False, debug=debug)
    target_b_centers = find_center_matches(f'{file_prefix}_sdb.out', b_centers, isTarget=False, debug=debug)
    a_template_matches, _ = read_gesamt_pairs(f'{file_prefix}_sda.out', return_unmatched=False)
    b_template_matches, _ = read_gesamt_pairs(f'{file_prefix}_sdb.out', return_unmatched=False)
    if debug:
        print(f"[DEBUG] assign_indexing: The matched centers of the target GAIN domain are:{target_a_centers = }\n\t{target_b_centers = }")

    a_out = list(create_compact_indexing(gain_obj, 'a', target_a_centers, threshold=3, outlier_cutoff=outlier_cutoff,
                                         template_matches=a_template_matches,
                                         template_extents=tdata.element_extents[best_a],
                                         template_centers=tdata.sda_centers[best_a],
                                         hard_cut=hard_cut, prio=tdata.center_priority, pseudocenters=pseudocenters,
                                         debug=debug) )
    b_out = list(create_compact_indexing(gain_obj, 'b', target_b_centers, threshold=1, outlier_cutoff=outlier_cutoff,
                                         template_matches=b_template_matches,
                                         template_extents=tdata.element_extents[best_b],
                                         template_centers=tdata.sdb_centers[best_b],
                                         hard_cut=hard_cut, prio=tdata.center_priority, pseudocenters=pseudocenters,
                                         debug=debug) )
    highest_split = max([a_out[4], b_out[4]])

    # Patch the GPS into the output of the indexing methods.
    if patch_gps:
        b_gps = tdata.sdb_gps_res[best_b]
        gps_matches = find_center_matches(f'{file_prefix}_sdb.out', b_gps, isTarget=False, debug=debug)
        gps_resids = sorted([v[0] for v in gps_matches.values() if v[0] is not None])
        if debug:
            print(f"[DEBUG] assign_indexing: Attempting Patching GPS\n\t{b_gps = }\n\t{gps_matches = }\n\t{gps_resids = }")
        # Append the GPS into the subdomain B output
        if gps_resids:
            b_out[0]["GPS"] = [gps_resids[0], gps_resids[-1]] # GPS interval
            if "GPS.-1" in gps_matches.keys():
                b_out[1]["GPS"] = gps_matches["GPS.-1"][0] # The residue number of the GPS-1
            else:
                b_out[1]["GPS"] = gps_matches[gps_matches.keys()[-1]][0]

            for label, v in gps_matches.items():
                b_out[2][label] = v[0]
        else:
            print("[NOTE] assign_indexing: No GPS matches have been detected. It will not be patched.")

    params = {"sda_template":best_a, "sdb_template":best_b, "split_mode":highest_split, "receptor":agpcr_type}
    #      elements+intervals          element_centers             residue_labels              unindexed_elements   highest used split mode    
    return { **a_out[0], **b_out[0] }, { **a_out[1], **b_out[1]} , { **a_out[2], **b_out[2] }, a_out[3] + b_out[3], params


def mp_assign_indexing(mp_args:list):
    # wrapper function deciphering the list of args into the function call, for mp we need the args as an iterable.
    #  gain_obj:object, 
    #  file_prefix: str, 
    #  gain_pdb: str, 
    #  template_dir: str, 
    #  gesamt_bin:str, 
    #  template_json="tdata.json", 
    #  outlier_cutoff=10.0,
    #  hard_cut=None, 
    #  debug=False, 
    #  create_pdb=False, 
    #  patch_gps=False
    #   0       1                       2                           3           4       5       6                      7       8        9               10
    #[gain, f"{prefix}_{gain_idx}", find_pdb(gain.name, pdb_dir), template_dir, debug, False, {"S2":7,"S6":3,"H5":3}, True, gain_idx, template_json, gesamt_bin] 
    intervals, indexing_centers, indexing_dir, unindexed, params = assign_indexing(
        gain_obj=mp_args[0],
        file_prefix=mp_args[1],
        gain_pdb=mp_args[2],
        template_dir=mp_args[3],
        gesamt_bin=mp_args[10],
        template_json=mp_args[9],
        debug=mp_args[4],
        create_pdb=mp_args[5],
        hard_cut=mp_args[6],
        patch_gps=mp_args[7]
        )
    return [intervals, indexing_centers, indexing_dir, unindexed, params, mp_args[8]]


def create_compact_indexing(gain_obj, subdomain:str, actual_centers:dict,
                            template_matches=None, template_extents=None, template_centers=None,
                            threshold=3,  outlier_cutoff=10.0, hard_cut=None, prio=None, pseudocenters=None, debug=False):
    ''' 
    Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
    Prints out the final list and writes it to file if outdir is specified

    Parameters
    ----------
    gain_obj: GainDomain object, required
        GainDomain object with corresponding parameters
    subdomain: character, required
        the name of the subdomain "a", "b"
    actual_centers : dict, required
        Dictionary of centers with each corresponding to the matched template center. This should already be the residue of this GAIN domain
    template_matches : dict, optional
        Dictionary of int->int matches of every template residue to the target. Needs to be provided along $template_extents
    template_extent : dict, optional
        Dictionary of the residue extents of the Template elements. These are TEMPLATE residue indices, not the target ones!
    center_dict : dict, required
        Dictionary where each center index is assigned a name (H or S followed by a greek letter for enumeration)
    offset : int,  optional (default = 0)
        An offsset to be incorporated (i.e. for offsetting model PDBs against UniProt entries)
    silent : bool, optional (default = False)
        opt in to run wihtout so much info.
    outdir : str, optional
        Output directory where the output TXT is going to be written as {gain_obj.name}.txt

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
    def split_segments(sse:list, found_centers:list, res2center:dict, coiled_residues:list, outlier_residues:list, hard_cut=None, debug=debug):
        # Split a segment with multiple occurring centers into its respective segments.
        # Use coiled_residues first, then outlier_residues with lower priority
        # Returns a value matching the line of splitting used (0 = best, 3 = worst).
        if debug:
            print(f"[DEBUG] split_segments: called with \n\t{hard_cut = }\n\t{found_centers = }\n\telements: {[res2center[i] for i in found_centers]}")
        n_segments = len(found_centers)
        sorted_centers = sorted(found_centers) # Just in case, order the centers
        n_boundaries = {}
        c_boundaries = {}

        # Start of first segment, end os last segment are pre-defined.
        n_boundaries[sorted_centers[0]] = sse[0]
        c_boundaries[sorted_centers[-1]] = sse[1]

        for i in range(n_segments-1):
            n_center, c_center = sorted_centers[i], sorted_centers[i+1]
            hard_cut_flag = False
            # Find breaker residue in between these two segments:
            coiled_breakers = [r for r in coiled_residues if r > n_center and r < c_center]
            outlier_breakers = [r for r in outlier_residues if r > n_center and r < c_center]

            # By convention, split the along the most N-terminal occurence of a break residue.
            # Here is the logic to detect what is used for breaking up the segment. The order of checks:
            # First line of splitting: Check of coiled or outlier residues present in the segment.
            split_mode = 1
            if len(coiled_breakers) > 0:
                breakers = coiled_breakers
            elif len(outlier_breakers) > 0:
                split_mode = 2
                breakers = outlier_breakers
            else:
                split_mode = 3
                # Second line of splitting: Check if there is a proline or glycine. If so, use this as a breaker.
                segment_sequence = gain_obj.sequence[sse[0]-gain_obj.start:sse[1]-gain_obj.start]
                print(f"[NOTE] split_segments : Last resort with \n\t{segment_sequence = }\n\t{sse = }")
                if "P" in segment_sequence:
                    breakers = [b + sse[0] for b in list(np.where(segment_sequence == 'P')[0])]
                    if debug:
                        print(f"[DEBUG] Proline {breakers = }")
                    print(f"[WARNING] No breakers, but Proline detected in the target segment. Using this for splitting:\n\t{segment_sequence = }\n\t{sse[0]}-{sse[1]} | P@{breakers[0]}")
                elif "G" in segment_sequence:
                    breakers = [b + sse[0] for b in list(np.where(segment_sequence == 'G')[0])]
                    if debug:
                        print(f"[DEBUG] Glycine {breakers = }")
                    print(f"[WARNING] No breakers, but Glycine detected in the target segment. Using this for splitting:\n\t{segment_sequence = }\n\t{sse[0]}-{sse[1]} | G@{breakers[0]}")
                else:
                    split_mode = 4
                    # Third line of splitting: Check if there is a hard cut provided in hard_cut for one of the centers (S7?). If so, break it after n residues,
                    # keep all residues in the segment (the breaker is added to the C-terminal segment)
                    if hard_cut is not None and res2center[n_center] in hard_cut.keys():
                        hard_cut_flag = True
                        breakers = [ n_center+hard_cut[res2center[n_center]] ]
                        print(f"[WARNING] HARD CUT BREAKER @ {breakers[0]} between {res2center[n_center]} | {res2center[c_center]}")
                    else:
                        split_mode = 5
                        # Last line of splitting: prio
                        if debug:
                            print("[DEBUG]: lst line of splitting invoked. Using center priorities to override ambiuguity")
                        if prio is None:
                            print(f"[ERROR]: No breakers and no Proline detected between {n_center = }:{res2center[n_center]} {c_center = }:{res2center[c_center]}\n{segment_sequence = }")
                            raise IndexError(f"No Breaker detected in multi-center ordered segment: \n\t{sse[0]}-{sse[1]}\n\t{gain_obj.name} Check this.")
                        if prio is not None:
                            priorities = [ prio[res2center[a]] for a in found_centers ]
                            best_center = found_centers[np.argmax(priorities)]
                            return {best_center:sse}, split_mode

            c_boundaries[n_center] = breakers[0]-1
            n_boundaries[c_center] = breakers[0]+1
            # Include the hard cut residue in the C-terminal element (decrement n-terminal boundary by 1)
            if hard_cut_flag:
                n_boundaries[c_center] -= 1
        # Merge the segments and return them
        segments = {a:[n_boundaries[a], c_boundaries[a]] for a in sorted_centers}
        return segments, split_mode

    def create_name_list(sse, ref_res, sse_name):
            ''' Creates a indexing list for an identified SSE with reference index as X.50,
                Also returns a tuple containing the arguments, enabling dict casting for parsing '''
            number_range = range(50+sse[0]-ref_res, 51+sse[1]-ref_res)
            name_list = [f"{sse_name}.{num}" for num in number_range]
            return name_list, (sse, ref_res, sse_name)

    def cast(nom_list, indexing_dir, indexing_centers, sse_x, name_list, cast_values):
        # Updates nom_list,  indexing_dir and indexing_centers with sse_x, name_list and cast_values.
        if debug:
            print(f"DEBUG CAST",sse_x, name_list, cast_values)

        nom_list[sse_x[0] : sse_x[1]+1] = name_list
        indexing_dir[cast_values[2]] = cast_values[0] # all to sse 
        indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the center is located

        return nom_list, indexing_dir, indexing_centers

    def truncate_segment(outliers:dict, center_res:int, sse:list, outlier_cutoff:float, debug=False):
        if debug:
            print(f"[DEBUG] INVOKED create_compact_indexing.truncate_segment:\n\t{outliers = }\n\t{center_res = }\n\t{sse = }\n\t{outlier_cutoff = }")
        # Truncate the segment if outliers are present that exceed the float multiple of the outlier_cutoff
        segment_outliers = [k for k,v in outliers.items() if k >= sse[0] and k <= sse[1] and v > outlier_cutoff]
        if segment_outliers:
            # As a fallback, use the sse boundaries as truncation.
            # go C- and N-terminal from the center residue up until the first segment outlier. Truncate there.
            n_terminus = max([x for x in segment_outliers if x < center_res], default=sse[0]-1)
            c_terminus = min([x for x in segment_outliers if x > center_res], default = sse[1]+1)
            if debug and n_terminus != sse[0]-1:
                print(f"[DEBUG] create_compact_indexing.truncate_segment:\n\tTruncated segment N@{n_terminus} (prev {sse[0]})")
            if debug and c_terminus != sse[1]+1:
                print(f"[DEBUG] create_compact_indexing.truncate_segment:\n\tTruncated segment C@{n_terminus} (prev {sse[1]})")
            return [n_terminus+1, c_terminus-1]
        return sse

    if debug:
        print(f"[DEBUG] create_compact_indexing: passed arguments:\n\t{gain_obj.name = }\n\t{subdomain = }\n\t{actual_centers = }")

    ### END OF FUNCTION BLOCK
    # Initialize Dictionaries
    indexing_dir = {}
    indexing_centers = {}
    named_residue_dir = {}
    unindexed = []
    split_mode = 0
    # Catch an empty match; return all empty
    if not actual_centers:
        print("[WARNING] create_compact_indexing. Umatched subdomain. Returning all empty entries.")
        return {}, {}, {}, [], 0

    # Invert the actual centers dict to match the GAIN residue to the named center
    res2center = {v[0]:k for k,v in actual_centers.items()} # v is a tuple  (residue, distance_to_template_center_residue)
    if debug: print(f"[DEBUG] create_compact_indexing: {res2center = }")
    # One-indexed Indexing list for each residue, mapping for the actual residue index
    nom_list = np.full(shape=[gain_obj.end+1], fill_value='      ', dtype='<U7')

    if subdomain.lower() == 'a':
        sses = gain_obj.sda_helices
    elif subdomain.lower() == 'b':
        sses = gain_obj.sdb_sheets

    # Go through each individual SSE in the GAIN SSE dictionary
    for idx, sse in enumerate(sses):

        first_res, last_res = sse[0]+gain_obj.start, sse[1]+gain_obj.start # These are PDB-matching indices

        if debug:print(f"[DEBUG] create_compact_indexing : \nNo. {idx+1}: {sse}\n{first_res = }, {last_res = }")

        # find residues matching centers within the ordered segment
        found_centers = [sse_res for sse_res in range(first_res, last_res+1) if sse_res in res2center.keys()]

        n_found_centers = len(found_centers)

        if n_found_centers == 0:
            # If no center is found, move to $unindexed and check for overlap of unindexed template elements in a second pass.
            # Check of overlapping non-indexed template elements. This should be done AFTER all normal matches are there

            if sse[1]-sse[0] >= threshold-1:
                if debug:
                    print(f"[DEBUG] create_compact_indexing : No center found! \n {[first_res, last_res] = }")
                unindexed.append([first_res, last_res])
            continue

        if n_found_centers == 1:
            center_res = found_centers[0]
            if debug: print(f"SINGLUAR ANCHOR FOUND: @ {center_res = }, {res2center[center_res]}")
            sse_name = res2center[center_res]
            tr_sse = truncate_segment(outliers=gain_obj.outliers,
                                      center_res=center_res,
                                      sse=[first_res, last_res], outlier_cutoff=outlier_cutoff, debug=debug)
            name_list, cast_values = create_name_list(tr_sse, center_res, sse_name)
            # name_list has the assignment for the SSE, cast_values contains the passed values for dict casting
            nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, tr_sse, name_list, cast_values)
            # Also write split stuff to the new dictionary
            for entryidx, entry in enumerate(name_list):
                named_residue_dir[entry] = entryidx + first_res
            continue

        if n_found_centers > 1:
            # When multiple centers are present, split the segment by finding coiled and outlier residues.
            if debug:
                print(f"[DEBUG] create_compact_indexing : ANCHOR AMBIGUITY in this SSE:\n\t{sse = }\n\t{found_centers = }")

            # parse the sse_sequence to find coiled and outlier residues
            coiled_residues = []
            outlier_residues = []
            for i in range(first_res+1, last_res): # The start and end of a segment are never breaks.
                if gain_obj.sse_sequence[i] ==  "C" or gain_obj.sse_sequence[i] == "T":
                    coiled_residues.append(i)
                if gain_obj.sse_sequence[i] in ['h', 'e']:
                    outlier_residues.append(i)

            if debug:
                print(f"[DEBUG] create_compact_indexing :\n\t{coiled_residues  = }\n\t{outlier_residues = }")
            split_segs, split_mode = split_segments(sse=[first_res, last_res],
                                                    found_centers=found_centers,
                                                    res2center=res2center,
                                                    coiled_residues=coiled_residues,
                                                    outlier_residues=outlier_residues,
                                                    hard_cut=hard_cut,
                                                    debug=debug)
            for center_res, segment in split_segs.items():
                tr_segment = truncate_segment(outliers=gain_obj.outliers,
                                      center_res=center_res,
                                      sse=segment, outlier_cutoff=outlier_cutoff, debug=debug)
                name_list, cast_values = create_name_list(tr_segment, center_res, res2center[center_res])
                # cast them also?
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, tr_segment, name_list, cast_values)
                 # Also write split stuff to the new dictionary
                for entryidx, entry in enumerate(name_list):
                    named_residue_dir[entry] = entryidx+tr_segment[0]

    # Second pass: Check for overlapping unindexed template elements:
    if template_extents is not None:
        # Filter only extents of unassigned elements
        unassigned_template_extents = {k:v for k,v in template_extents.items() if f"{k}.50" not in indexing_centers.keys()}
        if debug:
            print(f"[DEBUG] create_compact_indexing :\n\t{unassigned_template_extents = }\n\t{unindexed = }")

        if unassigned_template_extents:

            for sse in unindexed[::-1]:

                hasPseudocenter = False

                for name, extent in unassigned_template_extents.items():

                    if debug:
                        print(f"EXTENT OF UNMATCHED SEGMENT: {extent[0] = }, {extent[1] = }\n{template_matches = }\n{len(template_matches) = }")

                    exmatch = [template_matches[res][0] for res in range(extent[0], extent[1]+1) if template_matches[res][0] is not None]

                    if debug:
                        if exmatch is not None:
                            print(f"{exmatch = }\t")
                        print(f"\t{extent = }\t{sse[0] = }\t{sse[1] = }")

                    if not exmatch:
                        continue

                    # check for overlap in the elements
                    if sse[0] in exmatch or sse[1] in exmatch: # With overlap, either the last or first residue must be in the template element
                        target_center_res = actual_centers[name][0] # This is the target residue corresponding to the center
                        # target_center_res is the template resdiue. We need the fitting target residue

                        if target_center_res is None:
                            hasPseudocenter = True
                            print(f"[WARNING] create_compact_indexing:\n\tANCHOR MATCH NOT FOUND\n\t{actual_centers = }")
                            target_matches = {v[0]:k for k,v in template_matches.items() if v[0] is not None}
                            offset_dict = {target_matches[res]-template_centers[name]:res for res in exmatch}
                            # Check whether N-terminal or C-terminal offsets. Should only be one of them 
                            #   since the center_res is not directly included in exmatch
                            # Find the longest segment for determining the pseudocenter
                            if len([k for k in offset_dict.keys() if k < 0]) > len([k for k in offset_dict.keys() if k > 0]):
                                n_terminal = True
                                offset = max([k for k in offset_dict.keys() if k<0])
                            else:
                                n_terminal = False
                                offset = min([k for k in offset_dict.keys() if k>0])
                            target_center_res = offset_dict[offset]-offset # This is the pseudocenter - Even if it is unmatched, we pretend that it is.
                            if debug:
                                print(f"\tFound Pseudocenter @ {target_center_res}:{name}\n\tPlease validate manually.")
                            # Write to File if specified. This is for Pseudocenter statistics.
                            if pseudocenters is not None:
                                open(pseudocenters, "a").write(f"{gain_obj.name},{target_center_res},{name}\n")
                        if debug:
                            print(f"[DEBUG] create_compact_indexing : OFFSET CASE\n\t{first_res = } {last_res = } \n\t{extent = }")
                            print(f"\t{actual_centers = }\n\t{name = }")
                            print(f"\t{target_center_res = }\t{target_center_res = }")

                        name_list, cast_values = create_name_list(sse, target_center_res, name)

                        # Pseudocenter truncation check. We only want the overlapping part, not something like bulges which have negilgible accuracy.
                        if hasPseudocenter and n_terminal:
                            name_list = [l for l in name_list if int(l.split(".")[-1])<50]
                            cast_values[0][1] = cast_values[0][0]+len(name_list)-1
                        if hasPseudocenter and not n_terminal:
                            name_list = [l for l in name_list if int(l.split(".")[-1])>50]
                            cast_values[0][0] = cast_values[0][1]-len(name_list)+1

                        # Update with new elements
                        nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse, name_list, cast_values)

                        for entryidx, entry in enumerate(name_list):
                            named_residue_dir[entry] = entryidx+sse[0]

                        if debug:
                            print(f"[DEBUG] create_compact_indexing :\n\tFound offset Element {name} @ {target_center_res}\n\tTemplate: {extent}\n\t Target: {sse}\n\t{name_list = }\n\t{cast_values = }")
                        unassigned_template_extents.pop(name)
                        break

    return indexing_dir, indexing_centers, named_residue_dir, unindexed, split_mode

def add_grn_labels(gain_collection, gain_indexing):
    # Modifies a GainCollection to have their GainDomain objects contain GRN information resid --> GRN and vice cersa
    for gain in gain_collection.collection:

        ac = gain.name.split("_")[0].split("-")[0]
        gain_index = gain_indexing.accessions.index(ac)
        named_dir = gain_indexing.indexing_dirs[gain_index]

        gain.grn_labels = named_dir
        gain.reverse_grn_labels = {v: k for k, v in named_dir.items()}

    return gain_collection

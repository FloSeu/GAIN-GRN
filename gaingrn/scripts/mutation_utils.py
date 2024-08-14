# mutation_utils.py
# Functions for Analysis of Cancer mutations and natural variants of the GAIN-GRN dataset.
import re, json
import gaingrn.scripts.io
import gaingrn.scripts.assign

def retrieve_json_vars(name,jsons):
    # A function retrieving variants from a JSON File as exported by TCGA/NCGA, returning them as a dictionary
    # The list of {jsons} is filtered to find {name}
    # Returns a loaded dict from the json.
    try: 
        identifier = name.split("AGR")[1][:2] # When its a normal aGPCR
    except: 
        identifier = "C"+name.split("CELR")[1][0]

    json_identifier = identifier.lower()
    json_file = [j for j in jsons if json_identifier in j.lower()]
    #print(f"DEBUG : retrieve_mutation_json: {json_file = }" )
    if len(json_file) > 1 : print("WARNING: Multiple matching JSON Files detected:", json_file)
    with open(json_file[0]) as j_open:
        j_data = j_open.read()
    mutation_data = json.loads(j_data) # Will contain a list of dicts
    return mutation_data

def retrieve_csv_vars(name, csv_files, filter_str=None, with_resid=True):
    # A function retrieving variants from a CSV File as exported by gnomAD, returning them as a dictionary
    # The list of {csv_files} is filtered to find {name}, filter_str is used to find specific mutations, i.e. missense
    # With_resid also adds the residue and the standard AA at this position to the mutation entries.
    # Returns a list of dict, where each dict corresponds to one mutation
    try: 
        identifier = name.split("AGR")[1][:2] # When its a normal aGPCR
    except: 
        identifier = "C"+name.split("CELR")[1][0]

    csv_identifier = identifier.lower()
    csv_file = [c for c in csv_files if csv_identifier in c.lower()]
    print(f"DEBUG : parse_snp_csv: {csv_file = }" )
    
    # the target CSV files have 53 items per row. RETURNS columns: dict where key points to axis 1 index of np.array; val_arr: value array 
    with open(csv_file[0]) as cc:
        data = cc.readlines()

    # Pre-filter the data if specified.
    if filter_str is not None:
        newdata = [l for l in data[1:] if filter_str in l]
        data = [data[0]]
        for l in newdata: 
            data.append(l)

    oneletter = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H',
                 'Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Pyl':'O','Ser':'S','Thr':'T',
                'Trp':'W','Tyr':'Y','Val':'V'}

    variant_list = []
    variant_keys = data[0].strip().split(",")

    for i,row in enumerate(data[1:]):
        row = re.sub(r'\".*\"', 'REMOVED', row)   # Removes combined irrelevant entries that contain a "," within string denotation. This confuses the split routine.
        var_dict = {variant_keys[i]:val for i,val in enumerate(row.strip().split(","))}
        #print("DEBUG", var_dict)
        if with_resid:
            var_string = var_dict['Protein Consequence']
            #print(f"DEBUG: {var_string = }")
            if len(var_string) < 2:
                var_dict['resid'] = None
                var_dict['resname'] = None
            else:
                var_dict['resname'] = oneletter[re.search(r'[A-z]{3}', var_string).group(0)]
                var_dict['resid'] = int(re.search(r'[\d]+', var_string).group(0))
        variant_list.append(var_dict)

    return variant_list

def extract_variants(gain, files, resid_key, csvs):
    # Wrapper function for getting variants / mutations in the correct format.
    # Draws the list of vars/muts from the respective list of jsons/csvs abd gets the individual positions either at entry 
    # resid_key : 'x' (json from NCGA) / 'resid' (csv from gnomAD)

    if files[0][-3:].lower() == 'csv':
        print("List of CSV files detected.")
        vars = retrieve_csv_vars(gain.name, csvs, filter_str='missense_variant',with_resid=True)
    else:
        print("This should be a list of JSON Files.")
        v_dict = retrieve_json_vars(gain.name, files,)
        vars = [mut for mut in v_dict['mutations']]
        # returns a dictionary with a list of dicts, each dict corresponding to a var.
    variant_dict = {}
    positions = []
    for var in vars:
        # The "x" key is the identifying position!
        resid = int(var[resid_key]) 
        #
        # Here, one can define criteria for skipping said var. For now, use all.
        #
        # Add the receptor name to the var dictionary
        var["receptor"] = gain.name

        if resid in variant_dict.keys():
            variant_dict[resid].append(var)
        else:
            variant_dict[resid] = [var]
        positions.append(resid)
    return positions, variant_dict

def get_loop_stats(indexing_dir, sequence):
    # Returns a named dict with loop lengths, i.e. {"H1-H2":13, "H8-S1":12}
    inverted_dir = {sse[0] : (sse[1],ki) for ki, sse in indexing_dir.items() if "GPS" not in ki} # The begin of each sse is here {0:(13, "H2")}
    loop_loc = {}
    loop_dir = {}
    ordered_starts = sorted(inverted_dir.keys())
    for i, sse_start in enumerate(ordered_starts):
        if i == 0: 
            continue # Skip the first and go from the second SSE onwards, looking in N-terminal direction.
        c_label = inverted_dir[sse_start][1]
        n_end, n_label = inverted_dir[ordered_starts[i-1]]
        loop_loc[f"{n_label}-{c_label}"] = (n_end, sse_start-1)
        loop_dir[f"{n_label}-{c_label}"] = sequence[n_end+1:sse_start] # The one-letter-coded seqeuence. Will be a list of lists
    return loop_loc, loop_dir

def compose_vars(aGainCollection, files, resid_key, aa_key, fasta_offsets, merge_unique=False):
    # Take the Gain Collection and retrieve variant/mutation info from the files list via extract_variants()
    # Collect these for each GAIN in the aGainCollection.collection and compose them together to enable addressing each individual labeled position.
    # Returns the generalized (aligned) variants/mutations and their respective counts
    mismatch_flag = False
    generalized_vars = {}
    valid = 0
    invalid = 0

    for gain_ndx, gain in enumerate(aGainCollection.collection):
        gain_valid = 0
        gain_invalid = 0
        fasta_offset = fasta_offsets[gain_ndx]
        # Retrieve vars/mutations for respective receptor
        positions, gain_pos_dict = extract_variants(gain, files, resid_key)
        # Evaluate the vars/mutations. If the residue in question (corrected index) is named (i.e. "H6.50"), get its resepective label.
        for p_resid in gain_pos_dict.keys():
            # Since fasta_offset maps to the first gain residue (i.e. 2027 with gain.start being 459), an offset needs to be set by the difference
            corrected_resid = p_resid - fasta_offset + gain.start #-1 # [ISSUE] ROLE OF THIS OFFSET???
            #print(f"DEBUG {corrected_resid = }, {p_resid = },\n",
            #   f"TRUE GAIN INTERVAL {fasta_offset+gain.start} - {fasta_offset+gain.end}\nCORRECTED GAIN INTERVAL {gain.start}-{gain.end}")
            # map it to its corresponding label. If there is none, continue
            if corrected_resid < gain.start or corrected_resid >= gain.end:
                gain_invalid += 1
                continue
            try:
                p_label = gain.rev_idx_dir[corrected_resid] # This is the indexing label: i.e. 527 --> "H3.55"
                gain_valid +=1
            except KeyError:
                gain_invalid += 1
                continue
            # SANITY CHECK:
            if gain_pos_dict[p_resid][0][aa_key][0] != gain.sequence[corrected_resid-gain.start-1]: #-1 here since gain.sequence[0] is residue 1 in fasta terms
                print(f"{gain_pos_dict[p_resid][0][aa_key][0] = }", {gain.sequence[corrected_resid-gain.start]})
                print(f"MISMATCH! {p_resid} -> {corrected_resid}")
                print(f"{gain_pos_dict[p_resid][0][aa_key][0]} : {gain.sequence[corrected_resid-gain.start]} in GAIN",
                    f"{gain.sequence[corrected_resid-gain.start-2:corrected_resid-gain.start+3]}")
                mismatch_flag = True
            #else:
                #print(f"MATCH {p_resid} -> {corrected_resid} | {gain_mutation_dict[p_resid][0][aa_key][0]} : {gain.sequence[corrected_resid-gain.start]}")
            
            if merge_unique:
                # If the p_label is unique, extract the unique entry and map onto the common GRN
                p_label_parts = p_label.split(".")
                if len(p_label_parts) == 3:
                    print("NOTE: Corrected p_label:", p_label)
                    p_label = p_label_parts[0]+"."+p_label_parts[2]

            # with the Label, map into generalized dictionary
            if p_label not in generalized_vars.keys():
                generalized_vars[p_label] = gain_pos_dict[p_resid]
            else:
                [generalized_vars[p_label].append(pos) for pos in gain_pos_dict[p_resid]]
        valid += gain_valid
        invalid += gain_invalid
        print("VALID MAPPED POSITIONS: ", gain_valid, "\nINVALID POSITIONS     : ", gain_invalid, "\nTOTAL POSITIONS       : ", len(gain_pos_dict.keys()))
    if mismatch_flag: 
        print("[WARNING]: MISMATCHES HAVE BEEN FOUND. PLEASE CHECK THE OUTPUT.")
    else: 
        print("[NOTE]: NO MISMATCHES HAVE BEEN FOUND.")
    print(f"TOTAL VARS WITHIN GAIN:", valid, "\nTOTAL VARS OUTSIDE GAIN:", invalid, "\nTOTAL VARS:", valid+invalid)
    
    generalized_counts = {k:len(v) for k,v in generalized_vars.items()}

    return generalized_vars, generalized_counts

def loop2fasta(outfile, itemlist):
    # Write the collected loop sequences to a FASTA file for later alignment.
    with open(outfile, 'w') as out:
        for subdict in itemlist:
            out.write(f">{subdict['name']}\n{subdict['sequence']}\n")
    print("Done with", outfile)

def compose_loop_vars(aGainCollection, files, resid_key, aa_key, fasta_offsets):
    # Take the Gain Collection and retrieve variant/mutation info from the files list via extract_variants()
    # Collect these for each GAIN in the aGainCollection.collection and compose them together to enable addressing each individual labeled position.
    # Returns the generalized (aligned) variants/mutations and their respective counts
    mismatch_flag = False
    generalized_vars = {}
    valid = 0
    invalid = 0

    loop_valid = 0
    loop_invalid = 0
    for idx, gain in enumerate(aGainCollection.collection):
    # Retrieve vars/mutations for respective receptor
        positions, gain_pos_dict = extract_variants(gain, files, resid_key)
        fasta_offset = fasta_offsets[idx]
        # Find all vars/mutations whose RESID (fasta) matches the corrected GAIN domain INTERVAL(for fasta resids)
        within = [pos for pos in positions if pos in range(fasta_offset, fasta_offset + 1 + gain.end - gain.start)]
        #print("DEBUG: vars/mutations within GAIN space\n", sorted(within))
        idx_dir, _, _, _, _, = gaingrn.scripts.assign.assign_indexing(gain_obj=gain,
                                            file_prefix=f'../indexing_tmp/x{idx}',
                                            gain_pdb=gaingrn.scripts.io.find_pdb(gain.name, '../all_pdbs/'),
                                            template_dir='../r4_template_pdbs',
                                            template_json='template_data.json',
                                            gesamt_bin="/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt",
                                            hard_cut={"S2":7,"S6":3,"H5":3},
                                            debug=False,
                                            patch_gps=True)
        # i.e. 'H2-H3': (13, 20), 'H3-H4': (36, 42), 'H4-H5': (61, 75), 'H5-H6': (87, 89), ...., 'S11-S12': (316, 319), 'S12-S13': (327, 329)}
        i_loc, _ = get_loop_stats(idx_dir, gain.sequence)
        # Modify this to a position --> loop dict
        loop_locations = {}
        for loop, interval in i_loc.items():
            for i in range(interval[0], interval[1]+1):
                loop_locations[i+gain.start] = loop
        # Evaluate the vars/mutations. If the residue in question (corrected index) is named (i.e. "H6.50"), get its resepective label.
        print("DEBUG")
        #[print(i,k) for i,k in loop_locations.items()]

        for p_resid in gain_pos_dict.keys():
            # Since fasta_offset maps to the first gain residue (i.e. 2027 with gain.start being 459), an offset needs to be set by the difference
            corrected_resid = p_resid - fasta_offset + gain.start -1
            # Check in which loop this is located and collect it in a dictionary.
            try:
                loop_label = loop_locations[corrected_resid]
                loop_valid +=1
            except KeyError:
                loop_invalid += 1
                continue
            # SANITY CHECK:
            if gain_pos_dict[p_resid][0][aa_key][0] != gain.sequence[corrected_resid-gain.start]:
                print(f"{gain_pos_dict[p_resid][0][aa_key][0] = }", {gain.sequence[corrected_resid-gain.start]})
                print(f"MISMATCH! {p_resid} -> {corrected_resid}")
                print(f"{gain_pos_dict[p_resid][0][aa_key][0]} : {gain.sequence[corrected_resid-gain.start]} in GAIN",
                    f"{gain.sequence[corrected_resid-gain.start-2:corrected_resid-gain.start+3]}")
                mismatch_flag = True
            # with the Label, map into generalized dictionary
            if loop_label not in generalized_vars.keys():
                generalized_vars[loop_label] = gain_pos_dict[p_resid]
            else:
                [generalized_vars[loop_label].append(pos) for pos in gain_pos_dict[p_resid]]
        valid += loop_valid
        invalid += loop_invalid
        print("VALID LOOP POSITIONS: ", loop_valid, "\nINVALID POSITIONS     : ", loop_invalid, "\nTOTAL POSITIONS       : ", len(gain_pos_dict.keys()))
    if mismatch_flag: 
        print("[WARNING]: MISMATCHES HAVE BEEN FOUND. PLEASE CHECK THE OUTPUT.")
    else: 
        print("[NOTE]: NO MISMATCHES HAVE BEEN FOUND.")
    print(f"TOTAL VARS WITHIN GAIN:", valid, "\nTOTAL VARS OUTSIDE GAIN:", invalid, "\nTOTAL VARS:", valid+invalid)
    
    generalized_counts = {k:len(v) for k,v in generalized_vars.items()}

    return generalized_vars, generalized_counts

def json2text(j:str, mutation_data:dict):
    print(j.split("/")[-1].split(".")[0].upper(), "CANCER GENOME ATLAS MUTATIONS\n","_"*70)
    positions = []
    for mutation in mutation_data['mutations']:
        for key in mutation.keys():
            print(key.ljust(30), mutation[key])
        print("_"*70)
        resid = int(mutation["x"]) # X is the residue ID
        positions.append(resid)
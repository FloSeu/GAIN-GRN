# Functions for finding a template via GainDomain subselection and GESAMT pairwise structural alignments.
import glob, subprocess, os, re, shutil, math
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
import json
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

def run_command(cmd, logfile=None, outfile=None):
    # The command would be the whole command line, in this case the gesamt command.
    # also : shlex.split("cmd")
    p = subprocess.Popen(cmd, shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            bufsize=10, 
            universal_newlines=True)
    #exit_code = p.poll()
    outs, errs = p.communicate()
    if outfile is not None:
        f_out = open(outfile, "w")
        f_out.write(outs)
        f_out.close()
    if logfile is not None:
        f_err = open(logfile, "a")
        f_err.write(f"\nCalled Function run_command with: {cmd}\nOutput to: {logfile}")
        f_err.write(errs)
        f_err.close()
    if outfile is None:
        return outs
    return

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

def run_gesamt_execution(list_of_gain_obj, outfolder, pdb_folder='../all_pdbs', domain='sda', n_threads=6, max_struc=400, no_run=False, template=None):
    # Will compare all PDBS of a given list of GainDomain objects. It will compare all vs. all PDB files in the selection in the PDB folder
    # It will further create the given outfolder, if not existing, write a BASH-File with the commands for GESAMT and run if not otherwise specified.
    
    # CHECKS
    if domain not in ['sda', 'sdb']:
        print("No domain specified. Exiting"); return None
    
    if template is not None and not os.path.isfile(template):
        print("Template specified but file {template} not found. Aborting run."); return None

    n_gains = len(list_of_gain_obj)
    if n_gains > max_struc:
        list_of_gain_obj = list_of_gain_obj[:max_struc]; n_gains=max_struc; print(f"Too many structures in selection. Truncating after {max_struc} GAIN domains.")

    allpdbs = glob.glob(f'{pdb_folder}/*.pdb')

    extent_list = []
    selection_pdbs = []

    bashfile = f'{outfolder}/run_gesamt.sh'  #  Output File that also will be run as command.
    if not os.path.isdir(outfolder):
        print(f"Created {outfolder}")
        os.mkdir(outfolder)

    # Grab all PDB files corresponding to the GAIN domains in selection.
    for i, gain in enumerate(list_of_gain_obj):
        identifier = gain.name.split("-")[0]
        gain_pdb = [x for x in allpdbs if identifier in x][0]
        selection_pdbs.append(gain_pdb) # the PDB name associated with the gain domain

        # Getting and writing the EXTENT of the subdomain!
        pdb_start, sda_boundary, sdb_boundary, pdb_end = get_pdb_extents(gain_pdb, gain.subdomain_boundary)
        #start, end  = gain.start, gain.end
        #if gain.start < pdb_start:
        #    start = pdb_start
        #if gain.end > pdb_end:
        #    end = pdb_end
        #exlist = [start, gain.subdomain_boundary, end]
        if domain=='sda':
            extent_list.append(f"{pdb_start}-{sda_boundary}")
        else:
            extent_list.append(f"{sdb_boundary}-{pdb_end}")
        # --> 0,1 for subdomain A - 1,2 for subdomain B; this is for specifying the alignment interval in GESAMT

    # Write blocks of $n_threads commands that constitute a block. We will run GESAMT of all gains against all gains with n² runs.
    myct=0 # multithreading counter
    cmd_string = "#!/bin/bash\n"
    #for i, prot1 in enumerate(struc_list):
    for i in range(n_gains):
        p1name = selection_pdbs[i]
        if template is None:
            for j in range(n_gains):
                p2name = selection_pdbs[j]
                myct +=1
                run_id = f"{i}_{j}"
                cmd_string += f'/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt {p1name} -s A/{extent_list[i]} {p2name} -s A/{extent_list[j]} > {outfolder}/{run_id}.out &\n'
                if myct % n_threads == 0:
                    cmd_string +="wait\n" # Delimiter for multiprocessing
            
        else: # No nested double for loop upon template specification
            myct +=1
            run_id = i
            cmd_string += f'/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt {template} {p1name} -s A/{extent_list[i]}  > {outfolder}/{run_id}.out &\n'
            if myct % n_threads == 0:
                cmd_string +="wait\n" # Delimiter for multiprocessing
    
    # Write the composed commands to bashfile and run
    with open(bashfile,'w') as sh: # 
        sh.write(cmd_string)
    print(f"Written a total of {myct} GESAMT commands to file.")
    if not no_run:
        print(f"Running set of GESAMT comparisons with {n_threads} threads.")
        run_command(f"bash {bashfile}", f"{outfolder}/gesamt.log", f"{outfolder}/gesamt.err")
    print("done.")

def evaluate_gesamt_files(gesamt_folder, n_prot, penalty_value = 6.0, remove=False):
    # Build a RMSD matrix of all target proteins aligned via GESAMT in $gesamt_folder to find the centroid via sum of RMSD
    rmsd_arr = np.full(shape=(n_prot,n_prot), fill_value=penalty_value)
    outfiles = glob.glob(f'{gesamt_folder}/*.out')
    unmatched = 0
    print(f"Found {len(outfiles)} GESAMT files in target folder {gesamt_folder}")
    for f in outfiles:
        with open(f) as ff:
            data = ff.read()
        match = re.search(r"RMSD\W+\:\W+[0-9]+\.[0-9]+",data)
        if match is None:
            print("[WARNING]: NO RMSD FOUND:", f, data)
            unmatched += 1
            continue
        val = float(match.group(0)[-5:])
        indices = [int(x) for x in f.split("/")[-1][:-4].split("_")]

        rmsd_arr[indices[0], indices[1]] = val
        #rmsd_arr[indices[1], indices[0]] = val # Not necessary since we run double (sadly :( )
    for k in range(n_prot):
        rmsd_arr[k,k] = 0
    if remove:
        confirm = input("Confirm deletion of directory: {gesamt_folder} with {unmatched} unmatched entries? > ")
        if confirm.upper() == "Y":
            shutil.rmtree(gesamt_folder, ignore_errors=False, onerror=None)
    return rmsd_arr

def plot_heirarchy(distances, groupname='', savename=None):
    
    #assert np.all(distances - distances.T < 1e-6)
    reduced_distances = squareform(distances, checks=False)
    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
    fig = plt.figure(figsize=[18,4], facecolor='w')
    plt.title(f'RMSD Average linkage hierarchical clustering: {groupname}')
    _ = scipy.cluster.hierarchy.dendrogram(linkage, count_sort='descendent', show_leaf_counts=True, leaf_font_size=3)
    if savename is not None:
        plt.savefig(f"{savename}.png", dpi=200)

def cluster_agglomerative(distances, list_of_gain_obj, n_cluster=9):
    clustering = AgglomerativeClustering(n_clusters=n_cluster, metric='precomputed', 
                            memory=None, connectivity=None, compute_full_tree='auto', linkage='complete', distance_threshold=None, compute_distances=False).fit(distances)
    n_struc = distances.shape[0]
    #print(clustering.labels_)
    new_order = np.zeros(shape=(n_struc), dtype=int)
    current_num = 0
    for i in range(n_cluster):
        for j, cluster_id in enumerate(clustering.labels_):
            if cluster_id == i :
                new_order[j] = current_num
                current_num += 1

    remap_dict = {old_idx:new_idx for old_idx, new_idx in enumerate(new_order)}
    inv_remap_dict = {v:k for k,v in remap_dict.items()}
    ordered_distances = np.zeros(shape=(n_struc,n_struc))
    for x in range(n_struc):
        new_x = remap_dict[x]
        for y in range(n_struc):
            new_y = remap_dict[y]
            ordered_distances[new_x,new_y] = distances[x,y]

    reordered_gain_list = [list_of_gain_obj[inv_remap_dict[k]] for k in range(n_struc)]
    reordered_clusters = [clustering.labels_[inv_remap_dict[k]] for k in range(n_struc)]

    _, cluster_starts = np.unique(reordered_clusters, return_index = True)
    cluster_intervals = [(cluster_starts[k], cluster_starts[k+1]) for k in range(n_cluster-1)]
    cluster_intervals.append((cluster_starts[-1], n_struc))
    print("Cluster Intervals of found and sorted clusters:\n", cluster_intervals)

    cl_best = []
    cl_size = []
    for cl_idx, cluster in enumerate(cluster_intervals):
        best_struc = cluster[0]+np.argmin(np.sum(ordered_distances[ cluster[0]:cluster[1] , cluster[0]:cluster[1] ], axis=0))
        cl_best.append((best_struc, cl_idx))
        cl_size.append(cluster[1]-cluster[0])
        print(f"Current cluster: {cl_idx} with size {cluster[1]-cluster[0]}:\n\tCentroid: {best_struc}\n\tName:    {reordered_gain_list[best_struc].name}\n")

    overall_best = np.argmin(np.sum(ordered_distances[ : , : ], axis=0))

    print(f"All structures:\n\tCentroid: {overall_best}\n\tName:    {reordered_gain_list[overall_best]}\n")

    return {"overall_best_gain":overall_best, 
            "cluster_best_gains":cl_best, 
            "reordered_gain_list":reordered_gain_list, 
            "reordered_clusters":reordered_clusters, 
            "reordered_distances":ordered_distances,
            "cluster_sizes": cl_size}

def plot_matrix(distances, title='', savename=None):
    fig = plt.figure(figsize=[6,4])
    fig.set_facecolor('w')
    plt.imshow(distances, cmap='Greys')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('RMSD [$\AA$]')
    if savename is not None:
        plt.savefig(f'{savename}.png',dpi=300)

def save2json(distances, names, savename):
    df = pd.DataFrame(distances, index = list(names), columns = list(names))
    df = df.to_json(orient='index')
    with open(f"../{savename}.json",'w') as c:
        c.write(json.dumps(df))

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
            if element[1]-element[0] < threshold: 
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

def match_gain2subdomain_template(gain_idx, template_anchors, gesamt_folder, penalty=None, debug=False):
    '''
    Match a GAIN domain to a template PDB. This function has two modes. 
    We match the closest residue to the template as the new anchor and output the residues and their distances.

    gain_idx indicates the index of the domain used for locating GESAMT alignment files.
    template PDBs are global, but not needed since we are handling only the aligned files / PDBs.
    '''
    n_anch = len(template_anchors.keys())

    target_gesamt_file = f'{gesamt_folder}/{gain_idx}.out'
    actual_anchors = find_anchor_matches(target_gesamt_file, template_anchors, debug=debug)
    if debug: print(f"[DEBUG]: return find_anchor_matches: {actual_anchors = }")
    all_dist = {k:v[1] for k,v in actual_anchors.items() } #{'H1': (653, 1.04) , ...}
    if debug: print(f"[DEBUG]: return find_anchor_matches: {all_dist = }") #{'S1': 4.56, 'S2': 2.09, ...}
    # From the anchor-residue distances
    # Fill a matrix with the individual distances, assign unmatched anchors a pre-set penalty value
    distances = np.full(shape=(n_anch), fill_value=penalty)
    for i, sse in enumerate(template_anchors.keys()):
        if debug: print(f"[DEBUG]: match_gain2subdomain_template: {i = }, {sse = }")
        if sse in all_dist.keys():
            if all_dist[sse] is not None:
                distances[i] = all_dist[sse]
        # Cap the distance maximum to the penalty, if specified
        for i,val in enumerate(distances):
            if penalty is not None and val > penalty: 
                distances[i] = penalty
    if debug: print(f"[DEBUG]: return find_anchor_matches: Final distances to return : \n\t{distances = }")
    return distances, actual_anchors

def find_anchor_matches(file, anchor_dict,  isTarget=False, return_unmatched=False, debug=False):
    # Takes a gesamt file and an anchor dictionary either of the target or the template and returns the matched other residue with the pairwise distance
    # as a dictionary: {'H1': (653, 1.04) , ...}
    template_pairs, mobile_pairs = read_gesamt_pairs(file, return_unmatched=return_unmatched)
    # Find the closest residue to template anchors
    matched_residues = {}
    if not isTarget:
        parsing_dict = template_pairs
    else:
        parsing_dict = mobile_pairs

    if debug: print(f"[DEBUG]: find_anchor_matches: {file = }, {parsing_dict = }")
    try:
        start, end = min(parsing_dict.keys()), max(parsing_dict.keys())
    except:
        print("NOT MATCHED:", parsing_dict, file)
    #start, end = min(parsing_dict.keys()), max(parsing_dict.keys())
    for anchor_name, anchor_res in anchor_dict.items():
        # If the anchor lies outside the aligned segments, pass empty match (None, None)
        if anchor_res < start or anchor_res > end:
            matched_residues[anchor_name] = (None, None)
            continue
        matched_residues[anchor_name] = parsing_dict[anchor_res]

    return matched_residues

def read_gesamt_pairs(gesamtfile, return_unmatched=True):
    with open(gesamtfile) as f:
        pair_lines = [line for line in f.readlines() if line.startswith("|")][2:]
        #|H- A:LEU 720 | <**1.21**> |H- A:LEU 633 |
        #| - A:ALA 721 | <..1.54..> | + A:GLN 634 |
        #| + A:GLU 722 | <..1.75..> | + A:PRO 635 |
        #| + A:GLU 723 |            | + A:GLN 636 |
        #| + A:ASN 724 |            | - A:ALA 637 |
        #|H+ A:ARG 725 | <..2.08..> |H- A:LEU 638 |
    # construct a data structure with indices of both sides (fixed, mobile)
    mobile_pairs = {}
    template_pairs = {}
    for pair in pair_lines:
        template_str, distance_str, mobile_str = pair[9:13].strip(), pair[19:23].strip(), pair[36:40].strip()
        #print(template_str, distance_str, mobile_str, "\n", pair)
        # If either residue is empty, let the pair point to None
        if len(template_str) == 0:
            mobile_pairs[int(mobile_str)] = (None, None)
            continue
        if len(mobile_str) == 0:
            template_pairs[int(template_str)] = (None, None)
            continue
        if len(distance_str) == 0:
            dist = None
        else:
            dist = float(distance_str)

        if return_unmatched:
            template_pairs[int(template_str)] = (int(mobile_str), dist)
            mobile_pairs[int(mobile_str)] = (int(template_str), dist)
        if not return_unmatched and dist is None:
            template_pairs[int(template_str)] = (None, None)
            mobile_pairs[int(mobile_str)] = (None, None)
        if not return_unmatched and dist is not None:
            template_pairs[int(template_str)] = (int(mobile_str), dist)
            mobile_pairs[int(mobile_str)] = (int(template_str), dist)

    return template_pairs, mobile_pairs

def create_subdomain_indexing(gain_obj, subdomain, actual_anchors, threshold=3, padding=1, split_mode='single', silent=False,  debug=False):
    ''' 
    Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
    Prints out the final list and writes it to file if outdir is specified
        
    Parameters
    ----------
    gain_obj: GainDomain object, requred
        GainDomain object with corresponding parameters
    actual_anchors : dict, required
        Dictionary of anchors with each corresponding to the matched template anchor. This should already be the residue of this GAIN domain
    anchor_dict : dict, required
        Dictionary where each anchor index is assigned a name (H or S followed by a greek letter for enumeration)
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
            raise IndexError(f"No Break residues in between to Anchors:\n\t{stored_res = }\n\t{new_res = }\n\t{sse = }\n\t{coiled_res = }\n\t{outlier_res = }")

        # Check if there is a break residue in between the two conflicting anchors
        adjusted_break_residues = [res+sse[0] for res in break_residues]

        seg_stored = terminate(sse, stored_res, adjusted_break_residues)
        seg_new = terminate(sse, new_res, adjusted_break_residues)

        if not silent: print(f"[NOTE] disambiguate_segments: Split the segment into: {seg_stored = }, {seg_new = }")

        stored_name_list, stored_cast_values = create_name_list(seg_stored, stored_res, res2anchor[stored_res])
        new_name_list, new_cast_values = create_name_list(seg_new, new_res, res2anchor[new_res])

        if not silent: print(f"[NOTE] disambiguate_segments: Successful SSE split via BREAK.")
        return [(seg_stored, stored_name_list, stored_cast_values),(seg_new, new_name_list, new_cast_values)]

    def cast(nom_list, indexing_dir, indexing_centers, sse_x, name_list, cast_values):
        # Inserts the names of an SSE into the nom_list list of strings, returns the modified version
        if debug: 
            print(f"DEBUG CAST",sse_x, name_list, cast_values)

        nom_list[sse_x[0] : sse_x[1]+1] = name_list
        indexing_dir[cast_values[2]] = cast_values[0] # all to sse 
        indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the anchor is located

        return nom_list, indexing_dir, indexing_centers

    if debug: print(f"[DEBUG] create_subdomain_indexing: passed arguments:\n\t{gain_obj.name = }\n\t{subdomain = }\n\t{actual_anchors = }")
    ### END OF FUNCTION BLOCK
    # Initialize Dictionaries
    indexing_dir = {}
    indexing_centers = {}
    named_residue_dir = {}
    unindexed = []

    # Invert the actual anchors dict to match the GAIN residue to the named anchor
    res2anchor = {v[0]:k for k,v in actual_anchors.items()} # v is a tuple  (residue, distance_to_template_center_residue)
    if debug: print(f"[DEBUG] create_subdomain_indexing: {res2anchor = }")
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
            # Find the anchor within, since the anchors here are already aligned!
                    
            if sse_res in res2anchor.keys():

                if exact_match == False:  
                    if debug: print(f"ANCHOR FOUND: @ {sse_res = }, {res2anchor[sse_res]}")
                    sse_name = res2anchor[sse_res]
                    #stored_anchor = sse_name
                    stored_res = sse_res
                    name_list, cast_values = create_name_list([first_res, last_res], sse_res, sse_name)
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

                coiled_residues = []
                outlier_residues = []
                for i in range(sse[0]+gain_obj.start, sse[1]+1+gain_obj.start):
                    if gain_obj.sse_sequence[i] ==  "C":
                        coiled_residues.append(i-gain_obj.start)
                    if gain_obj.sse_sequence[i] == 'h':
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

        # if no exact match is found, the anchor does not exist in this domain
        if exact_match == False:
            if debug: print("Anchors not found in this sse", idx, sse)
            # expand anchor detection to +1 and -1 of SSE interval
            for res in res2anchor:
                if res == sse[0]-padding or res == sse[1]+padding:
                    fuzzy_match = True
                    if not silent: print(f"[DEBUG] GainDomain.create_indexing : Interval search found anchor @ Residue {res}")
                    # Find the closest residue to the anchor column index. N-terminal wins if two residues tie.
                    name_list, cast_values = create_name_list([first_res, last_res], res, res2anchor[res])

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
                if sse[1]-sse[0] > threshold:
                    if debug: print(f"[DEBUG] GainDomain.create_indexing : No anchor found! \n {first_res = } \ns{last_res = }")
                    unindexed.append(first_res)
    # Patch the GPS into the nom_list
    """        labels = ["GPS-2","GPS-1","GPS+1"]
    for i, residue in enumerate(gain_obj.GPS.residue_numbers[:3]):
        #print(residue)
        nom_list[residue] = labels[i]
        indexing_dir["GPS"] = gain_obj.GPS.residue_numbers
        # Also cast this to the general indexing dictionary
        named_residue_dir[labels[i]] = gain_obj.GPS.residue_numbers[i]"""
           
    return indexing_dir, indexing_centers, named_residue_dir, unindexed

def construct_structural_alignment(template_gain_domain, list_of_gain_obj, gain_indices, gesamt_folder=str, outfile=None, debug=False):
    # This will construct an UNGAPPED MSA based on the pairwise match to the template residues.
    msa_array = np.full(shape=(len(list_of_gain_obj)+1, len(template_gain_domain.sequence)), fill_value='-', dtype = '<U2')
    msa_array[0,:] = template_gain_domain.sequence
    all_names = [template_gain_domain.name]
    #for gesamtfile in glob.glob(f"{gesamt_folder}/*.out"):
    for gain_idx, gain in enumerate(list_of_gain_obj):
        if gain.name == template_gain_domain.name: # QC: ensure the template is not aligned to itgain_obj.
            continue
        all_names.append(gain.name)
        if debug: 
            print(gain.name)
            with open(f'{gesamt_folder}/{gain_indices[gain_idx]}.out') as out:
                data = out.readlines()[42:46]
                print(data)
            
        template_pairs, _ = read_gesamt_pairs(f'{gesamt_folder}/{gain_indices[gain_idx]}.out')
        # This is a dict: { 516: (837, 1.08),... }
        if debug:
            print(f"[DEBUG] {template_pairs = } {len(gain.sequence)} {gain.start = } {gain.end = }")
        res_matches = {k-template_gain_domain.start : gain.sequence[v[0]-gain.start] for k,v in template_pairs.items() if v[0] is not None and v[0]-gain.start < len(gain.sequence) and v[0]-gain.start >=0}
        # This is a dict matching the aln columns : {0: "V", 1, "A", 4:"C"}, non-matched residues are not present (None value)
        # map to alignment array
        for resid, res in res_matches.items():
            msa_array[gain_idx+1,resid] = res

    # If outfile is specified, write as FASTA formatted MSA.
    if outfile is not None:
        if os.path.isfile(outfile):
            print(f"[NOTE]: {outfile} overwritten.")
        with open(outfile, 'w') as alnfile:
            for i, name in enumerate(all_names):
                seq = msa_array[i,:]#[x.decode('ascii') for x in msa_array[i,:]]
                alnfile.write(f'>{name}\n{"".join(seq)}\n')

    # Make a dictionary corresponding to already implemented alignment format as dictionary.
    msa_dict = {name:msa_array[i,:] for i, name in enumerate(all_names)}
    return msa_dict

def plot_pca(distance_matrix, cluster_intervals, n_components, name, plot3D=False, save=True):
    colorlist = ['blue','red','green','yellow','orange','purple','forestgreen','limegreen','firebrick']
    #X = anchor_distance_matrix
    X = distance_matrix
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    X_r = pca.fit(X).transform(X)
    print(X_r.shape)

    fig = plt.figure(figsize=[5,5])
    fig.set_facecolor('w')
    if plot3D:
        ax = ax = fig.add_subplot(projection='3d')
        for i, interval in enumerate(cluster_intervals):
            ax.scatter(X_r[interval[0]:interval[1],0], X_r[interval[0]:interval[1],1], X_r[interval[0]:interval[1],2], marker='o', s=8, c=colorlist[i])
    else:
        ax = fig.add_subplot()
        for i, interval in enumerate(cluster_intervals):
            ax.scatter(X_r[interval[0]:interval[1],0], X_r[interval[0]:interval[1],1], marker='o', s=8, c=colorlist[i])
    #cbar = plt.colorbar()
    #plt.figure()
    #plt.imshow(anchor_distance_matrix[::30])
    #print(np.max(anchor_distance_matrix))
    #plt.colorbar()
    ax.set_title(f'PCA of MiniBatchKMeans - {name}')
    ax.set_xlabel('PC 0')
    ax.set_ylabel('PC 1')
    if plot3D:
        ax.set_zlabel('PC 2')
    if save:
        plt.savefig(f'{name}_pca.png', dpi=300)
        
def cluster_k_means(matrix, list_of_gain_obj, n_cluster=9):
    struc_list = [gain.name for gain in list_of_gain_obj]
    clust = MiniBatchKMeans(n_clusters=n_cluster,
                            random_state=0,
                            batch_size=6,
                            max_iter=10,
                            n_init="auto").fit(matrix)
    #clust = AgglomerativeClustering(n_clusters=n_cluster, metric='euclidean', 
    #                        memory=None, connectivity=None, compute_full_tree='auto', linkage='complete', distance_threshold=None, compute_distances=True).fit(anchor_distance_matrix)
    clustering=clust.labels_
    #print(np.unique(clustering, return_counts=True))
    n_struc, n_distances = matrix.shape # 14432, 21
    #print(clustering.labels_)
    new_order = np.zeros(shape=(n_struc), dtype=int)
    current_num = 0
    for i in range(n_cluster):
        for j, cluster_id in enumerate(clustering):
            if cluster_id == i :
                new_order[j] = current_num
                current_num += 1
    #print(new_order)
    remap_dict = {old_idx:new_idx for old_idx, new_idx in enumerate(new_order)}
    inv_remap_dict = {v:k for k,v in remap_dict.items()}
    reordered_matrix = np.zeros(shape=(n_struc,n_distances))
    for x in range(n_struc):
        new_x = remap_dict[x]
        reordered_matrix[new_x,:] = matrix[x,:]

    #fig = plt.figure(figsize=[20,1])
    #fig.set_facecolor('w')
    #plt.imshow(ordered_distances.transpose(), cmap='Greys')
    #cbar = plt.colorbar()
    #cbar.set_label('RMSD [$\AA$]')
    #plt.savefig('../test_largedist.png',dpi=300)

    reordered_names = [struc_list[inv_remap_dict[k]] for k in range(n_struc)]
    reordered_clusters = [clustering[inv_remap_dict[k]] for k in range(n_struc)]

    _, cluster_starts = np.unique(reordered_clusters, return_index = True)
    cluster_intervals = [(cluster_starts[k], cluster_starts[k+1]) for k in range(n_cluster-1)]
    cluster_intervals.append((cluster_starts[-1], n_struc))
    #print(cluster_intervals)

    return reordered_matrix, cluster_intervals, reordered_names

def get_anchor_coords(pdbfile, anchor_dict, multistate=False):
    # Find the CA coordinates of the anchor residue in the template PDB, return dictionary with the coords for each labeled anchor
    with open(pdbfile) as p:
        if multistate:
            data = p.read().split("ENDMDL")[1]
        if not multistate:
            data = p.read()
        mdl2data = [l for l in data.split("\n") if l.startswith("ATOM")]
        ca_data = [atom for atom in mdl2data if " CA " in atom]
        #print(anchor_dict, pdbfile)
    # Find the PDB coords of the CA of each atom and get them as a dict with  {'myanchor':(x,y,z), ...}
    res_ca_dict = {int(line[22:26]):(float(line[30:38]), float(line[38:46]), float(line[46:54])) for line in ca_data}
    coord_dict={}
    for anchor_name, anchor_res in anchor_dict.items():
        if anchor_res in res_ca_dict.keys():
            coord_dict[anchor_name]=res_ca_dict[anchor_res]
        else:
            "FALLBACK. Eliminiating Residue Anchor."
            coord_dict[anchor_name]=(1000.0, 1000.0, 1000.0)
    #coord_dict = {anchor_name:res_ca_dict[anchor_res] for anchor_name, anchor_res in anchor_dict.items()}
    
    return coord_dict

def space_distance(coords1, coords2):
    # Pythagorean distance of two sets of coords
    return round(math.sqrt(abs(coords1[0]-coords2[0])**2 + abs(coords1[1]-coords2[1])**2 + abs(coords1[2]-coords2[2])**2 ),3)

def calculate_anchor_distances(template_anchor_coords, mobile_pdb, mobile_anchors, threshold=10):
    # template_anchor_coords are a precalculated set of coords to save calc time.
    # Matches always the closest Helix and sheet anchors, respectively, regardless of label.
    mobile_anchor_coords = get_anchor_coords(mobile_pdb, mobile_anchors, multistate=True)
    #distances = pd.DataFrame()
    t_anchor_names = list(template_anchor_coords.keys())
    t_coord = list(template_anchor_coords.values())
    #print(t_coord)
    anchor_occupied = np.zeros(shape=(len(template_anchor_coords.keys())), dtype=bool)
    min_dists = []
    matched_anchors = []
    double_keys = []

    mob_coords = list(mobile_anchor_coords.values())
    n_template_keys = len(list(template_anchor_coords.keys()))

    for m_coord in mob_coords:
        distances = [space_distance(m_coord, coords) for coords in t_coord]
        min_idx = np.argmin(distances)
        if anchor_occupied[min_idx] == True:
            double_keys.append(t_anchor_names[min_idx])
        anchor_occupied[min_idx] = True
        matched_anchors.append(t_anchor_names[min_idx])
        min_dists.append(distances[min_idx])

    if len(double_keys) == 0:
        return dict(zip(mobile_anchors.keys(), matched_anchors)), dict(zip(mobile_anchors.keys(), min_dists))

    # If anchors are present multiple times, delete the further distant entry and make None, None
    for doublet in double_keys:
        #print(f'{doublet = }, {matched_anchors = }, {type(matched_anchors) = }')
        indices = [i for i,x in enumerate(matched_anchors) if doublet == x]
        minimum = np.argmin(np.array(min_dists)[indices])
        indices.remove(indices[minimum])
        # Only the non-minimum values will remain in the indices list that need to be re-evaluated
        newdists = np.empty(shape=(n_template_keys))
        for idx in indices:

            for i, coord in enumerate(t_coord):
                if anchor_occupied[i] == True:
                    newdists[i] = None
                    continue
                newdists[i] = space_distance(list(mobile_anchor_coords.values())[idx], coord)
            newmindist = np.min(newdists)
            if newmindist < threshold:
                print('Found Alternative')
                anchor_occupied[np.argmin(newdists)] = True
                min_dists[idx] = newmindist
                matched_anchors[idx] = list(mobile_anchor_coords.keys())[idx]
            else: 
                #print('Removed double occurrence')
                min_dists[idx] = None
                matched_anchors[idx] = ''

    return dict(zip(mobile_anchors.keys(), matched_anchors)), dict(zip(mobile_anchors.keys(), min_dists))

def find_best_templates(unknown_gain_obj, unknown_gain_pdb: str, sda_templates: dict, sdb_templates: dict, debug=False):
    # Put in an unknown GAIN domain object and its corresponding PDB path to match against the set of templates.
    # This will then give the best match for subdomain A and subdomain B based on GESAMT pairwise alignment
    pdb_start, sda_boundary, sdb_boundary, pdb_end = get_pdb_extents(unknown_gain_pdb, unknown_gain_obj.subdomain_boundary)
    # For each template, run GESAMT with the corresponding subdomain and find lowest RMSD; maybe future: Highest Q-Score
    if debug:
        print(f"[DEBUG] find_best_templates:\n\t{unknown_gain_obj.name}\n\t{sda_templates = }\n\t{sdb_templates = }")
    a_rmsds = []
    b_rmsds = []
    a_names = []
    b_names = []

    for sda_id, sda_pdb in sda_templates.items():
        cmd_string = f'/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt {sda_pdb} {unknown_gain_pdb} -s A/{pdb_start}-{sda_boundary}'
        output = run_command(cmd_string)
        # Read the RMSD value from the ouptut string returned by the function. No files are created here.
        match = re.search(r"RMSD\W+\:\W+[0-9]+\.[0-9]+",output)
        if match is None:
            print("[WARNING]: NO RMSD FOUND:", cmd_string, output)
            val = 100.0 # penalty for non-matching template
        else:
            val = float(match.group(0)[-5:])
        a_names.append(sda_id)
        a_rmsds.append(val)

    best_sda = a_names[a_rmsds.index(min(a_rmsds))]

    for sdb_id, sdb_pdb in sdb_templates.items():
        cmd_string = f'/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt {sdb_pdb} {unknown_gain_pdb} -s A/{sdb_boundary}-{pdb_end}'
        output = run_command(cmd_string)
        # Read the RMSD value from the ouptut string returned by the function. No files are created here.
        match = re.search(r"RMSD\W+\:\W+[0-9]+\.[0-9]+",output)
        if match is None:
            print("[WARNING]: NO RMSD FOUND:", cmd_string, output)
            val = 100.0 # penalty for non-matching template
        else:
            val = float(match.group(0)[-5:])
        b_names.append(sdb_id)
        b_rmsds.append(val)
        
    best_sdb = b_names[b_rmsds.index(min(b_rmsds))]
    if debug:
        print(f"[DEBUG] find_best_templates: RESULT\n\t{best_sda}\n\t{best_sdb}")
    return best_sda, best_sdb

def get_agpcr_type(name):
    queries = [('AGR..', name, lambda x: x[-1][-2:]), #
                ('ADGR..', name, lambda x: x[-1][-2:]), 
                ('cadher.*receptor.', name.lower(), lambda x: f"C{x[-1][-1]}"),
                ('cels?r.', name.lower(), lambda x: f"C{x[-1][-1]}"), 
                ('latrophilin.*protein-?\d', name.lower(), lambda x: f"L{x[-1][-1]}"),
                ('latrophilin-?\d', name.lower(), lambda x: f"L{x[-1][-1]}"),
                ('GP?R133', name.upper(),lambda x: 'D1'),
                ('GP?R126', name.upper(),lambda x: 'G6'),
                ('GP?R?124', name.upper(),lambda x: 'A2'),
                ('GP?R?125', name.upper(),lambda x: 'A3'),
                ('GP?R112', name.upper(),lambda x: 'G4'),
                ('GP?R116', name.upper(),lambda x: 'F5'),
                ('GP?R144', name.upper(),lambda x: 'D2'),
                ('ag-?.*-?coupled-?receptor-?.-?\d', name.lower(),lambda x: x[-1].replace('-','')[-2:].upper()),
                ('brain-?specific-?angiogenesis-?inhibitor-?\d', name.lower(), lambda x: f"B{x[-1][-1]}"),
                ('emr\d', name.lower(), lambda x: f"E{x[-1][-1]}"),
                ]
    for pattern, searchstring, output in queries:
        match = re.findall(pattern, searchstring)
        if match != []:
            #if output(match) == '': print(name)
            return output(match)
    return 'X'

def assign_indexing(gain_obj, file_prefix: str, gain_pdb: str, template_dir: str, hard_cut=None, debug=False, create_pdb=False):
    if debug:
        print(f"[DEBUG] assign_indexing: {gain_obj.start = }\n\t{gain_obj.end = }\n\t{gain_obj.subdomain_boundary = }\n\t{gain_pdb = }")
    # Arbitrarily defined data for templates. Receptor type -> template ID
    type_2_sda_template = {
                    'A1':'A', 'A2':'A', 'A3':'A', 
                    'B1':'A', 'B2':'A', 'B3':'A',
                    'C1':'C', 'C2':'C', 'C3':'C',
                    'D1':'D','D2':'G7',
                    'E1':'E1', 'E2':'E1', 'E3':'E1', 'E4':'E1', 'E5':'E5', 'E' :'E1',
                    'F1':'F5','F2':'F4', 'F3':'F5', 'F4':'F4', 'F5':'F5', 'F' :'F5',
                    'G1':'G7', 'G2':'G7', 'G3':'G7', 'G4':'G7', 'G5':'G7', 'G6':'G7', 'G7':'G7',
                    'L1':'L', 'L2':'L', 'L3':'L', 'L4':'L4',
                    'V1':'V'
                    }
    type_2_sdb_template = {
                    'A1':'E5b', 'A2':'E5b', 'A3':'E5b', 
                    'B1':'E5b', 'B2':'E5b', 'B3':'E5b',
                    'C1':'E5b', 'C2':'E5b', 'C3':'E5b',
                    'D1':'E5b','D2':'E5b',
                    'E1':'E5b', 'E2':'E5b', 'E3':'E5b', 'E4':'E5b', 'E5':'E5b', 'E' :'E5b',
                    'F1':'E5b','F2':'E5b', 'F3':'E5b', 'F4':'E5b', 'F5':'E5b', 'F' :'E5b',
                    'G1':'G5b', 'G2':'E5b', 'G3':'G5b', 'G4':'E5b', 'G5':'G5b', 'G6':'E5b', 'E5b':'E5b',
                    'L1':'E5b', 'L2':'E5b', 'L3':'E5b', 'L4':'E5b',
                    'V1':'E5b'
                    }
    # Predefined x.50 residues for each set template (= center)
    sda_centers = {
                    'A' :{'H1'   :416 , 'H2':439 , 'H3':454 , 'H4':489 , 'H5':496 , 'H6':514 },
                    'C' :{'H1'   :463 , 'H2':484 , 'H3':498 , 'H4':541 , 'H5':549 , 'H6':567 },
                    'D' :{'H1.D1':390 , 'H2':420 , 'H3':435 , 'H4':480 , 'H5':488 , 'H6':506 },
                    'E1':{'H1.E1':142 ,            'H3':159 , 'H4':198 ,            'H6':221 },
                    'E5':{                         'H3':268 , 'H4':303 ,            'H6':329 },
                    #'F5':{'H1'   :133 , 'H2':145 , 'H3':158 , 'H4':303 , 'H5':201 , 'H6':218 },
                    'F4':{'H1.F4':133 , 'H2':145 , 'H3':158 ,            'H5':201 , 'H6':218 },
                    'G7':{'H1'   :148 , 'H2':164 , 'H3':178 , 'H4':212 , 'H5':219 , 'H6':239 },
                    'L' :{'H1'   :494 , 'H2':509 , 'H3':521 , 'H4':580 , 'H5':588 , 'H6':608 },
                    'L4':{'H1'   :197 , 'H2':212 , 'H3':224 , 'H4':266 , 'H5':274 , 'H6':294 },
                    'V' :{'H1'   :528 , 'H2':546 , 'H3':558 , 'H4':589 ,            'H6':607 }
                  }
    sdb_centers = {
                    'E5b':	{'S1':324, 'S2':333, 'S3':350, 'S4':357, 'S5':375, 'S6':407, 'S7':414, 'S8':431, 'S9':450, 'S10':459, 'S11':464, 'S12':476 ,'S13':487},
                    'G5b':	{'S1': 65, 'S2':74,  'S3':88 , 'S4':105 ,'S5':124,			 'S7':149 ,'S8':167 ,'S9':183 ,'S10':198 ,'S11':203 ,'S12':215 ,'S13':226}
                  }
    # Anchor priority to finally define override
    anchor_priority  =  { 
                        "H1" :60, "H2" :70, "H3" :80, "H4" :90, "H5" :80, "H6" :100,
                        "S1" :70, "S2" :80, "S3" :90, "S4" :80, "S5" :85, "S6" :50, "S7" :90, "S8" :95, "S9" :90, "S10":80, "S11":98, "S12":99, "S13":100
                        }
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
    # In case the file_prefix specifies a folder, check if the folder exists. If not, create it.
    if "/" in file_prefix:
        target_path = "/".join(file_prefix.split("/")[:-1])
        if not os.path.isdir(target_path):
            print(f"Created {target_path}")
            os.mkdir(target_path)

    # Get the agpcr-type and the corresponding templates to be matched.
    agpcr_type = get_agpcr_type(gain_obj.name)
    # If the type is unknown, get the best matching templates by performing RMSD after alignment via GESAMT
    if agpcr_type not in type_2_sda_template.keys():
        if agpcr_type[0] not in type_2_sda_template.keys():

            best_a, best_b = find_best_templates(gain_obj, gain_pdb, sda_templates, sdb_templates, debug=debug)
            if debug:
                print(f"[DEBUG] assign_indexing: running template search with unknown receptor.\n{best_a = } {best_b = }")
    else:
        best_a = type_2_sda_template[agpcr_type] # This is the best template ID, i.e. "A"
        best_b = type_2_sdb_template[agpcr_type] #              -"-                   "E5b"
        if debug: 
            print(f"Looking up best template for {agpcr_type = }: {best_a = } {best_b = }")

    a_centers = sda_centers[best_a]
    b_centers = sdb_centers[best_b]

    if debug:
        print(f"[DEBUG] assign_indexing: Found receptor type and best corresponding templates:\n\t{agpcr_type = }\n\t{best_a = }\n\t{best_b = }\n\t{a_centers = }\n\t{b_centers = }")

    pdb_start, sda_boundary, sdb_boundary, pdb_end = get_pdb_extents(gain_pdb, gain_obj.subdomain_boundary)
    a_cmd_string = f'/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt {sda_templates[best_a]} {gain_pdb} -s A/{pdb_start}-{sda_boundary}'
    b_cmd_string = f'/home/hildilab/lib/xtal/ccp4-8.0/bin/gesamt {sdb_templates[best_b]} {gain_pdb} -s A/{sdb_boundary}-{pdb_end}'
    
    if create_pdb:
        a_cmd_string = a_cmd_string + f" -o {file_prefix}_sda.pdb"
        b_cmd_string = b_cmd_string + f" -o {file_prefix}_sdb.pdb"
    # run GESAMT, keep the file to read later and documentation
    run_command(a_cmd_string, logfile=None, outfile=f'{file_prefix}_sda.out')
    run_command(b_cmd_string, logfile=None, outfile=f'{file_prefix}_sdb.out')

    # Not needed if the template chains in the template PDBs are already !A
    """    def rewrite_chains(pdb):
        with open(pdb) as p:
            data = p.read()
        dx = data.split("ENDMDL", 1)
        with open(pdb,'w') as p:
            p.write(dx[0])
            p.write("ENDMDL")
            p.write(dx[1].replace(" A ", " B "))

    if create_pdb:
        rewrite_chains(f"{file_prefix}_sda.pdb")
        rewrite_chains(f"{file_prefix}_sdb.pdb")"""

    target_a_centers = find_anchor_matches(f'{file_prefix}_sda.out', a_centers, isTarget=False, debug=debug)
    target_b_centers = find_anchor_matches(f'{file_prefix}_sdb.out', b_centers, isTarget=False, debug=debug)

    if debug:
        print(f"[DEBUG] assign_indexing: The matched anchors of the target GAIN domain are:{target_a_centers = }\n\t{target_b_centers = }")

    a_out = list(create_compact_indexing(gain_obj, 'a', target_a_centers, threshold=3, padding=1, hard_cut=hard_cut, debug=debug) ) # [dict, dict, dict, list]
    b_out = list(create_compact_indexing(gain_obj, 'b', target_b_centers, threshold=1, padding=1, hard_cut=hard_cut, debug=debug) )

    #      elements+intervals          element_centers             residue_labels              unindexed_elements        
    return { **a_out[0], **b_out[0] }, { **a_out[1], **b_out[1]} , { **a_out[2], **b_out[2] }, a_out[3] + b_out[3]

    # merge outputs of both functions and return them in a combined way.

def create_compact_indexing(gain_obj, subdomain:str, actual_anchors:dict, threshold=3, padding=1, hard_cut=None, debug=False):
    ''' 
    Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
    Prints out the final list and writes it to file if outdir is specified
        
    Parameters
    ----------
    gain_obj: GainDomain object, required
        GainDomain object with corresponding parameters
    subdomain: character, required
        the name of the subdomain "a", "b"
    actual_anchors : dict, required
        Dictionary of anchors with each corresponding to the matched template anchor. This should already be the residue of this GAIN domain
    anchor_dict : dict, required
        Dictionary where each anchor index is assigned a name (H or S followed by a greek letter for enumeration)
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
    def split_segments(sse:list, found_anchors:list, res2anchor:dict, coiled_residues:list, outlier_residues:list, hard_cut=None, debug=debug):
        # Split a segment with multiple occurring anchors into its respective segments.
        # Use coiled_residues first, then outlier_residues with lower priority
        if debug: 
            print(f"[DEBUG] split_segments: called with \n\t{hard_cut = }\n\t{found_anchors = }\n\telements: {[res2anchor[i] for i in found_anchors]}")
        n_segments = len(found_anchors)
        sorted_anchors = sorted(found_anchors) # Just in case, order the anchors
        n_boundaries = {}
        c_boundaries = {}

        # Start of first segment, end os last segment are pre-defined.
        n_boundaries[sorted_anchors[0]] = sse[0]
        c_boundaries[sorted_anchors[-1]] = sse[1]

        for i in range(n_segments-1):
            n_anchor, c_anchor = sorted_anchors[i], sorted_anchors[i+1]
            hard_cut_flag = False
            # Find breaker residue in between these two segments:
            coiled_breakers = [r for r in coiled_residues if r > n_anchor and r < c_anchor]
            outlier_breakers = [r for r in outlier_residues if r > n_anchor and r < c_anchor]

            # By convention, split the along the most N-terminal occurence of a break residue.
            # Here is the logic to detect what is used for breaking up the segment. The order of checks:
            # First line of splitting: Check of coiled or outlier residues present in the segment.
            if len(coiled_breakers) > 0:
                breakers = coiled_breakers
            elif len(outlier_breakers) > 0:
                breakers = outlier_breakers
            else:
                # Second line of splitting: Check if there is a proline or glycine. If so, use this as a breaker.
                segment_sequence = gain_obj.sequence[sse[0]-gain_obj.start:sse[1]-gain_obj.start]
                print(f"[DEBUG] split_segments : Last resort with \n\t{segment_sequence = }\n\t{sse = }")
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
                    # Last line of splitting: Check if there is a hard cut provided in hard_cut for one of the anchors (S6?). If so, break it after n residues,
                    # keep all residues in the segment (the breaker is added to the C-terminal segment)
                    if hard_cut is not None and res2anchor[n_anchor] in hard_cut.keys():
                        hard_cut_flag = True
                        breakers = [n_anchor+hard_cut[res2anchor[n_anchor]]]
                        print(f"[WARNING] HARD CUT BREAKER @ {breakers[0]} between {res2anchor[n_anchor]} | {res2anchor[c_anchor]}")
                    else:
                        print(f"[ERROR]: No breakers and no Proline detected between {n_anchor = }:{res2anchor[n_anchor]} {c_anchor = }:{res2anchor[c_anchor]}\n{segment_sequence = }")
                        raise IndexError(f"No Breaker detected in multi-anchor ordered segment: \n\t{sse[0]}-{sse[1]}\n\t{gain_obj.name} Check this.")
            
            c_boundaries[n_anchor] = breakers[0]-1
            n_boundaries[c_anchor] = breakers[0]+1
            # Include the hard cut residue in the C-terminal element (decrement n-terminal boundary by 1)
            if hard_cut_flag:
                n_boundaries[c_anchor] -= 1
        # Merge the segments and return them
        segments = {a:[n_boundaries[a], c_boundaries[a]] for a in sorted_anchors}
        return segments

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
        indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the anchor is located

        return nom_list, indexing_dir, indexing_centers

    if debug: print(f"[DEBUG] create_compact_indexing: passed arguments:\n\t{gain_obj.name = }\n\t{subdomain = }\n\t{actual_anchors = }")
    ### END OF FUNCTION BLOCK
    # Initialize Dictionaries
    indexing_dir = {}
    indexing_centers = {}
    named_residue_dir = {}
    unindexed = []

    # Invert the actual anchors dict to match the GAIN residue to the named anchor
    res2anchor = {v[0]:k for k,v in actual_anchors.items()} # v is a tuple  (residue, distance_to_template_center_residue)
    if debug: print(f"[DEBUG] create_compact_indexing: {res2anchor = }")
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

        # find residues matching anchors within the ordered segment
        found_anchors = [sse_res for sse_res in range(first_res, last_res+1) if sse_res in res2anchor.keys()]  

        n_found_anchors = len(found_anchors)

        if n_found_anchors == 0:
            isPadded = False
            # Try finding the anchor with extending the residue by padding
            for res in res2anchor:
                if res == first_res-padding or res == last_res+padding:
                    if debug:
                        print(f"[DEBUG] create_compact_indexing : Found padded anchor \n{res = }\n{res2anchor[res] = }\n\t{[first_res, last_res] = }")
                    # Find the closest residue to the anchor column index. N-terminal wins if two residues tie.
                    name_list, cast_values = create_name_list([first_res, last_res], res, res2anchor[res])
                    isPadded =True
                    break
            # If there is an unadressed SSE with length 3 or more, then add this to unindexed.
            if not isPadded and sse[1]-sse[0] > threshold:
                if debug: 
                    print(f"[DEBUG] create_compact_indexing : No anchor found! \n {[first_res, last_res] = }")
                unindexed.append(first_res)

            continue

        if n_found_anchors == 1:
            anchor_res = found_anchors[0]
            if debug: print(f"SINGLUAR ANCHOR FOUND: @ {anchor_res = }, {res2anchor[anchor_res]}")
            sse_name = res2anchor[anchor_res]
            name_list, cast_values = create_name_list([first_res, last_res], anchor_res, sse_name)
            # name_list has the assignment for the SSE, cast_values contains the passed values for dict casting
            nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, [first_res, last_res], name_list, cast_values)
            # Also write split stuff to the new dictionary
            for entryidx, entry in enumerate(name_list): 
                named_residue_dir[entry] = entryidx + first_res
            continue

        if n_found_anchors > 1:
            # When multiple anchors are present, split the segment by finding coiled and outlier residues.
            if debug: 
                print(f"[DEBUG] create_compact_indexing : ANCHOR AMBIGUITY in this SSE:\n\t{sse = }\n\t{found_anchors = }")

            # parse the sse_sequence to find coiled and outlier residues
            coiled_residues = []
            outlier_residues = []
            for i in range(first_res+1, last_res): # The start and end of a segment are never breaks.
                if gain_obj.sse_sequence[i] ==  "C":
                    coiled_residues.append(i)
                if gain_obj.sse_sequence[i] == 'h':
                    outlier_residues.append(i)
            
            if debug:
                print(f"[DEBUG] create_compact_indexing :\n\t{coiled_residues  = }\n\t{outlier_residues = }")
                
            split_segments = split_segments(sse=[first_res, last_res], 
                                            found_anchors=found_anchors,
                                            res2anchor=res2anchor,
                                            coiled_residues=coiled_residues, 
                                            outlier_residues=outlier_residues, 
                                            hard_cut=hard_cut, 
                                            debug=debug)
            for anchor_res, segment in split_segments.items():
                name_list, cast_values = create_name_list(segment, anchor_res, res2anchor[anchor_res])
                # cast them also?
                nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, segment, name_list, cast_values)
                 # Also write split stuff to the new dictionary
                for entryidx, entry in enumerate(name_list): 
                    named_residue_dir[entry] = entryidx+segment[0]

    # Patch the GPS into the nom_list
    """        labels = ["GPS-2","GPS-1","GPS+1"]
    for i, residue in enumerate(gain_obj.GPS.residue_numbers[:3]):
        #print(residue)
        nom_list[residue] = labels[i]
        indexing_dir["GPS"] = gain_obj.GPS.residue_numbers
        # Also cast this to the general indexing dictionary
        named_residue_dir[labels[i]] = gain_obj.GPS.residue_numbers[i]"""
           
    return indexing_dir, indexing_centers, named_residue_dir, unindexed
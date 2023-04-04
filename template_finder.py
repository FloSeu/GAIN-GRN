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

"""class Template:
    def __init__(self, subdomain, name, pdb_file, anchor_dict):
        self.subdomain = subdomain
        self.anchor_dict = anchor_dict
        self.pdb_file = pdb_file
        self.identifier = name.split("-")[0]"""

def run_command(cmd, logfile, outfile):
    # The command would be the whole command line, in this case the gesamt command.
    with open(logfile,"a") as l: 
        l.write(f"\nCalled Function run_command with: {cmd}\nOutput to: {logfile}")
    f_out = open(outfile, "w")
    f_err = open(logfile, "w")
    # also : shlex.split("cmd")
    p = subprocess.Popen(cmd, shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            bufsize=10, 
            universal_newlines=True)
    #exit_code = p.poll()
    outs, errs = p.communicate()
    f_out.write(outs)
    f_err.write(errs)
    f_out.close()
    f_err.close()
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

def find_anchor_matches(file, anchor_dict,  isTarget=False, debug=False):
    # Takes a gesamt file and an anchor dictionary either of the target or the template and returns the matched other residue with the pairwise distance
    # as a dictionary: {'H1': (653, 1.04) , ...}
    template_pairs, mobile_pairs = read_gesamt_pairs(file)
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

def read_gesamt_pairs(gesamtfile):
    with open(gesamtfile) as f:
        pair_lines = [line for line in f.readlines() if line.startswith("|")][2:]
        #|H- A:LEU 720 | <**1.21**> |H- A:LEU 633 |
        #| - A:ALA 721 | <..1.54..> | + A:GLN 634 |
        #| + A:GLU 722 | <..1.75..> | + A:PRO 635 |
        #| + A:GLU 723 | <==2.03==> | + A:GLN 636 |
        #| + A:ASN 724 | <..1.91..> | - A:ALA 637 |
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
        template_pairs[int(template_str)] = (int(mobile_str), dist)
        mobile_pairs[int(mobile_str)] = (int(template_str), dist)
    
    return template_pairs, mobile_pairs

def create_subdomain_indexing(self, subdomain, actual_anchors, anchor_occupation, silent=False, split_mode='single',debug=False):
        ''' 
        Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
        Prints out the final list and writes it to file if outdir is specified
        
        Parameters
        ----------
        actual_anchors : dict, required
            Dictionary of anchors with each corresponding to the matched template anchor. This should already be the residue of this GAIN domain
        anchor_occupation : dict, required
            Dict of Occupation values corresponding to each of the template anchors (actual_anchors) for resolving ambiguity conflicts
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
                    name_list, cast_values = create_name_list(sse, new_res, res2anchor[new_res])
                else: 
                    name_list, cast_values = create_name_list(sse, stored_res, res2anchor[stored_res])
                
                return [(sse, name_list, cast_values)], False

            adjusted_break_residues = [res+sse[0] for res in break_residues]

            # Check if there is a break residue in between the two conflicting anchors
            for break_res in adjusted_break_residues:
                if stored_res < break_res and break_res < new_res:
                    priority_mode = False 
                if new_res < break_res and break_res < stored_res:
                    priority_mode = False

            if priority_mode == True:
                if not silent: 
                    print(f"DEBUG gain_classes.disambiguate_anchors : no break found between anchors, will just overwrite.")
                if new_anchor_weight > stored_anchor_weight:
                    name_list, cast_values = create_name_list(sse, new_res, res2anchor[new_res])
                else: 
                    name_list, cast_values = create_name_list(sse, stored_res, res2anchor[stored_res])
                
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
            stored_name_list, stored_cast_values = create_name_list(seg_stored, stored_res, res2anchor[stored_res])
            new_name_list, new_cast_values = create_name_list(seg_new, new_res, res2anchor[new_res])

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
            # if no exact match is found, the anchor does not exist in this domain
            if exact_match == False:
                if debug: print("Anchors not found in this sse", idx, sse)
                # expand anchor detection to +1 and -1 of SSE interval
                for res in res2anchor:
                    if res == sse[0]-1 or res == sse[1]+1:
                        fuzzy_match = True
                        if not silent: print(f"[DEBUG] GainDomain.create_indexing : Interval search found anchor @ Residue {res}")
                        # Find the closest residue to the anchor column index. N-terminal wins if two residues tie.
                        name_list, cast_values = create_name_list(sse, res, res2anchor[res])

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

def construct_structural_alignment(template_gain_domain, list_of_gain_obj, gain_indices, gesamt_folder=str, outfile=None):
    # This will construct an UNGAPPED MSA based on the pairwise match to the template residues.
    msa_array = np.full(shape=(len(list_of_gain_obj)+1, len(template_gain_domain.sequence)), fill_value='-', dtype = '<U2')
    msa_array[0,:] = template_gain_domain.sequence
    all_names = [template_gain_domain.name]
    #for gesamtfile in glob.glob(f"{gesamt_folder}/*.out"):
    for gain_idx, gain in enumerate(list_of_gain_obj):
        if gain.name == template_gain_domain.name: # QC: ensure the template is not aligned to itself.
            continue
        all_names.append(gain.name)
        template_pairs, _ = read_gesamt_pairs(f'{gesamt_folder}/{gain_indices[gain_idx]}.out')
        # This is a dict: { 516: (837, 1.08),... }
        res_matches = {k-template_gain_domain.start : gain.sequence[v[0]-gain.start] for k,v in template_pairs.items() if v[0] is not None}
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
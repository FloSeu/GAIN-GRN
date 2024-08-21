# Functions for finding a template via GainDomain subselection and GESAMT pairwise structural alignments.
import glob, os, re, shutil, math
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from gaingrn.scripts.io import read_gesamt_pairs, run_logged_command, get_agpcr_type
from gaingrn.scripts.structure_utils import get_pdb_extents

def space_distance(coords1, coords2):
    # Pythagorean distance of two sets of coords
    return round(math.sqrt(abs(coords1[0]-coords2[0])**2 + abs(coords1[1]-coords2[1])**2 + abs(coords1[2]-coords2[2])**2 ),3)

def run_gesamt_execution(list_of_gain_obj, outfolder, gesamt_bin, pdb_folder='../all_pdbs', domain='sda', n_threads=6, max_struc=400, no_run=False, template=None):
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
        if sda_boundary is None:
            print("WARNING. NO BOUNDARIES DETECTED. THIS GAIN MAY BE FAULTY!", gain.name, gain_pdb, sep="\n\t")
            continue
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
                cmd_string += f'{gesamt_bin} {p1name} -s A/{extent_list[i]} {p2name} -s A/{extent_list[j]} > {outfolder}/{run_id}.out &\n'
                if myct % n_threads == 0:
                    cmd_string +="wait\n" # Delimiter for multiprocessing
            
        else: # No nested double for loop upon template specification
            myct +=1
            run_id = i
            cmd_string += f'{gesamt_bin} {template} {p1name} -s A/{extent_list[i]}  > {outfolder}/{run_id}.out &\n'
            if myct % n_threads == 0:
                cmd_string +="wait\n" # Delimiter for multiprocessing
    
    # Write the composed commands to bashfile and run
    with open(bashfile,'w') as sh: # 
        sh.write(cmd_string)
    print(f"Written a total of {myct} GESAMT commands to file.")
    if not no_run:
        print(f"Running set of GESAMT comparisons with {n_threads} threads.")
        run_logged_command(f"bash {bashfile}", f"{outfolder}/gesamt.log", f"{outfolder}/gesamt.err")
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

def cluster_agglomerative(distances, list_of_gain_obj, n_cluster=9):
    # Perform agglomerative clustering on the selection of GAIN domain objects with the specified number of clusters.
    clustering = AgglomerativeClustering(n_clusters=n_cluster, metric='precomputed', 
                            memory=None, connectivity=None, compute_full_tree='auto', linkage='complete', distance_threshold=None, compute_distances=False).fit(distances)
    n_struc = distances.shape[0]
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

    # Extract the best structure OVERALL and the best structure for EACH CLUSTER
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

def cluster_k_means(matrix, list_of_gain_obj, n_cluster=9, plot=False):
    # For testing, clustering can also be performed via k-means
    struc_list = [gain.name for gain in list_of_gain_obj]
    clust = MiniBatchKMeans(n_clusters=n_cluster,
                            random_state=0,
                            batch_size=6,
                            max_iter=10,
                            n_init="auto").fit(matrix)
    clustering=clust.labels_
    n_struc, n_distances = matrix.shape
    new_order = np.zeros(shape=(n_struc), dtype=int)
    current_num = 0

    # Re-order by best / largest clusters.
    for i in range(n_cluster):
        for j, cluster_id in enumerate(clustering):
            if cluster_id == i :
                new_order[j] = current_num
                current_num += 1

    remap_dict = {old_idx:new_idx for old_idx, new_idx in enumerate(new_order)}
    inv_remap_dict = {v:k for k,v in remap_dict.items()}
    reordered_matrix = np.zeros(shape=(n_struc,n_distances))

    for x in range(n_struc):
        new_x = remap_dict[x]
        reordered_matrix[new_x,:] = matrix[x,:]

    reordered_names = [struc_list[inv_remap_dict[k]] for k in range(n_struc)]
    reordered_clusters = [clustering[inv_remap_dict[k]] for k in range(n_struc)]

    _, cluster_starts = np.unique(reordered_clusters, return_index = True)
    cluster_intervals = [(cluster_starts[k], cluster_starts[k+1]) for k in range(n_cluster-1)]
    cluster_intervals.append((cluster_starts[-1], n_struc))
    return reordered_matrix, cluster_intervals, reordered_names

def plot_matrix(distances, title='', savename=None):
    # plot the RMSD matrix
    fig = plt.figure(figsize=[6,4])
    fig.set_facecolor('w')
    plt.imshow(distances, cmap='Greys')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('RMSD [$\AA$]')
    if savename is not None:
        plt.savefig(f'{savename}.png',dpi=300)
    else:
        plt.show()
    del fig

def plot_heirarchy(distances, groupname='', savename=None):
    # plots the heirarchical clustering to assess groups of structures.
    reduced_distances = squareform(distances, checks=False)
    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
    fig = plt.figure(figsize=[18,4], facecolor='w')
    plt.title(f'RMSD Average linkage hierarchical clustering: {groupname}')
    _ = scipy.cluster.hierarchy.dendrogram(linkage, count_sort='descendent', show_leaf_counts=True, leaf_font_size=3)
    if savename is not None:
        plt.savefig(f"{savename}.png", dpi=200)
    else:
        plt.show()
    del fig

def plot_pca(distance_matrix, cluster_intervals, n_components, name, plot3D=False, save=True):
    # Creates and plots a Principal component analysis for assessing variance in between respective clusters
    colorlist = ['blue','red','green','yellow','orange','purple','forestgreen','limegreen','firebrick']
    #X = center_distance_matrix
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
    ax.set_title(f'PCA of MiniBatchKMeans - {name}')
    ax.set_xlabel('PC 0')
    ax.set_ylabel('PC 1')
    if plot3D:
        ax.set_zlabel('PC 2')
    if save:
        plt.savefig(f'{name}_pca.png', dpi=300)

def match_gain2subdomain_template(gain_idx, template_centers, gesamt_folder, penalty=None, debug=False):
    '''
    Match a GAIN domain to a template PDB. This function has two modes. 
    We match the closest residue to the template as the new center and output the residues and their distances.

    gain_idx indicates the index of the domain used for locating GESAMT alignment files.
    template PDBs are global, but not needed since we are handling only the aligned files / PDBs.
    '''
    n_anch = len(template_centers.keys())

    target_gesamt_file = f'{gesamt_folder}/{gain_idx}.out'
    actual_centers = find_center_matches(target_gesamt_file, template_centers, debug=debug)
    if debug: 
        print(f"[DEBUG]: return find_center_matches: {actual_centers = }")
    all_dist = {k:v[1] for k,v in actual_centers.items() } #{'H1': (653, 1.04) , ...}
    if debug: 
        print(f"[DEBUG]: return find_center_matches: {all_dist = }") #{'S1': 4.56, 'S2': 2.09, ...}
    # From the center-residue distances, fill a matrix with the individual distances, assign unmatched centers a pre-set penalty value
    distances = np.full(shape=(n_anch), fill_value=penalty)
    for i, sse in enumerate(template_centers.keys()):
        if debug: print(f"[DEBUG]: match_gain2subdomain_template: {i = }, {sse = }")
        if sse in all_dist.keys():
            if all_dist[sse] is not None:
                distances[i] = all_dist[sse]
        # Cap the distance maximum to the penalty, if specified
        for i,val in enumerate(distances):
            if penalty is not None and val > penalty: 
                distances[i] = penalty
    if debug: print(f"[DEBUG]: return find_center_matches: Final distances to return : \n\t{distances = }")
    return distances, actual_centers

def construct_structural_alignment(template_gain_domain, list_of_gain_obj, gain_indices, gesamt_folder=str, outfile=None, debug=False):
    # This will construct an UNGAPPED MSA based on the pairwise match to the template residues.
    msa_array = np.full(shape=(len(list_of_gain_obj)+1, len(template_gain_domain.sequence)), fill_value='-', dtype = '<U2')
    msa_array[0,:] = template_gain_domain.sequence
    all_names = [template_gain_domain.name]
    template_seq_len = len(template_gain_domain.sequence)
    #for gesamtfile in glob.glob(f"{gesamt_folder}/*.out"):
    for gain_idx, gain in enumerate(list_of_gain_obj):
        if gain.name == template_gain_domain.name: # QC: ensure the template is not aligned to itgain_obj.
            continue
        all_names.append(gain.name)
        if debug:
            print("[DEBUG] template_utils.construct_structural_alignment :")
            print("\tGAIN NAME :", gain.name)
            with open(f'{gesamt_folder}/{gain_indices[gain_idx]}.out') as out:
                data = out.readlines()[43]
                print(f"\t READ DATA : {gesamt_folder}/{gain_indices[gain_idx]}.out\n\t", data)
            
        template_pairs, _ = read_gesamt_pairs(f'{gesamt_folder}/{gain_indices[gain_idx]}.out')
        if not template_pairs: # Check if the dict is empty
            print(f"[NOTE]: GESAMT Alignment has failed. Skipping: {gain.name = } with \n\t{gain.sda_helices =}\n\t{gain.sdb_sheets = }")
            continue
        # This is a dict: { 516: (837, 1.08),... }
        if debug: 
            print(f"[DEBUG] {template_pairs = }\n\t{len(gain.sequence)} {gain.start = } {gain.end = }\n\t{template_gain_domain.start = }\n\t{template_gain_domain.end = }")
        
        res_matches = {}

        for k,v in template_pairs.items():
            #  has a match       |  is in target GAIN     |  is in in target GAIN (c)              |  | Template RES is in template GAIN (pot. calc. in fasta offset)
            if v[0] is not None and v[0]-gain.start >= 0 and v[0]-gain.start < len(gain.sequence) and k-template_gain_domain.start < template_seq_len:
                res_matches[k-template_gain_domain.start] = gain.sequence[v[0]-gain.start]
        #res_matches = {k-template_gain_domain.start : gain.sequence[v[0]-gain.start] for k,v in template_pairs.items() if v[0] is not None and v[0]-gain.start < len(gain.sequence) and v[0]-gain.start >=0}
        # This is a dict matching the aln columns : {0: "V", 1, "A", 4:"C"}, non-matched residues are not present (None value)
        # map to alignment array
        if debug:
            print(f"[DEBUG]\n\t{res_matches = }\n\t{msa_array.shape = }")
        for resid, res in res_matches.items():
            msa_array[gain_idx+1,resid] = res
        if debug: print(f"After populating the array:\n{msa_array[gain_idx+1,:] = }")

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

def get_center_coords(pdbfile, center_dict, multistate=False):
    # Find the CA coordinates of the center residue in the template PDB, return dictionary with the coords for each labeled center
    with open(pdbfile) as p:
        if multistate:
            data = p.read().split("ENDMDL")[1]
        if not multistate:
            data = p.read()
        mdl2data = [l for l in data.split("\n") if l.startswith("ATOM")]
        ca_data = [atom for atom in mdl2data if " CA " in atom]
        #print(center_dict, pdbfile)
    # Find the PDB coords of the CA of each atom and get them as a dict with  {'mycenter':(x,y,z), ...}
    res_ca_dict = {int(line[22:26]):(float(line[30:38]), float(line[38:46]), float(line[46:54])) for line in ca_data}
    coord_dict={}
    for center_name, center_res in center_dict.items():
        if center_res in res_ca_dict.keys():
            coord_dict[center_name]=res_ca_dict[center_res]
        else:
            "FALLBACK. Eliminiating Residue Anchor."
            coord_dict[center_name]=(1000.0, 1000.0, 1000.0)
    #coord_dict = {center_name:res_ca_dict[center_res] for center_name, center_res in center_dict.items()}
    
    return coord_dict

def calculate_center_distances(template_center_coords, mobile_pdb, mobile_centers, threshold=10):
    # template_center_coords are a precalculated set of coords to save calc time.
    # Matches always the closest Helix and sheet centers, respectively, regardless of label.
    mobile_center_coords = get_center_coords(mobile_pdb, mobile_centers, multistate=True)
    #distances = pd.DataFrame()
    t_center_names = list(template_center_coords.keys())
    t_coord = list(template_center_coords.values())
    #print(t_coord)
    center_occupied = np.zeros(shape=(len(template_center_coords.keys())), dtype=bool)
    min_dists = []
    matched_centers = []
    double_keys = []

    mob_coords = list(mobile_center_coords.values())
    n_template_keys = len(list(template_center_coords.keys()))

    for m_coord in mob_coords:
        distances = [space_distance(m_coord, coords) for coords in t_coord]
        min_idx = np.argmin(distances)
        if center_occupied[min_idx] == True:
            double_keys.append(t_center_names[min_idx])
        center_occupied[min_idx] = True
        matched_centers.append(t_center_names[min_idx])
        min_dists.append(distances[min_idx])

    if len(double_keys) == 0:
        return dict(zip(mobile_centers.keys(), matched_centers)), dict(zip(mobile_centers.keys(), min_dists))

    # If centers are present multiple times, delete the further distant entry and make None, None
    for doublet in double_keys:
        #print(f'{doublet = }, {matched_centers = }, {type(matched_centers) = }')
        indices = [i for i,x in enumerate(matched_centers) if doublet == x]
        minimum = np.argmin(np.array(min_dists)[indices])
        indices.remove(indices[minimum])
        # Only the non-minimum values will remain in the indices list that need to be re-evaluated
        newdists = np.empty(shape=(n_template_keys))
        for idx in indices:

            for i, coord in enumerate(t_coord):
                if center_occupied[i] == True:
                    newdists[i] = None
                    continue
                newdists[i] = space_distance(list(mobile_center_coords.values())[idx], coord)
            newmindist = np.min(newdists)
            if newmindist < threshold:
                print('Found Alternative')
                center_occupied[np.argmin(newdists)] = True
                min_dists[idx] = newmindist
                matched_centers[idx] = list(mobile_center_coords.keys())[idx]
            else: 
                #print('Removed double occurrence')
                min_dists[idx] = None
                matched_centers[idx] = ''

    return dict(zip(mobile_centers.keys(), matched_centers)), dict(zip(mobile_centers.keys(), min_dists))

def find_center_matches(file, center_dict,  isTarget=False, return_unmatched=False, debug=False):
    # Takes a gesamt file and an center dictionary either of the target or the template and returns the matched other residue with the pairwise distance
    # as a dictionary: {'H1': (653, 1.04) , ...}
    template_pairs, mobile_pairs = read_gesamt_pairs(file, return_unmatched=return_unmatched)
    # Find the closest residue to template centers
    matched_residues = {}
    # is target determines the order of the file reading.
    if not isTarget:
        parsing_dict = template_pairs
    else:
        parsing_dict = mobile_pairs

    if debug: print(f"[DEBUG]: find_center_matches: {file = }, {parsing_dict = }")

    try:
        start, end = min(parsing_dict.keys()), max(parsing_dict.keys())
    except:
        print("NOT MATCHED:", parsing_dict, file)
        return {}
    #start, end = min(parsing_dict.keys()), max(parsing_dict.keys())
    for center_name, center_res in center_dict.items():
        # If the center lies outside the aligned segments, pass empty match (None, None)
        if center_res < start or center_res > end:
            matched_residues[center_name] = (None, None)
            continue
        matched_residues[center_name] = parsing_dict[center_res]

    return matched_residues

def find_best_templates(unknown_gain_obj, gesamt_bin, unknown_gain_pdb: str, sda_templates: dict, sdb_templates: dict, debug=False, template_mode='extent', sda_mode='rmsd'):
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
    a_qvals = []

    for sda_id, sda_pdb in sda_templates.items():
        cmd_string = f'{gesamt_bin} {sda_pdb} {unknown_gain_pdb} -s A/{pdb_start}-{sda_boundary}'
        output = run_logged_command(cmd_string)
        # Read the RMSD value from the ouptut string returned by the function. No files are created here.
        match = re.search(r"RMSD\W+\:\W+[0-9]+\.[0-9]+",output)
        qmatch = re.search(r"Q-score\W+\:\W+[0-9]+\.[0-9]+",output)
        if match is None:
            print("[WARNING]: NO RMSD FOUND:", cmd_string, output)
            val = 100.0 # penalty for non-matching template
            qval = 0.0
        else:
            val = float(match.group(0)[-5:])
            qval = float(qmatch.group(0)[-5:])
        a_names.append(sda_id)
        a_rmsds.append(val)
        a_qvals.append(qval)

    if debug:
        print(f"HERE ARE THE RMSD VALUES FOR SUBDOAMIN A TEMPLATES:\n\t{a_names}\n\t{a_rmsds}\n\t{a_qvals}")
    if sda_mode == 'rmsd':
        best_sda = a_names[a_rmsds.index(min(a_rmsds))]
    if sda_mode == 'q':
        best_sda = a_names[a_qvals.index(max(a_qvals))]

    for sdb_id, sdb_pdb in sdb_templates.items():
        cmd_string = f'{gesamt_bin} {sdb_pdb} {unknown_gain_pdb} -s A/{sdb_boundary}-{pdb_end}'
        output = run_logged_command(cmd_string)
        # Read the RMSD value from the ouptut string returned by the function. No files are created here.
        match = re.search(r"RMSD\W+\:\W+[0-9]+\.[0-9]+",output)
        if match is None:
            print("[WARNING]: NO RMSD FOUND:", cmd_string, output)
            val = 100.0 # penalty for non-matching template
        else:
            val = float(match.group(0)[-5:])
        b_names.append(sdb_id)
        b_rmsds.append(val)
    

    if template_mode == 'extent':
        if debug:
            print("Here are the RMSD values for the EXTENT mode of Subdomain B templates:")
        for i, name in enumerate(b_names):
            print(f"{name}:{b_rmsds[i]}")
            if name == "G4b":
                if b_rmsds[i] < 3.0 : 
                    if debug:
                        print("Selecting G4b template.")
                    best_sdb = "G4b"
                else:
                    if debug:
                        print("G4b match quality is too low for maximalist assignment. Switching to lowest-RMSD template.")
                    template_mode = 'rmsd'

    if template_mode == 'rmsd':
        best_sdb = b_names[b_rmsds.index(min(b_rmsds))]
    
    if debug:
        print(f"[DEBUG] find_best_templates: RESULT\n\t{best_sda}\n\t{best_sdb}")

    return best_sda, best_sdb

def calc_identity(aln_matrix, return_best_aa=False):
    # This takes an alignment matrix with shape=(n_columns, n_sequences) and generates counts based on the identity matrix.
    # Returns the highest non "-" residue count as the most conserved residue and its occupancy based on count("-") - n_struc
    n_struc = aln_matrix.shape[0]
    quality = []
    occ = []
    aa = []
    for col in range(aln_matrix.shape[1]):
        chars, count = np.unique(aln_matrix[:,col], return_counts=True)
        dtype = [('aa', 'S1'), ('counts', int)]
        values = np.array(list(zip(chars,count)), dtype=dtype)
        s_values = np.sort(values, order='counts')

        if s_values[-1][0] == b'-':
            q = s_values[-2][1]
            aa.append(s_values[-2][0])
        else:
            q = s_values[-1][1]
            aa.append(s_values[-1][0])
        x = np.where(chars == '-')[0][0]
        occ.append(n_struc - count[x])
        quality.append(q)
    if not return_best_aa:
        return quality, occ
    if return_best_aa:
        return quality, occ, aa

def get_struc_aln_centers(gain, aln_dict, subdomain='a', threshold=3, silent=False):
    aln_matrix = np.array([list(seq) for seq in aln_dict.values()])
    # Get the identity scores from the alignment
    quality, occ, aa = calc_identity(aln_matrix, return_best_aa=True)
    # The columns here exactly correspond to the template sequence order
    if subdomain.lower() == 'a':
        sse = gain.sda_helices
        d_string = "HELIX "
        sse_type = "H"
    elif subdomain.lower() == 'b':
        sse = gain.sdb_sheets
        d_string = "STRAND"
        sse_type = "S"
    else:
        print("NO SUBDOMAIN specified. EXITING.")
    
    center_quality = {}
    centers = {}
    counter = 1

    for i,element in enumerate(sse):
        if element[1]-element[0] <= threshold:
            #print("Element length below threshold. Skipping.", element)
            continue
        if subdomain =='a' and gain.start+element[0] > gain.subdomain_boundary:
            print("Skipping Subdomain A Helix", element)
            continue

        q = quality[element[0]:element[1]+1]
        label = f'{sse_type}{counter}'
        max_id = element[0]+np.argmax(q)
        max_res = gain.sequence[max_id]

        res_id = gain.start+max_id

        if not silent:
            print(f"{d_string} #{i+1}: {max_res}{res_id} @ SSE residue {max_id-element[0]} | q = {np.max(q)} with res_idx {max_id} | MOST CONSERVED: {aa[max_id]} | PDB-res {gain.start+element[0]+1}-{gain.start+element[1]+1}")
        center_quality[label] = np.max(q)
        centers[label] = max_id
        counter += 1
        pdb_centers = {v:k+gain.start+1 for v,k in centers.items()}
        
    return centers, center_quality, pdb_centers

def get_template_information(identifier, gain_collection, subdomain='a', threshold=3, no_input=True):
    for gain in gain_collection.collection:
        if identifier in gain.name:
            print(gain.name, gain.start, gain.subdomain_boundary, gain.end, "\n")

            if subdomain.lower() == 'a':
                sse = gain.sda_helices
                d_string = "HELIX "
                sse_type = "H"
            elif subdomain.lower() == 'b':
                sse = gain.sdb_sheets
                d_string = "STRAND"
                sse_type = "S"
            else:
                print("NO SUBDOMAIN specified. EXITING.")
        
            #print(sse)
            center_quality = {}
            centers = {}
            counter = 1
            aln_indices = []
            for i,element in enumerate(sse):
                if element[1]-element[0] <= threshold:
                    #print("Element length below threshold. Skipping.", element)
                    continue
                if subdomain =='a' and gain.start+element[0] > gain.subdomain_boundary:
                    print("Skipping Subdomain A Helix", element)
                    continue
                label = f'{sse_type}{counter}'
                q = [ gain.residue_quality[res] for res in range(element[0], element[1]+1)]
                max_id = element[0]+np.argmax(q)
                max_res = gain.sequence[max_id]
                #aln_idx = gain.alignment_indices[max_id]
                res_id = gain.start+max_id+1
                print(f"{d_string} #{i+1}: {max_res}{res_id} @ SSE residue {max_id-element[0]} | q = {np.max(q)} with res_idx {max_id} | {q} | {gain.start+element[0]}-{gain.start+element[1]}")
                if not no_input:
                    confirm = input(f"{d_string} #{i+1}: {max_res}{res_id} @ SSE re {max_id-element[0]} | q={np.max(q)} w res_idx {max_id} | {gain.start+element[0]}-{gain.start+element[1]}. Keep?")
                    if confirm.lower() != "y":
                        print("Skipping this center.");continue
                center_quality[label] = np.max(q)
                centers[label] = max_id
                aln_indices.append(gain.alignment_indices[max_id])
                counter += 1
            pdb_centers = {v:k+gain.start+1 for v,k in centers.items()}
            print("__________")
            return centers, center_quality, aln_indices, pdb_centers
    
def gain_set_to_template(list_of_gains, index_list, template_centers, gesamt_folder, penalty=None, subdomain='sda', return_unmatched_mode='quality', threshold=3, debug=False):
    '''This matches a subselection of GAIN domains (i.e. all ADGRB1 proteins) against a given template with which GESAMT calculation have already been run.
    list_of_gains :         list of GainDomain objects
    index_list :            list of indiced for each GainDomain object according to its index in original collection
    template_centers :      dict of templatecenter residues (with actual PDB-matching residue number, since this is was GESAMT will be taking)
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
    distances:              np.array of shape (len(list_of_gains), #template_centers) filled with distances in A for each center, penalty if unmatched.
    all_matched_centers:    list,  with dict as entries, corresponding to every GainDomain with Anchor label, the residue and the distance : {'H1': (653, 1.04) , ...}
    unindexed_elements:     dict,  gain.name:[[elementstart/start_alignment_column, elementend/end_alignment_column, elementlength], [e2], ...] for the unmatched elements
    unindexed_counter:      int, a total counter of unindexed elements. This can also include non-conserved initital SSE that have no center.
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

    n_anch = len(template_centers.keys())
    distances = np.full(shape=(len(list_of_gains), n_anch), fill_value=penalty)
    all_matched_centers = []
    unindexed_elements = {}
    unindexed_counter = 0
    #for gain_idx in range(len(list_of_gains)):
    # Now, match every single GAIN here to its template via the GESAMT file
    for gain_idx, gain in enumerate(list_of_gains):
        gain_distances, matched_centers = match_gain2subdomain_template(index_list[gain_idx],
                                                                            template_centers=template_centers,
                                                                            gesamt_folder=gesamt_folder,
                                                                            penalty=penalty,
                                                                            debug=debug)
        # check if there are unindexed elements in the GAIN larger than 3 residues
        center_residues = [v[0] for v in matched_centers.values()]
        if debug: print(f"[DEBUG]: gain_set_to_template {center_residues = }")

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
                      f'and {center_residues = }')
        # Introduce a wider detection range for the helical match by widening the SSE interval edge by 1. 
        # This has no relevance to the actual indexing and it just for Detection purposes.
            isMatch = [a in range(element[0]+gain.start-1, element[1]+gain.start+2) for a in center_residues] # <- -1 and +2 for widened edge
            if not np.any(isMatch):
                composed_element = compose_element(element=element, gain=gain, mode=return_unmatched_mode)
                if gain.name not in unindexed_elements:
                    unindexed_elements[gain.name]=[composed_element]
                else:
                    unindexed_elements[gain.name].append(composed_element)

        if debug: print(f"[DEBUG]: gain_set_to_template : {gain_distances = }")
        distances[gain_idx,:] = gain_distances
        all_matched_centers.append(matched_centers)
    return distances, all_matched_centers, unindexed_elements, unindexed_counter

def evaluate_template(template_gain_obj, list_of_gain_obj, gesamt_folder, subdomain, threshold, aln_output=None, gain_indices=None):
    #wrapper function for combining the creation of the structural alignment with the analysis of the alignment.
    structural_alignment = construct_structural_alignment(template_gain_domain=template_gain_obj,
                                                          list_of_gain_obj=list_of_gain_obj,
                                                          gain_indices=gain_indices,
                                                          gesamt_folder=gesamt_folder,
                                                          outfile=aln_output,
                                                          debug=False)
                                                            
    indices, center_quality, centers = get_struc_aln_centers(gain=template_gain_obj,
                                                             aln_dict=structural_alignment,
                                                             subdomain=subdomain,
                                                             threshold=threshold,
                                                             silent=False)
    
    return indices, center_quality, centers, structural_alignment

def analyze_template_matches(template_ids, template_centers, valid_collection, gesamt_folders, receptors, receptor_list):
    # Analyze a set of templates by matching the respective GESAMT pairwise analysis and stacking the information for segment center matches and distances.
    t_occupancies = {}
    t_distances = {}
    unmatched = {}
    unmatched_counters = {}
    y = len(receptors)

    for t_id in template_ids:

        t_centers = template_centers[t_id]

        n_anch = len(t_centers.keys())
        u_list = np.zeros(shape=(y), dtype=dict)
        u_counters = np.zeros(shape=(y), dtype=int)

        center_index = {k:i for i, k in enumerate(t_centers.keys())}
        assigned_center_freq = np.zeros(shape=(y,n_anch))
        all_center_averages =  np.full(shape=(y,n_anch), fill_value=None)
        all_center_occupancy = np.zeros(shape=(y,n_anch))

        if "b" in t_id:
            sd_string = "sda"
        else:
            sd_string = "sdb"

        for fam_idx, r in enumerate(receptors):
            gain_subset =   [ gain for i, gain in enumerate(valid_collection.collection) if receptor_list[i] == r ]
            gain_idx_list = [ i for i,gain in enumerate(receptor_list) if gain == r ]

            element_occupation = {k:0 for k in t_centers.keys()}

            for key, val in element_occupation.items():
                assigned_center_freq[fam_idx, center_index[key]] = float(val)/len(gain_subset)
            
            #gain_set_to_template(list_of_gains, index_list, template_centers, gesamt_folder, penalty=None, subdomain='sda', return_unmatched_mode='quality', threshold=3, debug=False):
            fam_distances, fam_matched_centers, unmatched_elements, unmatched_counter =gain_set_to_template(gain_subset, 
                                                                                                                gain_idx_list, 
                                                                                                                t_centers, 
                                                                                                                gesamt_folders[t_id], 
                                                                                                                penalty=None,
                                                                                                                subdomain=sd_string,
                                                                                                                return_unmatched_mode='index', 
                                                                                                                debug=False)
            mean_dist = np.empty(shape=(n_anch))
            occ = np.zeros(shape=(n_anch))
            
            for j in range(n_anch):
                occ_values = np.array([d for d in fam_distances[:,j] if d is not None])
                if len(occ_values) != 0:
                    mean_dist[j] = round(np.mean(occ_values), 3)
                    occ[j] = round(np.count_nonzero(fam_distances[:,j])/len(gain_idx_list), 3)
            all_center_averages[fam_idx,:] = mean_dist
            all_center_occupancy[fam_idx,:] = occ
            u_counters[fam_idx] = unmatched_counter
            u_list[fam_idx] = unmatched_elements
        
        print(f"Done with Template {t_id}.\n", "_"*30)

        t_distances[t_id] = all_center_averages
        t_occupancies[t_id] = all_center_occupancy
        unmatched[t_id] = u_list
        unmatched_counters[t_id] = u_counters

    return t_distances, t_occupancies, unmatched, unmatched_counters

def write_unmatched_elements(output_file,
                             template_ids,
                             receptor_counts,
                             receptor_list,
                             unmatched):
    with open(output_file, 'w') as outfile:
    
        outfile.write("Temp  Grp   nGrp  alnIdx  nNoMat  avgLen  %unmat\n")
        adress_matrix = [] 
        col_names = {}
        skip = 3

        for t_index, t_id in enumerate(template_ids[skip:]):
            print(t_id)
            u_list = unmatched[t_id]

            for receptor_unmatched_dict in u_list:
                print(len(list(receptor_unmatched_dict.keys())))
                e_length = []
                e_res = []
                res_len = {}
                all_items = []
                
                for lst in receptor_unmatched_dict.values():
                    lengths = [int(i[2]) for i in lst]
                    e_length = e_length+lengths
                    e_res += [i[0] for i in lst]
                    
                    for i in lst:
                        if int(i[0]) not in res_len.keys():
                            res_len[int(i[0])] = [i[2]]
                        else:
                            res_len[int(i[0])].append(i[2])
                        all_items.append(i)

                res_av_len = {k:np.average(v) for k,v in res_len.items()}

                resid, ct = np.unique(e_res, return_counts=True)

                sel_length = len(list(receptor_unmatched_dict.keys())) # How many total are in this selection of unmatched segments
                receptor_name = get_agpcr_type(list(receptor_unmatched_dict.keys())[0])
                for idx, count in enumerate(ct):
                    if count > 0.1*sel_length and res_av_len[resid[idx]] > 3.5: # more than 10% of selection have this

                        unindexed_freq = count/sel_length
                        column_name = f"{receptor_name}-{str(resid[idx]).ljust(4)}"
                        if column_name not in col_names.keys(): 
                            name_idx = len(col_names.keys())
                            col_names[column_name] = name_idx
                            
                        else:
                            name_idx = col_names[column_name]
                        adress_matrix.append( (t_index, name_idx, unindexed_freq) )

                        outfile.write(f"{t_id}{receptor_name.rjust(7)}{str(sel_length).rjust(8)}")
                        outfile.write(f"{str(resid[idx]).rjust(8)}{str(count).rjust(8)}{str(round(res_av_len[resid[idx]],1)).rjust(8)}{str(round(count*100/sel_length)).rjust(7)}%   ")
                        for value in all_items[idx]:
                            outfile.write(str(value).rjust(8))#plt.bar(resid[idx], count)
                        outfile.write("\n")
                        #plt.annotate(f"{round(res_av_len[resid[idx]],1)}", (resid[idx],count))
        outfile.close()
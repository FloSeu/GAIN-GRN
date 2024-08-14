# Functions for finding a template via GainDomain subselection and GESAMT pairwise structural alignments.
import glob, os, re, shutil, math
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from gaingrn.scripts.io import find_anchor_matches
from gaingrn.scripts.io import read_gesamt_pairs, run_logged_command
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
    ax.set_title(f'PCA of MiniBatchKMeans - {name}')
    ax.set_xlabel('PC 0')
    ax.set_ylabel('PC 1')
    if plot3D:
        ax.set_zlabel('PC 2')
    if save:
        plt.savefig(f'{name}_pca.png', dpi=300)

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
    if debug: 
        print(f"[DEBUG]: return find_anchor_matches: {actual_anchors = }")
    all_dist = {k:v[1] for k,v in actual_anchors.items() } #{'H1': (653, 1.04) , ...}
    if debug: 
        print(f"[DEBUG]: return find_anchor_matches: {all_dist = }") #{'S1': 4.56, 'S2': 2.09, ...}
    # From the anchor-residue distances, fill a matrix with the individual distances, assign unmatched anchors a pre-set penalty value
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
            print(gain.name)
            with open(f'{gesamt_folder}/{gain_indices[gain_idx]}.out') as out:
                data = out.readlines()[43]
                print(data)
            
        template_pairs, _ = read_gesamt_pairs(f'{gesamt_folder}/{gain_indices[gain_idx]}.out')
        if not template_pairs: # Check if the dict is empty
            print(f"[NOTE]: GESAMT Alignment has failed. Skipping: {gain.name = } with \n\t{gain.sda_helices =}\n\t{gain.sdb_sheets = }")
            continue
        # This is a dict: { 516: (837, 1.08),... }
        if debug:
            print(f"[DEBUG] {template_pairs = }\n\t{len(gain.sequence)} {gain.start = } {gain.end = }\n\t{template_gain_domain.start = }")
        
        res_matches = {}
        
        for k,v in template_pairs.items():
            if v[0] is not None and v[0]-gain.start >= 0 and v[0]-gain.start < len(gain.sequence) and k-template_gain_domain.start < template_seq_len:
                #print(f"[DEBUG] {v[0] = }")
                res_matches[k-template_gain_domain.start] = gain.sequence[v[0]-gain.start]
        #res_matches = {k-template_gain_domain.start : gain.sequence[v[0]-gain.start] for k,v in template_pairs.items() if v[0] is not None and v[0]-gain.start < len(gain.sequence) and v[0]-gain.start >=0}
        # This is a dict matching the aln columns : {0: "V", 1, "A", 4:"C"}, non-matched residues are not present (None value)
        # map to alignment array
        if debug:
            print(f"[DEBUG] {min(list(res_matches.keys())) = }, {max(list(res_matches.keys())) = }\n\t{msa_array.shape = }")
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


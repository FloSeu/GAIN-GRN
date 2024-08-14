## indexing_utils.py
# Contains functions for handling and analyzing the completely indexed dataset via StAlIndexing class instances

import logomaker
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FixedLocator)
import gaingrn.scripts.io

def get_loops(indexing_dir):
    # Returns a named dict with loop lengths, i.e. {"H1-H2":13, "H8-S1":12}
    inverted_dir = {sse[0] : (sse[1],ki) for ki, sse in indexing_dir.items()} # The begin of each sse is here {0:(13, "H2")}
    loop_dir = {}
    ordered_starts = sorted(inverted_dir.keys())
    for i, sse_start in enumerate(ordered_starts):
        if i == 0: 
            continue # Skip the first and go from the second SSE onwards, looking in N-terminal direction.
        c_label = inverted_dir[sse_start][1]
        n_end, n_label = inverted_dir[ordered_starts[i-1]]
        loop_dir[f"{n_label}-{c_label}"] = sse_start - n_end - 1
    return loop_dir

def get_sse_len(indexing_dir, total_keys):
    # Returns a dict with the length of each SSE in respective GAIN domain.
    len_dir = {x:0 for x in total_keys}
    for ki in indexing_dir.keys():
        start = indexing_dir[ki][0]
        end = indexing_dir[ki][1]
        len_dir[ki] = end - start + 1
    return len_dir

def get_pos_res(pos_dir, gain):
    # Returns a dict with the One-Letter-Code of each SSE position in the respective GAIN domain.
    pos_res = {k : gain.sequence[v-gain.start] for k,v in pos_dir.items() if v is not None and v-gain.start < len(gain.sequence)}
    return pos_res

def match_dirs(single_dir, collection_dir, exclude=[]):
    for k, v in single_dir.items():
        if v in exclude:
            continue
        if k not in collection_dir.keys():
            collection_dir[k] = [v]
            continue
        collection_dir[k].append(v)
    return collection_dir

def plot_hist(datarow, color, name, length):
    max = np.max(datarow)
    try: 
        dens = stats.gaussian_kde(datarow)
    except:
        print(np.unique(datarow))
        return
    fig = plt.figure(figsize=[4,2])
    fig.set_facecolor('w')
    n, x, _ = plt.hist(datarow, bins=np.linspace(0,max,max+1), histtype=u'step', density=True, color='white',alpha=0)
    plt.plot(x, dens(x),linewidth=2,color=color,alpha=1)
    plt.fill_between(x,dens(x), color=color,alpha=0.1)
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    val_string = f'{round(np.average(datarow),2)}±{round(np.std(datarow),2)}'
    plt.text(max, ymax*0.95, name, horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.text(max, ymax*0.8, val_string, horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.text(max, ymax*0.65, f"{round(len(datarow)/length*100, 1)}%", horizontalalignment='right', fontsize=14, verticalalignment='top')
    plt.xlabel('Element Length [Residues]')
    plt.ylabel('Relative density [AU]')
    plt.savefig(f'{name}_hist.svg')
    plt.show()
    plt.close(fig)

def parse_conservation(datarow, length):
    total = len(datarow)
    letters, counts = np.unique(np.array(datarow), return_counts=True)

    resid_counts = {}
    for i, res in enumerate(letters):
            resid_counts[int(counts[i])] = res
    
    sorted_counts = sorted(resid_counts.keys())[::-1]

    occupancy = round(total/length*100, 1)
    conserv_string = []
    residue_occupancies = [ int( x*100 / total ) for x in sorted_counts]
    for idx, occ in enumerate(residue_occupancies):
        if occ >= 5: conserv_string.append(f"{resid_counts[sorted_counts[idx]]}:{occ}%")

    return occupancy, ", ".join(conserv_string)

def construct_identifiers(intervals:dict, center_dir:dict, plddt_values:dict, max_id_dir:dict, name:str, seq=None, gain_start=0, debug=False):
    id_dir = {}
    plddts = {}
    sse_seq = {}
    if debug:
        print("DEBUG",f"{len(plddt_values) = }", f"{len(seq) = }", f"{gain_start = }", sep="\n\t")
    for sse in intervals.keys():
        if sse == 'GPS' :
            continue
        start = intervals[sse][0]
        end = intervals[sse][1]
        if end-start > 45:
            print(f"NOTE: SKIPPDING TOO LONG SSE WITH LENGTH {end-start}\n{name}: {sse}")
            continue
        center_resid = center_dir[f"{sse}.50"]
        first_resid = 50 - center_resid + start
        for k in range(end-start+1):
            if sse not in max_id_dir.keys():
                max_id_dir[sse] = []
            if first_resid+k not in max_id_dir[sse]:
                max_id_dir[sse].append(first_resid+k)
        id_dir[sse] = [first_resid+k for k in range(end-start+1)]
        plddts[sse] = [plddt_values[k] for k in range(start, end+1)]
        if seq is not None:
            sse_seq[sse] = [seq[k-gain_start] for k in range(start, end+1) if k-gain_start<len(seq)]
    if seq is None:
        sse_seq = None
    return max_id_dir, id_dir, plddts, sse_seq

def get_plddt_dir(file='all_plddt.tsv'):
    plddt_dir = {}
    with open(file) as f:
        data = [l.strip() for l in f.readlines()[1:]]
        for l in data:
            i,v  = tuple(l.split("\t"))
            plddt_dir[i] = [float(val) for val in v.split(",")]
    return plddt_dir

def make_id_list(id_dir):
    id_list = []
    for sse in id_dir.keys():
        for res in id_dir[sse]:
            id_list.append(f"{sse}.{res}")
    return id_list #np.array(id_list)

def compact_label_positions(id_collection, plddt_collection, sse_keys, debug=False):
    # Stacks label positions on one another
    label_plddts = {}
    for sse in sse_keys:
        label_plddts[sse] = {}

    for i in range(len(id_collection)):
        gain_positions = id_collection[i]
        plddt_positions = plddt_collection[i]
        if debug: 
            print(i,gain_positions, plddt_positions, sep="\n")
        for sse, v in gain_positions.items():
            if v == []:
                continue
            for j, pos in enumerate(v):
                pos = int(pos)
                if j >= len(plddt_positions[sse]):
                    continue
                if pos not in label_plddts[sse].keys():
                    label_plddts[sse][pos] = [plddt_positions[sse][j]]
                else:
                    label_plddts[sse][pos].append(plddt_positions[sse][j])

    return label_plddts

def construct_id_occupancy(indexing_dirs, center_dirs, length, plddt_dir, names, seqs, starts:list, debug=False):
    newkeys = ['H1','H1.D1','H1.E1','H1.F4','H2','H3','H4','H5','H6','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14']
    id_collection = []
    plddt_collection = []
    seq_collection = []
    all_id_dir = {x:[] for x in newkeys}
    for k in range(length):
        identifier = names[k].split("-")[0]
        plddt_values = plddt_dir[identifier]
        all_id_dir, id_dir, plddts, sse_seq = construct_identifiers(indexing_dirs[k], center_dirs[k], plddt_values, all_id_dir, names[k], seqs[k], starts[k], debug=debug)
        #print(k, sse_seq)
        id_collection.append(id_dir)
        #print(id_dir)
        plddt_collection.append(plddts)
        seq_collection.append(sse_seq)
    print("Completed creating value collection.")
    print(id_collection[0])
    print(plddt_collection[0])

    # Here, parse through the id_dirs to count the occurrence of positions per SSE
    # Dictionary to map any label identifier to a respective position.
    id_map = {}
    i = 0
    for sse in newkeys:
        for res in all_id_dir[sse]:
            id_map[f'{sse}.{res}'] = i 
            i += 1
    
    max_id_list = []
    for i, id_dict in enumerate(id_collection):
        max_id_list.append(make_id_list(id_dict))
    flat_id_list = np.array([item for sublist in max_id_list for item in sublist])
    print("Finished constructing flat_id_list.")
    labels, occ = np.unique(flat_id_list, return_counts=True)
    # Parse through labels, occ to generate the sse-specific data
    occ_dict = {labels[u]:occ[u] for u in range(len(labels))}
    # Transform occ_dict to the same format as label_plddts (one dict per sse):
    label_occ = {}
    for sse in newkeys:
        label_occ[sse] = {int(k[-2:]):v for k,v in occ_dict.items() if sse in k}
    #print(labels, occ)
    label_plddts = compact_label_positions(id_collection, plddt_collection, newkeys, debug=debug)
    label_seq = compact_label_positions(id_collection, seq_collection, newkeys, debug=debug)
    #print(labels)
    return label_plddts, label_occ, label_seq
    #[print(k, len(v)) for k,v in label_plddts.items()]

def mark_seg_cons(stal_indexing, rr_occ, elements, uniprot_id, pdbfile, outfile, fill_b=None):

    # create a b_factor map, mapping segment occupancy to the segmentis in human ADGRA2, Q96PE1
    segment_occ = np.mean(rr_occ, axis=0)
    #print(segment_occ)
    occ_dict = dict(zip(elements,segment_occ))
    for idx, ac in enumerate(stal_indexing.accessions):
        if uniprot_id in ac:
            gain_index = idx
            break
    print(gain_index)

    gain_elements = stal_indexing.indexing_dirs[gain_index]

    res2value = {}

    for label, resid in gain_elements.items():
        segment = label.split(".")[0]
        if "GPS" in segment: continue
        occ = occ_dict[segment]
        res2value[resid] = occ

    gaingrn.scripts.io.label2b(pdbfile, outfile, res2value=res2value, fill_b=fill_b)
    print("gaingrn.scripts.indexing_utils.mark_seg_cons : Done.")

def mark_pos_cons(stal_indexing, pos_occ_dict, uniprot_id, pdbfile, outfile, fill_b=None):
    # create a b_factor map, mapping POSITION occupancy in a given GAIN
    for idx, ac in enumerate(stal_indexing.accessions):
        if uniprot_id in ac:
            gain_index = idx
            break
    print(gain_index)

    gain_elements = stal_indexing.indexing_dirs[gain_index]

    res2value = {}

    for label, resid in gain_elements.items():
        if "GPS" in label: continue
        occ = pos_occ_dict[label]/14435 # divide absolute counts by number of total GAINs to normalize.
        res2value[resid] = occ

    gaingrn.scripts.io.label2b(pdbfile, outfile, res2value=res2value, fill_b=fill_b)
    print("gaingrn.scripts.indexing_utils.mark_pos_cons : Done.")

def get_elem_seq(uniprot, stal_indexing, valid_collection, segment):
    # Get the sequence of a specific element from a specific protein from the dataset.
    for gain in valid_collection.collection:
        #print(gain.name)
        if uniprot in gain.name:
            print(gain.name, "found.")
            break
    for i,ac in enumerate(stal_indexing.accessions):
        if uniprot == ac:
            idx = i
            break
    myseg = [(k,v) for k,v in stal_indexing.indexing_dirs[idx].items() if k.split(".")[0] == segment]
    # get min and max of the element
    print(myseg[0], myseg[-1])


def plot_segment_statistics(sse, xvals=None, y_plddt=None, y_occupancy=None, savename=None, show=False):
    fig, ax = plt.subplots(figsize=[5,2])
    fig.set_facecolor('w')
    ax.xaxis.set_minor_locator(MultipleLocator(1)) #AutoMinorLocator())
    ax.xaxis.set_major_locator(FixedLocator([a for a in range(2,100,3)]))#MultipleLocator(3)))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=6)

    if y_plddt is not None:
        plt.bar(xvals,y_plddt, color='silver', alpha=0.7)

    if y_occupancy is not None:
        plt.plot(xvals, y_occupancy, color='dodgerblue')
    
    if y_occupancy is not None and y_plddt is not None:
        normalized_plddt = np.array(y_plddt)*np.array(y_occupancy)
        plt.bar(xvals, normalized_plddt, color='xkcd:lightish red', alpha=0.1)

    plt.title(f'Element Composition ({sse})')
    plt.yticks(ticks = [0, 0.2, 0.4, 0.6, 0.8, 1], labels = ['0%', '20%', '40%', '60%', '80%', '100%'])
    #plt.ylabel('')
    ax.set_xticklabels([f'{sse}.{str(int(v))}' for v in ax.get_xticks()], rotation=90)
    if savename is not None:
        plt.savefig(f'', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_logo_segment(dataframe, sse, threshold=0.05, savename=None):
    # Note down the first and last row where the occupation threshold is met.
    firstval = None
    for i, r in dataframe.iterrows():
        if np.sum(r) > threshold: 
            if firstval is None:
                firstval = i
            lastval = i
    print(firstval, lastval)
    subframe = dataframe.truncate(before=firstval, after=lastval)

    # With the specified interval, create the Logoplot
    fig, ax = plt.subplots(figsize=[5,2])
    cons_logo = logomaker.Logo(subframe,
                                ax=ax,
                                color_scheme='chemistry',
                                show_spines=False,
                                font_name='FreeSans')

    fig.set_facecolor('w')

    ax.xaxis.set_minor_locator(MultipleLocator(1)) #AutoMinorLocator())
    ax.xaxis.set_major_locator(FixedLocator([a for a in range(2,100,3)]))#MultipleLocator(3))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=6)
    ax.set_xticklabels([f'{sse}.{str(int(v))}' for v in ax.get_xticks()], rotation=90)

    cons_logo.draw()

    fig.tight_layout()
    fig.set_facecolor('w')

    plt.savefig(savename, bbox_inches='tight')
    plt.close(fig)
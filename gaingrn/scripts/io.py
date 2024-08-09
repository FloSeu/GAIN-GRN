## scripts/io.py
#   contains functions for file input/output operations and running commands like STRIDE as command line.

import os, json, re, glob, shlex
from subprocess import PIPE, Popen
import numpy as np
import pandas as pd


# executing binaries

def run_command(cmd, out_file=None, cwd=None):
     # wrapper for running a command line argument
    if not cwd:
        cwd = os.getcwd()
    if out_file:
        out = open(out_file, 'w')
        out.flush()
        p = Popen(shlex.split(cmd), stdout=out, stderr=PIPE, bufsize=10, universal_newlines=True, cwd=cwd)
    else:
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, bufsize=10, universal_newlines=True, cwd=cwd)
        for line in p.stdout:
            print(line)
    if out_file:
        for line in p.stderr:
            print(line)

    exit_code = p.poll()
    # Discard all outputs
    _ = p.communicate()

    if out_file:
        out.close()

    p.terminate()

    return exit_code

def run_logged_command(cmd, logfile=None, outfile=None):
    # The command would be the whole command line, in this case the gesamt command.
    # also : shlex.split("cmd")
    p = Popen(cmd, shell=True,
            stdout=PIPE,
            stderr=PIPE,
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

#read_write

def read_sse_loc(file):
    '''
    STRIDE output file parsing function, returns the FULL sequence of SSE data (N->C direction)
    Parameters : 
        file : str, required
            STRIDE file to be read. 
    Returns 
        sseDict : dict
            Dictionary containing all SSE with respective names and intervals by residue
    '''
    sseDict = {}

    with open(file) as f:

        for l in f.readlines():

            if l.startswith("LOC"):     # ASG is the per-residue ASSIGNMENT of SSE
                                        # LOC is already grouped from [start - ]
                items = l.split(None)   # [24] has the SSE one-letter code
                sse, first, last = items[1], int(items[3]), int(items[6])

                if sse not in sseDict.keys():
                    sseDict[sse] = [(first, last)]
                else:
                    sseDict[sse].append((first,last))
    return sseDict


def read_sse_asg(file):
    '''
    STRIDE output file parsing function, used for modified stride files contaning outliers as lowercase letters (H - h, G - g)
    Parameters : 
    file : str, required
            STRIDE file to be read. 
    Returns 
        residue_sse : dict
            DICT containing a sequence of all letters assigned to the residues with the key being the present residue
        outliers    : dict
            DICT containing all outlier residues outside of 2 sigma with the respectve float multiple of StdDev
    '''

    with open(file) as f:
        lines = f.readlines()
    asgs = [l for l in lines if l.startswith("ASG")]

    first_res = int(asgs[0].split(None)[3])
    last_res = int(asgs[-1].split(None)[3])
    residue_sse = {k:"X" for k in range(first_res, last_res+1)}
    outliers = {}
    for l in asgs:              # ASG is the per-residue ASSIGNMENT of SSE
                                # LOC is already grouped from [start - ]
        items = l.split(None)   # [24] has the SSE one-letter code
        # Example lines:
        # items[i]:
        #  0    1 2    3    4    5             6         7         8         9        10
        #ASG  THR A  458  453    E        Strand   -123.69    131.11       4.2      ~~~~
        #ASG  SER A  459  454    e        Strand    -66.77    156.86      10.4      2.97
        # 3 is the PDB index, 4 is the enumerating index, this is crucial for avoiding offsets, always take 3
        residue_sse[int(items[3])] = items[5]
        # if there is missing keys, just label them "X", since these are residues skipped by AlphaFold2 (i.e. "X")
        if len(items) >= 11 and items[10] != "~~~~":
            outliers[int(items[3])] = float(items[10])
    return residue_sse, outliers


def get_stride_seq(file):
    seq_lines = [l for l in open(file).readlines() if l.startswith("SEQ")]
    return list("".join([l.split()[2] for l in seq_lines]))


def read_stride_angles(file, filter_letter=None):
    '''
    STRIDE output file parsing function, used for modified stride files contaning outliers as lowercase letters (H - h, G - g)
    Parameters : 
    file : str, required
            STRIDE file to be read. 
    filter_letter : str, optional
            Filters entries to match a pre-assigneed secondary structure letter (E, H, ...) --> items[3]
    Returns 
        residue_sse : dict
            DICT containing PHI and PSI float values for each residue number (PDB) as key.
    '''
    # Example lines:
    # items[i]:
    #  0    1 2    3    4    5             6         7         8         9        10
    #
    #ASG  SER A  459  454    E        Strand    -66.77    156.86      10.4      ~~~~
    # 3 is the PDB index, 4 is the enumerating index, this is crucial for avoiding offsets, always take 3  
    with open(file) as f:
        asgs = [l for l in f.readlines() if l.startswith("ASG")]

    angles = {}
    for l in asgs:
        items = l.split(None)   # [24] has the SSE one-letter code
        if filter_letter is None or filter_letter == items[5]:
            angles[int(items[3])] = [float(items[7]), float(items[8])]

    return angles


def get_angle_outlier(sse:list, stride_file:str, phi_mean_sd, psi_mean_sd, psi_prio=True):
    # For strand outliers, prioritize PSI over PSI - with precalculated values of PHI and PSI mean+SD:
    angles = read_stride_angles(stride_file)
    phipsi = [phi_mean_sd, psi_mean_sd]
    sse_angles = np.array([ angles[i] for i in range(sse[0],sse[1]) ])

    if psi_prio:
        order = [0,1]
    else:
        order = [1,0]
    for i in order:
        xangles = [ x+360 if x<0 else x for x in sse_angles[:,i]] # remove negative values for wrapping in negative angle values
        max_deviance = np.argmax( [abs(x - phipsi[i][0]) for x in xangles] )
        #print(f"{abs(xangles[max_deviance] - phipsi[i][0]) = } | {2*phipsi[i][1] = }")
        if abs(xangles[max_deviance] - phipsi[i][0]) > 2*phipsi[i][1]:
            return sse[0]+max_deviance
    # If not outside 2 sigma, return None.
    return None


def read_seq(file, return_name=False):
    '''
    Read a sequence from a FASTA file, return sequence name if specified.
    Parameters:
        file : str, required
            The FASTA file to be read
        return_name : bool, optional
            return sequence name if true. Default = False
    Returns:
        seq : str
            The sequence of the FASTA file
        name : str
            The name of the sequence specified in the first line.
    '''
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if return_name == False:
                    continue
                name = line[1:].strip()
            else:
                seq = line.strip(" \t\r\n")
                if return_name == False:
                    return seq

                return name, seq


def read_multi_seq(file):
    '''
    Build a sequences object from a FASTA file contaning multiple sequences.

    Parameters:
        file : str, required
            The FASTA file to be read
    Returns:
        sequences : object
        A list contaning one tuple per sequence with (name, sequence)
    '''
    with open(file) as f:
        data = f.read()
        entries = data.strip().split(">")
        entries = list(filter(None, entries))
    sequences = np.empty([len(entries)], dtype=tuple)

    for i, entry in enumerate(entries):
        name, sequence = entry.strip().split("\n")
        sequences[i] = (name, sequence)

    return sequences


def read_alignment(alignment, cutoff=-1):
    '''
    Load all the data from an alignment into a matrix, stopping at the cutoff column

    Parameters:
        alignment :     str, required
            An Alignment file in FASTA format to be read
        cutoff :        int, required
            The index of the last alignment column to be read

    Returns:
        sequences : dict
            a dictionary with {sequence_name}:{sequence} as items
    '''
    sequences = {}

    with open(alignment) as f:

        data = f.read()
        seqs = data.split(">")

        for seq in seqs[1:]: # First item is empty

            sd = seq.splitlines()
            try:
                sd[0] = sd[0].split("/")[0]
            except:
                pass

            sequences[sd[0]] = "".join(sd[1:])[:cutoff]

    return sequences


def read_quality(jal):
    '''
    extracts ONLY BLOSUM62 quality statements from the specified annotation file
    This jal file can be generated by exporting Annotation from an Alignment in JALVIEW

    Parameters:
        jal : str, required
            A JALVIEW exported annotation file. The file extension is arbitrary.

    Returns:
        cut_data : list
            A list contaning all the Blosum62 quality values per alignment column
    '''
    with open(jal) as annot:

        data = None

        for line in annot:

            if "Blosum62" in line[:200]:
                data = line.strip(" \t\r\n").split("|")
            else:
                continue

        if not data: # Sometimes, BLOSUM62 data is not contained in the annotation file

            print(f"ERROR: Blosum62 indication not found in {jal}")
            return None

    # Process the raw data into a list
    cut_data=[]
    [cut_data.append(float(i.split(",")[1][:5])) for i in data if len(i) > 0]

    return cut_data


def find_pdb(name, pdb_folder):
    # Finds a PDB within a directory containing the UniProtKB Accession of the provided Gain name.
    identifier = name.split("-")[0]
    target_pdb = glob.glob(f"{pdb_folder}/*{identifier}*.pdb")[0] # raises IndexError if not found.
    return target_pdb


def find_stride_file(name, path="stride_out/*_0.stride"):
    '''
    Finds the STRIDE file in a collection of stride files,
    then reads SSE info from this found file via read_sse_loc()
    Used in the base dataset calculation

    Parameters:
        name : str, required
            The name of the sequence, corresponding to the search string
        path : str, optional
            The glob.glob() string to find the STRIDE file collection. default = stride_out/*_0.stride

    Returns:
        sse_dict : dict
            The dictionary containng SSE data as in read_sse_loc()
    '''
    stride_files = glob.glob(path) #_0 indicates that only the best model SSE data is evaluated
    strides = [st for st in stride_files if name in st]

    if len(strides) == 0:
        print("ERROR: Stride files not found in here. {name = }")
        return None

    sse_dict = read_sse_loc(strides[0])
    return sse_dict


def filter_by_list(sequences, selection):
    # Same as filter_by_receptor, instead using a list as input
    new_list = []
    for seq_tup in sequences:
        for it in selection:
            if it in seq_tup[0]:
                new_list.append(seq_tup)
    return new_list


# DETECTION AND SELECTION BLOCK

def filter_by_receptor(sequences, selection):
    # Filter a selection for the receptor
    new_list = []
    for seq_tup in sequences:
        if selection in seq_tup[0]:
            new_list.append(seq_tup)
    return new_list


def write2fasta(sequence, name, filename):
    '''
    Construct a standard sequence fasta file

    Parameters:
        sequence: str, required
            The string of the one-letter amino acid sequence
        name:     str, required
            The name to be put into the header of the FASTA
        filename: str, required
            The file name'''
    with open(filename, 'w') as fa:
        fa.write(f'>{name}\n{sequence}')
    print(f'NOTE: Written {name} to fasta in {filename}.')


def run_mafft(mafft_bin, args, copied_fasta):
	# call MAFFT twice, once for the map (where the truncating can be refined), once for outputting alignment
	# the fasta should be in the outdir/alignment, since the map will be created there too
	mafft_map_command = f"{mafft_bin} --add {copied_fasta} --keeplength --thread {args.nt} --mapout {args.source_alignment}"
	mafft_aln_command = f"{mafft_bin} --add {copied_fasta} --keeplength --thread {args.nt} {args.source_alignment}"

	run_command(mafft_map_command, out_file=f"{args.outdir}/alignment/tmp.fa")
	run_command(mafft_aln_command, out_file=f"{args.outdir}/alignment/appended_alignment.fa")

	return f"{copied_fasta}.map"


def run_stride(pdb_file, out_file, stride_bin):
	# Executes the STRIDE binary via the wrapper function
	stride_command = f'{stride_bin} {pdb_file} -f{out_file}'
	run_command(stride_command)


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
        return {}
    #start, end = min(parsing_dict.keys()), max(parsing_dict.keys())
    for anchor_name, anchor_res in anchor_dict.items():
        # If the anchor lies outside the aligned segments, pass empty match (None, None)
        if anchor_res < start or anchor_res > end:
            matched_residues[anchor_name] = (None, None)
            continue
        matched_residues[anchor_name] = parsing_dict[anchor_res]

    return matched_residues


def save2json(distances, names, savename):
    # stores an RMSD matrix to a JSON file
    df = pd.DataFrame(distances, index = list(names), columns = list(names))
    df = df.to_json(orient='index')
    with open(f"../{savename}.json",'w') as c:
        c.write(json.dumps(df))


def get_agpcr_type(name):
    # Queries the protein name to find the receptor type. If not found, returns "X" to indicate unknown type
    queries = [('AGR..', name, lambda x: x[-1][-2:]),
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
                ('adhesion.?g.*coupled.?receptor...', name.lower(), lambda x: x[-1][-2:].upper())
                ]
    for pattern, searchstring, output in queries:
        match = re.findall(pattern, searchstring)
        if match != []:
            #if output(match) == '': print(name)
            return output(match)
    return 'X'
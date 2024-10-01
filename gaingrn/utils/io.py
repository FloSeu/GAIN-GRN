## utils/io.py
#   contains functions for file input/output operations and running commands like STRIDE as command line.

import os, json, re, glob, shlex, shutil, requests, tarfile
from subprocess import PIPE, Popen
import numpy as np
import pandas as pd
import multiprocessing as mp

# Get the data; the link might change TODO
def download_data(url='https://zenodo.org/uploads/12515545/gaingrn_data.tgz', target_directory=f"{os.getcwd().replace('/gaingrn','/')}"):
    # Ensure the target directory exists 
    os.makedirs(target_directory, exist_ok=True)
    # Download the file 
    response = requests.get(url, stream=True)
    response.raise_for_status() # Check for request errors
    # # Define the path for the temporary file 
    temp_file_path = os.path.join(target_directory, "gaingrn_data.tgz") 
    # Write the content to a temporary file
    with open(temp_file_path, 'wb') as file: 
        for chunk in response.iter_content(chunk_size=8192): 
            file.write(chunk) 
    # Extract the tar.gz file
    with tarfile.open(temp_file_path, 'r:gz') as tar: 
        tar.extractall(path=target_directory)

    print("Download and Extraction complete.")

# Get the PDBs; the link might change TODO
def download_pdbs(url='https://zenodo.org/uploads/12515545/agpcr_gains.tgz', target_directory=str):
    # Ensure the target directory exists 
    os.makedirs(target_directory, exist_ok=True)
    # Download the file 
    response = requests.get(url, stream=True)
    response.raise_for_status() # Check for request errors
    # # Define the path for the temporary file 
    temp_file_path = os.path.join(target_directory, "gaingrn_data.tgz") 
    # Write the content to a temporary file
    with open(temp_file_path, 'wb') as file: 
        for chunk in response.iter_content(chunk_size=8192): 
            file.write(chunk) 
    # Extract the tar.gz file
    with tarfile.open(temp_file_path, 'r:gz') as tar: 
        tar.extractall(path=target_directory)

    print("Download and Extraction complete.")

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

def compile_stride_mp_list(pdbs, stride_folder, stride_bin):
    # Compiles a list of arguments for multithreaded STRIDE execution
    stride_mp_list = []
    
    for pdb in pdbs:
        pdb_name = pdb.split("/")[-1]
        name = pdb_name.split("_unrelaxed_")[0]
        out_file = f"{stride_folder}/{name}.stride"
        arg = [pdb, out_file, stride_bin]
        
        stride_mp_list.append(arg)
        
    return stride_mp_list

def execute_stride_mp(stride_mp_list, n_threads=10):
    # multiprocessed variant wrapper
    stride_pool = mp.Pool(n_threads)
    stride_pool.map(run_stride, stride_mp_list)
    print("Completed mutithreaded creation of STRIDE files!")

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

def read_plddt_tsv(file='all_plddt.tsv'):
    # Load the pLDDT file into a dictionary
    plddt_dir = {}
    with open(file) as f:
        data = [l.strip() for l in f.readlines()[1:]]
        for l in data:
            i,v  = tuple(l.split("\t"))
            plddt_dir[i] = [float(val) for val in v.split(",")]
    return plddt_dir

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

def label2b(pdbfile, outfile, res2label, clear_b=False):
    data = open(pdbfile).readlines()
    newdata = []
    for l in data:
        if not l.startswith("ATOM"):
            newdata.append(l)
            continue
        if not l[13:15] == "CA" or int(l[22:26]) not in res2label.keys():
            #print(l[13:14], int(l[22:26]))
            if clear_b:
                k = l[:60]+"      "+l[67:]
            newdata.append(k)
            continue
        k = l[:60]+res2label[int(l[22:26])].rjust(6)+l[67:]
        newdata.append(k)
    open(outfile, 'w').write("".join(newdata))
    print(f"Written residue labels to PDB file CA entries : {outfile}")

def score2b(input_pdb, output_pdb, metric):
    # Write a given metric to a PDB file
    
    # line [61:66] = xx.xx b factor
    # set to zero if not in metric, set to val otherwise
    with open(input_pdb) as ipdb:
        data = ipdb.readlines()
    newdata = []
    for l in data:
        if not l.startswith("ATOM"):
            newdata.append(l)
            continue
        
        resid = int(l[22:26])

        if resid not in metric.keys():
            l = l[:61]+"00.00"+l[66:]
            newdata.append(l)
            continue

        l = l[:61]+f'{metric[resid]:5.2f}'+l[66:]
        newdata.append(l)

    with open(output_pdb, 'w') as opdb:
        opdb.write("".join(newdata))
    
    print("Done.")

def grn_score2b(pdb, outpdb, rev_idx_dir, score_dict):
    # Write a given score to a PDB file. For this, we need a GAIN that has a corresponding rev_idx_dir to adress GAIN_GRN label:position
    with open(pdb) as inpdb:
        pdb_data = inpdb.readlines()

    new_data = []
    for line in pdb_data:

        if not line.startswith("ATOM"):
            new_data.append(line)
            continue

        resid = int(line[22:26])

        try:
            position_label = rev_idx_dir[resid] # i.e. "H2.54" account for 0-indexed residues in the dictionary!
            try: 
                b = score_dict[position_label]
            except KeyError:
                b = 0.0

        except KeyError: 
            b = -20.0 # negative value for non-structured residues

        new_data.append(line[:61]+"%6.2f"%(b)+line[68:])

    with open(outpdb,"w") as opdb:
        for line in new_data:
            opdb.write(line)
    print("Done.")

def grn2csv(res2label, outfile, target_gain): 
    with open(outfile, "w") as csv:
        csv.write("RESNR,RESNAME,LABEL\n")
        for k in range(target_gain.start, target_gain.end+1):
            if k in res2label.keys():
                csv.write(f"{k},{target_gain.sequence[k-target_gain.start]},{res2label[k]}\n")
            else:
                csv.write(f"{k},{target_gain.sequence[k-target_gain.start]},\n")

# run commands


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

# Function for Parsing out specific Files from the overall dataset based on selection
def grab_selection(parse_string, stride_path, pdb_list, sequences, profile_path, target_dir, seqs=None):
    # grabs PDB file, stride file, profiles, sequence from FASTA and copies to target dir.
    if seqs is None:
        sub_seqs = [seq for seq in sequences if parse_string.lower() in seq[0].lower()]
    else: sub_seqs = seqs
    print(f"Found {len(sub_seqs)} sequences.")
    strides = glob.glob(stride_path+"*.stride")#
    profiles = glob.glob(profile_path+"*.png")
    
    sub_strides = []
    sub_profiles = []
    sub_pdbs = []
    
    for seq in sub_seqs:
        ac = seq[0].split("-")[0]
        [sub_profiles.append(prof) for prof in profiles if ac in prof]
        [sub_strides.append(stride) for stride in strides if ac in stride]
        [sub_pdbs.append(pdb) for pdb in pdb_list if ac in pdb]
    
    for prof in sub_profiles:
        name = prof.split("/")[-1]
        shutil.copyfile(prof, target_dir+"profiles/"+name)
    
    for stride in sub_strides:
        name = stride.split("/")[-1]
        shutil.copyfile(stride, target_dir+"strides/"+name)
    
    for pdb in sub_pdbs:
        name = pdb.split("/")[-1]
        shutil.copyfile(pdb, target_dir+"pdbs/"+name)
        
    for seq in sub_seqs:
        write2fasta(seq[1]+"\n", seq[0], target_dir+"seqs/"+seq[0]+".fa")
        
    print(f"Copied {len(sub_pdbs)} PDB files, {len(sub_strides)} STRIDE files,",
          f" {len(sub_profiles)} Profiles and {len(sub_seqs)} Sequences",
          f"for Selection {parse_string}")

def get_gain(identifier, a_gain_collection, return_index=False):
    # pick a GainDomain object with an identifier from a GainCollection objects
    for i, gain in enumerate(a_gain_collection.collection):
        if identifier in gain.name:
            if return_index:
                return gain, i
            return gain
    print("ERROR: GAIN not found!")
    return None

def check_3rd_party(GESAMT_BIN, STRIDE_BIN):
    GESAMT_BIN = __run_check(GESAMT_BIN, 'GESAMT_BIN', '/home/username/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt')
    STRIDE_BIN = __run_check(STRIDE_BIN, 'STRIDE_BIN', '/home/username/lib/stride/stride')
    return GESAMT_BIN, STRIDE_BIN

def __run_check(bin, bin_name, example_path):
    if bin is None:
        bin = os.environ.get(bin_name)
        if bin is None:
            bin = input(
                f"You need a '{bin_name.rstrip('_BIN'.lower())}' binary in your system, e.g. in '{example_path}' ."
                f"\nPlease type the path to the binary or set the value of '{bin_name}=None' above this code-block to the path'.")
    else:
        if not os.path.exists(bin):
            raise FileNotFoundError(f"Can't find the '{bin_name}' you provided: '{bin}'")
    return bin

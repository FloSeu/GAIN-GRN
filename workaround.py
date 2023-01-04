''' A new workaround file for the workflow
    PROBLEM:
        Upon adding more distantly related GPS-containing proteins (i.e. PKD1 or invertebrate proteins), the MAFFT command run via
        "mafft --keelength might" delete residues from the original sequence
        There therefore needs to be a routine checking for the following:
        - evaluate the .map file where residues are truncated
        - if truncation exists:
            modify the sequence and the STRIDE information into an internal, truncated version
            check if the truncated residues are within a SSE - if so, spit out a warning
            Create a GAIN indexing based on the internal truncated version --> new "stride", and new "sequence"
            In case of no SSE truncations, this should be sufficient without mapping back. '''
import numpy as np
from sse_func import read_sse_asg
def check_internal_truncation(map_file):
    # checks the mapfile and returns a boolean matrix of length = len(sequence) where the "True" indicated truncated residues.
    # mapfile structure:
    has_truncation = False
    ''' MAPFILE STRUCTURE
        >U4PRL9-U4PRL9_CAEEL-GPSdomain-containingprotein-Caenorhabditis_elegans.
        # letter, position in the original sequence, position in the reference alignment
        M, 1, 403
        S, 2, 404 '''
        # A truncation exists where the last number is "-", since the residue is not represented within the alignment
    with open(map_file) as m:
        data = m.readlines()[2:]
    is_truncated = np.zeros([len(data)], dtype=bool)
    aln_start_res = None

    for i,line in enumerate(data):
        if line.strip().split(", ")[2] == "-":
            is_truncated[i] = 1
            has_truncation = True
            continue
        if aln_start_res is None:
            aln_start_res = int(line.strip().split(", ")[2])
    
    return aln_start_res, is_truncated, has_truncation

def check_truncated_SSE(sequence, sse_sequence, is_truncated):
    # The one-letter code must not be: GEH
    trunc_stride = sse_sequence[is_truncated == True]
    if "G" in trunc_stride or "E" in trunc_stride or "H" in trunc_stride:
        # Detailed residue search:
        sse_conflict_list = []
        for i, tr_idx in enumerate(is_truncated):
            if tr_idx and sse_sequence[i] in "GEH":
                sse_conflict_list.append(f"Residue {sequence[i]} {i+1} : sse-code {sse_sequence[i]}")
        print("WARNING: Found SSE residue within the truncated region. Please check location of the following residues:", "\n".join(sse_conflict_list))
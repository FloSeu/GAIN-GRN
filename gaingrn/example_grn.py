# THIS IS AN EXAMPLE FILE FOR ASSIGNING THE GAIN_GRN INDEXING TO A SINGLE MODEL.
#       In this case, we feed in the GAIN pdb from human PKD1 alongside its STRIDE output file

# LOCAL IMPORTS
import gaingrn.scripts.assign
from gaingrn.scripts.gain_classes import GainDomainNoAln
import os

try: 
    GESAMT_BIN = os.environ.get('GESAMT_BIN')
except:
    GESAMT_BIN = '/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt'

# First, generate the GainDomain object fromt the STRIDE file
pkd_gain = GainDomainNoAln(start=None,
                           subdomain_boundary=None, 
                           end=None, 
                           fasta_file=None, 
                           name='hPKD1', 
                           sequence=None, 
                           explicit_stride_file="../data/example/PKD1_1.stride")

# With the object, feed in the PDB for structural alignment and the template data
element_intervals, element_centers, residue_labels, unindexed_elements, params = gaingrn.scripts.assign.assign_indexing(pkd_gain, 
                                                                            file_prefix=f"../data/example/hpkd1", 
                                                                            gain_pdb="../data/example/PKD1_HUMAN_unrelaxed_rank_1_model_3.pdb",
                                                                            template_dir='../data/template_pdbs/',
                                                                            gesamt_bin=GESAMT_BIN,
                                                                            debug=False, 
                                                                            create_pdb=True,
                                                                            template_json='../data/template_data.json',
                                                                            hard_cut={"S2":7,"S6":3,"H5":3},
                                                                            patch_gps=True
                                                                            )
offset = 2271 # resid 1 in PDB = 2272 in Sequence
offset_labels = {k:v+offset for k,v in residue_labels.items()}
print(offset_labels)
with open("../data/example/pkd1_grn.csv", "w") as c:
    c.write("GRN,residue\n")
    for k,v in offset_labels.items():
        c.write(f"{k},{v}")
        c.write("\n")
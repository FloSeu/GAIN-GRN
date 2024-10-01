import unittest
import sys, os, re, glob

sys.path.append('/home/hildilab/agpcr_nom/repo/')

import gaingrn.utils.alignment_utils
import gaingrn.utils.io
from gaingrn.utils.gain_classes import GainCollection, GainDomainNoAln
import gaingrn.utils.structure_utils
import gaingrn.utils.template_utils
import gaingrn.utils.assign
import pandas as pd
from gaingrn.utils.variant_classes import *
# Test functions have to start with "test_"

class TestBinaries(unittest.TestCase):

    #STRIDE_BIN = '/home/hildilab/lib/stride/stride'
    #GESAMT_BIN = '/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt'

    def test_stride(self):
        
        STRIDE_BIN = os.environ.get('STRIDE_BIN')
        pdb_file = "test_data/A_A0A2Y9F628.pdb"
        out_file = "stride_test.out"
        self.assertTrue(os.path.isfile(STRIDE_BIN))
        stride_command = f"{STRIDE_BIN} {pdb_file} -f{out_file}"
        exit_code = gaingrn.utils.io.run_command(stride_command)
        self.assertTrue(exit_code == 0)
        self.assertTrue(os.path.isfile(out_file))
        with open(out_file) as of:
            stride_data = of.read()
        helix1 = """LOC  AlphaHelix   ASP   414 B      GLN    424 B"""
        self.assertTrue(helix1 in stride_data)
        os.remove(out_file)

    def test_gesamt(self):

        GESAMT_BIN = os.environ.get('GESAMT_BIN')
        template_pdb = "test_data/A_A0A2Y9F628.pdb"
        mobile_pdb = "test_data/Q8IWK6_Q6UXK9_Q86SQ5_Q8TC55-AGRA3_HUMAN-AGRA3-Homo_sapiens_gain.pdb"

        out_file = "gesamt.test.out"

        self.assertTrue(os.path.isfile(GESAMT_BIN))
        gesamt_command = f'{GESAMT_BIN} {template_pdb} {mobile_pdb}'
        gaingrn.utils.io.run_command(gesamt_command, out_file=out_file)

        self.assertTrue(os.path.isfile(out_file))
        with open(out_file) as of:
            gesamt_data = of.read()

        qscore = re.compile(r"Q-score +: 0\.3[0-9]+")
        rmsd = re.compile(r"RMSD *: 0\.1[0-9]+")
        qmatch = re.search(qscore, gesamt_data).group()
        rmsdmatch = re.search(rmsd, gesamt_data).group()
        print("\n== GESAMT ALIGNMENT METRICS ==", qmatch, rmsdmatch, sep="\n")

        self.assertTrue(type(qmatch) == str)
        self.assertTrue(type(rmsdmatch) == str)

        self.assertTrue("Gesamt:  Normal termination" in gesamt_data)
        os.remove(out_file)

class TestFunctions(unittest.TestCase):
    # Test underlying large functions from sse_func
    def test_assign_indexing(self):

        GESAMT_BIN = os.environ.get('GESAMT_BIN')

        # First, generate the GainDomain object fromt the STRIDE file
        pkd_gain = GainDomainNoAln(start=None,
                                subdomain_boundary=None, 
                                end=None, 
                                fasta_file=None, 
                                name='hPKD1', 
                                sequence=None, 
                                explicit_stride_file="../data/example/PKD1_1.stride")

        # With the object, feed in the PDB for structural alignment and the template data
        print("\nIndexing human PKD1 onto the GAIN-GRN. Expect a number of unindexed segments on the extra Subdomain...")
        element_intervals, element_centers, residue_labels, unindexed_elements, params = gaingrn.utils.assign.assign_indexing(pkd_gain, 
                                                                                    file_prefix=f"hpkd1/", 
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
        self.assertTrue(os.path.isdir("./hpkd1"))
        self.assertTrue("H2" in element_intervals.keys())
        print("Found the following indexed segments: ", element_intervals.keys(), sep="\n")
        self.assertTrue("H2.50" in residue_labels.keys())
        print("Example GRN label: H2.50: ", residue_labels["H2.50"])
        print(f"The GAIN boundaries: \n\tN:{pkd_gain.start}\n\tA/B boundary:{pkd_gain.subdomain_boundary}\n\tC: {pkd_gain.end}")
        print(f"Found {len(unindexed_elements)} unindexed segments")
        print(f"Since this is an unknown GAIN domain without assigned receptor type, the following templates were used:\n\tSubdomain A template: {params['sda_template']} \
              \n\tSubdomain B template: {params['sdb_template']}")

        with open("./hpkd1/pkd1_grn.csv", "w") as c:
            c.write("GRN,residue\n")
            for k,v in offset_labels.items():
                c.write(f"{k},{v}")
                c.write("\n")
        self.assertTrue(os.path.isfile("./hpkd1/pkd1_grn.csv"))
        for f in glob.glob("./hpkd1/*"):
            os.remove(f)
        os.rmdir("./hpkd1")
    
    def test_get_template_information(self):
        # Test template_utils via getting infomation of a template.
        valid_collection = pd.read_pickle("../data/valid_collection.pkl")
        centers, center_quality, aln_indices, pdb_centers= gaingrn.utils.template_utils.get_template_information(identifier='A0A6G1Q0B9', gain_collection=valid_collection, subdomain='a')
        self.assertTrue(len(centers.keys()) == 6)
    
class TestClasses(unittest.TestCase):
    # Test Class instance generation and in that regard, also the underlying functions

    def test_GainCollection(self):
        # This tests the GainCollection and GainDomain class with the more complex ALIGNMENT CASE. Therefore, validity in the NON-ALIGNMENT CASE is ensured.
        alignment_file = "./test_data/gain_collection/test_seqs.mafft.fa"
        quality_file =   "./test_data/gain_collection/offset_test_seqs.jal"
        full_seqs =      "./test_data/gain_collection/full_test_seqs.fa"
        stride_files = glob.glob("./test_data/gain_collection/stride/*.stride")
        gps_minus_one = 6553
        aln_cutoff = 6567
        alignment_dict = gaingrn.utils.io.read_alignment(alignment_file, aln_cutoff)
        quality = gaingrn.utils.io.read_quality(quality_file)

        valid_seqs = gaingrn.utils.io.read_multi_seq("./test_data/gain_collection/offset_test_seqs.fa") # MODEL SEQUENCES
        full_seqs = gaingrn.utils.io.read_alignment( "./test_data/gain_collection/full_test_seqs.fa") # UNIPROT QUERY SEQUENCES
 
        valid_adj_seqs = gaingrn.utils.alignment_utils.offset_sequences(full_seqs=full_seqs, short_seqs=valid_seqs)
                    
        print(f"Adjusted the sequence offset of {len(valid_adj_seqs)} sequences.")

        alignment_dict = gaingrn.utils.io.read_alignment(alignment_file, aln_cutoff)
        quality = gaingrn.utils.io.read_quality(quality_file)

        test_collection = GainCollection(  alignment_file = alignment_file,
                                    aln_cutoff = aln_cutoff,
                                    quality = quality,
                                    gps_index = gps_minus_one,
                                    stride_files = stride_files,
                                    sequence_files=None,
                                    sequences=valid_adj_seqs,
                                    alignment_dict = alignment_dict,
                                    is_truncated = True,
                                    coil_weight=0.00,
                                    stride_outlier_mode=True,
                                    debug=False)
        self.assertTrue(len(test_collection.collection) == 5)

        # EACH OF THE GAIN DOMAIN OBJECTS SHOULD BE VALID, HAVE 7 HELICAL SEGMENTS (SUBDOMAIN A) AND 14 STRAND SEGMENTS (SUBDOMAIN B)
        for gain in test_collection.collection:
            self.assertTrue(gain.isValid)
            self.assertTrue(len(gain.sda_helices) == 7)
            self.assertTrue(len(gain.sdb_sheets) == 14)
        print(f"Checked Test GainCollection with {len(test_collection.collection)} GainDomain objects. All are valid.")
    
    def test_MutationAnalysis(self):
        # Initialize the human GAIN collection

        jsons = glob.glob("../data/gain_json/*.json")
        csvs = glob.glob("../data/snp_mane/*csv")

        human_collection = pd.read_pickle("../data/human_collection.pkl")

        human_accessions = [gain.name.split("-")[0].split("_")[0] for gain in human_collection.collection]
        human_sequences = ["".join(gain.sequence) for gain in human_collection.collection]
        seq_file = '../data/seq_aln/all_query_sequences.fasta'

        human_fasta_offsets = gaingrn.utils.alignment_utils.find_offsets(seq_file,
                                        human_accessions, 
                                        human_sequences)

        human_indexing = np.load("../data/human_indexing.pkl", allow_pickle=True)

        appended_human_collection = gaingrn.utils.assign.add_grn_labels(human_collection, human_indexing)

        segments = ['H1','H2','H3','H4','H5','H6','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','GPS']

        GainMutations = MutationAnalysis(appended_human_collection, segments, jsons, csvs, human_fasta_offsets)

        self.assertTrue(len(GainMutations.generalized_mutations['H1.39']) == 3)

if __name__ == '__main__':
    unittest.main()
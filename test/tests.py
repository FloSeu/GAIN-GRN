import unittest
import sys, os, re, glob

sys.path.append('/home/hildilab/agpcr_nom/repo/')
from gaingrn import sse_func
from gaingrn import execute
from gaingrn import gain_classes
from gaingrn import template_finder
# Test functions have to start with "test_"

class TestBinaries(unittest.TestCase):

    STRIDE_BIN = '/home/hildilab/lib/stride/stride'
    GESAMT_BIN = '/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt'

    def test_stride(self):
        pdb_file = "test_data/A_A0A2Y9F628.pdb"
        out_file = "stride_test.out"
        self.assertTrue(os.path.isfile(self.STRIDE_BIN))
        stride_command = f"{self.STRIDE_BIN} {pdb_file} -f{out_file}"
        exit_code = execute.run_command(stride_command)
        self.assertTrue(exit_code == 0)
        self.assertTrue(os.path.isfile(out_file))
        with open(out_file) as of:
            stride_data = of.read()
        helix1 = """LOC  AlphaHelix   ASP   414 B      GLN    424 B"""
        self.assertTrue(helix1 in stride_data)
        os.remove(out_file)

    def test_gesamt(self):
        template_pdb = "test_data/A_A0A2Y9F628.pdb"
        mobile_pdb = "test_data/Q8IWK6_Q6UXK9_Q86SQ5_Q8TC55-AGRA3_HUMAN-AGRA3-Homo_sapiens_gain.pdb"

        out_file = "gesamt.test.out"

        self.assertTrue(os.path.isfile(self.GESAMT_BIN))
        gesamt_command = f'{self.GESAMT_BIN} {template_pdb} {mobile_pdb}'
        execute.run_command(gesamt_command, out_file=out_file)

        self.assertTrue(os.path.isfile(out_file))
        with open(out_file) as of:
            gesamt_data = of.read()

        qscore = re.compile("Q-score +: 0\.3[0-9]+")
        rmsd = re.compile("RMSD *: 0\.1[0-9]+")
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

        GESAMT_BIN = '/home/hildilab/lib/xtal/ccp4-8.0/ccp4-8.0/bin/gesamt'

        # First, generate the GainDomain object fromt the STRIDE file
        pkd_gain = gain_classes.GainDomainNoAln(start=None,
                                subdomain_boundary=None, 
                                end=None, 
                                fasta_file=None, 
                                name='hPKD1', 
                                sequence=None, 
                                explicit_stride_file="../data/example/PKD1_1.stride")

        # With the object, feed in the PDB for structural alignment and the template data
        print("\nIndexing human PKD1 onto the GAIN-GRN. Expect a number of unindexed segments on the extra Subdomain...")
        element_intervals, element_centers, residue_labels, unindexed_elements, params = template_finder.assign_indexing(pkd_gain, 
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
        alignment_dict = sse_func.read_alignment(alignment_file, aln_cutoff)
        quality = sse_func.read_quality(quality_file)

        valid_seqs = sse_func.read_multi_seq("./test_data/gain_collection/offset_test_seqs.fa") # MODEL SEQUENCES
        full_seqs = sse_func.read_alignment( "./test_data/gain_collection/full_test_seqs.fa") # UNIPROT QUERY SEQUENCES
 
        valid_adj_seqs = sse_func.offset_sequences(full_seqs=full_seqs, short_seqs=valid_seqs)
                    
        print(f"Adjusted the sequence offset of {len(valid_adj_seqs)} sequences.")

        alignment_dict = sse_func.read_alignment(alignment_file, aln_cutoff)
        quality = sse_func.read_quality(quality_file)

        test_collection = gain_classes.GainCollection(  alignment_file = alignment_file,
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

if __name__ == '__main__':
    # Get STRIDE and GESAMT binary from SYSTEM.
    TestBinaries.STRIDE_BIN = os.environ.get('STRIDE_BIN', TestBinaries.STRIDE_BIN)
    TestBinaries.GESAMT_BIN = os.environ.get('GESAMT_BIN', TestBinaries.GESAMT_BIN)

    unittest.main()
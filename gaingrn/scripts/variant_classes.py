## variant_classes.py
# Contains classes for SNP and cancer mutation analysis.
import gaingrn.scripts.mutation_utils
import numpy as np

class MutationAnalysis:
    def __init__(self, appended_gain_collection, segments, jsons, csvs, fasta_offsets ):
        self.segments = segments
        self.generalized_counts = {}     # For every nomenclature label, the number of MUTATION entries is denoted
        self.generalized_mutations = {}  #   ^ The info behind the entries is here.
        self.valid = 0
        self.invalid = 0
        self.total_within = 0

        mismatch_flag = False

        # Count GRN label occurrences in the gain collection
        # For each GAIN domain in the collection, count the occurence of a specific label
        self.occupancy_dict = {}
        for gain in appended_gain_collection.collection:
            #rev_dir = {v: k for k, v in gain.named_dir.items()}
            for key in gain.grn_labels.keys():
                if key not in self.occupancy_dict.keys():
                    self.occupancy_dict[key] = 1
                else:
                    self.occupancy_dict[key] += 1

        # Inititalize des compacted SNP and CANCER MUTATION DATA HERE (not generalized via Label)
        self.mutations, self.mut_counts = gaingrn.scripts.mutation_utils.compose_vars(appended_gain_collection, jsons, 'x', 'aa_change', fasta_offsets, merge_unique=True)
        self.snps,      self.snp_counts = gaingrn.scripts.mutation_utils.compose_vars(appended_gain_collection, csvs, 'resid', 'resname', fasta_offsets, merge_unique=True)

        # Here, we initialize the GENERALIZED MUTATIONS (per GRN label)
        for gain_ndx, gain in enumerate(appended_gain_collection.collection):

            gain_valid = 0
            gain_invalid = 0
            fasta_offset = fasta_offsets[gain_ndx]

            # Retrieve mutations for respective receptor
            positions, gain_mutation_dict = gaingrn.scripts.mutation_utils.extract_variants(gain, jsons, 'x')
            
            # Find all mutations whose RESID (fasta) matches the corrected INTERVAL(for fasta resids)
            within = [pos for pos in positions if pos in range(fasta_offset, fasta_offset + 1 + gain.end - gain.start)]

            # Go through the gain_mutation_dict and add into the two dicts
            self.total_within += len(within)

            # Evaluate the mutations. If the residue in question (corrected index) is named (i.e. "H6.50"), get its resepective label.
            for mutated_resid in gain_mutation_dict.keys():
                # Since fasta_offset maps to the first gain residue (i.e. 2027 with gain.start being 459), an offset needs to be set by the difference
                corrected_resid = mutated_resid - fasta_offset + gain.start #-1  [ISSUE]this one was the one!
                # map it to its corresponding label. If there is none, continue
                if corrected_resid < gain.start or corrected_resid >= gain.end:
                    gain_invalid += 1
                    continue
                try:
                    mutated_label = gain.reverse_grn_labels[corrected_resid]
                    #print(mutated_label)
                    gain_valid +=1
                except KeyError:
                    gain_invalid += 1
                    continue
                # SANITY CHECK:
                if gain_mutation_dict[mutated_resid][0]['aa_change'][0] != gain.sequence[corrected_resid-gain.start-1]:
                    print(f"MISMATCH! {mutated_resid} -> {corrected_resid}",
                        f"{gain_mutation_dict[mutated_resid][0]['aa_change'][0]} : {gain.sequence[corrected_resid-gain.start]} in GAIN",
                        f"{gain.sequence[corrected_resid-gain.start-2:corrected_resid-gain.start+3]}")
                    mismatch_flag = True
                
                # with the Label, map into generalized dictionary
                if mutated_label not in self.generalized_mutations.keys():
                   self. generalized_mutations[mutated_label] = gain_mutation_dict[mutated_resid]
                else:
                    [self.generalized_mutations[mutated_label].append(mutation) for mutation in gain_mutation_dict[mutated_resid]]
        
            self.valid += gain_valid
            self.invalid += gain_invalid

        if mismatch_flag: 
            print("[WARNING]: MISMATCHES HAVE BEEN FOUND. PLEASE CHECK THE OUTPUT.")
        else: 
            print("[NOTE]: NO MISMATCHES HAVE BEEN FOUND.")
        print(f"TOTAL MUTATIONS WITHIN GAIN:", self.valid, "\nTOTAL MUTATIONS OUTSIDE GAIN:", self.invalid, "\nTOTAL MUTATIONS:", self.valid+self.invalid)

    def generate_data_array(self, return_list=False):
        # Returns data in Array form, i.e. for writing to File.
        varmut_arr = [] # list of lists with 3 items to be converted into numpy array later
        for sse in self.segments:

            varkeys = [k for k in self.mut_counts.keys() if sse in k]+[k for k in self.snp_counts.keys() if sse in k]

            if sse == "GPS":
                x_range = (1,2,3)
                mut_y = [self.mut_counts["GPS.-2"], self.mut_counts["GPS.-1"], self.mut_counts["GPS.+1"]]
                var_y = [self.snp_counts["GPS.-2"], self.snp_counts["GPS.-1"], self.snp_counts["GPS.+1"]]
                print(x_range, mut_y, var_y, "FOR GPS")
            if sse != 'GPS':
                x_positions = [int(x.split('.')[-1]) for x in varkeys]
                x_range = range(min(x_positions), max(x_positions)+1)

                mut_y = gaingrn.scripts.mutation_utils.compose_y(x_range, sse, self.mut_counts)
                var_y = gaingrn.scripts.mutation_utils.compose_y(x_range, sse, self.snp_counts)
            
            score_y = gaingrn.scripts.mutation_utils.score(mut_y,var_y)

            for i in range(len(x_range)):
                varmut_arr.append([sse, x_range[i], int(mut_y[i]), int(var_y[i]), score_y[i]])

        if return_list:
            return varmut_arr
        # make the data parseable
        vm_arr = np.array(varmut_arr, dtype = [("segment", str),
                                            ("position", int),
                                            ("n_mut", int),
                                            ("n_snp", int),
                                            ("score", float)])
        return vm_arr
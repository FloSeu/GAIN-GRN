# These are indexing classes for the GPCRDB-specific indexing of multiple GAIN domains.

import numpy as np
import gaingrn.scripts.assign
import glob
import multiprocessing as mp

class StAlIndexing:
    def __init__(self, list_of_gain_obj, prefix:str, pdb_dir:str,  template_dir:str, template_json:str, gesamt_bin:str, 
                 n_threads=1, outlier_cutoff=10.0, fasta_offsets=None, pseudocenters=None, debug=False):

        def find_pdb(name, pdb_folder):
            identifier = name.split("-")[0]
            target_pdb = glob.glob(f"{pdb_folder}/*{identifier}*.pdb")[0]
            return target_pdb

        length = len(list_of_gain_obj)
        total_keys = []

        if fasta_offsets is None:
            self.fasta_offsets = np.zeros([length])
        if fasta_offsets is not None: 
            # The existing FASTA offsets do not account for the residue starting not at 0,
            # Therefore the value of the starting res (gain.start) needs to be subtracted.
            corrected_offsets = [fasta_offsets[i]-list_of_gain_obj[i].start for i in range(length)]
            self.fasta_offsets = np.array(corrected_offsets, dtype=int)

        all_centers = np.empty([length], dtype=dict) # <-- center_dirs
        all_indexing_dir = np.empty([length], dtype=dict) # <-- indexing_dirs
        all_unindexed = np.empty([length], dtype=list) # <-- total_unindexed
        all_params = np.empty([length], dtype=dict) # <-- a_templates, b_templates, receptor_types
        all_intervals = np.empty([length], dtype=dict)

        if n_threads > 1 :
            print(f"StAlIndexing: Assigning indexing via multiprocessing with {n_threads} threads.")
            pool = mp.Pool(n_threads)
            # Construct an iterable with the arguments for the wrapper function mp_assign_indexing
            mp_arglist = [ 
                            [gain, f"{prefix}_{gain_idx}", find_pdb(gain.name, pdb_dir), template_dir, debug, False, {"S2":7,"S6":3,"H5":3}, True, gain_idx, template_json, gesamt_bin] 
                            for gain_idx, gain in enumerate(list_of_gain_obj)
                         ]
            print("Completed composing arg_list for assigning multithreaded indexing.", mp_arglist[:5],"\n", mp_arglist[-5:])
            for result in pool.imap_unordered(gaingrn.scripts.assign.mp_assign_indexing, mp_arglist):
                # this is each instance of the above function return with the result[4] being the index
                all_intervals[result[5]] = result[0]
                all_centers[result[5]] = result[1]
                all_indexing_dir[result[5]] = result[2]
                all_unindexed[result[5]] = result[3]
                all_params[result[5]] = result[4]
            print("Done with multiprocessing.")
            pool.close()
            pool.join()

        if n_threads == 1:
            print(f"StAlIndexing: Assigning indexing via single process.")
            for gain_index, gain in enumerate(list_of_gain_obj):
                intervals, indexing_centers, indexing_dir, unindexed, params = gaingrn.scripts.assign.assign_indexing(gain, 
                                                                                        file_prefix=f"{prefix}_{gain_index}", 
                                                                                        gain_pdb=find_pdb(gain.name, pdb_dir), 
                                                                                        template_dir=template_dir, 
                                                                                        gesamt_bin=gesamt_bin,
                                                                                        template_json=template_json,
                                                                                        outlier_cutoff=outlier_cutoff,
                                                                                        debug=debug, 
                                                                                        create_pdb=False,
                                                                                        pseudocenters=pseudocenters,
                                                                                        hard_cut={"S2":7,"S6":3,"H5":3},
                                                                                        patch_gps=True
                                                                                        )

                all_centers[gain_index] = indexing_centers
                all_indexing_dir[gain_index] = indexing_dir
                all_unindexed[gain_index] = unindexed
                all_params[gain_index] = params
                all_intervals[gain_index] = intervals

        for indexing_dir in all_indexing_dir:
            for key in indexing_dir.keys():
                if key not in total_keys:
                    total_keys.append(key)

            #params = {"sda_template":best_a, "sdb_template":best_b, "split_mode":highest_split}

        # Assign data structures
        self.indexing_dirs = all_indexing_dir
        self.center_dirs = all_centers
        self.names = [gain.name for gain in list_of_gain_obj]
        self.length = length
        self.intervals = all_intervals
        #self.offsets = [gain.start for gain in list_of_gain_obj]
        self.accessions = [gain.name.split("-")[0].split("_")[0] for gain in list_of_gain_obj]
        self.sequences = ["".join(gain.sequence) for gain in list_of_gain_obj]
        self.total_keys = sorted(total_keys)
        self.center_keys = [f"H{i}.50" for i in range(1,7)]+[f"S{i}.50" for i in range(1,15)]
        self.unindexed = all_unindexed
        self.a_templates = [params["sda_template"] for params in all_params]
        self.b_templates = [params["sdb_template"] for params in all_params]
        self.receptor_types = [params["receptor"] for params in all_params]
        self.split_modes = [params["split_mode"] for params in all_params]

        print("Total of keys found in the dictionaries:\n", self.total_keys, self.center_keys)
        print("First entry", self.indexing_dirs[0], self.center_dirs[0])
    
    def construct_data_matrix(self, overwrite_gps=True, unique_sse=False):
        # the unique_sse boolean specifies if unique element identifiers should be kept (i.e. "H1.E1" instead of "H1")
        # overwrite_gps specifies whether GPS entries will be unique - they will be removed from other entries i.e S13 / S14
                 
        #header = "Receptor,Accession," + ",".join(self.total_keys) + ",".join(self.center_keys)
        header_dict = {}
        header = ["Accession","Name","Species","type", "GPS-2", "GPS-1", "GPS+1"]
        for j in self.center_keys:
            header.append(f"{j[:-3]}.start")
            header.append(f"{j[:-3]}.anchor")
            header.append(f"{j[:-3]}.end")
        
        if unique_sse:
            # We find the unique entries in the intervals:
            unique_headers = []
            for interval_dict in self.intervals:
                unique = [k for k in interval_dict.keys() if "." in k]
                unique_headers = unique_headers + unique
            individual_headers = np.unique(unique_headers)
            print(f"[DEBUG] construct_data_matrix: Found the following unique headers:\n\t{individual_headers = }")

            for j in individual_headers:
                header.append(f"{j}.start")
                header.append(f"{j}.anchor")
                header.append(f"{j}.end")

        for idx, key in enumerate(header):
            header_dict[key] = idx
        # if unique_sse is specified, find and add unique entries to the dictionary

        print(header_dict)

        data_matrix = np.full([self.length, len(header_dict.keys())], fill_value='', dtype=object)
        # Go through each of the sub-dictionaries and populate the dataframe:
        for row in range(self.length):
                # Q5T601_Q5KU15_..._Q9H615-AGRF1_HUMAN-AGRF1-Homo_sapiens.fa
                # 0                        1           2     3
            name_parts = self.names[row].replace('AGR', 'ADGR').replace(',','').split("-")

            data_matrix[row, header_dict["Name"]] = "".join(name_parts[1:-1])
            data_matrix[row, header_dict["Accession"]] = name_parts[0].split("_")[0]
            data_matrix[row, header_dict["Species"]] = name_parts[-1].replace(".","")
            data_matrix[row, header_dict["type"]] = self.receptor_types[row]
            fa_offset = self.fasta_offsets[row]

            # Find GPS residues.
            gps_res = []
            gps_col = {"GPS.-2":4,"GPS.-1":5, "GPS.+1":6}
            for key in self.indexing_dirs[row].keys():
                if "GPS" in key:
                    if self.indexing_dirs[row][key] is not None:
                        data_matrix[row, gps_col[key]] = str(self.indexing_dirs[row][key]+fa_offset)
                        if overwrite_gps:
                            gps_res.append(self.indexing_dirs[row][key]+fa_offset)
                    else:
                        data_matrix[row, gps_col[key]] = " "
            # Write start and end residues with offset to dataframe
            for key in self.intervals[row].keys():
                if key == "GPS":
                    continue
                #print("[DEBUG]", self.intervals[row][key], type(self.intervals[row][key]), sep="\n\t")
                sse = [int(x+fa_offset) for x in self.intervals[row][key]] # should be a list of [start, end] for each interval
                # A "." denotes individual elements, that means they will NOT be in the data_matrix unless $unique_sse is specified.
                if "." in key and not unique_sse:
                    key = key.split(".")[0] # "H1.E1" --> "H1"
                # Overwrite_gps: If any Start of end coincides with GPS residues, incr/decrement this value so that GPS-residues are not included
                if overwrite_gps:
                    if sse[0] in gps_res:
                        sse[0] = sse[0]+1
                    if sse[1] in gps_res:
                        sse[1] = sse[1]-1
                data_matrix[row, header_dict[f"{key}.start"]] = str(sse[0])
                data_matrix[row, header_dict[f"{key}.end"]] = str(sse[1])
            # Write center residues with offset to dataframe

            for key in self.center_dirs[row].keys():
                if key == "GPS":
                    continue
                if "." in key and not unique_sse:
                    data_matrix[row, header_dict[f"{key.split('.')[0]}.anchor"]] = str(self.center_dirs[row][key]+fa_offset)
                else:
                    data_matrix[row, header_dict[key.replace(".50",".anchor")]] = str(self.center_dirs[row][key]+fa_offset)
            
            data_header = ",".join(header)
            data_matrix = data_matrix
        
        return data_header, data_matrix

    def data2csv(self, data_header, data_matrix, outfile):
        with open(outfile, "w") as f:
            f.write(data_header+"\n")
            for row in range(data_matrix.shape[0]):
                f.write(",".join(data_matrix[row,:])+"\n")
        print("Completed file", outfile, ".")

class GPCRDBIndexing:
    def __init__(self, aGainCollection, anchors, anchor_occupation, anchor_dict, fasta_offsets=None, split_mode='single') :
        
        length = len(aGainCollection.collection)
        names = np.empty([length], dtype=object)
        indexing_dirs = np.empty([length], dtype=object)
        center_dirs = np.empty([length], dtype=object)
        offsets = np.zeros([length], dtype=int)
        total_keys = []
        center_keys = []
        total_unindexed = []
        if fasta_offsets is None:
            self.fasta_offsets = np.zeros([length])
        if fasta_offsets is not None: 
            corrected_offsets = []
            for i in range(length):
                # The existing FASTA offsets do not account for the residue starting not at 0,
                # Therefore the value of the starting res (gain.start) needs to be subtracted.
                corrected_offsets.append(fasta_offsets[i]-aGainCollection.collection[i].start)
            self.fasta_offsets = np.array(corrected_offsets, dtype=int)
            
        for gain_index, gain in enumerate(aGainCollection.collection):
            print("\n\n","_"*70,"\n",gain.name,"\n",
                 f"{gain.a_breaks = }\n {gain.b_breaks = }\n")
            indexing_dir, indexing_centers, _, unindexed = gain.create_indexing(anchors, 
                                                            anchor_occupation, 
                                                            anchor_dict,
                                                            split_mode=split_mode)
                    
            print(indexing_dir, indexing_centers)
            for key in indexing_dir.keys():
                if key not in total_keys:
                    total_keys.append(key)
                    
            for key in indexing_centers.keys():
                if key not in center_keys:
                    center_keys.append(key)         
                            
            if unindexed != []:
                for item in unindexed : 
                    total_unindexed.append(item)

            indexing_dirs[gain_index] = indexing_dir
            center_dirs[gain_index] = indexing_centers
            # Patch ADGRC/CELSR naming
            names[gain_index] = gain.name.replace("CadherinEGFLAGseven-passG-typereceptor", "CELSR")
            offsets[gain_index] = gain.start

        self.indexing_dirs = indexing_dirs
        self.center_dirs = center_dirs
        self.names = names
        self.length = length
        self.offsets = offsets
        self.accessions = [gain.name.split("-")[0].split("_")[0] for gain in aGainCollection.collection]
        self.sequences = ["".join(gain.sequence) for gain in aGainCollection.collection]
        self.total_keys = sorted(total_keys)
        self.center_keys = sorted(center_keys)
        self.unindexed = total_unindexed

        print("Total of keys found in the dictionaries:\n", self.total_keys, self.center_keys)
        print("First entry", self.indexing_dirs[0], self.center_dirs[0])
        
        header_list = ["Receptor", "Accession"] + self.total_keys + self.center_keys
        #header = "Receptor,Accession," + ",".join(self.total_keys) + ",".join(self.center_keys)
        new_header = ["Receptor", "Accession", "GPS.-2", "GPS.-1", "GPS.+1"]
        for j in self.center_keys:
            new_header.append(f"{j[:-3]}.start")
            new_header.append(f"{j[:-3]}.anchor")
            new_header.append(f"{j[:-3]}.end")

        header_dict = {}
        for idx, item in enumerate(header_list):
            header_dict[item] = idx
        gpcr_header_dict = {}
        for idx, key in enumerate(new_header):
            gpcr_header_dict[key] = idx
        print(gpcr_header_dict)
        data_matrix = np.full([self.length, len(gpcr_header_dict.keys())], fill_value='', dtype=object)
        # Go through each of the sub-dictionaries and populate the dataframe:
        for row in range(self.length):
                # Q5T601_Q5KU15_..._Q9H615-AGRF1_HUMAN-AGRF1-Homo_sapiens.fa
                # 0                        1           2     3
            name_parts = self.names[row].replace(',','').split("-")
            data_matrix[row, gpcr_header_dict["Receptor"]] = name_parts[2]
            data_matrix[row, gpcr_header_dict["Accession"]] = name_parts[0].split("_")[0]
            offset = self.offsets[row]
            fa_offset = self.fasta_offsets[row]

            for key in self.indexing_dirs[row].keys():
                if key == "GPS":
                    sse=[int(x+fa_offset) for x in self.indexing_dirs[row][key]]
                    data_matrix[row, 2:5] = [str(sse[0]), str(sse[1]), str(sse[2])]
                else:
                    sse = [int(x+offset+fa_offset) for x in self.indexing_dirs[row][key]]
                    data_matrix[row, gpcr_header_dict[f"{key}.start"]] = str(sse[0])
                    data_matrix[row, gpcr_header_dict[f"{key}.end"]] = str(sse[1])

            for key in self.center_dirs[row].keys():
                data_matrix[row, gpcr_header_dict[key.replace(".50",".anchor")]] = str(self.center_dirs[row][key]+offset+fa_offset)
            
            self.data_header = ",".join(new_header)
            self.data_matrix = data_matrix

    def data2csv(self, outfile):
        with open(outfile, "w") as f:
            f.write(self.data_header+"\n")
            for row in range(self.length):
                f.write(",".join(self.data_matrix[row,:])+"\n")
        print("Completed file", outfile, ".")

# Another, mostly unused class for sorting the indexing

class Indexing:
    def __init__(self, aGainCollection, anchors, anchor_occupation, anchor_dict, fasta_offsets=None, split_mode='double',):

        total_unindexed = []
        length = len(aGainCollection.collection)
        names = np.empty([length], dtype=object)
        indexing_dirs = np.empty([length], dtype=object)
        center_dirs = np.empty([length], dtype=object)
        offsets = np.zeros([length], dtype=int)
        total_keys = []
        center_keys = []
        if fasta_offsets is None:
            self.fasta_offsets = np.zeros([length])
        if fasta_offsets is not None: 
            corrected_offsets = []
            for i in range(length):
                # The existing FASTA offsets do not account for the residue starting not at 0,
                # Therefore the value of the starting res (gain.start) needs to be subtracted.
                corrected_offsets.append(fasta_offsets[i]-aGainCollection.collection[i].start)
            self.fasta_offsets = np.array(corrected_offsets, dtype=int)
            
        for gain_index, gain in enumerate(aGainCollection.collection):
            indexing_dir, indexing_centers, unindexed = gain.create_indexing(anchors, 
                                                                  anchor_occupation, 
                                                                  anchor_dict,
                                                                  split_mode=split_mode)
            #print(indexing_dir, indexing_centers)
            for key in indexing_dir.keys():
                if key not in total_keys:
                    total_keys.append(key)
                    
            for key in indexing_centers.keys():
                if key not in center_keys:
                    center_keys.append(key)                 
            if unindexed != []:
                for item in unindexed : 
                    total_unindexed.append(item)

            indexing_dirs[gain_index] = indexing_dir
            center_dirs[gain_index] = indexing_centers
            # Patch ADGRC/CELSR naming
            names[gain_index] = gain.name.replace("CadherinEGFLAGseven-passG-typereceptor", "CELSR")
            offsets[gain_index] = gain.start

        self.indexing_dirs = indexing_dirs
        self.center_dirs = center_dirs
        self.names = names
        self.length = length
        self.offsets = offsets
        self.accessions = [gain.name.split("-")[0].split("_")[0] for gain in aGainCollection.collection]
        self.sequences = ["".join(gain.sequence) for gain in aGainCollection.collection]
        self.total_keys = sorted(total_keys)
        self.center_keys = sorted(center_keys)
        self.unindexed = total_unindexed
        
        print("Total of keys found in the dictionaries:\n", self.total_keys, self.center_keys)
        print("First entry", self.indexing_dirs[0], self.center_dirs[0])
        
        header_list = ["Receptor", "Accession"] + self.total_keys + self.center_keys
        #header = "Receptor,Accession," + ",".join(self.total_keys) + ",".join(self.center_keys)
              
        header_dict = {}
        for idx, item in enumerate(header_list):
            header_dict[item] = idx

        data_matrix = np.full([self.length, len(header_dict.keys())], fill_value='', dtype=object)
        # Go through each of the sub-dictionaries and populate the dataframe:
        for row in range(self.length):
                # Q5T601_Q5KU15_..._Q9H615-AGRF1_HUMAN-AGRF1-Homo_sapiens.fa
                # 0                        1           2     3
            name_parts = self.names[row].split("-")
            data_matrix[row, header_dict["Receptor"]] = name_parts[2]
            data_matrix[row, header_dict["Accession"]] = name_parts[0].split("_")[0]
            offset = self.offsets[row]
            fa_offset = self.fasta_offsets[row]

            for key in self.indexing_dirs[row].keys():
                if key == "GPS":
                    sse=[int(x+fa_offset) for x in self.indexing_dirs[row][key]]
                    data_matrix[row, header_dict[key]] = f"{sse[0]}-{sse[-1]}"
                else:
                    sse = [int(x+offset+fa_offset) for x in self.indexing_dirs[row][key]]
                    data_matrix[row, header_dict[key]] = f"{sse[0]}-{sse[1]}"

            for key in self.center_dirs[row].keys():
                data_matrix[row, header_dict[key]] = str(self.center_dirs[row][key]+offset+fa_offset)
            
            self.data_header = ",".join(header_list)
            self.data_matrix = data_matrix

    def data2csv(self, outfile):
        with open(outfile, "w") as f:
            f.write(self.data_header+"\n")
            for row in range(self.length):
                f.write(",".join(self.data_matrix[row,:])+"\n")
        print("Completed file", outfile, ".")
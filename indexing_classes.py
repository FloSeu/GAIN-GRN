# These are indexing classes for the GPCRDB-specific indexing of multiple GAIN domains.

import numpy as np
import template_finder as tf
import glob

class StAlIndexing:
    def __init__(self, aGainCollection, prefix:str, pdb_dir:str,  template_dir:str, fasta_offsets=None):

        def find_pdb(name, pdb_folder):
            identifier = name.split("-")[0]
            target_pdb = glob.glob(f"{pdb_folder}/*{identifier}*.pdb")[0]
            return target_pdb

        length = len(aGainCollection.collection)
        names = np.empty([length], dtype=object)
        indexing_dirs = np.empty([length], dtype=object)
        center_dirs = np.empty([length], dtype=object)
        offsets = np.zeros([length], dtype=int)
        total_keys = []
        total_unindexed = []
        receptor_types = np.empty([length], dtype='<U2')
        a_templates = np.empty([length], dtype='<U4')
        b_templates = np.empty([length], dtype='<U2')

        if fasta_offsets is None:
            self.fasta_offsets = np.zeros([length])
        if fasta_offsets is not None: 
            corrected_offsets = []
            for i in range(length):
            # The existing FASTA offsets do not account for the residue starting not at 0,
            # Therefore the value of the starting res (gain.start) needs to be subtracted.
                corrected_offsets.append(fasta_offsets[i]-aGainCollection.collection[i].start)
            self.fasta_offsets = np.array(corrected_offsets, dtype=int)

        total_keys = []
        total_unindexed = []

        for gain_index, gain in enumerate(aGainCollection.collection):
            _, indexing_centers, indexing_dir, unindexed, params = tf.assign_indexing(gain, 
                                                                                       file_prefix=f"{prefix}_{gain_index}", 
                                                                                       gain_pdb=find_pdb(gain.name, pdb_dir), 
                                                                                       template_dir=template_dir, 
                                                                                       debug=False, 
                                                                                       create_pdb=False,
                                                                                       hard_cut={"S2":7,"S6":3,"H5":3},
                                                                                       patch_gps=True)

            for key in indexing_dir.keys():
                if key not in total_keys:
                    total_keys.append(key)      
                            
            if unindexed != []:
                for item in unindexed : 
                    total_unindexed.append(item)

            indexing_dirs[gain_index] = indexing_dir
            center_dirs[gain_index] = indexing_centers

            names[gain_index] = gain.name
            offsets[gain_index] = gain.start
            #params = {"sda_template":best_a, "sdb_template":best_b, "split_mode":highest_split}
            a_templates[gain_index] = params["sda_template"]
            b_templates[gain_index] = params["sdb_template"]
            receptor_types[gain_index] = params["receptor"]

        self.indexing_dirs = indexing_dirs
        self.center_dirs = center_dirs
        self.names = names
        self.length = length
        self.offsets = offsets
        self.accessions = [gain.name.split("-")[0].split("_")[0] for gain in aGainCollection.collection]
        self.sequences = ["".join(gain.sequence) for gain in aGainCollection.collection]
        self.total_keys = sorted(total_keys)
        self.center_keys = [f"H{i}.50" for i in range(1,7)]+[f"S{i}.50" for i in range(1,14)]
        self.unindexed = total_unindexed
        self.a_templates = a_templates
        self.b_templates = b_templates
        self.receptor_types = receptor_types

        print("Total of keys found in the dictionaries:\n", self.total_keys, self.center_keys)
        print("First entry", self.indexing_dirs[0], self.center_dirs[0])
        
        header_list = ["Receptor", "Accession"] + self.total_keys + self.center_keys
        #header = "Receptor,Accession," + ",".join(self.total_keys) + ",".join(self.center_keys)
        new_header = ["Receptor", "Accession", "GPS-2", "GPS-1", "GPS+1"]
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

            gps_col = {"GPS-2":2,"GPS-1":3, "GPS+1":4}
            for key in self.indexing_dirs[row].keys():
                if "GPS" in key:
                    data_matrix[row, gps_col[key]] = [str(self.indexing_dirs[row][key])]
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
        new_header = ["Receptor", "Accession", "GPS-2", "GPS-1", "GPS+1"]
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
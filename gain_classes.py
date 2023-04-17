# classes.py

# Object structure:
#

# GainCollection ---|- GainDomain ---|- Anchors
#                   |- GainDomain    |- GPS
#                   |- ...

import matplotlib.pyplot as plt
import numpy as np
import sse_func
from tqdm import tqdm
import scipy.stats as stats

class GainCollection:
    ''' 
    A collection of GainDomain objects with collected Anchors.
    This is used for generating the information needed to hardcode the indexing info

    Attributes
    ----------

    collection : list
        List of GainDomain instances

    anchor_hist : list
        List of anchor alignment indices

    left_limit : int
        Alignment column index of the most N-terminal anchor residue occurring.

    alignment_subdomain_boundary : int
        The column where the subdomain boundary is located

    alignment_quality : list
        The list of quality values corresponding to each alignment column

    alignment_length : int
        The length of the underlying alignment
    
    gps_minus_one_column : int
        The column index of the GPS-1 (Leu) residue
    
    Methods
    ----------
    print_gps():
        Prints info about all GainDomain GPS subinstances

    write_all_seq(savename):
        Writes all sequences in the Collection to one file

    transform_alignment(self, input_alignment, output_alignment, aln_cutoff):
        Transforms a given alignment with SSE data to a SSE-assigned version of the alignment

    find_anchors(cutoff)_
        Constructs an anchor and anchor_occupation array to use for constructing the numbering scheme
      '''
    def __init__(self,
                alignment_file, 
                aln_cutoff,
                quality,
                gps_index,
                alignment_dict=None,
                stride_files=None,
                sequence_files=None,
                sequences=None,
                subdomain_bracket_size=20,
                domain_threshold=20,
                coil_weight=0,
                is_truncated=False,
                stride_outlier_mode=False): 
        '''
        Constructs the GainCollection objects by initializing one GainDomain instance per given sequence
        
        Parameters
        ----------
        sequence_files:     list, optional
            A list of sequences to be read as the collection. Either this list or a large file containing all seqs must be specified

        alignment_file:     str, required
            The base dataset alignment file

        aln_cutoff:         int, required
            Integer value of last alignment column

        quality:            list, required
            List of quality valus for each alignment column. Has to have $aln_cutoff items
            By default, would take in the annotated Blosum62 values from the alignment exported from Jalview

        gps_index:          int, required
            Alignment column index of the GPS-1 residue (consensus: Leu)

        alignment_dict:     dict, optional
            A dictionary containing the parsed entries of the alignment. Speeds up computation significantly

        stride_files:       list, required
            A list of stride files corresponding to the sequences contained within.

        sequences:          object, optional
            A list of tuples containing a parsed multi-sequence fasta file.

        subdomain_bracket_size: int, optional
            Smoothing window size for the signal convolve function. Default = 20.

        domain_threshold:   int, optional
            Minimum size of a helical segment to be considered candidate for Subdomain A of GAIN domain. Default = 20.

        coil_weight:        float, optional
            Weight assigned to unordered residues during Subdomain detection. Enables decay of helical signal
            default = 0. Recommended values < +0.2 for decay

        is_truncated :      bool, optional
            indicates if the sequences are already truncated to only contain the GAIN domain

        stride_outlier_modes : bool, optional
            indicates of the STRIDE files contan "h" and "g" to indicate outlier SSE residues used to refine the SSE detection

        Returns
        ----------
        None
        '''
        if sequence_files:

            # Compile all sequence files to a sequences object
            sequences = np.empty([len(sequence_files)])

            for i, seq_file in enumerate(sequence_files):

                name, seq = sse_func.read_seq(seq_file, return_name = True)
                sequences[i] = (name, seq)

        elif (sequences is not None):
            print(f"[NOTE] GainCollection.__init__: Found sequences object.")
        else: 
            print(f"ERROR: no sequence_files or sequences parameter found. Aborting compilation.")
            return None

        # Initialize collection (containing all GainDomain instances) and the anchor-histogram
        self.collection = np.empty([len(sequences)], dtype=object)
        anchor_hist = np.zeros([aln_cutoff])
        subdomain_boundaries = []
        # Create a GainDomain instance for each sequence file contained in the list
        #progressbar = tqdm(total=len(sequences))
        invalid_count = 0
        for i,seq_tup in enumerate(sequences):

            name, sequence = seq_tup[0], seq_tup[1]

            explicit_stride = [stride for stride in stride_files if name.split("-")[0] in stride]
            if len(explicit_stride) == 0:
                print(f"Stride file not found for {name}")
            if len(explicit_stride) > 1:
                print(f"WARNING: AMBIGUITY in STRIDE files: {explicit_stride}") 
            #print("DEBUG:", name)
            newGain = GainDomain(alignment_file = alignment_file, 
                                  aln_cutoff = aln_cutoff,
                                  quality = quality,
                                  gps_index = gps_index,
                                  alignment_dict = alignment_dict,
                                  name = name,
                                  sequence = sequence,
                                  subdomain_bracket_size = subdomain_bracket_size,
                                  domain_threshold = domain_threshold,
                                  coil_weight = coil_weight,
                                  explicit_stride_file = explicit_stride[0],
                                  without_anchors = False,
                                  skip_naming = True,
                                  is_truncated = is_truncated,
                                  stride_outlier_mode=stride_outlier_mode
                                  )

            # Check if the object is an actual GAIN domain
            if newGain.isValid == True and newGain.GPS.isConsensus == True:

                # process the Anchors and add them to the anchor-histogram
                for j in newGain.Anchors.alignment_indices:
                    anchor_hist[j] += 1
                
                self.collection[i] = newGain
                # additionally, get the subdomain boundary value and store it.
                subdomain_boundaries.append(newGain.alignment_indices[newGain.subdomain_boundary-newGain.start])
            else:
                invalid_count += 1

            #progressbar.update(1)
        
        print(f"Completed collection with {invalid_count} invalid structures.")
        # Kick out empty items from the Collection
        self.collection = self.collection[self.collection.astype(bool)] 
        #print(f"DEBUG: {anchor_hist = }, {np.nonzero(anchor_hist)}")

        # Get the leftmost anchor, which denotes the absolute first "ordered" structure in the alignment
        left_limit = np.where(anchor_hist!=0)[0][0]

        self.anchor_hist = anchor_hist 
        self.leftmost = left_limit

        # Initialize detecting the overall subdomain boundary.

        self.alignment_subdomain_boundary = round(np.average(subdomain_boundaries))
        self.alignment_quality = quality
        self.alignment_length = aln_cutoff
        self.gps_minus_one_column = gps_index

        #progressbar.close()

    def print_gps(self):
        '''
        Prints information about the GPS of each GAIN domain.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        '''
        for i, gain in enumerate(self.collection):
            try:
                gain.GPS.info()
            except:
                print(f"No GPS data for {gain.name}. Likely not a GAIN Domain!")


    def write_all_seq(self, savename):
        '''
        Write all GAIN sequences of the GainCollection into one fasta file.

        Parameters
        ----------
        savename: str, required
            Output name of the fasta file.

        Returns
        ----------
        None
        '''
        with open(savename, 'w') as f:
            for gain in self.collection:
                f.write(f">{gain.name[:-3]}\n{''.join(gain.sequence)}\n")

    def transform_alignment(self, input_alignment, output_alignment, aln_cutoff):
        ''' 
        Transform any input alignment containing all sequences in the GainCollection 
        into one where each residue is replaced with the respective 
        Secondary Structure from the STRIDE files

        Parameters
        ----------
        input_alignment: str, required
            Input alignment file
        output_alignment: str, required
            Output alignment file
        aln_cutoff: int, required
            Last alignment column to be read from the Input Alignment

        Returns
        ---------
        None
        '''
        #initial_dict = sse_func.read_alignment(input_alignment, aln_cutoff)
        out_dict = {}
        for gain in self.collection:
            sse_alignment_row = np.full([aln_cutoff], fill_value='-', dtype='<U1')
            mapper = sse_func.get_indices(gain.name, gain.sequence, input_alignment, aln_cutoff)
            for res_id, sse_letter in gain.sse_sequence.items():
                sse_alignment_row[mapper[res_id]] = sse_letter
            out_dict[gain.name[:-3]] = sse_alignment_row

        # Write to file
        with open(output_alignment, "w") as f:
            for key in out_dict.keys():
                f.write(f">{key}\n{''.join(out_dict[key])}\n")
        print(f"Done transforming alignment {input_alignment} to {output_alignment} with SSE data.")
            
    def plot_sse_hist(self, title=None, n_max = 15, alpha=0.1, savename=None):
        n_bins = n_max + 1
        sheets = []
        helices = []

        for Gain in self.collection:
            helices.append(len(Gain.sda_helices))
            sheets.append(len(Gain.sdb_sheets))

        cols = ["cornflowerblue", "darkorange"]

        fig = plt.figure(figsize=[4,3])
        fig.set_facecolor("w")

        if title:
            fig.suptitle(title, fontsize=14)

        for i, values in enumerate([helices, sheets]):
            n, x, _ = plt.hist(values, bins = np.linspace(0,n_max, n_bins), histtype=u'bar', density=True, color=cols[i],alpha=1, align='left')
            #density = stats.gaussian_kde(values)
            #plt.plot(x, density(x), color=cols[i])
            #if alpha!=0: plt.fill_between(x,density(x), color=cols[i],alpha=alpha)
        plt.xticks(ticks = range(n_max+1), labels = range(n_max+1))

        if savename:
            plt.savefig(f"{savename}.png", dpi=600, bbox_inches='tight')
        plt.show()

    def find_anchors(self, cutoff=0):
        """
        Finds anchors and anchor_occupation depending on the provided cutoff value.
        Parameters :
            Cutoff : int
                The minimum occupation of the desired anchor
        Returns
            anchors : np.array
                The list of alignment columns corresponding to the anchors
            anchor_occupation : np.array
                The corresponding anchor occupation to resolve possible conflicts in anchor assignment (not expected.)
        """
        anchors = np.where(self.anchor_hist >= cutoff)[0]
    
        print(f"Found Anchors at cutoff {cutoff} with "
         f"{len(np.where(anchors < self.alignment_subdomain_boundary)[0])} Helix Anchors and "
         f"{len(np.where(anchors > self.alignment_subdomain_boundary)[0])} Sheet Anchors.")
        return anchors, self.anchor_hist[anchors]

class GainDomain:
    '''
    Instantiates the data from stride and the alignment, checking if the object is a GAIN domain
    and if so, creating analysis and establishing the indexing based on the results

    Attributes
    ----------
    name : str
        The name of the GAIN sequence
    isValid : bool
        Is True if the roughest GAIN domain criteria are met. Can be used for skipping invalid sequences.
    hasSubdomain : bool
        Is True if the subdomain detection has been successful. Can be used for filtering GAIN criteria.
    complete_sse_dict : dict
        The complete dictionary of all secondary structure elements (SSE) in the whole sequence
    sse_dict : dict
        The truncated dictionary of all SSE within the GAIN domain only
    start : int
        Start index of the GAIN domain
    end : int
        End index of the GAIN domain
    subdomain_boundary : int
        Residue index of the GAIN domain subdomain boundary (A/B)
    index : list
        List of residue indices belonging to the GAIN domain
    sequence : numpy array
        List of amino acid one letter codes of the GAIN domain
    alignment_indices : list
        List of alignment indices corresponding to each GAIN domain residue
    sse_name_map : list
        List of names for each residue corresponding to the SELF-CONSISTENT naming method (NOT NOMENCLATURE!)
    sse_sequence : list
        List of all indices of residues with their corresponding Helix or Strand assignment
    stride_outlier_mode : bool, optional
            indicates of the STRIDE files contan "h" and "g" to indicate outlier SSE residues used to refine the SSE detection

    Objects
    ----------
    GPS : object
        Instance of the GPS class containing info about the GPS
    Anchors : object
        Instance of the Anchors class containing info about the Anchors

    Methods
    ----------
    plot_profile(self, outdir=None):
        Plots a profile of the Protein showing details about GAIN domain detection

    plot_helicality(self, bracket_size=30, domain_threshold=20, coil_weight=0, savename=None):
        Plots a signal profile used for detection of subdomain A and the GAIN domain boundaries

    write_sequence(self, savename):
        Thin wrapper for writing the name and sequence to a fasta file

    create_indexing(self, anchors, anchor_occupation, anchor_dict, outdir=None):
        Creates a indexing list and writes it to a file

    write_gain_pdb(self, outfile=None)
        Writes a truncated PDB file to specified outfile
    '''
    def __init__(self, 
                 alignment_file, 
                 aln_cutoff,
                 quality,
                 gps_index, 
                 alignment_dict=None,
                 fasta_file=None,
                 name=None,
                 sequence=None,
                 subdomain_bracket_size=20,
                 domain_threshold=20,
                 coil_weight=0,
                 explicit_stride_file=None,
                 seq_name=None,
                 without_anchors=False,
                 skip_naming=False,
                 is_truncated=False,
                 stride_outlier_mode=False,
                 truncation_map=None,
                 aln_start_res=None):   
        ''' Initilalizes the GainDomain object and checks for GainDomain criteria to be fulfilled
        Parameters
        ----------
        alignment_file :    str, required
            The alignment file where the sequence can be found. For base dataset, this is default,
            for new GAIN domains, this is the extended alignment file

        aln_cutoff :        int, required
            Index of the last Alignment column to be read

        quality:            list, required
            List of quality valus for each alignment column. Has to have $aln_cutoff items
            By default, would take in the annotated Blosum62 values from the alignment exported from Jalview

        gps_index:          int, required
            Alignment column index of the GPS-1 residue (consensus: Leu)

        alignment_dict :    dict, optional
            A dictionary containing all entries by name from the input alignment. Speeds up computation significantly

        fasta_file :    str, optional
            The fasta file containing the sequence to be analyzed. Based on that, the STRIDE file 
            from the base dataset will be parsed if not explicitly stated. Either this or name+sequence have to be specifed
        
        name :          str, optional
            The name of the sequence. Enables reading from a compliled sequences object instead of individual files.

        sequence :      str, optional
            The one-letter coded sequence as string. Enables reading from a compliled sequences object instead of individual files.

        subdomain_bracket_size: int, optional
            Smoothing window size for the signal convolve function. Default = 20.

        domain_threshold:   int, optional
            Minimum size of a helical segment to be considered candidate for Subdomain A of GAIN domain. Default = 20.

        coil_weight:        float, optional
            Weight assigned to unordered residues during Subdomain detection. Enables decay of helical signal
            default = 0. Recommended values < +0.2 for decay

        explicit_stride_file: str, optional
            Explicit existing STRIDE file for new GAIN domains, when not present in the base dataset. Will manually parse this
            default = None

        seq_name :          str, optional
            Explicit sequence name to be found in the alignment file. Should be specified for GAIN domains not in the base dataset. 
            Overwrites other name variables.

        without_anchors :   bool, optional
            Skips the Detection and calculation of self.Anchors for filtering purposes.

        skip_naming :       bool, optional
            Skips the alternative naming of SSE - for filtering purposes

        is_truncated :      bool, optional
            indicates if the sequence is already truncated to only contain the GAIN domain
        
        truncation_map :     np.array(boolean), optional
            For the workflow.py - if a sequence is added and there is truncation present, this map indicates the truncated residues.

        aln_start_res :     int, optional
            if already predetermined for mafft --add sequences, also pass the start column wtihin the alignment.
        Returns
        ----------
        None
        '''
        #Initialize self.name for finding the correspondent alignment row!
        if name is not None:
            self.name = name
        else:
            if fasta_file:
                self.name = fasta_file.split("/")[-1] # This is how the name would be in the pre-calculated alignment
            else:
                print("No name specified. Exiting")
                return None
        #print(f"\n[DEBUG] gain_classes.GainDomain : Initializing GainDomain Object \n Name : {name}")#.\n{fasta_file = }, {name = }, {sequence = }")

        # Initalize SSE Dict (LOC) and the SSE sequence (ASG) from STRIDE outfiles.
        # Either from the standard folder (base dataset) or from an explicitly stated STRIDE file. (new GAIN)
        if explicit_stride_file:
            # Read directly from the explicitly stated STRIDE file. (new GAIN)
            self.complete_sse_dict = sse_func.read_sse_loc(explicit_stride_file)
            self.sse_sequence = sse_func.read_sse_asg(explicit_stride_file)
        else:
            # Find SSE data in corresponding STRIDE files (base data). Extract corresponding STRIDE files from the list and read SSE from that. (base dataset)
            self.complete_sse_dict = sse_func.find_stride_file(self.name.replace(".fa","")) # previously: fasta_file.split("/")[-1][:-3]
            self.sse_sequence = sse_func.read_sse_asg(self.name.replace(".fa",""))
        # Try to detect GAIN-like order of SSE. Frist criterion is a C-terminal strand being present (= Stachel/TA)
        try: 
            self.end = self.complete_sse_dict['Strand'][-1][1]
            self.isValid = True
        except: 
            print("No Strands detected. This is not a GAIN domain.")
            self.isValid = False
            return      

        # Find the domain boundaries (this includes a check whether the sequence is in fact a GAIN)
        # Will return (None, None) if checks fail. 
        self.start, self.subdomain_boundary = sse_func.find_boundaries(self.complete_sse_dict, 
                                                                       self.end, 
                                                                       bracket_size=subdomain_bracket_size, 
                                                                       domain_threshold=domain_threshold,
                                                                       coil_weight=coil_weight)
        if (self.start is not None):
            self.hasSubdomain = True

        if self.start == None:
            print("No Subdomain boundaries detected. Possible Fragment found.")
            self.hasSubdomain = False
            # For possible Fragment detection (i.e. Subdomain B only sequences), set start as the N-terminal res. of the first beta sheet    
            self.start = np.amin(np.array(self.complete_sse_dict["Strand"]))
        
        # Initialize residue indices as list, starting form zero, indexing EXISTING residues including "X" etc.
        self.index = list(range(0, self.end-self.start+1))
        # Initialize one-letter GAIN sequence as list
        #print(f"[DEBUG] gain_classes.GainDomain :\n\t{self.start = }\n\t{self.end = }\n\t{len(sequence) = }\n\t{self.end-self.start+1 = }")
        if is_truncated:
            # SANITY CHECK: There might occur a case where the GAIN domain is detected anew (i.e. when different parameters are used). There might be a truncation therefore.
            #               If that is the case, truncate the sequence N-terminally to that only ((self.end-self.start+1)) residues are included
            if len(sequence) > (self.end-self.start+1):
                print(f"[DEBUG] gain_classes.GainDomain : {self.name}\nDETECTED ALTERED GAIN DOMAIN DETECTION. TRUNCATING @ RESIDUE : {len(sequence)-self.end+self.start}"
                    f"\n\t{self.start = }\t{self.end = }\n\t{len(sequence) = }\n\t{self.end-self.start+1 = }")
                self.sequence = np.asarray(list(sequence[len(sequence)-self.end+self.start:])) # Begin with the new first residue, end normally
                print(f"[DEBUG]: gain_classes.GainDomain : \n\t {len(sequence) = }, {len(self.sequence) = }\n{sequence}\n{''.join(self.sequence)}")

            elif len(sequence) < (self.end-self.start+1): 
                # This is an edge case where the signal detection identifies a Sheet-segment in Subdomain A. Therefore, non-cons. GAIN domain.
                print(f"[DEBUG] gain_classes.GainDomain : {self.name}\nSEQUENCE LENGTH SHORTER THAN DETECTED GAIN BOUNDARIES.!\n"
                    f"IT WILL BE DECLARED INVALID.\n{len(sequence) =}\n{self.end+self.start = }")
                self.sse_dict = sse_func.cut_sse_dict(self.start, self.end, self.complete_sse_dict)
                print(f"[DEBUG] gain_classes.GainDomain.__init__():\n {self.subdomain_boundary = }, {type(self.subdomain_boundary) = }")
                if self.subdomain_boundary is None :
                    self.subdomain_boundary = 0
                self.plot_helicality(savename=f"{self.name}_SEQSHORT_SKIP.png")
                self.isValid = False
                self.hasSubdomain = False
                return

            else:
                self.sequence = np.asarray(list(sequence))
        if sequence and not is_truncated: 
            self.sequence = np.asarray(list(sequence[self.start:self.end+1]))
        if fasta_file and not is_truncated:
            self.sequence = np.asarray(list(sse_func.read_seq(fasta_file)))[self.start:self.end+1]
        ''' Find the indices of the Alignment where each residue of the sequence is located.
            For base dataset, this will be the base dataset alignment,
            For new GAIN, this will be the alignment appended by the adding method.
            Returns empty list if failed. '''
        #print(f"DEBUG", self.sequence, type(self.sequence), self.sequence.shape)
        #print(f"DEBUG: Getting alignment indices with: {self.name}, {self.sequence.shape = }, {alignment_file} {type(alignment_dict)}")
        #print(f"{self.sequence = }\n{self.start = }\n{self.end = }\n{len(self.sequence) = }")
        
        if truncation_map is not None: 
            cut_truncation_map = truncation_map[self.start:self.end+1]
        else:
            cut_truncation_map = None

        self.alignment_indices = sse_func.get_indices(  name = self.name, 
                                                        sequence = self.sequence, 
                                                        alignment_file = alignment_file, 
                                                        aln_cutoff = aln_cutoff, 
                                                        alignment_dict = alignment_dict,
                                                        truncation_map = cut_truncation_map,
                                                        aln_start_res = aln_start_res
                                                        ) # str(self.sequence) is a Patch!
        #print(f"DEBUG: {self.alignment_indices = }")

        # Check if sse_func.get_indices failed
        try:
            temp = self.alignment_indices[0]
        except: 
            print("[WARNING]: Empty alignment indices detected. If this is unintended, check the alignment file.\n", self.name)
            self.isValid = False
            return

        # Cut down the SSE dictionary down to the GAIN only
        self.sse_dict = sse_func.cut_sse_dict(self.start, self.end, self.complete_sse_dict)

        # get a name map based on enumerating the SSE segments,
        # THIS IS NOT THE ACTUAL NOMENCLATURE BUT A SELF-CONSISTENT METHOD FOR OVERVIEW PURPOSES
        """self.sse_name_map = None
        if not skip_naming:
            self.sse_name_map = sse_func.name_sse(self.sse_dict, 
                                              self.subdomain_boundary, 
                                              self.start, 
                                              self.end,
                                              self.sse_sequence)"""
        
        # Find the GPS residues (triad) based on the alignment column of gps-minus-one (GPS-1 N-terminal residue before cleavage site)
        self.GPS = GPS(self.alignment_indices, 
                       self.sse_dict, 
                       self.index, 
                       self.sequence, 
                       self.start,
                       gps_minus_one=gps_index)
        
        # parse the Quality from the input quality LIST, not the quality file
        # The input as a list is deliberate to make the quality parameter more flexible,
        # You could input any kind of quality signal here
        self.residue_quality = sse_func.get_quality(self.alignment_indices, quality)

        if self.hasSubdomain == True:
            # enumeration + evaluation of subdomain SSE composition
            alpha, beta, a_breaks, b_breaks = sse_func.get_subdomain_sse(self.sse_dict, 
                                                                         self.subdomain_boundary,
                                                                         self.start, 
                                                                         self.end,
                                                                         self.sse_sequence,
                                                                         stride_outlier_mode=stride_outlier_mode)
            # offset correction. Since the PDB residues are one-indexed, we convert them to python zero-indexed. This is obsolete when the start is already at 0.
            diff = self.start-1
            if diff < 0: 
                diff = 0
            self.sda_helices = np.subtract(alpha, diff)
            #print(f"[DEBUG] gain_classes.GainDomain : {alpha = } ,{self.sda_helices = }")
            self.sdb_sheets = np.subtract(beta, diff)
            self.a_breaks = a_breaks
            self.b_breaks = b_breaks
            #print(f"{a_breaks = }, \n {self.a_breaks = }")
        
        if not without_anchors:
            # Gather the respective anchors for this GainDomain
            if not hasattr(self, 'sda_helices'):
                if self.subdomain_boundary is None :
                    self.subdomain_boundary = 0
                self.plot_helicality(savename=f"{self.name}_NO_HELICES.png")
                print(f"[ERROR] gain_classes.__init()__ : NO SDA HELICES DETECTED\n{self.name}")
                self.isValid = False 
                self.hasSubdomain = False
                return
            self.Anchors = Anchors(self)

        if without_anchors:
            self.Anchors = None 

        ### FUTURE CHANGE - The list is mainly for transforming the Alignment for visualization, might be removed later
        def build_sse_sequence(self):
            '''Create a sequence-like list where instead of the residue identifiers, sheets and helices are denoted.'''
            sequence = np.full([self.end-self.start], fill_value='-', dtype='<U1')

            if self.hasSubdomain == True:
                for hel in self.sda_helices:
                    sequence[hel[0]:hel[1]+1] = 'H'

            for she in self.sdb_sheets:
                sequence[she[0]:she[1]+1] = 'E'

            return sequence
        
        if not without_anchors:
            self.sse_sequence_list = build_sse_sequence(self)
    
    #### GainDomain METHODS

    def plot_profile(self, outdir=None, savename=None, noshow=True):
        '''
        Plots the SSE profile and quality profile of the GainDomain. Can be saved into {outdir}
        Also denotes whether this object is a GAIN domain or not.

        Parameters
        ----------
        outdir : str, optional
            Output directory if the Figure should be saved

        Returns
        ----------
        None     
        '''
        # Initialize Figure
        fig = plt.figure(figsize=[8,2])
        fig.set_facecolor("w")
        plt.title(self.name)
        
        # Plot the subdomain A section in blue, subdomain B section in orange
        # If there is not a subdomain boundary, plot everything in black
        if self.subdomain_boundary:
            plt.plot(np.arange(self.start,self.subdomain_boundary,1),
                self.residue_quality[:self.subdomain_boundary-self.start], 
                 color='dodgerblue')
            plt.plot(np.arange(self.subdomain_boundary,self.end+1,1), #self.end+1 or not +1? Seems to cause errors sometimes, depending on truncation method
                 self.residue_quality[self.subdomain_boundary-self.start:], 
                 color='darkorange')
        else:
            plt.plot(np.arange(self.start,self.end+1,1),
                 self.residue_quality, 
                 color='black')

        # The bottom part has the SSE profile plotted as horizontal lines colored to SSE type
        if "Strand" in self.sse_dict.keys():
            for sheet in self.sse_dict["Strand"]:
                plt.hlines(0,sheet[0],sheet[1], linewidth=5, color='darkorange')
                
        if "AlphaHelix" in self.sse_dict.keys():
            for helix in self.sse_dict["AlphaHelix"]:
                plt.hlines(0,helix[0],helix[1], linewidth=5, color='dodgerblue')

        if "310Helix" in self.sse_dict.keys():
            for helix in self.sse_dict["310Helix"]:
                plt.hlines(0,helix[0],helix[1], linewidth=5, color='navy')

        if self.subdomain_boundary:
            plt.vlines(self.subdomain_boundary,0,500, color='black', linewidth=2)
        plt.xlabel("Index")
        plt.ylabel("Conservation Quality")

        # Denote GPS either green (if consensus), red (if not)
        if self.GPS.isConsensus == True:
            col = 'forestgreen'
        else:
            col = 'firebrick'
        ymax = np.amax(self.residue_quality)
        if self.GPS.residue_numbers is not None:
            for idx in self.GPS.residue_numbers:
                plt.vlines(idx,ymax*0.1,ymax*0.9, color=col, linewidth = 1, alpha=0.5)

        # store or show   
        if savename is None and outdir: 
            savename = "%s/%s.png"%(outdir,self.name[:-3])
            savename = savename.replace(":","") # PATCH FOR RARE SPECIAL CHARACTER ":"
        if savename is not None:
            plt.savefig(savename, bbox_inches='tight',dpi=150)
        if not noshow:
            plt.show()
        plt.close(fig)
    
    def plot_helicality(self, bracket_size=30, domain_threshold=20, coil_weight=0, savename=None, debug=False, noshow=True):
        '''
        Plots the SSE profile used for detecting the subdomains. All parameters should be consistent with the ones
        used during actual detection of the GAIN domain
        
        Parameters
        ----------
        bracket_size :      int, optional
            Bracket size for convolving the SSE signal, default = 30
        domain_threshold :  int, optional
            Threshold for a helical block to be considered candidate for Subdomain A, default = 20
        coil_weight :       int, optional
            Coil Weight for decaying helical blocks. Should be 0 (no decay) or below + 0.2, default = 0
        savename :          str, optional
            Output file name for saving, default = None

        Returns
        ----------
        None
        '''
        # Check if the dictionary even contains AlphaHelices, and also check for 310Helix
        helices = []
        sheets = []
        if 'AlphaHelix' not in self.sse_dict.keys() or 'Strand' not in self.sse_dict.keys():
            print("This is not a GAIN domain.")
            return None, None
        [helices.append(h) for h in self.sse_dict['AlphaHelix']]
        if '310Helix' in self.sse_dict.keys(): 
            [helices.append(h) for h in self.sse_dict['310Helix']]
        [sheets.append(s) for s in self.sse_dict['Strand']]

        # Create a scoring matrix similar to the one for finding the Subdomains, 
        # helices are assigned -1, sheets +1
        # Coil_weight emulates the value used in the finding algorithm aswell, replicating the exact profile.
        scored_seq = np.full([self.end], fill_value=coil_weight)
        for element_tuple in helices:
            scored_seq[element_tuple[0]:element_tuple[1]] = -1
        for element_tuple in sheets:
            scored_seq[element_tuple[0]:element_tuple[1]] = 1
        
        # Smooth the SSE signal with np.convolve 
        signal = np.convolve(scored_seq, np.ones([bracket_size]), mode='same')
        if debug:
            print(signal)
        boundaries = sse_func.detect_signchange(signal, exclude_zero=True)
        #print(boundaries)

        # Initialize Figure
        fig = plt.figure(figsize=[8,2])
        fig.set_facecolor("w")
        plt.ylabel("SSE signal")
        plt.xlabel("index")
        plt.plot([val*10 for val in scored_seq], color="gray")
        plt.plot(signal[:self.end],color='dodgerblue')
        plt.hlines(0,0,self.end)
        plt.fill_between(range(self.end), 
                         signal[:self.end], 
                         np.zeros([self.end]),
                        color='dodgerblue',
                        alpha=0.2)
        if boundaries is not None:
            for b in boundaries:
                plt.vlines(b,-20,20, color='gray', linestyle='dashed', linewidth=1)
        plt.vlines(self.subdomain_boundary, -30, 30, color='black', linewidth=1.5)
        if savename != None:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        if not noshow:
            plt.show()
        plt.close(fig)

    def write_sequence(self, savename):
        '''
        Super thin wrapper for calling write2fasta as a GainDomain inherent function
        .
        Parameters
        ----------
        savename : str, required
            Name of the Output file
        .
        Returns
        ----------
        None
        '''
        sse_func.write2fasta(self.sequence, self.name, savename)

    #### CREATING THE NOMENCLATURE

    def create_indexing(self, anchors, anchor_occupation, anchor_dict, outdir=None, offset=0, silent=False, split_mode='single',debug=False):
        ''' 
        Makes the indexing list, this is NOT automatically generated, since we do not need this for the base dataset
        Prints out the final list and writes it to file if outdir is specified
        
        Parameters
        ----------
        anchors : list, required
            List of anchors with each corresponding to an alignment index
        anchor_occupation : list, required
            List of Occupation values corresponding to each anchor for resolving ambiguity conflicts
        anchor_dict : dict, required
            Dictionary where each anchor index is assigned a name (H or S followed by a greek letter for enumeration)
        offfset : int,  optional (default = 0)
            An offsset to be incorporated (i.e. for offsetting model PDBs against UniProt entries)
        silent : bool, optional (default = False)
            opt in to run wihtout so much info.
        outdir : str, optional
            Output directory where the output TXT is going to be written as {self.name}.txt

        Returns
        ---------
        indexing_dir : dict
            A dictionary containing the residue indices for each assigned SSE and the GPS
        indexing_centers : dict
            A dictionary containing the XX.50 Residue for each assigned SSE
        named_residue_dir : dict
            A dictionary mapping individual consensus labels to their respective position.
        unindexed : list
            A list of detected SSE that remain unindexed.
        '''
        # recurring functional blocks used within this function

        def create_name_list(sse, ref_res, sse_name):
            ''' Creates a indexing list for an identified SSE with reference index as X.50,
                Also returns a tuple containing the arguments, enabling dict casting for parsing '''
            number_range = range(50+sse[0]-ref_res, 51+sse[1]-ref_res)
            name_list = [f"{sse_name}.{num}" for num in number_range]
            cast_values = (sse, ref_res, sse_name)
            return name_list, cast_values

        def indexing2longfile(self, nom_list, outfile, offset):
            ''' Creates a file where the indexing will be denoted line-wise per residue'''
            with open(outfile, "w") as file:
                file.write(self.name+"\nSequence length: "+str(len(self.sequence))+"\n\n")
                print("[DEBUG] GainDomain.create_indexing.indexing2longfile\n", nom_list, "\n", outfile)
                for j, name in enumerate(nom_list[self.start:]):
                    file.write(f"{self.sequence[j]}  {str(j+self.start+offset).rjust(4)}  {name.rjust(7)}\n")

        def disambiguate_anchors(self, stored_anchor_weight, stored_res, new_anchor_weight, new_res, sse, break_residues, mode='single'):
            #print(f"[DEBUG] disambiguate_anchors: {stored_anchor_weight = }, {stored_res = }, {new_anchor_weight = }, {new_res = }, {sse = }")
            def terminate(sse, center_res, adj_break_res):
                #print(f"DEBUG terminate : {sse = }, {sse[0] = }, {sse[1] = }, {center_res = }, {adj_break_res = }")
                # neg bound
                offset = 0
                while center_res-offset >= sse[0]:
                    if center_res-offset == sse[0]: 
                        N_boundary = center_res - offset
                        break
                    if center_res-offset in adj_break_res:
                        N_boundary = center_res - offset + 1   # Do not include the BREAK residue itself
                        break
                    offset+=1
                
                offset = 0
                while center_res+offset <= sse[1]:
                    if center_res+offset == sse[1]: 
                        C_boundary = center_res + offset
                        break
                    if center_res+offset in adj_break_res:
                        C_boundary = center_res + offset - 1   # Do not include the BREAK residue itself
                        break
                    offset+=1

                return [N_boundary, C_boundary]

            # this mode indicates if the anchor will just be overwritten instead of split,
            #   this will be set to OFF if there are breaks between both anchors
            priority_mode = True

            # First case, there are no break residues
            if break_residues == []:
                if new_anchor_weight > stored_anchor_weight:
                    name_list, cast_values = create_name_list(sse, new_res, anchor_dict[self.alignment_indices[new_res]])
                else: 
                    name_list, cast_values = create_name_list(sse, stored_res, anchor_dict[self.alignment_indices[stored_res]])
                
                return [(sse, name_list, cast_values)], False

            adjusted_break_residues = [res+sse[0] for res in break_residues]

            # Check if there is a break residue in between the two conflicting anchors
            for break_res in adjusted_break_residues:
                if stored_res < break_res and break_res < new_res:
                    priority_mode = False 
                if new_res < break_res and break_res < stored_res:
                    prority_mode = False

            if priority_mode == True:
                if not silent: 
                    print(f"DEBUG gain_classes.disambiguate_anchors : no break found between anchors, will just overwrite.")
                if new_anchor_weight > stored_anchor_weight:
                    name_list, cast_values = create_name_list(sse, new_res, anchor_dict[self.alignment_indices[new_res]])
                else: 
                    name_list, cast_values = create_name_list(sse, stored_res, anchor_dict[self.alignment_indices[stored_res]])
                
                return [(sse, name_list, cast_values)], False

            # Find the closest break_residue to the lower weight anchor
            if mode == 'single':
                if stored_anchor_weight > new_anchor_weight:
                    lower_res = new_res
                else:
                    lower_res = stored_res

                
                print(f"{adjusted_break_residues = }, {lower_res = }")
                # Go in left and right direction, check where there is the first break
                breaker = None
                offset = 1
                while offset < 20:
                    if lower_res + offset in adjusted_break_residues:
                        offset_idx = adjusted_break_residues.index(lower_res + offset)
                        breaker = lower_res + offset
                        break
                    if lower_res - offset in adjusted_break_residues:
                        offset_idx = adjusted_break_residues.index(lower_res - offset)
                        breaker = lower_res - offset
                        break

                    offset += 1 

                if breaker is None: 
                    print("[ERROR] BREAKER residue not found.") 
                    return None, None

                # Divide SSE via BREAKER into two segments, create two separate name_list instances for them
                seg_N = [sse[0],adjusted_break_residues[offset_idx]-1]
                seg_C = [adjusted_break_residues[offset_idx]+1, sse[1]]

                if stored_res < breaker:
                    seg_stored = seg_N
                    seg_new = seg_C
                else:
                    seg_stored = seg_C
                    seg_new = seg_N

            if mode == 'double':
                seg_stored = terminate(sse, stored_res, adjusted_break_residues)
                seg_new = terminate(sse, new_res, adjusted_break_residues)

            if not silent: print(f"[NOTE] disambiguate_anchors: Split the segment into: {seg_stored = }, {seg_new = }")
            stored_name_list, stored_cast_values = create_name_list(seg_stored, stored_res, anchor_dict[self.alignment_indices[stored_res]])
            new_name_list, new_cast_values = create_name_list(seg_new, new_res, anchor_dict[self.alignment_indices[new_res]])

            #print(f"[NOTE] disambiguate_anchors: Successful SSE split via BREAKER residue @ {breaker}")

            return [(seg_stored, stored_name_list, stored_cast_values),(seg_new, new_name_list, new_cast_values)], True # True indicates if lists were split or not!

        def cast(nom_list, indexing_dir, indexing_centers, sse_x, name_list, cast_values):
            #print(f"DEBUG CAST",sse_x, name_list, cast_values)
            nom_list[sse_x[0]+self.start : sse_x[1]+1+self.start] = name_list
            indexing_dir[cast_values[2]] = cast_values[0] # all to sse 
            indexing_centers[cast_values[2]+".50"] = cast_values[1] # sse_res where the anchor is located
            return nom_list, indexing_dir, indexing_centers

    ### END OF FUNCTION BLOCK

        # Initialize Dictionaries
        indexing_dir = {}
        indexing_centers = {}
        named_residue_dir = {}
        unindexed = []
        # One-indexed Indexing list for each residue, mapping for the actual residue index
        nom_list = np.full([self.end+1], fill_value='      ', dtype='<U7')
        breaks = [self.a_breaks, self.b_breaks]
        if not silent: print(f"{breaks = }")
        for i,typus in enumerate([self.sda_helices, self.sdb_sheets]): # Both types will be indexed separately
            # Where the corresponding breaks are located
            type_breaks = breaks[i]
            if not silent:
                print(type_breaks)

            # Go through each individual SSE in the GAIN SSE dictionary
            for idx, sse in enumerate(typus):
                # Get first and last residue of this SSE
                try:first_col = self.alignment_indices[sse[0]]
                except: continue
                # Error correction. Sometimes the detected last Strand exceeds the GAIN boundary.
                if debug: print(f"DEBUG {sse[1] = }; {self.end-self.start = }, {len(self.alignment_indices) = }")
                if sse[1] > self.end-self.start-1:
                    last_col = self.alignment_indices[-1]
                    sse_end = sse[1]-1
                else:
                    last_col = self.alignment_indices[sse[1]]
                    sse_end = sse[1]
                exact_match = False                             # This is set to True, otherwise continue to Interval search
                fuzzy_match = False                             # Flag for successful Interval search detection
                ambiguous = False                               # Flag for ambiguity
                if debug:print(f"[DEBUG] GainDomain.create_indexing : \n{typus} No. {idx+1}: {sse}")
                if debug:print(f"[DEBUG] GainDomain.create_indexing : \n{first_col = }, {last_col = }")
                
                for sse_res in range(sse[0],sse_end+1):
                    # Find the corresponding alignment index for that residue:
                    if sse_res < len(self.alignment_indices):
                        sse_idx = self.alignment_indices[sse_res]
                    else:
                        continue
                    
                    if sse_idx in anchors:

                        if exact_match == False:  
                            if debug: print(f"PEAK FOUND: @ {sse_res = }, {sse_idx = }, {anchor_dict[sse_idx]}")
                            sse_name = anchor_dict[sse_idx]
                            anchor_idx = np.where(anchors == sse_idx)[0][0]
                            if debug: print(f"{np.where(anchors == sse_idx)[0] = }, {sse_idx = }, {anchor_idx = }")
                            stored_anchor_weight = anchor_occupation[anchor_idx]
                            stored_anchor = anchors[anchor_idx]
                            stored_res = sse_res
                            name_list, cast_values = create_name_list(sse, sse_res, anchor_dict[sse_idx])
                            # name_list has the assignment for the SSE, cast_values contains the passed values for dict casting
                            exact_match = True
                            continue
                        ''' HERE is an ANCHOR AMBIGUITY CASE
                               There might occur the case where two anchors are within one SSE, 
                               check for present break residues in between the two anchors, 
                                > If there are some, eliminate that residue and break the SSE it into two.
                                    >   If there are multiple break residues, use the one closest to the lower occupancy anchor
                                > If there is no break residue the anchor with highest occupation wins. '''
                        ambiguous = True
                        if not silent: 
                            print(f"[NOTE] GainDomain.create_indexing : ANCHOR AMBIGUITY in this SSE:")
                            print(f"\n\t {sse_idx = },")
                            print(f"\n\t {anchor_dict[sse_idx] = },")
                            print(f"{type_breaks = } \t {idx = }")
                            print(f"\n\t {type_breaks[idx] = }")
                        # if the new anchor is scored better than the first, replace!
                        disambiguated_lists, isSplit = disambiguate_anchors(self,
                                                                            stored_anchor_weight=stored_anchor_weight,
                                                                            stored_res=stored_res,
                                                                            new_anchor_weight=anchor_occupation[np.where(anchors == sse_idx)[0][0]],
                                                                            new_res=sse_res,
                                                                            sse=sse,
                                                                            break_residues=type_breaks[idx],
                                                                            mode=split_mode)
                        if not silent: print(disambiguated_lists)
                        sse_adj, name_list, cast_values = disambiguated_lists[0]
                        if isSplit:
                            sse_adj_2, name_2, cast_2 = disambiguated_lists[1]
                            if not silent: print(f"[DEBUG] GainDomain.create_indexing : Found a split list:\n"
                                f"{sse_adj_2  = },\t{name_2 = },\t{cast_2 =  }")
                            nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj_2, name_2, cast_2)
                            # Also write split stuff to the new dictionary
                            for entryidx, entry in enumerate(name_2): 
                                named_residue_dir[entry] = entryidx+sse_adj[0]+self.start
                        # if anchor_occupation[np.where(anchors == sse_idx)[0]] > stored_anchor_weight:
                        #    name_list, cast_values = create_name_list(sse, sse_res, anchor_dict[sse_idx])
                # if no exact match is found, continue to Interval search and assignment.
                if exact_match == False:
                    # expand anchor detection to +1 and -1 of SSE interval
                    ex_first_col = self.alignment_indices[sse[0]-1]
                    try:
                        ex_last_col = self.alignment_indices[sse[1]+1]
                    except:
                        ex_last_col = last_col
                    # Construct an Interval of alignment columns corresp. to the SSE residues
                    if debug:print(f"[DEBUG] GainDomain.create_indexing : \nNo exact match found: extindeing search.\n{typus} No. {idx+1}: {sse}")
                    for peak in anchors: # Look if any peak is contained here
                        
                        if ex_first_col <= peak and ex_last_col >= peak:
                            fuzzy_match = True
                            if not silent: print(f"[DEBUG] GainDomain.create_indexing : Interval search found anchor @ Column {peak}")
                            # Find the closest residue to the anchor column index. N-terminal wins if two residues tie.
                            peak_dists = [abs(self.alignment_indices[res]-peak) \
                                                   for res in range(sse[0], sse_end+1)]

                            ref_idx = peak_dists.index(min(peak_dists))
                            ref_res = range(sse[0], sse_end+1)[ref_idx]

                            if not silent: print(f"NOTE: GainDomain.create_indexing : Interval search found SSE:"
                                                    f"{peak = }, {peak_dists = }, {ref_res = }. \n"
                                                    f"NOTE: GainDomain.create_indexing : This will be named {anchor_dict[peak]}")

                            name_list, cast_values = create_name_list(sse, ref_res, anchor_dict[peak])

                # Finally, if matched, write the assigned nomeclature segment to the array
                
                if ambiguous == False and exact_match == True or fuzzy_match == True:
                    nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse, name_list, cast_values)
                    # Also cast to general indexing dictionary
                    for namidx, entry in enumerate(name_list):
                        named_residue_dir[entry] = namidx+sse[0]+self.start
                elif ambiguous == True:
                    nom_list, indexing_dir, indexing_centers = cast(nom_list, indexing_dir, indexing_centers, sse_adj, name_list, cast_values)
                    # Also cast to general indexing dictionary
                    for namidx, entry in enumerate(name_list): 
                        named_residue_dir[entry] = namidx+sse_adj[0]+self.start
                else: # If there is an unadressed SSE with length 3 or more, then add this to unindexed.
                    if sse[1]-sse[0] > 3:
                        if debug: print(f"[DEBUG] GainDomain.create_indexing : No anchor found! \n {self.alignment_indices[sse[0]] = } \ns{self.alignment_indices[sse_end] = }")
                        unindexed.append(self.alignment_indices[sse[0]])
        # Patch the GPS into the nom_list
        labels = ["GPS-2","GPS-1","GPS+1"]
        for i, residue in enumerate(self.GPS.residue_numbers[:3]):
            #print(residue)
            nom_list[residue] = labels[i]
            indexing_dir["GPS"] = self.GPS.residue_numbers
            # Also cast this to the general indexing dictionary
            named_residue_dir[labels[i]] = self.GPS.residue_numbers[i]
        # FUTURE CHANGE : GPS assignment maybe needs to be more fuzzy -> play with the interval of the SSE 
        #       and not the explicit anchor. When anchor col is missing, the whole SSE wont be adressed
        # print([DEBUG] : GainDomain.create_indexing : ", nom_list)

        # Create a indexing File if specified
        if outdir is not None:
            indexing2longfile(self, nom_list, f"{outdir}/{self.name}.txt", offset=offset)

        return indexing_dir, indexing_centers, named_residue_dir, unindexed


    def write_gain_pdb(self, pdb_file, outfile=None):

        with open(pdb_file, "r") as initial_pdb:
            data = initial_pdb.readlines()

        atom_index = 1                      # For re-numbering the atoms

        with open(outfile, "w") as trunc_pdb:
            for line in data:
                # parse the residue indices of the ATOM entries, only including ones within start and end interval
                if line.startswith("ATOM"):

                    res_id = int(line[22:26])

                    if self.start > res_id: # Cut all residues before the start resId
                        continue
                    if self.end < res_id:   # Cut all residues after the end resId
                        continue

                    trunc_pdb.write(f"{line[:6]}{str(atom_index).rjust(5)}{line[11:]}")
                    atom_index += 1

                elif line.startswith("TER "): # Patch to eliminate the last atom index from the default modeling
                    trunc_pdb.write("TER\n")
                else:
                    trunc_pdb.write(line)

        print(f"[DEBUG]: gain_classes.GainDomain.write_gain_pdb : Created PDB with {atom_index-1} atom entries in {outfile}.")

class Anchors:
    '''
    Subclass of GainDomain which contains all info about the respective anchors of this domain
    The metric of establishing the anchors is based on the quality list (by default generated from Blosum62 in .jal Annotation file)
    But this list can be any list that has the length of the alignment column number
    This will be instantiated once per GainDomain object

    Attributes
    ----------
    sse_names : numpy array
        Contains the anchors with their respective SSE (based on the SELF-CONSISTENT naming scheme)

    quality_values : numpy array
        Contains the individual quality values for each anchor residue

    alignment_indices : numpy array
        Contains the individual alignment index for each anchor Residue. 
        This is used to construct the anchor histogram in the GainCollection class

    gain_residues : numpy array
        Contains the individual residue of the GAIN domain associated with the anchor

    relative_positions : numpy array
        Contains the realtive best position in the SSE (when all SSE residues are zero-indexed)

    Methods
    ---------
    anchor_slice(index)
        Returns info about the anchor at index in the anchor list
    info()
        Prints details from anchor_slice(index) for each anchor
    '''
    def __init__(self, a_gain_domain):
        sse_names = []
        quality_values = []
        alignment_indices = []
        gain_residues = []
        relative_positions = []
        # Mush together the smoothened SSE from both subdomains
        all_sse = np.concatenate((a_gain_domain.sda_helices, a_gain_domain.sdb_sheets), axis=0)

        # print(f"{a_gain.domain.sdb_sheets = }")
        
        #print(f"[DEBUG] gain_classes.Anchors : {a_gain_domain.residue_quality}, \n {all_sse = } ")

        # Get the residue within each SSE of the highest value of the quality metric
        for i in range(all_sse.shape[0]):
            try:
                best_index = all_sse[i,0] + \
                         np.argmax(a_gain_domain.residue_quality[all_sse[i,0]:all_sse[i,1]])
            except:
                continue

            # For which (self-consistently enumerated) SSE is this
            #sse_names.append(a_gain_domain.sse_name_map[best_index])

            # What is the associated quality value
            quality_values.append(a_gain_domain.residue_quality[best_index])   
            # In which alignment column is it
            alignment_indices.append(a_gain_domain.alignment_indices[best_index]) 
            # What is the residue name
            gain_residues.append(a_gain_domain.sequence[best_index])
            # What is the relative best quality position in the respective SSE
            relative_positions.append(best_index)

        # Feed into class attributes
        self.sse_names = np.array(sse_names)
        self.quality_values = np.array(quality_values)
        self.alignment_indices = np.array(alignment_indices)
        self.gain_residues = np.array(gain_residues)
        self.relative_positions = np.array(relative_positions)
        self.count = all_sse.shape[0]
    
    def anchor_slice(self, index):
        '''returns all info about a given anchor at INDEX as list'''
        return [self.sse_names[index], 
                self.quality_values[index], 
                self.alignment_indices[index], 
                self.gain_residues[index],
                self.relative_positions[index]]
    
    def info(self):
        '''plot some general info about the anchor object'''
        for i in range(self.count):
            print(self.anchor_slice(i))
            
class GPS:
    '''
    Subclass of GainDomain checking the GPS and containing all info about it

    Attributes
    ----------
    isConsensus : bool
        Is True if the GPS matches the assigned alignment columns
    indices : list
        List of the GPS residues
    sequence : list
        One-letter amino acid codes of the GPS residues
    alignment_indices : list
        Alignment indices for the GPS residues
    residue_numbers : list
        Residue numbers of the GPS residues in the GAIN domain

    Methods
    ----------
    info()
        Prints info about this particular GPS


    '''
    def __init__(self, alignment_indices, sse_dict, index,  
                 sequence, start, gps_minus_one):
        ''' Initializes the GPS and checks if the GPS column is occupied or not
            If we can detect the -1 (Leu), this is consensus, output -2 to +1 residue
            we don't find it, output the sequence between the two C-terminal beta-sheets of Subdomain B
            The GPS.isConsensus denotes if the GPS is within the consensus in the alignment or not.

        Parameters
        ----------
        alignment_indices : list, required
            Alignment indices of a GainDomain object
        sse_dict :          dict, required
            SSE dictionary of a GainDomain object
        index :             list, required
            residue index list of a GainDomain object
        sequence :          list, required
            One-letter-coded amino acid sequence of a GainDomain object
        start :             int, required
            starting index of the GainDomain
        gps_minus_one :     int, required
            The alignment index of the GPS-1 residue N-terminal of the cleavage site

        Returns
        ----------
        None
        '''
        minus_one_residue = sse_func.detect_GPS(alignment_indices, gps_minus_one)
        #print(f"[DEBUG] gain_classes.GPS : {minus_one_residue = }, {start = }")
        if minus_one_residue:
            #print(f"DEBUG: {start = }")
            self.isConsensus = True
            self.indices = index[minus_one_residue-1:minus_one_residue+2]
            if len(self.indices) < 2:
                print(f"[DEBUG] gain_classes.GPS : not enough GPS residues! {self.indices}")
                #print(f"[DEBUG] gain_classes.GPS : indices specified: {minus_one_residue = }, \n, "
                    #f"{self.indices = }, {index = } {len(index) = }\n,{alignment_indices = },\n {len(alignment_indices) = }, {gps_minus_one = }")
            
        else:
            self.isConsensus = False
            last_sheets = sse_dict["Strand"][-2:]
            print(f"[DEBUG] gain_classes.GPS : ALTERNATIVE GPS DETECTION!",
                f"DEBUG:{last_sheets = }")
            if len(last_sheets) < 2: 
                print("Only one sheet Found. No GPS.")
                self.indices = None
            else:
                self.indices = index[last_sheets[0][1]+1-start:last_sheets[1][0]-start]
        if self.indices is not None:
        #print(f"[DEBUG] gain_classes.GPS : {self.indices = }")
            self.sequence = sequence[self.indices[0]:self.indices[-1]+1]
            self.alignment_indices = alignment_indices[self.indices[0]:self.indices[-1]+1]
            self.residue_numbers = [start+idx for idx in self.indices]

        else:
            self.sequence = None
            self.alignment_indices = None 
            self.residue_numbers = None 

    def info(self):
        ''' prints info about this particular GPS. 
            Returns None '''
        print(f"Info about this GPS:\n{self.isConsensus = }\n{self.indices = }\n"
            f"{self.sequence = }\n{self.alignment_indices = }\n{self.residue_numbers = }")
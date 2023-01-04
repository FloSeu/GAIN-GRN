# Main Script for running the workflow
# Establishing a nomenclature for an input-sequence

# Currently does not contain te data processing method to establish the data underlying the nomenclature from the base dataset!

import argparse, sys, shutil, os
from datetime import date
import glob
import sse_func
from gain_classes import *
import execute
import workaround
# path to binaries which will be called. Can be adjusted accordingly
mafft_bin = "mafft"
stride_bin = "/home/hildilab/lib/stride/stride"

# A slightly modified AlphaFold2 run_docker.py script which enables to pass the output folder as a variable.
# The default script has the output path hardcoded. One could also think about running the python script from this package as well.
alphafold_script = "/home/hildilab/projects/alphafold_db/alphafold/docker/run_docker_nom.py"

#### TEMPORARY: The underlying alignment file to be used as default. Corresponds to --source_alignment
# Can be used for testing. We should have a solid one as basis for the final variant.
# The threshold is used for ensuring that the GPS residues are covered in the aligned sequence (They are some of the most conserved anyway)
		# align_file = '/home/hildilab/projects/agpcr_nom/onlygain/onlygain.mafft.fa'
		# align_threshold = 801
align_file = '/home/hildilab/projects/agpcr_nom/app_gain_gain.mafft.fa'
align_threshold = 6782 # corresponding to GPS+2 (HL--T)

#### TEMPORARY: This hardcoded data corresponds to the data processing output of the default alignment file.
# For generating this data one needs all AlphaFold2 Models and STRIDE data of each working sequence in the alignment!
# Therefore, right now this is static (as the alignment also is)

alignment_length = 6826 #812
alignment_subdomain_boundary = 3425 # 362
gps_minus_one_column = 6781 #799
# Columns of the anchor residues
# onlygain.mafft.jal
#		anchors = [122, 169, 258, 310, 344, 374, 387, 415, 482, 570, 637, 642, 680, 739, 765, 772, 793, 805] 
#    How many sequences for each anchor in case of conflicts
#		anchor_occupation = [150.0, 188.0, 237.0, 197.0, 243.0, 138.0, 189.0, 210.0, 212.0, 241.0, 181.0, 199.0, 164.0, 213.0, 221.0, 242.0, 209.0, 238.0] 
anchors = [ 662, 1194, 1912, 2490, 2848, 3011, 3073, 3260, 3455, 3607, 3998, 4279, 4850, 5339, 5341, 5413, 5813, 6337, 6659, 6696, 6765, 6808 ]

anchor_occupation = [ 4594.0,  6539.0, 11392.0, 13658.0,  8862.0,  5092.0,  3228.0, 14189.0,  
					  9413.0, 12760.0, 9420.0, 11201.0, 12283.0,  3676.0,  4562.0, 13992.0, 
					  12575.0, 13999.0, 14051.0, 14353.0, 9760.0, 14215.0]
#### TEMPORARY: This quality data is corresponding to the default base dataset alignment Blosum62 values. This can be changed accordingly
# 				or specified via --source_quality
# default alignment = '/home/hildilab/projects/agpcr_nom/app_gain_gain.mafft.fa'
default_quality = '/home/hildilab/projects/agpcr_nom/app_gain_gain.mafft.jal' #'/home/hildilab/projects/agpcr_nom/annot/onlygain.mafft.jal'


today = date.today()

# Parse arguments
parser = argparse.ArgumentParser(description='Getting input strings.')
parser.add_argument('--fasta', help='path to your input fasta sequence.', type=str, required=True)
parser.add_argument('--outdir', help='Output directory', type=str, required=True)
parser.add_argument('--alphafold_preset', help='database preset used by alphafold. ["full_dbs", "reduced_dbs"], default = full_dbs', type=str, required=False, default='full_dbs')
parser.add_argument('--alphafold_date', help='Date used for the Alphafold template. Default is today. Format=YYYY-MM-DD', type=str, required=False, default=today.strftime("%Y-%m-%d"))
parser.add_argument('--source_alignment', help='The file used as the source alignment (Alpha!)', type=str, required=False, default=align_file) # This isnt dynamic! Change it for final
parser.add_argument('--forcedelete', help='Forcefully skip the confirmation dialogue for removing the target output-directory', required=False, default=False, action='store_true')
parser.add_argument('--source_quality', help='The file containing the Quality profile (i.e. BLOSUM62) respective to each alignment column.', required=False, default=default_quality, type=str)
parser.add_argument('--existing_pdb', help='If you have an existing structure of your protein and want to skip AlphaFold Modelling, give the file here', required=False, default=None, type=str)
parser.add_argument('--no_mafft', help='Skip MSA step', required=False, default=False, action='store_true')
parser.add_argument('--nodelete', help='Skip deleting files.', required=False, default=False, action='store_true')
parser.add_argument('--nt', help='Number of threads for MAFFT executable', required=False, default=1)
args = parser.parse_args()

print("DEBUG", args)
# Fitting the GAIN-nomenclature to a new GAIN Domain
# workflow for putting in a new sequence

# 0) Initializing folder structure

#outputpath = "%s/%s"%(os.getcwd(),args.outdir)
outputpath = args.outdir
print("DEBUG: outputpath: %s"%outputpath)
print("DEBUG: does it extist?")
print(os.path.exists(outputpath))

# Ensure there will not be full deletion of existing folders!
if not args.nodelete and os.path.exists(outputpath) :

    print("Output directory already exists. Please ensure a fresh new directory.")
    if args.forcedelete == False:

        inp = input("[IMPORTANT NOTE]: Please confirm deletion of directory %s and all its contents: [Y] \nType any other key to abort.\n >> "%outputpath).upper()
        if inp == "Y": shutil.rmtree(args.outdir); print("[NOTE]: Deleted %s and all its contents."%(outputpath))

    else: 
    	print("[IMPORTANT NOTE]: FORCED DELETION OF %s BY --forcedelete FLAG."%args.outdir); shutil.rmtree(outputpath)

if not args.nodelete:
	os.mkdir(outputpath)  
	os.mkdir(f"{outputpath}/alignment")
	print(f"[NOTE]: Created folder {args.outdir} and pdbs subfolder.")

# Copy the sequence file into the respective output_directory

copied_fasta = f"{outputpath}/alignment/{args.fasta.split('/')[-1]}"
shutil.copyfile(args.fasta, copied_fasta)

# 1) Fit in the input sequence and Push it to align onto an existing MSA

seq_name, sequence = sse_func.read_seq(args.fasta, return_name=True)
print(f"NOTE: Sequence of length {len(sequence)} read from input file: {args.fasta}")

# 2) Truncate the sequence after the Stachel for a potential NTF to remain
#      - Ideally, there should be manual confirmation of the truncation. Generally, the GPS ProSite Rule should be handy enough to qualify that
#              https://prosite.expasy.org/PDOC50221
#      - Alternatively, user can provide an already truncated sequence
#	   - Maybe one could implement a visual way to confirm that "correct" truncation occurred.

# For now we use MAFFT --add :
# 		https://mafft.cbrc.jp/alignment/software/
#		10.1093/bioinformatics/bts578

# run MAFFT and extract the MAP file to get the corresponding alignment columns
if not args.no_mafft:
	map_file = execute.run_mafft(mafft_bin, args, copied_fasta)

	truncated_seq = sse_func.truncate_sequence(map_file, sequence, right_threshold=align_threshold)
	print(f"DEBUG {truncated_seq = }")
	# Fetch the existing copied fasta file and add "_tr.fa", write the truncated seq into it
	trunc_file_name = f"{outputpath}/{args.fasta.split('/')[-1][:-3]}_tr.fa"
	print("Writing truncated FASTA file:", trunc_file_name)
	sse_func.write2fasta(sequence=truncated_seq, name=f"{args.fasta.split('/')[-1]}_tr", filename=trunc_file_name)
else:
	map_file = glob.glob(f"{outputpath}/alignment/*.map")[0]
	trunc_file_name = glob.glob(f"{outputpath}/*_tr.fa")[0]
# 3) Feed the truncated SEQ into AlphaFold2
#	   - Define temporary or manually assigned output-dir
#      - Skip AF2 of an existing PDB has been specified

if args.existing_pdb:
	print("NOTE: Existing PDB file specified. Skipping AlphaFold2 for prediction")
	# There exists a case where the provided PDB file is longer than the specified truncated sequence.
	# If that is the case, we want to truncate the PDB to only contain residues that are the FASTA_tr file.
	shutil.copyfile(args.existing_pdb, f'{outputpath}/best_model.pdb')
else:
	print("Running AlphaFold2.")
	execute.run_alphafold(trunc_file_name, outputpath, args.alphafold_date, alphafold_script, alphafold_preset='full_dbs')
	print("Completed AlphaFold2.")
	shutil.copyfile(f'{outputpath}/{trunc_file_name[:-3]}/ranked_0.pdb', f'{outputpath}/best_model.pdb')

# 4) Grab the SSE info from $output-dir/ranked_0.pdb with the STRIDE binary
#      - Define output-dir, one could put the .stride file also directly in the AlphaFold2 results folder
print("Executing STRIDE.")
stride_file = f"{args.outdir}/best_model.stride"
execute.run_stride(f'{outputpath}/best_model.pdb', stride_file, stride_bin)

# 5) Create a GainDomain Object and perform the python analysis
#      - nomeclate it with a pre-set group of anchors that are encoded based on the underlying dataset

# First, check for internal truncation
aln_start_res, truncation_map, has_truncation = workaround.check_internal_truncation(map_file)
# If there is internal truncation, check if this occurs within an SSE!
# if has_truncation:
sse_sequence = np.array(sse_func.read_sse_asg(stride_file))
trunc_stride = workaround.check_truncated_SSE(sequence, sse_sequence, truncation_map)

anchor_dict = sse_func.make_anchor_dict(anchors, alignment_subdomain_boundary)
# Maybe one could impose a way to make that dynamic with a set of presets or somthing

# Read in the quality from the quality file
quality = sse_func.read_quality(args.source_quality)[:alignment_length]
print("DEBUG: Proceeding to GAIN domain evaluation.")
newGain = GainDomain(fasta_file = trunc_file_name,
						alignment_file = f"{outputpath}/alignment/appended_alignment.fa", # This needs to be the appended alignment file - previous: args.source_alignment
						aln_cutoff = alignment_length,
						quality = quality,
						gps_index = gps_minus_one_column,
						explicit_stride_file = stride_file,
						name = seq_name,
						truncation_map = truncation_map,
						aln_start_res = aln_start_res)  
						# Keep in mind that the hardcoded data from the alignment is used here! Alter all the data if you run with another alignment!

newGain.create_indexing(anchors, anchor_occupation, anchor_dict, outdir=outputpath) # presets = ["sequential", "fasta-like"], keep it sequential for now

# 6) Metrics for quality control ? For now, use the quality plot to quickly assess the protein profile with respect to any GAIN detection

newGain.plot_profile(outdir=outputpath)

# NOTE for TESTING: Check if it would work for PKD with extra fold! (That would be fun) -> Maybe we could establish an extra "enhanced nomenclature" for that

print("Done.")
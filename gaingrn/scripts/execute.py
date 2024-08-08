# execute.py
# Function block for running the binaries of MAFFT, AlphaFold2 and STRIDE

import shlex, os
from subprocess import Popen, PIPE

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

def run_mafft(mafft_bin, args, copied_fasta):
	# call MAFFT twice, once for the map (where the truncating can be refined), once for outputting alignment
	# the fasta should be in the outdir/alignment, since the map will be created there too
	mafft_map_command = f"{mafft_bin} --add {copied_fasta} --keeplength --thread {args.nt} --mapout {args.source_alignment}"
	mafft_aln_command = f"{mafft_bin} --add {copied_fasta} --keeplength --thread {args.nt} {args.source_alignment}"

	run_command(mafft_map_command, out_file=f"{args.outdir}/alignment/tmp.fa")
	run_command(mafft_aln_command, out_file=f"{args.outdir}/alignment/appended_alignment.fa")
	
	return f"{copied_fasta}.map"

def run_alphafold(truncated_fasta_file, outputpath, af2date, af2script, alphafold_preset='full_dbs'):
    # call AlphaFold2 from python initializing with truncated seq
    # This requires a working alphafold2 script
    # Note that this is not the batch version of CoabFold, which was used to the complete dataset.
	af2path = "/".join(af2script.split("/")[:-1])
	script_name =  af2script.split("/")[-1]
	alphafold_command = f"python3 {script_name} --fasta_paths={truncated_fasta_file} --preset={alphafold_preset} --outdir={outputpath} --max_template_date={af2date}" #{af2path}/
	run_command(alphafold_command, cwd=af2path)

def run_stride(pdb_file, out_file, stride_bin):
    # Executes the STRIDE binary via the wrapper function
	stride_command = f'{stride_bin} {pdb_file} -f{out_file}'
	run_command(stride_command)

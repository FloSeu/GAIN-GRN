import sys, os

# autocompletion
#import readline
#import rlcompleter
#readline.parse_and_bind('tab: complete')

# pymol environment
#moddir='/opt/pymol-svn/modules'
#sys.path.insert(0, moddir)
#os.environ['PYMOL_PATH'] = os.path.join(moddir, 'pymol/pymol_path')

# pymol launching: quiet (-q), without GUI (-c) and with arguments from command line
import pymol
test_pdb1 = '../human_31/uniprot_indexed_pdbs/O14514-AGRB1_HUMAN-AGRB1-Homo_sapiens_gain.pdb'
test_pdb2 = '../human_31/uniprot_indexed_pdbs/Q6QNK2_B2CKK9_B7ZLF7_Q2M1L3_Q6ZMQ1_Q7Z7M2_Q86SM4-AGRD1_HUMAN-AGRD1-Homo_sapiens_gain.pdb'
pymol.pymol_argv = ['pymol','-qc']#,f'{test_pdb1}', f'{test_pdb2}']#+ sys.argv[1:]
pymol.finish_launching()
#cmd = pymol.cmd


#print(sys.argv)
#pdb1 = sys.argv[1]
#pdb2 = sys.argv[2]

pymol.cmd.load(test_pdb1)
pymol.cmd.load(test_pdb2)
pymol.cmd.alignto(test_pdb1)
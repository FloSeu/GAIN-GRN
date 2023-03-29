import sys
from pymol import cmd

pdb1 = sys.argv[1]
pdb2 = sys.argv[2]

cmd.load(pdb1)
cmd.load(pdb2)
alignto(pdb1)
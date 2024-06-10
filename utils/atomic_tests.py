from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *

from pyrosetta.rosetta.protocols.carbohydrates import *
from pyrosetta.rosetta.core.select.residue_selector import *
from pyrosetta.rosetta.core.simple_metrics.metrics import *
from pyrosetta.rosetta.core.simple_metrics.composite_metrics import *
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import *
from pyrosetta.rosetta.core.pose import pose_from_saccharide_sequence

from scipy.spatial.transform import Rotation as R

from carb_utils import *

options = """
-beta
-include_sugars
-alternate_3_letter_codes pdb_sugar

-write_pdb_link_records
-auto_detect_glycan_connections
-ignore_unrecognized_res
-out:level 100
"""

#-out:level 100

init(" ".join(options.split('\n')))

import os
import numpy as np
import pandas as pd
import copy

input_dir = "./"
os.chdir(input_dir)

if __name__ == '__main__':

    #make the glycan in rosetta
    sc = get_score_function()
    fr = pyrosetta.rosetta.protocols.relax.FastRelax()
    fr.set_scorefxn(sc)
    fr.max_iter(50)

    s = '-alpha-D-Manp->4)-alpha-D-ManpNAc->2)-alpha-D-Glcp->2)-alpha-D-Galp'
    #carb = pose_from_saccharide_sequence("a-Manp6Ac-(1->3)-[b-GlcpA-(1->2)]-a-Manp6Ac-(1->3)-a-Manp6Ac-(1->3)-[b-Xylp-(1->2)]-a-Manp-(1->3)-[b-GlcpA-(1->2)]-a-Manp6Ac-(1->3)-a-Manp6Ac-(1->3)-[b-Xylp-(1->2)]-a-Manp-(1->3)-a-Manp6Ac-(1->3)")
    carb = pose_from_saccharide_sequence(s)
    fr.apply(carb)

    #make the polymer
    p = pyrosetta_to_poly(carb)
    p2 = copy.deepcopy(p)


    #calculate rmsd
    print('RMSD\tRing\tAtom')
    print('\t',ring_rmsd(p,p2),'\t',atom_rmsd(p,p2))

    print("Angle_diff: ",np.sum(get_angle_diff(p,p2)))
    #print("Coms:\t",p.monos[0].get_ring_com(),'\t',p.monos[-1].get_ring_com())

    #Align the stuff and recalculate
    #align(p,p2)
    #print('ALIGN WITH NOTHING: RMSD\tRing\tAtom')
    #print('\t',ring_rmsd(p,p2),'\t',atom_rmsd(p,p2))


    print("\nRotating p \n")

    for jj in range(50):
        #find edge and then perturb
        r = np.random.randint(len(p.monos))
        edges = p.edges[r]
        a = np.random.randint(len(edges) )
        e = edges[ a ]
        deg = np.random.normal(0,1)

        p.dihedral_rotation(r,e,deg,degrees=True)


    #print("Coms:\t",p.monos[0].get_ring_com(),'\t',p.monos[-1].get_ring_com())

    print('Unaligned RMSD\tRing\tAtom')
    print('\t',ring_rmsd(p,p2),'\t',atom_rmsd(p,p2))

    out = "MODEL    1" + '\n'
    out += p2.output_pdb()
    out += 'TER\nENDMDL\n'
    out += "MODEL    2" + '\n'
    out += p.output_pdb()
    out += 'TER\nENDMDL\n'

    f = open('../test_pdbs/un_aligned.pdb','w+')
    f.write(out)
    f.close()

    print('\nAligning')
    align(p,p2)
    #print("\nComs:\t",p.monos[0].get_ring_com(),'\t',p.monos[-1].get_ring_com())

    print('RMSD\tRing\tAtom')
    print('\t',ring_rmsd(p,p2),'\t',atom_rmsd(p,p2))
    print("Angle_diff: ",np.sum(get_angle_diff(p,p2)))

    out = "MODEL    1" + '\n'
    out += p2.output_pdb()
    out += 'TER\nENDMDL\n'
    out += "MODEL    2" + '\n'
    out += p.output_pdb()
    out += 'TER\nENDMDL\n'

    f = open('../test_pdbs/aligned.pdb','w+')
    f.write(out)
    f.close()

    print("Big Rotation")
    for jj in range(150):
        #find edge and then perturb
        r = np.random.randint(len(p.monos))
        edges = p.edges[r]
        a = np.random.randint(len(edges) )
        e = edges[ a ]
        deg = np.random.normal(0,10)

        p.dihedral_rotation(r,e,deg,degrees=True)

    align(p,p2)
    #print("\nComs:\t",p.monos[0].get_ring_com(),'\t',p.monos[-1].get_ring_com())

    print('\nRMSD\tRing\tAtom')
    print('\t',ring_rmsd(p,p2),'\t',atom_rmsd(p,p2))
    print("Angle_diff: ",np.sum(get_angle_diff(p,p2)))

import sys
import os
import csv

# PATH TO molfile_to_params.py
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from alignment import *

import pandas as pd
import argparse

class Conformer():
    # A class that stores all of the necessary information about conformers to be used by the other scripts
    
    def __init__(self, pose, conf_num, mol_name, mol_id, lig_aid, t_aid = None, t_coords = None, ligand_residue = 1):
        self.pose = pose
        self.conf_num = conf_num
        self.name = mol_name
        self.id = mol_id
        self.lig_aid = lig_aid
        self.t_aid = t_aid
        self.t_coords = t_coords
        self.ligand_residue = ligand_residue

        self.pose.pdb_info().name(f"{self.name}, {self.id}")

    def align_to_target(self, res = None):
        if res:
            align_ligand_to_residue(self.pose, res, self.lig_aid, self.t_aid)
        else:
            align_ligand_to_target_coords(self.pose, self.lig_aid, self.t_coords)

    def check_collision(self, grid):
        return grid.check_collision(self.pose.residue(self.ligand_residue))

    def determine_functional_groups(self, verbose = False, bioisostere = False):
        """
        Determines the functional groups for a ligand
        If bioisostere is turned on it will approximate some functional groups as
        vdM-compatible ones

        Returns
        List of tuples of (group name, atomnos, xyz coordinate matrix)
        e.g. (coo, [atomno1, atomno2, atomno3], [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])


        """
        lig = self.pose.residue(self.ligand_residue)
        ligtype = lig.type()

        # List of tuples of (group name, atomnos, xyz coordinate matrix)
        # e,g, (coo, [atomno1, atomno2, atomno3], [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        groups = []

        for atomno in range(1, lig.natoms() + 1):
            #print(ligtype.bonded_neighbor_types(atomno))
            elem = lig.atom_type(atomno).element()

            nbrnos = list(ligtype.nbrs(atomno))
            nbrs = [lig.atom_type(nbrno).element() for nbrno in nbrnos]
            bonds = [str(ligtype.bond_type(atomno, nbrno)) for nbrno in nbrnos]
            sorted_nbrs = sorted(nbrs)
            #print(elem, nbrs)


            if elem == "C":

                if sorted_nbrs == ["C", "O", "O"]:
                    nbrnos_o = [e for e in ligtype.nbrs(atomno) if lig.atom_type(e).element() == "O"]

                    if  [lig.atom_type(nbrnbrno).element() for nbrnbrno in list(ligtype.nbrs(nbrnos_o[0])) + list(ligtype.nbrs(nbrnos_o[1]))].count("H") == 1:
                        
                        # CARBOXYLIC ACIDS (CD, OE1, OE2 (OH))
                        if verbose: print("Detected: coo")
                        fg_name = "coo"
                        xyz_c = lig.xyz(atomno)
                    
                        xyz_oh = lig.xyz(nbrnos_o[0])
                        xyz_o = lig.xyz(nbrnos_o[1])
                        xyzs = np.array([xyz_c, xyz_o, xyz_oh])
                        atomnos = np.array([atomno, nbrnos_o[1], nbrnos_o[0]])
                        groups.append((fg_name, atomnos, xyzs))
                        continue 

                    else:
                        
                        if bioisostere:
                            # ESTERS, treat esters like CCO
                            if verbose: print("Detected: cco, ester")
                            fg_name = "bb_cco"
                            xyz_c = lig.xyz(atomno)
                            xyz_ca = lig.xyz(nbrnos[nbrs.index("C")])

                            for bond, nbr, nbrno in zip(bonds, nbrs, nbrnos):
                                if bond == "BondName.DoubleBond" and nbr == "O":
                                    xyz_o = lig.xyz(nbrno)
                                    break
                            xyzs = np.array([xyz_ca, xyz_c, xyz_o])
                            atomnos = np.array([nbrnos[nbrs.index("C")], atomno, nbrno])
                            groups.append((fg_name, atomnos, xyzs))  
                            continue      



                elif ("O" in nbrs and "N" in nbrs and "BondName.DoubleBond" in [e for i, e in enumerate(bonds) if nbrs[i] == "O"]):
                    
                    # AMIDES (C O N)
                    if verbose: print("Detected: conh2")
                    fg_name = "conh2"
                    xyz_c = lig.xyz(atomno)
                    xyz_n = lig.xyz(nbrnos[nbrs.index("N")])

                    for nbr, nbrno, bond in zip(nbrs, nbrnos, bonds):
                        if nbr == "O" and bond == "BondName.DoubleBond":
                            xyz_o = lig.xyz(nbrno)
                            break
                    xyzs = np.array([xyz_c, xyz_o, xyz_n])
                    atomnos = np.array([atomno, nbrno, nbrnos[nbrs.index("N")]])
                    groups.append((fg_name, atomnos, xyzs))
                    continue 

                elif nbrs.count("O") == 1 and "C" in nbrs:
                    
                    for bond, nbr,nbrno in zip(bonds, nbrs, nbrnos):
                        if nbr == "O" and bond == "BondName.DoubleBond":
                            
                            # CARBONYLS (CA, C, O)
                            if verbose: print("Detected: bb_cco")
                            fg_name = "bb_cco"
                            xyz_c = lig.xyz(atomno)
                            xyz_o = lig.xyz(nbrno)

                            xyz_ca = lig.xyz(nbrnos[nbrs.index("C")])
                            xyzs = np.array([xyz_ca, xyz_c, xyz_o])
                            atomnos = np.array([nbrnos[nbrs.index("C")], atomno, nbrno])
                            groups.append((fg_name, atomnos, xyzs))
                            break


            elif elem == "S":
                
                if sorted_nbrs == ["C", "C"]:
                    
                    # THIOETHERS (CG, SD, CE)
                    if verbose: print("Detected: csc")
                    fg_name = "csc"
                    xyz_s = lig.xyz(atomno)
                    xyz_c1 = lig.xyz(nbrnos[0])
                    xyz_c2 = lig.xyz(nbrnos[1])
                    xyzs = np.array([xyz_c1, xyz_s, xyz_c2])
                    atomnos = np.array([nbrnos[0], atomno, nbrnos[1]])
                    groups.append((fg_name, atomnos, xyzs))
                    continue 
                
                elif sorted_nbrs == ["C", "H"]:
                    
                    # THIOLS (CB, SG, HG)
                    if verbose: print("Detected: csh")
                    fg_name = "csh"
                    xyz_s = lig.xyz(atomno)
                    xyz_c = lig.xyz(nbrnos[nbrs.index("C")])
                    xyz_h = lig.xyz(nbrnos[nbrs.index("H")])
                    xyzs = np.array([xyz_c, xyz_s, xyz_h])
                    atomnos = np.array([nbrnos[nbrs.index("C")], atomno, nbrnos[nbrs.index("H")]])
                    groups.append((fg_name, atomnos, xyzs))
                    continue 
                
                elif sorted_nbrs == ["C"]:
                    
                    if bioisostere:
                    # THIOKETONE
                        if verbose: print("Detected: cco, thioketone")
                        fg_name = "bb_cco"
                        xyz_s = lig.xyz(atomno)
                        xyz_c = lig.xyz(nbrnos[0])
                        nbrnbrnos = list(ligtype.nbrs(nbrnos[0]))
                        for nbrnbrno in nbrnbrnos:
                            if lig.atom_type(nbrnbrno).element() == "O":
                                continue
                            else:
                                xyz_ca = lig.xyz(nbrnbrno)
                                break
                        xyzs = np.array([xyz_ca, xyz_c, xyz_s])
                        atomnos = np.array([nbrnbrno, nbrnos[0], atomno])
                        groups.append((fg_name, atomnos, xyzs))
                        continue 


            elif elem == "O":

                if sorted_nbrs == ["C", "H"]:
                    for nbrno in nbrnos:
                        if lig.atom_type(nbrno).element() == "C":
                            nbrnbrnos = list(ligtype.nbrs(nbrno))
                            nbrnbrs = [lig.atom_type(nbrnbrno).element() for nbrnbrno in nbrnbrnos]
                            if sorted(nbrnbrs) != ["C", "O", "O"]:

                                bonds = [str(ligtype.bond_type(nbrno, nbrnbrno)) for nbrnbrno in ligtype.nbrs(nbrno)]
                                if sorted(nbrnbrs) == ["C", "C", "O"] and "BondName.DoubleBond" in bonds:
                                    
                                    # PHENOL (CZ, OH, HH)
                                    if verbose: print("Detected: phenol")
                                    fg_name = "phenol"
                                    xyz_o = lig.xyz(atomno)
                                    xyz_c = lig.xyz(nbrno)
                                    xyz_h = lig.xyz(nbrnos[nbrs.index("H")])
                                    
                                    c_nbrs = ligtype.nbrs(nbrno)
                                    xyzs_c = [lig.xyz(c_nbr) for c_nbr in c_nbrs if lig.atom_type(c_nbr).element() == "C"]
                                    xyzs = [xyz_c, xyz_o]
                                    atomnos = [nbrno, atomno]
                                
                                    xyzs += [xyzs_c[0], xyzs_c[1]]
                                    atomnos += [c_nbr for c_nbr in c_nbrs if lig.atom_type(c_nbr).element() == "C"]

                                    groups.append((fg_name, np.array(atomnos), np.array(xyzs)))

                                else:
                                    
                                    # HYDROXYL (CB, OG1, HG1)
                                    if verbose: print("Detected: coh")
                                    fg_name = "coh"
                                    xyz_o = lig.xyz(atomno)
                                    xyz_c = lig.xyz(nbrno)
                                    xyz_h = lig.xyz(nbrnos[nbrs.index("H")])
                                    xyzs = np.array([xyz_c, xyz_o, xyz_h])
                                    atomnos = np.array([nbrno, atomno, nbrnos[nbrs.index("H")]])
                                    groups.append((fg_name, atomnos, xyzs))

                            break

                if sorted_nbrs == ["C", "C"]:
                    if bioisostere:

                        # ETHER (C, O, C), treat like THIOETHER
                        if verbose: print("Detected: csc")
                        fg_name = "csc"
                        xyz_o = lig.xyz(atomno)
                        xyz_c1 = lig.xyz(nbrnos[0])
                        xyz_c2 = lig.xyz(nbrnos[1])
                        xyzs = np.array([xyz_c1, xyz_o, xyz_c2])
                        atomnos = np.array([nbrnos[0], atomno, nbrnos[1]])
                        groups.append((fg_name, atomnos, xyzs))
                        continue 

            elif elem == "N":

                if sorted_nbrs == ["C", "H", "H"]:
                    for nbrno in nbrnos:
                        nbrnbrs = [lig.atom_type(nbrnbrno).element() for nbrnbrno in lig.nbrs(nbrno)]
                        nbrnbrbonds = [str(ligtype.bond_type(nbrno, nbrnbrno)) for nbrnbrno in lig.nbrs(nbrno)]
                        for nbr, bond in zip(nbrnbrs, nbrnbrbonds):
                            if nbr == "O" and bond == "BondName.DoubleBond":
                                break
                        else:
                            continue
                        break
                    else:
                        
                        # PRIMARY AMINE
                        if verbose: print("Detected: ccn")
                        fg_name = "ccn"
                        xyz_n = lig.xyz(atomno)
                        xyz_c = lig.xyz(nbrnos[nbrs.index("C")])
                        xyz_h = lig.xyz(nbrnos[nbrs.index("H")])

                        xyzs = np.array([xyz_c, xyz_n, xyz_h])
                        atomnos = np.array([nbrnos[nbrs.index("C")], atomno, nbrnos[nbrs.index("H")]])
                        groups.append((fg_name, atomnos, xyzs))
                        continue

                elif sorted_nbrs == ["C", "C", "H"] or sorted_nbrs == ["C", "H", "N"]:
                    
                    break_flag = False
                    for nbrno in nbrnos:
                        if lig.atom_type(nbrno).element() == "C":
                            nbrnbrnos = list(lig.nbrs(nbrno))
                            nbrnbrs = [lig.atom_type(nbrnbrno).element() for nbrnbrno in nbrnbrnos]
                            nbrnbrbonds = [str(ligtype.bond_type(nbrno, nbrnbrno)) for nbrnbrno in lig.nbrs(nbrno)]

                            for nbrnbr, nbrbond, nbrnbrno in zip(nbrnbrs, nbrnbrbonds, nbrnbrnos):
                                if nbrnbr == "N" and nbrbond == "BondName.DoubleBond":
                                    
                                    # HISTIDINE
                                    if verbose: print("Detected: hid/hie")
                                    fg_name = "hie"
                                    xyz_nh = lig.xyz(atomno)
                                    xyz_c = lig.xyz(nbrno)
                                    xyz_n = lig.xyz(nbrnbrno)

                                    xyzs = np.array([xyz_nh, xyz_c, xyz_n])
                                    atomnos = np.array([atomno, nbrno, nbrnbrno])
                                    groups.append((fg_name, atomnos, xyzs))

                                    break_flag = True
                                    break
                            else:
                                continue
                            break
                    if break_flag:
                        continue

                    for nbrno in nbrnos:
                        nbrnbrs = [lig.atom_type(nbrnbrno).element() for nbrnbrno in lig.nbrs(nbrno)]
                        if len(nbrnbrs) == 3 and "O" in nbrnbrs:
                            break
                    else:
                        
                        # SECONDARY AMINE (N , CA, H)
                        if verbose: print("Detected: bb_cnh")
                        fg_name = "bb_cnh"
                        xyz_n = lig.xyz(atomno)
                        xyz_h = lig.xyz(nbrnos[nbrs.index("H")])
                        for nbrno in nbrnos:
                            if lig.atom_type(nbrno).element() == "H":
                                continue
                            else:
                                xyz_ca = lig.xyz(nbrno)
                                break
                        xyzs = np.array([xyz_n, xyz_ca, xyz_h])
                        atomnos = np.array([atomno, nbrno, nbrnos[nbrs.index("H")]])
                        groups.append((fg_name, atomnos, xyzs))
                        continue

        return groups

    def get_residue(self):
        return self.pose.residue(self.ligand_residue)


def yield_ligand_poses(df, path_to_conformers, post_accepted_conformers, ligand_residue = 1):
    """
    Python generator which lazily loads in conformers from params files and pdb files as needed to avoid
    memory issues

    Arguments:
    pkl_file with the following columns:
        Molecule Name, Molecule ID, Molecule File Stem, Conformer Range, Molecule Atoms, Residue Atoms

        Molecule name: Name of the molecule as determined by first line of SDF
        Molecule ID: ID of the molecule as determiend by the third line of SDF
        Molecule File Stem: Where in path_to_conformers the params file/pdb files are located and their naming scheme
            e.g. if use_mol_id is on and the id is 12, it would be 12/12.params and 12/12_0001.pdb, 12/12_0002.pdb etc
            all inside of path_to_conformers
        Conformer Range: How many conformers were used in generating the params file
            e.g. 1_1 for 1 conformer, 1_100 for 100 conformers etc
        Molecule Atoms: MANUAL ENTRY atom labels for atoms on the conformer pdbs that correspond to target atoms
            separated by dashes e.g. C1-C4-C6 denotes three atoms C1, C4, C6 in that order
        Target Atoms: MANUAL ENTRY atom labels for target atoms on a Residue Object
            separated by dashes e.g. CD2-CZ2-CZ3 denote three atoms CD2, CZ2, CZ3 in that order on the target residue

    path_to_conformers: String location for a directory which contains all of the ligand directories
    post_clash: Whether or not to consider accepted conformers or not

    Returns:
    generator: yields (lig, i, mol_name, mol_id, lig_aid, res_aid) where
        lig: Pose object for the ligand
        i: int conformer number
        mol_name: molecule name of the ligand
        mol_id: moleucle id of the ligand
        lig_aid: atom labels of the ligand in a list, e.g. ["C1", "C4", "C6"]
        res_aid: atom labels of the target residue in a list, e.g. ["CD2", "CZ2", "CZ3"]

    upon switching between rows in the csv_file it will yield None as a flag before continuing

    """

    if not post_accepted_conformers:
        for index, row in df.iterrows():
            lig = Pose()
            
            file_stem = row["Molecule File Stem"]
            conformer_range = [int(e) for e in row["Conformer Range"]]
            conformer_range[1] += 1

            params_file = f"{path_to_conformers}/{file_stem}.params"
            res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list = [params_file])


            for i in range(*conformer_range):
                conformation_file = f"{path_to_conformers}/{file_stem}_{i:04}.pdb"
                pose_from_file(lig, res_set, conformation_file)
                t_aln = row["Target Atoms"]
                try:
                    float(t_aln[0])

                    t_atoms = [float(e) for e in row["Target Atoms"]]
                    conf = Conformer(lig, i, row["Molecule Name"], row["Molecule ID"], row["Molecule Atoms"], t_coords = t_atoms, ligand_residue = ligand_residue)
                except ValueError:
                    conf = Conformer(lig, i, row["Molecule Name"], row["Molecule ID"], row["Molecule Atoms"], t_aid = t_aln, ligand_residue = ligand_residue)
                yield conf
            yield None
    else:     
        for index, row in df.iterrows():
            lig = Pose()
            
            file_stem = row["Molecule File Stem"]
            conformers = [int(e) for e in row["Accepted Conformers"]]

            params_file = f"{path_to_conformers}/{file_stem}.params"
            res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list = [params_file])


            for i in conformers:
                conformation_file = f"{path_to_conformers}/{file_stem}_{i:04}.pdb"
                pose_from_file(lig, res_set, conformation_file)
                t_aln = row["Target Atoms"]
                try:
                    float(t_aln[0])

                    t_atoms = [float(e) for e in row["Target Atoms"]]
                    conf = Conformer(lig, i, row["Molecule Name"], row["Molecule ID"], row["Molecule Atoms"], t_coords = t_atoms, ligand_residue = ligand_residue)
                except ValueError:
                    conf = Conformer(lig, i, row["Molecule Name"], row["Molecule ID"], row["Molecule Atoms"], t_aid = t_aln, ligand_residue = ligand_residue)
                yield conf
            else:
                yield None












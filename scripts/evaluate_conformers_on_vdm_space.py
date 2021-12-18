import sys
import os
import math
import time
from collections import defaultdict

from precalculate_vdm_space import orientation_dependent_metric, rmsd_distance_metric, vdM
from configparser import ConfigParser
import argparse



import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import BallTree
import prody
from prody import writePDB, AtomGroup, parsePDB


VDM_LABELS = {"bb_cco": ["CA", "C", "O"],
              "bb_cnh": ["N", "CA", "H"],
              "ccn-LYS": ["CE", "NZ", "HD2"],
              "coh-THR": ["CB", "OG1", "HG1"],
              "coh-SER": ["CB", "OG", "HG"],
              "conh2-ASN": ["CG", "OD1", "ND2"],
              "conh2-GLN": ["CD", "OE1", "NE2"],
              "coo-GLU": ["CD", "OE1", "OE2"],
              "coo-GLU-flip": ["CD", "OE2", "OE1"],
              "coo-ASP": ["CG", "OD1", "OD2"],
              "coo-ASP-flip": ["CG", "OD2", "OD1"],
              "csc-MET": ["CG", "SD", "CE"],
              "csc-MET-flip": ["CE", "SD", "CG"],
              "csh-CYS": ["CB", "SG", "HG"],
              "hie-HIS": ["NE2", "CE1", "ND1"],
              "phenol-TYR": ["CZ", "OH", "CE1", "CE2"],
              "phenol-TYR-flip": ["CZ", "OH", "CE2", "CE1"]
              }

THREE_TO_ONE = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


def determine_vdm_set(possibilities, num_fgs):
    """
    Given all of the possible vdMs, determine the best set of possible vdMs
    considering residuetype clashes at positions, and output some data

    Arguments
    possibilities: a dictionary mapping position residue numbers to lists that are vdMs
    [vdM score, Residue type, i, an index that maps to a unique FG on a ligand, instance, a df that contains info about the vdM such as sc coords]
    num_fgs: the number of FGs on the ligand, used to determine satisfaction of FGs

    Returns:
    vdm_set: A list of the best arrangement of vdMs [vdM score, Residuee type, i, an index that maps to a unique FG on a ligand, instance, a df that contains info about the vdM such as sc coords]
    position_residues: a dictionary mapping position residue numbers to residue types
    total_score: total vdW score for the vdw set
    satisfied: a boolean array containing whether or not each FG was satisfied

    """
    vdm_set = []
    total_score = 0
    satisfied = [False for _ in range(num_fgs)]
    position_residues = {}
    for position, vdms in possibilities.items():
        residues = defaultdict(float)


        vdms = pd.DataFrame(vdms)
        vdms.columns = ["Score", "Residue", "i", "Instance"]
        vdms = vdms.sort_values("Score").drop_duplicates(["Residue", "i"], keep = 'last')

        best_residue = vdms.groupby(["Residue"])["Score"].sum().idxmax()
        best_score = vdms.groupby(["Residue"])["Score"].sum()[best_residue]

        total_score += best_score
        position_residues[position] = best_residue
        vdms = vdms.loc[vdms["Residue"] == best_residue]
        for i, row in vdms.iterrows():
            vdm_set.append(list(row))

        for i in vdms["i"].values:
            satisfied[i] = True
    return vdm_set, position_residues, total_score, satisfied


def align_ligands_to_residue_and_score_from_generator(pose, res, path_to_conformers, df_file_name, pkl_file, vdm_file_stem, 
    rmsd_cutoff = 1.0, lerp = False, bioisostere = False, print_to_pdb = True, path_to_vdms = "../vdm_pdbs"):
    """
    Aligns ligands from pkl files to a certain residue, finds their vdM score based off the vdm space generated from
    precalculate_vdw_space, and outputs info into a pkl file database named df_file_name

    Arguments:
    pose: pose from which the vdm space was generated
    res: residue object to align each ligand to
    path_to_conformers: directory where conformer files are stored
    df_file_name: name for the final DataFrame pkl file
    pkl_files: Pickled DataFrame files which contain all of the conformer information
    vdm_file_stem: contains all of the information from the vdm space precalculation
    rmsd_cutoff: what cutoff value to use for the NN graph
    lerp: a boolean, whether to linear interpolate scores from 0 to score based off RMSD
    bioisostere: a boolean, whether to include similar functional groups to approximate other interactions
    print_to_pdb: a boolean, whether to print out the vdm interactions
    path_to_vdms: where to store the vdm interactions

    Outputs:
    A dataframe file containing information about all of the outputted vdMs as well as
    pdb files for each vdM created


    """
    t0 = time.time()

    print("\nBegan reading pkl files")
    trees = pd.read_pickle(f"{vdm_file_stem}-trees.pkl")
    info = pd.read_pickle(f"{vdm_file_stem}-info.pkl")
    print(f"Completed reading pkl files, time taken {(time.time() - t0)/60} minutes")

    t0 = time.time()
    print("\nBegan working on ligands")

    try:
        print(f"Attempting to make {path_to_vdms}")
        os.mkdir(path_to_vdms)
    except:
        print(f"{path_to_vdms} already made, continuing")

    # Precalculate protein collision grid
    pose_grid = collision_check.CollisionGrid(pose, bin_width = 1, vdw_modifier = 0.5, include_sc = False)


    df = []


    ligand_generator = conformer_prep.yield_ligand_poses(df = pd.read_pickle(pkl_file), path_to_conformers = path_to_conformers, post_accepted_conformers = True)
                        
    for count, pose_info in enumerate(ligand_generator):
        if not pose_info:
            continue

        # Grab conformer from generator
        conf = pose_info

        # Align to residue
        conf.align_to_target(res)


        ligand_grid = collision_check.CollisionGrid(conf.pose, bin_width = 0.5, vdw_modifier = 0.5, include_sc = True)
        fgs = conf.determine_functional_groups(bioisostere = bioisostere)
        
        possibilities = defaultdict(list)
        for i, (group_name, atomnos, coords) in enumerate(fgs):
            # Query the nearest neighbor graphs ofr nearby vdm interactions
            tree = trees[group_name]
            inds, dists = tree.query_radius([coords.flatten()], r = rmsd_cutoff, return_distance = True)
            inds = list(inds[0])
            dists = list(dists[0])

            for ind, dist in zip(inds, dists):
                vdM = info[group_name][ind]
                instance = vdM.get_instance()
                sc_residue = vdM.residue
                sc_coords = vdM.sc_coords
                

                # Check if the vdm interactions' sidechain collides with the ligand or the protein backbone
                sc_ligand_check = ligand_grid.check_collision_matrix(sc_coords)
                if sc_residue in ["LYS", "GLU", "ASN", "ASP", "GLN", "PHE", "TRP", "TYR", "MET", "ARG"]:
                    sc_pose_check = pose_grid.check_collision_matrix(sc_coords[8:])
                else:
                    sc_pose_check = False

                if not sc_ligand_check and not sc_pose_check:
                    score = vdM.score

                    if lerp:
                        score = (score) * (rmsd_cutoff - dist)

                    position = vdM.position
                    
                    possibilities[position].append([score, sc_residue, i, instance])
                    
               
        # Determine the optimal set of vdms that doesn't have clashing residue identites as a single position
        vdm_set, position_residues, total_score, fg_satisfaction = determine_vdm_set(possibilities, len(fgs))
                  
        num_vdms = len(vdm_set)

        # Number of functional groups satisfied by vdms (Satisfied by Vdms, fgs)
        svfgs = fg_satisfaction.count(True)

        # Number of functional groups satisfied by solvent (CURRENTLY NOT IMPLEMENTED) (Satisfied by Solvent, fgs)
        ssfgs = fg_satisfaction.count("by Solvent")

        # Number of functional groups satisfied by nothing (UnSatisfied, fgs)
        usfgs = fg_satisfaction.count(False)

        # Print out vdms to pdb files if asked to and if vdms exist
        if print_to_pdb and num_vdms > 0:
            lig_and_vdm_to_PDB(conf.pose, [e[3] for e in vdm_set] , f"{conf.id}_{conf.conf_num:04}_{count%100:02}", f"{path_to_vdms}/{conf.id}")


        # Magical spell to extract the mutations that the vdm interactions require
        mutations = [f"{pose.residue(position).name1()}" + \
                     f"{pose.pdb_info().pose2pdb(position).split()[0]}" + 
                     f"{THREE_TO_ONE[value]}" 
                        for position, value in position_residues.items() 
                        if (pose.residue(position).name1() != 
                            THREE_TO_ONE[value])
                    ]         


        # Pose numbered positions of all the vdm interactions
        positions = [position for position, _ in position_residues.items()]
        num_mutations = len(mutations)

        mutations = " ".join(mutations)

        if mutations == "":
            mutations = "WT"      

        print(conf.id, conf.conf_num, round(total_score, 4), mutations, sep = ", ") 

        if svfgs == 0:
            norm_score = 0
        else:
            norm_score = total_score/len(fgs)
        
        df.append([conf.name, conf.id, f"{conf.id}/{conf.id}", conf.conf_num, conf.lig_aid, conf.t_aid, 
                   total_score, norm_score, num_vdms, svfgs, ssfgs, usfgs, 
                   mutations, positions, num_mutations])
       

    big_df = pd.DataFrame(df)
    big_df.columns = ["Molecule Name", "Molecule ID", "Molecule File Stem", "Conformer Number", "Molecule Atoms", "Target Atoms", 
                      "vdM Score", "Norm. Score", "Num. vdMs",  "Sat. by vdM", "Sat. by Solvent", "Unsat.", 
                      "Mutations", "Positions","Num. Mutations"]

    for score in ["vdM Score", "Norm. Score"]:
        mean = big_df[score].mean()
        std = big_df[score].std()

        big_df[score] -= mean
        big_df[score] /= std

    print(big_df)
    
    big_df.to_pickle(df_file_name, protocol = 4)
    print(f"Completed evaluating ligands, time taken {(time.time() - t0) / 60} minutes")


def lig_and_vdm_to_PDB(lig, instances, file_stem, outpath ):
    try:
        os.mkdir(outpath)
    except:
        pass

    for i, instance in enumerate(instances):
        df = instance

        if 'resnum' not in df.columns:
            df['resnum'] = i + 1
        
        ag = AtomGroup()
        ag.setCoords(df[["c_x", "c_y", "c_z"]].values)
        ag.setResnums(df['resnum'].values)
        ag.setResnames(df['resname'].values)
        ag.setNames(df['name'].values)
        ag.setChids(df['chain'].values)
        ag.setSegnames(df['chain'].values)
        
        if 'beta' in df.columns:
            ag.setBetas(df['beta'].values)
        
        if 'occ' not in df.columns:
            df['occ'] = 1
        ag.setOccupancies(df["occ"].values)
        
        writePDB(f"{outpath}/{file_stem}_{i+1}.pdb", ag)
    
    lig.dump_pdb(f"{outpath}/{file_stem}.pdb")
    return


def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file",
                        help = "your config file",
                        default = "my_conf.txt")
    
    if len(argv) == 0:
        print(parser.print_help())
        return

    args = parser.parse_args(argv)

    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["vdm"]

    sys.path.append(default["PathToPyRosetta"])
    
    global pyrosetta, Pose, alignment, conformer_prep, collision_check

    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    pyrosetta.init("-mute all")
    prody.confProDy(verbosity = 'none')

    params_list = default["ParamsList"]

    print("Reading in Pre and Post PDBs")
    if len(params_list) > 0:
        pre_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(pre_pose, params_list = params_list.split(" "))
        pyrosetta.pose_from_file(pre_pose, res_set, default["PrePDBFileName"])

        post_pose = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(post_pose, params_list = params_list.split(" "))
        pyrosetta.pose_from_file(post_pose, res_set, default["PostPDBFileName"])
    else:
        pre_pose = pyrosetta.pose_from_pdb(default["PrePDBFileName"])
        post_pose = pyrosetta.pose_from_pdb(default["PostPDBFileName"])

    res = pre_pose.residue(pre_pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))

    vdm_space_file_stem = spec["VDMSpaceFileStem"]
    path_to_conformers = default["PathToConformers"]
    pkl_file_name = default["PKLFileName"]
    output_file_name = spec["OutputFileName"]
    rmsd_cutoff = spec["RMSDCutoff"]
    lerp = spec["LERP"] == "True"
    bioisostere = spec["Bioisostere"] == "True"
    print_to_pdb = spec["PrintToPDB"] == "True"
    path_to_vdms = spec["PathToVDMInteractions"]

    align_ligands_to_residue_and_score_from_generator(post_pose, res, path_to_conformers,  output_file_name, 
                                                        pkl_file_name, vdm_space_file_stem, rmsd_cutoff, lerp, bioisostere, print_to_pdb, path_to_vdms)

    
if __name__ == "__main__":
    main(sys.argv[1:])




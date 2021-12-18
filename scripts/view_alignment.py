import sys
import os
import time

import pandas as pd

from configparser import ConfigParser
import argparse


def align_conformers(pose, res, path_to_conformers, df, pkl_file, lig_res_num = 1):
    """
    Aligns conformers based off the input file

    Arguments
    poses: input poses of interest
    reses: input residues to map onto
    path_to_conformers: where conformers are stored
    pkl_files: pkl_files that contain all the necessary info as generated in conformer_prep
    """

    pmm = pyrosetta.PyMOLMover()
    pmm.keep_history(True)

    t0 = time.time()

    total_confs = 0

    for pose_info in conformer_prep.yield_ligand_poses(df = df, path_to_conformers = path_to_conformers, post_accepted_conformers = False, ligand_residue = lig_res_num):
        
        if not pose_info:
            print(f"{conf.name}, {conf.id}")
            continue

        # Grab the conformer from the generator
        conf = pose_info
        
        # Perform alignment
        conf.align_to_target(res)

        # Check for collision
        total_confs += 1
        
        conf.pose.pdb_info().name(f"{conf.name}, {conf.id}")
        pmm.apply(conf.pose)

    print(total_confs)


def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file",
                        help = "your config file",
                        default = "my_conf.txt")
    
    if len(argv) == 0:
        print(parser.print_help())
        return

    args = parser.parse_args(argv)

    # Parsing config file
    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["grade_conformers"]
    sys.path.append(default["PathToPyRosetta"])
    auto = default["AutoGenerateAlignment"] == "True"
    
    # Importing necessary dependencies
    global pyrosetta, Pose, alignment, conformer_prep, collision_check, csv_to_df_pkl


    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    import alignment
    import conformer_prep
    import collision_check
    from grade_conformers import csv_to_df_pkl

    pyrosetta.init("-mute all")  

    params_list = default["ParamsList"]

    # Reading in Pre and Post PDBs
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

    # Defining the residue to align to
    res = pre_pose.residue(pre_pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))


    # Reading in information from the config file
    path_to_conformers = default["PathToConformers"]
    pkl_file_name = default["PKLFileName"]

    if pkl_file_name.strip() == "":
        pkl_file_name = None
    print(pkl_file_name)

    lig_res_num = int(default["LigandResidueNumber"])

    # Use an existant pkl file when possible
    print("Attempting to read in .pkl file")
    try:
        df = pd.read_pickle(pkl_file_name)
    except:
        print(".pkl file not found, generating one instead (this is normal)")
        csv_to_df_pkl(default["CSVFileName"], pkl_file_name, auto, path_to_conformers, pre_pose, res, lig_res_num)
        df = pd.read_pickle(pkl_file_name)

    print("\nBeginning grading")
    align_conformers(post_pose, res, path_to_conformers, df, pkl_file_name,lig_res_num)

    

if __name__ == "__main__":  
    main(sys.argv[1:])

import sys
import os
import time

import pandas as pd

from configparser import ConfigParser
import argparse



def csv_to_df_pkl(csv_file_name, pkl_file_name = None):

    if pkl_file_name == None:
        pkl_file_name = f"{csv_file_name[:-4]}.pkl"

    df = pd.read_csv(f"{csv_file_name}")
    if len(df.columns) == 5:
        def create_file_stem(name):
            return f"{name}/{name}"
        df["Molecule File Stem"] = df["Molecule ID"].apply(create_file_stem)

    def split_crange(name):
        print(name)
        lst = name.split("_")
        return [lst[0], lst[1]]

    def split_alabels(name):
        if name == "default":
            return ["CD2", "CZ2", "CZ3"]

        lst = name.split("-")
        return [str(e) for e in lst]

    df["Conformer Range"] = df["Conformer Range"].apply(split_crange)
    df["Molecule Atoms"] = df["Molecule Atoms"].apply(split_alabels) 
    df["Target Atoms"] = df["Target Atoms"].apply(split_alabels)

    df.to_pickle(pkl_file_name)

def align_to_residue_and_check_collision(pose, res, path_to_conformers, df, pkl_file,
         bin_width = 1, vdw_modifier = 0.7, include_sc = False):
    """
    Aligns and then checks for collisions

    Arguments
    poses: input poses of interest
    reses: input residues to map onto
    path_to_conformers: where conformers are stored
    pkl_files: pkl_files that contain all the necessary info as generated in conformer_prep
    bin_width: grid width of the collision grid
    vdw_modifier: by what factor to multiply pauling vdw radii by in the grid calculation
    include_sc: whether to do just backbone or include sc atoms

    Writes to the provided pkl files with conformers that are accepted in a column called
    "Accepted Conformers"
    """

    pmm = pyrosetta.PyMOLMover()
    pmm.keep_history(True)

    grid = collision_check.CollisionGrid(pose, bin_width = bin_width, vdw_modifier = vdw_modifier, include_sc = include_sc)
    
    
    all_accepted_all_files = []
    total_confs_all_files = 0

    t0 = time.time()

    all_accepted = []
    accepted_conformations = []
    every_other = 0
    total_confs = 0

    for pose_info in conformer_prep.yield_ligand_poses(pkl_file = pkl_file, path_to_conformers = path_to_conformers, post_accepted_conformers = False):
        
        if not pose_info:
            print(f"{conf.name}, {conf.id}: {len(accepted_conformations)/conf.conf_num}")
            all_accepted.append(accepted_conformations)
            accepted_conformations = []
            continue

        conf = pose_info
        
        # Perform alignment
        conf.align_to_target(res)

        # Check for collision
        does_collide = conf.check_collision(grid)

        total_confs += 1

        if not does_collide:
            accepted_conformations.append(conf.conf_num)
            
            if every_other % 15 == 0:
                conf.pose.pdb_info().name(f"{conf.name}, {conf.id}")
                pmm.apply(conf.pose)

            every_other += 1


    print(f"\n\n---Output, {pkl_file}---")
    #print(f"List of Acceptances: {all_accepted}")
    print(f"\nNumber of Ligands: {len(all_accepted)}")
    ligands_accepted = len([e for e in all_accepted if e])
    print(f"Number of Ligands Accepted: {ligands_accepted}")
    print(f"Proportion of Ligands Accepted: {ligands_accepted/len(all_accepted)}")

    total_accepted = sum([len(e) for e in all_accepted])
    print(f"\nNumber of Conformers: {total_confs}")
    print(f"Number of Conformers Accepted: {total_accepted}")
    print(f"Proportion of Conformers Accepted: {total_accepted/total_confs}")

    tf = time.time()
    print(f"\nTime taken: {(tf - t0)/60} minutes")
    print(f"Conformers per minute: {total_confs/(tf-t0)*60}")
    
    df["Accepted Conformers"] = all_accepted
    df.to_pickle("Ligands_1msw.pkl")


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
    spec = config["grade_conformers"]
    sys.path.append(default["PathToPyRosetta"])
    
    global pyrosetta, alignment, collision_check, Pose, conformer_prep

    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    pyrosetta.init("-mute all")  

    params_list = default["ParamsList"]

    if len(params_list) > 0:
        pose = Pose()
        pmm = pyrosetta.PyMOLMover()
        res_set = pyrosetta.generate_nonstandard_residue_set(pose, params_list = params_list.split(" "))
        pyrosetta.pose_from_file(pose, res_set, default["PDBFileName"])
    else:
        pose = pyrosetta.pose_from_pdb(default["PDBFileName"])

    res = pose.residue(pose.pdb_info().pdb2pose(default["ChainLetter"], int(default["ResidueNumber"])))


    path_to_conformers = default["PathToConformers"]
    pkl_file_name = default["PKLFileName"]
    print(pkl_file_name)
    bin_width = float(spec["BinWidth"])
    vdw_modifier = float(spec["VDW_Modifier"])
    include_sc = spec["IncludeSC"] == "True"



    try:
        df = pd.read_pickle(pkl_file_name)
    except:
        csv_to_df_pkl(default["CSVFileName"], pkl_file_name)
        df = pd.read_pickle(pkl_file_name)

    align_to_residue_and_check_collision(pose, res, path_to_conformers, df, pkl_file_name, 
                                        bin_width, vdw_modifier, include_sc)

    




if __name__ == "__main__":  
    main(sys.argv[1:])

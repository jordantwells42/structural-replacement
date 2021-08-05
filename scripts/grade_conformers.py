import sys
import os
import time

import pandas as pd

from configparser import ConfigParser
import argparse



def csv_to_df_pkl(csv_file_name, pkl_file_name, auto, path_to_conformers, pose, target_res, lig_res_num):
    df = pd.read_csv(f"{csv_file_name}", index_col = False)
    if len(df.columns) == 5:
        def create_file_stem(name):
            return f"{name}/{name}"
        df["Molecule File Stem"] = df["Molecule ID"].apply(create_file_stem)

    if pkl_file_name == None:
        pkl_file_name = f"{csv_file_name[:-4]}.pkl"
    def split_crange(name):
        lst = name.split("_")
        return (lst[0], lst[1])

    df["Conformer Range"] = df["Conformer Range"].apply(split_crange)

    if auto:
        print("Auto Generating Alignments")
        for i, row in df.iterrows():
            print(i)
            print(f"{i+1}/{len(df)}", end = " ")
            lig = Pose()
            mol_id = row["Molecule ID"]
            conf_num = 1
            res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list = [f"{path_to_conformers}/{mol_id}/{mol_id}.params"])
            pyrosetta.pose_from_file(lig, res_set, f"{path_to_conformers}/{mol_id}/{mol_id}_{conf_num:04}.pdb")

            molecule_atoms, target_atoms = alignment.auto_align_residue_to_residue(lig, lig.residue(1), target_res)
            df.loc[i, "Molecule Atoms"] = "-".join(molecule_atoms)
            df.loc[i, "Target Atoms"] = "-".join(target_atoms)


    def split_alabels(name):
        if name == "default":
            return ("CD2", "CZ2", "CZ3")

        lst = name.split("-")
        return tuple([str(e) for e in lst])
    df["Molecule Atoms"] = df["Molecule Atoms"].apply(split_alabels) 
    df["Target Atoms"] = df["Target Atoms"].apply(split_alabels)

    df.to_pickle(pkl_file_name, protocol = 4)

def align_to_residue_and_check_collision(pose, res, path_to_conformers, df, pkl_file,
         bin_width = 1, vdw_modifier = 0.7, include_sc = False, lig_res_num = 1):
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


    excluded_residues = [pose.pdb_info().pdb2pose("A", e) for e in [882, 883]]
    grid = collision_check.CollisionGrid(pose, bin_width = bin_width, vdw_modifier = vdw_modifier, include_sc = include_sc, excluded_residues = excluded_residues)


    all_accepted_all_files = []
    total_confs_all_files = 0

    t0 = time.time()

    all_accepted = []
    accepted_conformations = []
    every_other = 0
    total_confs = 0
    for pose_info in conformer_prep.yield_ligand_poses(df = df, path_to_conformers = path_to_conformers, post_accepted_conformers = False, ligand_residue = lig_res_num):
        
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
            
            if every_other % 25 == 0:
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
    df.to_pickle(pkl_file, protocol = 4)


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
    auto = default["AutoGenerateAlignment"] == "True"
    
    global pyrosetta, alignment, collision_check, Pose, conformer_prep

    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    pyrosetta.init("-mute all")  

    params_list = default["ParamsList"]

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


    path_to_conformers = default["PathToConformers"]
    pkl_file_name = default["PKLFileName"]

    if pkl_file_name.strip() == "":
        pkl_file_name = None
    print(pkl_file_name)
    bin_width = float(spec["BinWidth"])
    vdw_modifier = float(spec["VDW_Modifier"])
    include_sc = spec["IncludeSC"] == "True"
    lig_res_num = int(default["LigandResidueNumber"])


    try:
        df = pd.read_pickle(pkl_file_name)
    except:
        csv_to_df_pkl(default["CSVFileName"], pkl_file_name, auto, path_to_conformers, pre_pose, res, lig_res_num)
        df = pd.read_pickle(pkl_file_name)

    align_to_residue_and_check_collision(post_pose, res, path_to_conformers, df, pkl_file_name, 
                                        bin_width, vdw_modifier, include_sc, lig_res_num)

    
   




if __name__ == "__main__":  
    main(sys.argv[1:])

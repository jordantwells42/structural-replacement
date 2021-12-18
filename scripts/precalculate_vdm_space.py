import sys
import os
import math
import time
from collections import defaultdict
import pickle

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

from configparser import ConfigParser
import argparse 

class vdM():
    """
    A class that stores all of the required information about
    a vdM
    """
    def __init__(self, score, position, residue, instance):
        self.score = score
        self.position = position
        self.residue = residue
        self.names = instance["name"].values
        self.vdm_residue = instance["resname"].values[0]
        self.vdm_coords = np.array(instance.loc[instance["chain"] == "Y"][["c_x", "c_y", "c_z"]].values)
        self.sc_coords = np.array(instance.loc[instance["chain"] == "X"][["c_x", "c_y", "c_z"]].values)
        self.probe_name = instance["probe_name"].values[0]
        self.rota = instance["rota"].values[0]
        self.CG = instance["CG"].values[0]

    def get_instance(self):
        """
        Returns a DataFrame instance of a vdM
        """
        df = []

        count = 0
        for row in self.vdm_coords:
            df.append(["Y", self.vdm_residue, self.names[count], row[0], row[1], row[2]])
            count += 1

        for row in self.sc_coords:
            df.append(["X", self.residue, self.names[count], row[0], row[1], row[2]])
            count += 1


        df = pd.DataFrame(df)
        df.columns = ["chain", "resname", "name", "c_x", "c_y", "c_z"]
        return df


def get_ABPLE(resn, phi, psi):
    """
    Returns the ABPLE assignment of a backbone position given its phi and psi and a residue type

    Arguments:
    resn: three letter AA code
    phi: phi value for the position of interest
    psi: psi value for the position of interest

    Returns:
    a character: A P B L E depending on the assignment
    """
    try:
        psi = int(np.ceil(psi / 10.0)) * 10
        phi = int(np.ceil(phi / 10.0)) * 10
        if psi == -180:
            psi = -170
        if phi == -180:
            phi = -170
        return ABPLE_DICT[resn][psi][phi]
    except ValueError:
        return 'n'


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


def resfile_parser(pose, resfile):
    """
    Takes in a pose of interest and a resfile and parses necessary information for VDM design

    Arguments:
    pose: a Pose object
    resfile: path/to/resfile.resfile

    Returns:
    positions, permissible_residues

    position:  list of all pose numbered positions of interest
    permissibile_residues: a dictionary mapping a pose numbered position of interest to
    a list of three letter coded residues that it can be designed to
    """
    commands_dict = {"ALLAA":
                "ACFGILMPVWYDEHKNQRST",
                "ALLAAwc":
                "ACFGILMPVWYDEHKNQRST",
                "ALLAAxc":
                "AFGILMPVWYDEHKNQRST",
                "POLAR":
                "DEHKNQRST",
                "APOLAR":
                "ACFGILMPVWY"
                }


    positions = []
    permissible_residues = defaultdict(list)

    with open(resfile, "r") as rf:
        started = False
        for row in rf:
            if row.lower().strip() == "start":
                started = True
                continue

            if not started:
                continue

            lst = row.split(" ")
            res, chain = int(lst[0]), lst[1]
            commands = lst[2:]

            if commands == ["NATRO"]:
                continue

            for i, command in enumerate(commands):
                if "\n" in command:
                    commands[i] = command[:-1]

            res_num = pose.pdb_info().pdb2pose(chain, res)
            positions.append(res_num)

            if commands == ["NATAA"]:
                permissible_residues[res_num].append(pose.residue(res_num).name3())
                continue

            if len(commands) > 1:
                AAs = commands_dict["ALLAA"]
                for i, command in enumerate(commands):
                    if command in commands_dict:
                        new_AAs = commands_dict[command]
                    else:
                        if command == "PROPERTY":
                            print("PROPERTY not currently supported")
                        elif command == "NOTAA":
                            new_AAs = "".join(list(set(list(commands_dict["ALLAA"])).difference((set(list(commands[i+1]))))))
                        elif command == "PIKAA":
                            new_AAs = commands[i+1]
                        else:
                            continue


                    AAs = "".join(list(set(list(AAs)).intersection((set(list(new_AAs))))))

            else:
                AAs = commands_dict[commands[0]]


            for char in AAs:
                residue = ONE_TO_THREE[char]
                permissible_residues[res_num].append(residue)

    return positions, permissible_residues


def rmsd_distance_metric(v1, v2):
    """
    Distance metric for the NN graph that is a calculation of RMSD
    
    Arguments:
    v1, v2: numpy arrays which are flattened representations of the coordinates of interest e.g.
    [x1,y1,z1,x2,y2,z2... xn,yn,zn]

    Returns:
    A caluclated value for the RMSD

    """
    mat1 = v1.reshape(len(v1)//3,3)
    mat2 = v2.reshape(len(v2)//3,3)

    return calc_rmsd(mat1, mat2)


def calc_rmsd(mat1, mat2):
    """
    Calculates RMSD given two numpy arrays of corresponding atoms

    Arguments:
    mat1, mat2: numpy arrays of corresponding atom coordinates of the form
    [[x1,y1,z1], [x2,y2,z2],...,[xn,yn,zn]

    Returns:
    A calculated value for the RMSD
    """
    dists = []
    for v1, v2 in zip(mat1, mat2):

        dist = np.linalg.norm(v1-v2)
        dists.append(dist)

    rmsd = np.sqrt((np.array(dists) ** 2).mean())
    return rmsd


def orientation_dependent_metric(v1, v2):
    """
    An orientation dependent RMSD metric for the NN graph
    computes the RMSD between two sets of atoms and then finds the minimum
    dot product between all vectors between the first three atom, and uses that 
    as a measure of "orientation"

    Arguments:
    v1, v2: numpy arrays which are flattened representations of the coordinates of interest e.g.
    [x1,y1,z1,x2,y2,z2... xn,yn,zn]

    Returns:
    A calculaed value for the orientation dependent RMSD, of the form
    rmsd * (1 - min(dot_product)/4), such that the range of the values is
    [0.75 * rmsd, 1.25 * rmsd] (*lower is better for a distance metric)
    """
    mat1 = v1.reshape(len(v1)//3,3)
    mat2 = v2.reshape(len(v2)//3,3)

    dots = []
    for n1, n2 in [(0,1), (0,2), (1,2)]:
        v1 = mat1[n1] - mat1[n2]
        v2 = mat2[n1] - mat2[n2]
        v1 /= np.sqrt(np.sum(v1**2))
        v2 /= np.sqrt(np.sum(v2**2))
        dot = np.dot(v1,v2)
        dots.append(dot)

    rmsd = calc_rmsd(mat1, mat2)
    return rmsd * (1 - min(dots)/4)
    

def precalculate_vdm_space(pose, resfile, pkl_file_stem, score_cutoff = 0):
    """
    Calculates the "vdM space" for a pose, given a resfile
    This vdM space is a projection of all possible residue-CG interactions, stored in
    a nearest neighbors graph, for the residues and positions in the resfile

    Arguments:
    pose: the pose of interest
    resfile: a resfile for that pose
    pkl_file_stem: the naming convention for the outputted information
        e.g. pkl_file_stem = "Dog", then outputs would be Dog-info.pkl, Dog-coords.pkl, Dog-trees.pkl
    score_cutoff: the minimum score cutoff for the selected vdMs, 0 is the mean

    Outputs:
    Three files
    {pkl_file_stem}-trees.pkl which contains a dictionary mapping a vdm CG type to a 
        sklearn BallTree Nearest Neighbors graphs with all of the positions and residues it could map back to
    {pkl_file_stem}-coords.pkl which contains all of the spatial coordinates of the vdM CGs
    {pkl_file_stem}-info.pkl which is a dictionary mapping a vdM CG type to a list of information
        for each information which corresponds to the outputted indices of the NN graph
    """

    ia = pd.read_pickle(f"{VDM_DIR}/ideal_alanine_bb_only.pkl")
    bb_ia = np.array([[float(ia.loc[ia["name"] == f"{elem}"][f"c_{d}"]) for d in ["x", "y", "z"]] for elem in ["N", "CA", "C"]])
        
    bin_width = 1 
    vdw_modifier = 0.7
    include_sc = False
    grid = collision_check.CollisionGrid(pose, bin_width = bin_width, vdw_modifier = vdw_modifier, include_sc = include_sc)
    
    residues = ["ARG", "ALA", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
                "MET", "ASN", "PRO", "GLN", "SER", "THR", "VAL", "TRP",
                "TYR", "CYS"]
    vdm_names = ["csc","phenol", "hie", "bb_cco", "coo", "bb_cnh", "ccn", "coh", "conh2", "csh"]

    positions, permissible_residues = resfile_parser(pose, resfile)
    
    vdm_coords = {}
    corresponding_info = {}
    trees = {}

    bb_positions = [np.array([pose.residue(position).atom(e).xyz() for e in ["N", "CA", "C"]]) for position in positions]
    phis = [pose.phi(position) for position in positions]
    psis = [pose.psi(position) for position in positions]

    # PRECALCULATION
    for vdm_name in vdm_names:
        print(f"\nStarted {vdm_name}")
        t0 = time.time()
        for residue in residues:
            parent_df = pd.read_pickle(f"{VDM_DIR}/vdms/{vdm_name}/{residue}.gz")

            gpdf = parent_df.groupby(["cluster_number", "ABPLE"])

            for position, bb_pos, phi, psi in zip(positions, bb_positions, phis, psis):
                if residue not in permissible_residues[position]:
                    continue
                abple = get_ABPLE(residue, phi, psi)

                # Get Rot, Trans matrices to transform vdm coordinates to bb position
                m_com, t_com, R = alignment.svd(bb_ia, bb_pos)

                for cluster_number in range(1, parent_df["cluster_number"].max() + 1):
                    # Grab the instance with that identifier
                    try:
                        instance = gpdf.get_group((cluster_number, abple))
                    except KeyError:
                        continue
                    if len(instance) == 0:
                        continue

                    # If it doesn't have a corresponding C-score for our bb config continue
                    try:
                        score = instance[f"C_score_abple_{abple}"].values[0]
                    except:
                        continue

                    # Only accept hydrogen-bonding interactions
                    if (True not in instance["contact_hb"].values) and (True not in instance["contact_wh"].values): 
                        continue
                         
                    # Only accept good scores
                    if score < score_cutoff or pd.isna(score):
                        continue
                    
                    Y_res = instance.loc[instance["chain"] == "Y"]["resname"].iloc[0]

                    if not "bb" in vdm_name:
                        vdm_label = f"{vdm_name}-{Y_res}"
                    else:
                        vdm_label = f"{vdm_name}"

                    labels = [vdm_label]
                    instance = instance.copy()
                    instance[["c_x", "c_y", "c_z"]] = (np.matmul(R, (instance[["c_x", "c_y", "c_z"]].values - m_com).transpose())).transpose() + t_com

                    # Allows for flipped versions of chemical groups that are symmetric
                    if f"{vdm_label}-flip" in VDM_LABELS: labels.append(f"{vdm_label}-flip")

                    for label in labels:    
                        instance_vdm_coords_transformed = np.array([[instance.loc[(instance["name"] == f"{elem}") & (instance["chain"] == "Y")][f"c_{d}"].values[0]
                                                      for d in ["x", "y", "z"]
                                                     ] 
                                                    for elem in VDM_LABELS[label]])

                        
                        if grid.check_collision_matrix(instance_vdm_coords_transformed):
                            continue    
                    
                        if vdm_name not in corresponding_info:
                            corresponding_info[vdm_name] = [vdM(score, position, residue, instance)]
                            vdm_coords[vdm_name] = np.array([instance_vdm_coords_transformed.flatten()])
                        else:
                            corresponding_info[vdm_name].append(vdM(score, position, residue, instance))
                            vdm_coords[vdm_name] = np.append(vdm_coords[vdm_name], [instance_vdm_coords_transformed.flatten()], axis = 0)


            print(f"Finished {vdm_name}-{residue} interaction")

        # Construct a nearest neighbors graph of the vdm interactions
        trees[vdm_name] = BallTree(vdm_coords[vdm_name], metric = orientation_dependent_metric)
        print(f"Completed {vdm_name}, time taken: {(time.time() - t0)/60} minutes")

    with open(f"{pkl_file_stem}-trees.pkl", "wb+") as t, open(f"{pkl_file_stem}-info.pkl", "wb+") as i, open(f"{pkl_file_stem}-coords.pkl", "wb+") as c:
        pickle.dump(trees, t, protocol = 4)
        pickle.dump(corresponding_info, i, protocol = 4)
        pickle.dump(vdm_coords, c, protocol = 4)

def main(argv):
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file",
                        help = "your config file",
                        default = "my_conf.txt")
    
    if len(argv) == 0:
        print(parser.print_help())
        return

    args = parser.parse_args(argv)

    # Parsing config
    config = ConfigParser()
    config.read(args.config_file)
    default = config["DEFAULT"]
    spec = config["vdm"]
    sys.path.append(default["PathToPyRosetta"])
    

    # Importing necessary dependencies
    global pyrosetta, Pose, VDM_DIR, ABPLE_DICT, alignment, conformer_prep, collision_check

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
        pyrosetta.pose_from_file(pose, res_set, default["PostPDBFileName"])
    else:
        pose = pyrosetta.pose_from_pdb(default["PostPDBFileName"])

    resfile = spec["Resfile"]
    vdm_space_file_stem = spec["VDMSpaceFileStem"]
    score_cutoff = float(spec["VDMScoreCutoff"])
    VDM_DIR = spec["VDM_Directory"]

    with open(f'{VDM_DIR}/abple_dict.pkl', 'rb') as infile:
        ABPLE_DICT = pickle.load(infile)



    precalculate_vdm_space(pose, resfile, vdm_space_file_stem, score_cutoff)
    print(f"\n\nCompleted Precalculation of vdM space, time taken: {(time.time() - t0)/60} minutes")
    
if __name__ == "__main__":
    main(sys.argv[1:])




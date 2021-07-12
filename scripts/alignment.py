from pyrosetta import *

from pyrosetta.rosetta.numeric import xyzMatrix_double_t, xyzVector_double_t
import math
import numpy as np


def svd_flatten(P, Q):
    """
    Perform Singular Value Decomposition given a list of corresponding
    position vectors between two structures to be aligned, finds translation vectors and a rotation
    matrix to perform alignment

    Arguments:
    P: 2D Python list, list of [x, y, z] for the object to be moved
    Q: 2D Python list, list of [x, y, z] for the target that correspond to the
    points for the object

    Returns:
    p_com: Center of mass of object
    t_com: Center of mass of target
    R: rotation matrix between object to targetx (as a list)
    """
    p_com = np.mean(P, 0)
    t_com = np.mean(Q, 0)
    
    P -= p_com
    Q -= t_com

    P = np.matrix(P)
    Q = np.matrix(Q)
    
    M = np.matmul(P.transpose(), Q)

    U, W, Vt = np.linalg.svd(M)

    R = np.matmul(Vt.transpose(),U.transpose())
    
    return [e for e in np.nditer(p_com)], [e for e in np.nditer(t_com)], [e for e in np.nditer(R)]


def svd(P, Q):
    """
    Perform Singular Value Decomposition given a list of corresponding
    position vectors between two structures to be aligned, finds translation vectors and a rotation
    matrix to perform alignment

    Arguments:
    P: 2D Python list, list of [x, y, z] for the object to be moved
    Q: 2D Python list, list of [x, y, z] for the target that correspond to the
    points for the object

    Returns:
    p_com: Center of mass of object
    t_com: Center of mass of target
    R: rotation matrix between object to targetx (as a list)
    """
    P = P.copy()
    Q = Q.copy()
    p_com = np.mean(P, 0)
    t_com = np.mean(Q, 0)
    
    P -= p_com
    Q -= t_com

    P = np.matrix(P)
    Q = np.matrix(Q)
    
    M = np.matmul(P.transpose(), Q)

    U, W, Vt = np.linalg.svd(M)

    R = np.matmul(Vt.transpose(),U.transpose())
    
    return p_com, t_com, np.array(R)


def align_pose_coords_to_target_coords(pose, p_xyzs, t_xyzs):  
    """
    Takes an input pose, and a mapping of its atom coordinates to another objects corresponding atom coordinates and
    aligns them using SVD

    Arguments:
    pose: Pose to be aligned
    p_xyzs: List of coordinates of the input pose
    t_xyzs: List of corresponding coordinates of target object 

    """

    p_xyz_mat = np.matrix(p_xyzs)
    t_xyz_mat = np.matrix(t_xyzs)

    p_com, t_com, R = svd_flatten(p_xyz_mat, t_xyz_mat)

    # Converting list types into Rosetta-friendly types
    R = xyzMatrix_double_t.rows(*R)
    p_com = xyzVector_double_t(*p_com)
    t_com = xyzVector_double_t(*t_com)
    null_R = xyzMatrix_double_t.rows(1, 0, 0, 0, 1, 0, 0, 0, 1)

    # Applying transformations
    # 1. Translation to origin (so that rotation will work)
    # 2. Rotation as determined by SVD
    # 3. Translation to target COM
    pose.apply_transform_Rx_plus_v(null_R, p_com.negated())
    pose.apply_transform_Rx_plus_v(R, xyzVector_double_t(0))
    pose.apply_transform_Rx_plus_v(null_R, t_com)


def atom_coords_from_aid(res, aid):
    """
    Uses a list of atom labels to get xyz coordinates for atoms in a residue

    Arguments:
    res: Residue of interest
    aid: List of atom labels e.g. ["C4", "C6", "C10"]

    """

    return [res.atom(e).xyz() for e in aid]


def align_ligand_to_residue(lig, res, lig_aid, res_aid):
    """
    Wrapper for align_pose_to_target_coords, which takes in a ligand pose object, a residue to map onto,
    and lists of atom labels for the ligand and residue

    Arguments
    lig: Ligand pose object
    res: Residue object to align onto
    lig_aid: List of atom labels for ligand e.g. ["C4", "C6", "C10"]
    res_aid: List of corresponding atoms for residue e.g. ["CD2", "CZ2", "CZ3"]

    """

    l_xyzs = atom_coords_from_aid(lig.residue(1), lig_aid)
    r_xyzs = atom_coords_from_aid(res, res_aid)

    align_pose_coords_to_target_coords(lig, l_xyzs, r_xyzs)


def align_pose_residue_to_target_residue(pose1, res1, res2, res1_aid, res2_aid):
    """
    Wrapper for align_pose_to_target_coords, which takes in a pose object to be moved, a residue it contains, a 
    residue to map onto and lists of atom labels for the pose and residue

    Arguments
    pose1: Pose object
    res1: Residue object in pose1 of interest
    res2: Residue object to map onto
    res1_aid: List of atom labels for res1 e.g. ["C4", "C6", "C10"]
    res2_aid: List of corresponding atoms for res2 e.g. ["CD2", "CZ2", "CZ3"]

    """
    res1_xyzs = atom_coords_from_aid(res1, res1_aid)
    res2_xyzs = atom_coords_from_aid(res2, res2_aid)
    align_pose_coords_to_target_coords(pose1, res1_xyzs, res2_xyzs)


def align_pose_residue_to_target_coords(pose1, res1, res1_aid, t_xyzs):
    res1_xyzs = atom_coords_from_aid(res1, res1_aid)
    align_pose_coords_to_target_coords(pose1, res1_xyzs, t_xyzs)

def align_ligand_to_target_coords(lig, lig_aid, t_xyzs):
    align_pose_residue_to_target_coords(lig, lig.residue(1), lig_aid, t_xyzs)

def main():
    lig = Pose()
    pmm = pyrosetta.PyMOLMover()
    res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list = ["conformers/271-34-1/271-34-1.params"])
    pose_from_file(lig, res_set, "conformers/271-34-1/271-34-1_0001.pdb")

    enz = pose_from_pdb("structures/1CEZ.pdb")
    lig_aid = ["C1", "C5", "N2"]
    w_aid = ["CD2", "CZ2", "CZ3"]

    pmm.apply(lig)
    pmm.apply(enz)

    w_res_num = 287
    w_res_num = enz.pdb_info().pdb2pose("A", w_res_num)
    w_res = enz.residue(w_res_num)

    l_xyzs = atom_coords_from_aid(lig.residue(1), lig_aid)
    w_xyzs = atom_coords_from_aid(w_res, w_aid)

    align_ligand_to_residue(lig, w_res, lig_aid, w_aid)
    pmm.apply(lig)
    input()


if __name__ == "__main__":
    init()
    main()

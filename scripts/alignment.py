from pyrosetta import *

from pyrosetta.rosetta.numeric import xyzMatrix_double_t, xyzVector_double_t
import math
import numpy as np
import sklearn.decomposition
from sklearn.neighbors import BallTree
import sklearn

def svd_new(P, Q):
    P = P.copy()
    Q = Q.copy()
    p_com = np.mean(P, 0)
    t_com = np.mean(Q, 0)
    
    P -= p_com
    Q -= t_com
    
    M = np.matmul(P.transpose(), Q)

    U, W, Vt = np.linalg.svd(M)

    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0
    

    if d:
        U[:, -1] = -U[:, -1]

    R = np.matmul(Vt.transpose(),U.transpose())
    
    return [e for e in np.nditer(p_com)], [e for e in np.nditer(t_com)], [e for e in np.nditer(R)]

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
    P = P.copy()
    Q = Q.copy()
    p_com = np.mean(P, 0)
    t_com = np.mean(Q, 0)
    
    P -= p_com
    Q -= t_com

    #P = np.matrix(P)
    #Q = np.matrix(Q)
    
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

    
    M = np.matmul(P.transpose(), Q)

    U, W, Vt = np.linalg.svd(M)

    R = np.matmul(Vt.transpose(),U.transpose())
    
    return p_com, t_com, R

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

        
    p_com, t_com, R = svd_new(p_xyz_mat, t_xyz_mat)

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


def auto_align_residue_to_residue(pose, res1, res2):  
    """
    Takes an input pose, and a mapping of its atom coordinates to another objects corresponding atom coordinates and
    aligns them using SVD

    Arguments:
    pose: Pose to be aligned
    p_xyzs: List of coordinates of the input pose
    t_xyzs: List of corresponding coordinates of target object 

    """
    pmm = PyMOLMover()
    start_res = 7
    p_xyz = np.array([res1.atom(i).xyz() for i in range(1, res1.natoms() + 1) if res1.atom_type(i).element() != "H"])
    t_xyz = np.array([res2.atom(i).xyz() for i in range(start_res, res2.natoms() + 1) if res2.atom_type(i).element() != "H"])
    p_elem = [res1.atom_type(i).element() for i in range(1, res1.natoms() + 1) if res1.atom_type(i).element() != "H"]
    t_elem = [res2.atom_type(i).element() for i in range(start_res, res2.natoms() + 1) if res2.atom_type(i).element() != "H"]
    p_ind = [i for i in range(1, res1.natoms() + 1) if res1.atom_type(i).element() != "H"]
    t_ind = [i for i in range(start_res, res2.natoms() + 1) if res2.atom_type(i).element() != "H"]
    null_R = xyzMatrix_double_t.rows(1, 0, 0, 0, 1, 0, 0, 0, 1)
    p_com = np.mean(p_xyz, 0)
    t_com = np.mean(t_xyz, 0)
    p_com = xyzVector_double_t(*p_com)
    t_com = xyzVector_double_t(*t_com)
    t_atoms = []
    p_atoms = []
    
    mcs = Pose()
    mcs_atoms = 0

    pose.apply_transform_Rx_plus_v(null_R, p_com.negated())
    pose.apply_transform_Rx_plus_v(null_R, t_com)

    for rnd in range(5000):
        p_xyz = np.array([res1.atom(i).xyz() for i in range(1, res1.natoms() + 1) if res1.atom_type(i).element() != "H"])
        t_xyz = np.array([res2.atom(i).xyz() for i in range(start_res, res2.natoms() + 1) if res2.atom_type(i).element() != "H"])


        pca = sklearn.decomposition.PCA(3)

        covar_p = np.matmul(p_xyz.transpose(), p_xyz)
        covar_t = np.matmul(t_xyz.transpose(), t_xyz)

        A = pca.fit(covar_p).components_
        B = pca.fit(covar_t).components_
        A /= np.sqrt(np.sum(A**2, 0))
        B /= np.sqrt(np.sum(B**2, 0))

        R1 = np.matmul(B, A.transpose())

        R = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        B = np.matmul(R, A)
        R2 = np.matmul(B, A.transpose())
        B = np.matmul(R, A)
        R = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        B = np.matmul(R, A)
        R3 = np.matmul(B, A.transpose())
        B = np.matmul(R, A)
        R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        B = np.matmul(R, A)
        R4 = np.matmul(B, A.transpose())

        Rs_traces = []
        for R in [R1, R2, R3, R4]:
            if abs(np.trace(R) + 1) < 1e-6 or abs(np.trace(R) - 1) < 1e-6:
                Rs_traces.append((R, math.pi))
            else:
                Rs_traces.append((R, math.acos((np.trace(R)-1)/2))) 

        Rs_traces = sorted(Rs_traces, key = lambda x: abs(x[1]))

        R = Rs_traces[0][0]        
        R = R.transpose()
        # Converting list types into Rosetta-friendly types
        R = xyzMatrix_double_t.rows(*[a for e in R for a in e])
        p_com = xyzVector_double_t(*p_com)
        t_com = xyzVector_double_t(*t_com)
        

        R = Rs_traces[0][0]        
        #R = R.transpose()
        # Converting list types into Rosetta-friendly types
        R = xyzMatrix_double_t.rows(*[a for e in R for a in e])
        pose.apply_transform_Rx_plus_v(null_R, t_com.negated())
        pose.apply_transform_Rx_plus_v(R, xyzVector_double_t(0))
        pose.apply_transform_Rx_plus_v(null_R, t_com)

        
        if rnd % 5 == 0:
            R = Rs_traces[rnd%4][0].transpose()
            R = xyzMatrix_double_t.rows(*[a for e in R for a in e])
            pose.apply_transform_Rx_plus_v(null_R, t_com.negated())
            pose.apply_transform_Rx_plus_v(R, xyzVector_double_t(0))
            pose.apply_transform_Rx_plus_v(null_R, t_com)
        


        p_xyz = np.array([res1.atom(i).xyz() for i in range(1, res1.natoms() + 1) if res1.atom_type(i).element() != "H"])
        t_xyz = np.array([res2.atom(i).xyz() for i in range(start_res, res2.natoms() + 1) if res2.atom_type(i).element() != "H"])

        
        

        hetatom_bonus = 2
        aligned_atoms = 0
        tree = sklearn.neighbors.BallTree(p_xyz)
        for req_dist in np.arange(0.25, 10, 0.25):
            p_corr = []
            t_corr = []
            approved_dists = []
            for i, xyz in enumerate(t_xyz):
                dists, inds = tree.query([xyz], k = 3)

                if dists[0][0] < req_dist:
                    approved_dists.append((dists[0][0], p_ind[inds[0][0]], t_ind[i]))
                    t_corr.append(xyz)
                    p_corr.append(p_xyz[inds[0][0]])
                    if p_elem[inds[0][0]] == t_elem[i] and t_elem[i] != "C":
                        aligned_atoms += hetatom_bonus


            if len(p_corr)/len(t_xyz) < 0.5 or len(p_corr) < 3:
                continue
            else:
                break


        aligned_atoms += len([e for e in approved_dists if e[0] < 1])

        p_com, t_com, R = svd_new(np.matrix(p_corr), np.matrix(t_corr))
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

        if aligned_atoms > mcs_atoms:
            approved_dists = sorted(approved_dists, key = lambda x: x[0])
            p_atoms = [res1.atom_name(e[1]).strip() for e in approved_dists[:3]]
            t_atoms = [res2.atom_name(e[2]).strip() for e in approved_dists[:3]]
            mcs_atoms = aligned_atoms
            mcs.assign(pose)

    pose.assign(mcs)
    pmm.apply(pose)

    return (p_atoms, t_atoms)
        


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
    mol_id = "91-33-8"
    conf_num = 1
    res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list = [f"conformers/{mol_id}/{mol_id}.params"])
    pose_from_file(lig, res_set, f"conformers/{mol_id}/{mol_id}_{conf_num:04}.pdb")

    enz = pose_from_pdb("structures/1CEZ.pdb")
    """
    lig_aid = ["C1", "C5", "N2"]
    w_aid = ["CD2", "CZ2", "CZ3"]
    """
    pmm.apply(lig)
    pmm.apply(enz)

    w_res_num = 287
    w_res_num = enz.pdb_info().pdb2pose("A", w_res_num)
    w_res = enz.residue(w_res_num)
    """
    l_xyzs = atom_coords_from_aid(lig.residue(1), lig_aid)
    w_xyzs = atom_coords_from_aid(w_res, w_aid)
    """

    #lig = pose_from_sequence("W")
    
    print(align_residue_to_residue(lig, lig.residue(1), w_res))
    pmm.apply(lig)


if __name__ == "__main__":
    init()
    main()

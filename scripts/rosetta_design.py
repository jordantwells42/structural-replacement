import sys
from configparser import ConfigParser
import argparse

import pandas as pd


def prepare_pose_for_design(pose, focus_pos, lig, lig_aid):
    complx = pyrosetta.Pose()
    complx.assign(pose)

    vrm = pyrosetta.rosetta.protocols.simple_moves.VirtualRootMover()
    vrm.apply(complx)

    complx.append_residue_by_jump(lig.residue(1), focus_pos, "", "", True)
    

    lig_pos = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    lig_pos.set_index(complx.size())

    ccg = pyrosetta.rosetta.protocols.constraint_generator.CoordinateConstraintGenerator()
    ccg.set_residue_selector(lig_pos)
    ccg.set_sidechain(True)
    ccg.set_bounded_width(0.1)
    ccg.set_sd(0.2)
    ccg.apply(complx)

    return complx

def make_mutations(pose, vdm_mutations):
    complx = Pose()
    complx.assign(pose)
    for position, residue in vdm_mutations:
        pyrosetta.toolbox.mutants.mutate_residue(complx, position, residue)
    return complx


def determine_mutations(pose, mutation_string):
    if mutation_string == "WT":
        pos_res = []
    else:
        mutations = mutation_string.split(" ")
        pos_res = [(int(mutation[1:-1]), mutation[-1:]) for mutation in mutations]

    return [(pose.pdb_info().pdb2pose("A", pos), res) for pos, res in pos_res]

def design(pose, pose_wt, focus_seqpos, resfile, vdm_positions, vdm_mutations, ref_seq, conf_id, fnr_bonus, rtc_bonus):
    
    # SET UP SCORE FUNCTION
    sf = pyrosetta.get_fa_scorefxn()
    unsat_penalty = 0.05
    sf.set_weight(pyrosetta.rosetta.core.scoring.res_type_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.buried_unsatisfied_penalty, unsat_penalty)


    # SET UP SCORE BONUSES
    fnr = pyrosetta.rosetta.protocols.protein_interface_design.FavorNativeResidue(pose, fnr_bonus)

    for position in vdm_positions:
        rtc = pyrosetta.rosetta.core.scoring.constraints.ResidueTypeConstraint(pose, position, rtc_bonus)
        pose.add_constraint(rtc)


    # SET UP TASK FACTORY
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.ReadResfile(resfile))
    

    # SET UP NEIGHBORHOOD RESIDUES and JUMP
    focus_pos = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    focus_pos.set_index(focus_seqpos)
    nbr = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr.set_distance(12)
    nbr.set_focus_selector(focus_pos)
    nbr.set_include_focus_in_subset(True)

    pose.update_residue_neighbors()
    nbr.apply(pose)
    rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(rlt, nbr, True))

    jump_num = pose.num_jump()
    jump_selector = pyrosetta.rosetta.core.select.jump_selector.JumpIndexSelector()
    jump_selector.jump(jump_num)


    # SET UP MOVEMAP
    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.add_chi_action(pyrosetta.rosetta.core.select.movemap.mm_enable, nbr)
    mmf.add_jump_action(pyrosetta.rosetta.core.select.movemap.mm_enable, jump_selector)


    # SET UP FASTRELAX
    fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in = sf, standard_repeats = 3)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.ramp_down_constraints(False)
    fr.min_type('lbfgs_armijo_nonmonotone')
    fr.max_iter(300)
    
    # Apply FastRelax
    fr.apply(pose)


    # Minimization without the buried unsatisfied penalty, recc. by Vmulligan
    sf.set_weight(pyrosetta.rosetta.core.scoring.buried_unsatisfied_penalty, 0.0)
    mm = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
    mm.score_function(sf)
    mm.movemap_factory(mmf)
    mm.apply(pose)

    pre_revert_seq = pose.sequence()
    revert = pyrosetta.rosetta.protocols.protein_interface_design.Revert(sf, 0.0)
    revert.apply(pose_wt, pose)
    post_revert_seq = pose.sequence()

    num_reverts = 0
    for a,b in zip(pre_revert_seq, post_revert_seq):
        if a != b:
            num_reverts += 1

    print(f"Reverting {num_reverts} weak mutations")

    mm.apply(pose)


    # Analyzing interface
    sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 0.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.res_type_constraint, 0.0)

    interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover(jump_num)
    interface_analyzer.set_scorefunction(sf)
    interface_analyzer.set_pack_separated(True)
    interface_analyzer.set_pack_rounds(3)
    interface_analyzer.set_compute_interface_sc(True)
    interface_analyzer.set_compute_interface_energy(True)
    interface_analyzer.set_compute_separated_sasa(True)
    interface_analyzer.set_compute_interface_delta_hbond_unsat(True)
    interface_analyzer.apply(pose)

    data = interface_analyzer.get_all_data()
    SC = data.sc_value
    IFE = data.complexed_interface_score[1] - data.separated_interface_score[1]
    ddG = interface_analyzer.get_interface_dG()
    dG_dSASA_ratio = data.dG_dSASA_ratio 
    dSASA = data.dSASA_sc[1]/data.dSASA[1]
    num_buns = data.delta_unsat_hbonds


    print("SC: ", SC, "IFE: ", IFE,"dSASA: ", dSASA, "ddG:", ddG, "dG/dSASA ratio:", dG_dSASA_ratio, "BUNs:", num_buns)

    # Determine mutations
    new_seq = pose.sequence()
    mutations = []
    all_mutations = []
    for i in range(len(ref_seq)):
        if new_seq[i] != ref_seq[i]:
            empty_string = " "
            mutations.append(f"{ref_seq[i]}{pose.pdb_info().pose2pdb(i+1).split(empty_string)[0]}({i+1}){new_seq[i]}")
            all_mutations.append(f"{ref_seq[i]}{pose.pdb_info().pose2pdb(i+1).split(empty_string)[0]}{new_seq[i]}")
            #print("Different than native")
            if (i+1, new_seq[i]) in vdm_mutations:
                print(f"Same as vdM: {ref_seq[i]}{pose.pdb_info().pose2pdb(i+1).split(empty_string)[0]}({i+1}){new_seq[i]}")
            else: 
                print(f"{ref_seq[i]}{pose.pdb_info().pose2pdb(i+1).split(empty_string)[0]}({i+1}){new_seq[i]}")

    all_mutations = " ".join(all_mutations)
    return (pose, all_mutations, mutations, SC, IFE, dSASA, ddG, dG_dSASA_ratio, num_buns)




def main(argv):
    # PARSING CONFIG FILE
    print("Parsing config file")

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
    spec = config["rosetta_design"]


    # Importing necessary dependencies
    print("Importing necessary dependencies")
    sys.path.append(default["PathToPyRosetta"])    
    global pyrosetta, alignment, collision_check, Pose, conformer_prep
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    import alignment
    import conformer_prep
    import collision_check

    pyrosetta.init("-mute all -multithreading:total_threads 10")

    # Grabbing values from config file
    print("Grabbing info from config file")
    params_list = default["ParamsList"]
    path_to_conformers = default["PathToConformers"]
    path_to_complexes = spec["PathToComplexes"]

    resfile = spec["Resfile"]
    fnr_bonus = float(spec["FNRBonus"])
    vdm_bonus = float(spec["VDMBonus"])
    pkl_file = spec["DesignPKLFile"]

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


    focus_pos = pre_pose.pdb_info().pdb2pose(spec["FocusChain"], int(spec["FocusResidueNumber"]))
    
    new_df = []
    df = pd.read_pickle(pkl_file)

    print("\n\nBeginning Design\n\n")
    score_type = "Norm. Score"
    df = df.sort_values(score_type, ascending = False).drop_duplicates(["Molecule ID"], keep = "first")
    

    for i, molecule_df in df.iterrows():    
        print("\n\nPreparing pose for design")
        pose = pyrosetta.pose_from_pdb(spec["DesignPDBFile"])
        ref_seq = pose.sequence()

        mol_id, conf_num, file_stem, lig_aid, r_aid, mutation_string, vdm_positions = molecule_df[[
        "Molecule ID", "Conformer Number", "Molecule File Stem", "Molecule Atoms", "Target Atoms",
        "Mutations", "Positions"]]

        print(f"{mol_id}, {conf_num}")

        lig = Pose()
        res_set = pyrosetta.generate_nonstandard_residue_set(lig, params_list = [f"{path_to_conformers}/{file_stem}.params"])
        pyrosetta.pose_from_file(lig, res_set, f"{path_to_conformers}/{file_stem}_{conf_num:04}.pdb")

        alignment.align_ligand_to_residue(lig, res, lig_aid, r_aid)

        vdm_mutations = determine_mutations(pose, mutation_string)

        pose_with_ligand = prepare_pose_for_design(pose, focus_pos, lig, lig_aid)

        pose_to_design = make_mutations(pose_with_ligand, vdm_mutations)
        
        conf_id = f"{mol_id}_{conf_num}"

        out_pose, all_mutations, mutations, SC, IFE, dSASA, ddG, dG_dSASA_ratio, num_buns = design(pose_to_design, pose_with_ligand, focus_pos, resfile, vdm_positions, vdm_mutations, ref_seq, conf_id, fnr_bonus = fnr_bonus, rtc_bonus = vdm_bonus)

        new_df.append(list(molecule_df[:-2]) + [mutations, all_mutations, SC, IFE, dSASA, ddG, dG_dSASA_ratio, num_buns])
        out_pose.dump_pdb(f"{path_to_complexes}/{mol_id}_{conf_num:04}.pdb")


    new_df = pd.DataFrame(new_df)
    new_df.columns = list(df.columns)[:-2] + ["Pose Numbered Mutations", "All Mutations", "SC", "IFE", "dSASA", "ddG", "dG/dSASA", "BUNs"]
    new_df.to_pickle(f"{path_to_complexes}/complex_scores.pkl")



    
if __name__ == "__main__":
    main(sys.argv[1:])
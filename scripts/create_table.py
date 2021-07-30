import sys
import os

import fnmatch
from configparser import ConfigParser
import argparse


def generate_params_pdb_and_table(mtp, csv_file_name, path_to_conformers, molecule_sdfs, use_mol_id = False, no_name = False):
    """
    Used internally to generate csv files for reading in by yield_ligand_pose, params files,
    and pdb objects for conformers, will create directories inside of path_to_conformers 
    which are either mol_id/ or mol_name/

    Arguments:
    csv_file: File to create or append new entries to
    path_to_confomers: directory where ligand conformer and params file directories should be held
    molecule_sdfs: list of strings of molec Moule sdfs to read in and generate params file/pdbs for
    use_mol_id: boolean, if a mol id is present on the third line of each molecule in the sdf
        uses that to name directories and files

    Creates:
    csv_file with the following columns:
        Molecule Name, Molecule ID, Molecule File Stem, Conformer Range, Molecule Atoms, Residue Atoms
        Molecule name: Name of the molecule as determined by first line of SDF
        Molecule ID: ID of the molecule as determiend by the third line of SDF (optional)
        Conformer Range: How many conformers were used in generating the params file
            e.g. 1-1 for 1 conformer, 1-100 for 100 conformers etc
            (Note this does assume that your conformers are in separate files)
        Molecule Atoms: MANUAL ENTRY atom labels for atoms on the conformer pdbs that correspond to target atoms
            e.g. C1-C4-C6 denotes three atoms C1, C4, C6 in that order
        Target Atoms: MANUAL ENTRY atom labels for target atoms on a Residue Object
            e.g. CD2-CZ2-CZ3 denote three atoms CD2, CZ2, CZ3 in that order on the target residue

    """
    try:
        os.mkdir(path_to_conformers)
    except:
        print(f"Directory {path_to_conformers} already made")
        pass


    with open(csv_file_name, "w+") as f:
        f.write("Molecule Name,Molecule ID,Conformer Range,Molecule Atoms,Target Atoms\n")
        for i, molecule_sdf in enumerate(molecule_sdfs):
            with open(molecule_sdf, "r", encoding='utf-8-sig') as sdf:
                lines = list(sdf)
                if len(lines) == 0:
                    continue
                mol_name = lines[0].strip()
                if use_mol_id:
                    mol_id = lines[2].split(" ")[0].strip()
                    file_stem = f"{mol_id}/{mol_id}"
                    dir_name = f"{path_to_conformers}/{mol_id}"
                elif not no_name:
                    mol_id = mol_name
                    file_stem =f"{mol_name}/{mol_name}"
                    dir_name = f"{path_to_conformers}/{mol_name}"

                else:
                    mol_name = i
                    mol_id = i
                    file_stem =f"{i}/{i}"
                    dir_name = f"{path_to_conformers}/{i}"

                try:
                    os.mkdir(dir_name)
                except:
                    print(f"Directory {dir_name} already made")
                    pass
                mtp.main([f"{molecule_sdf}", "-n", f"{path_to_conformers}/{file_stem}"])
                count = str(lines.count("$$$$\n"))

                f.write(",".join([f"\"{e}\"" for e in [mol_name, mol_id, f"{str(1)}_{count}_"]] + ["\n"]))


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
    spec = config["create_table"]
    ptr = "PathToRosetta"
    sys.path.append(f"{default[ptr]}/main/source/scripts/python/public")

    import molfile_to_params as mtp


    csv_file_name = default["CSVFileName"]
    path_to_conformers = default["PathToConformers"]
    input_molecule_sdfs = spec["MoleculeSDFs"].split(" ")


    molecule_sdfs = []
    for inp in input_molecule_sdfs:
        if "*" in inp:
            if "/" in inp:
                directory = "/".join([e for e in inp.split("/")[:-1]])
            else:
                directory = "."

            for file in os.listdir(directory):
                if fnmatch.fnmatch(file.lower(), inp.split("/")[-1]):
                    molecule_sdfs.append(f"{directory}/{file}")
        else:
            molecule_sdfs.append(inp)

    use_mol_id = spec["UseMoleculeID"] == "True"
    no_name = spec["NoName"] == "True"


    generate_params_pdb_and_table(mtp, csv_file_name, path_to_conformers, molecule_sdfs, use_mol_id, no_name)
    print(f"Succesfully generated table at {csv_file_name} and conformers at {path_to_conformers}")
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
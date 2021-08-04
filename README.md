<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 0.753 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β30
* Wed Aug 04 2021 11:48:09 GMT-0700 (PDT)
* Source doc: Running the Scripts
* Tables are currently converted to HTML tables.
----->
### Structural Replacement ###

This document will guide you through how to run the structural_replacement scripts for your project:

**Prerequisites:**

You do not need to know how to use any of the following, but they will be required by the scripts in order to run



* A local installation of PyRosetta
* A local installation of Rosetta (to use molfile_to_params script)
* The following Python Libraries (these can be installed in a terminal by typing “pip install lib-name”, more info can be found here [https://pypi.org/project/pip/](https://pypi.org/project/pip/))
    * [numpy](https://numpy.org/) - very useful for math operations
    * [pandas](https://pandas.pydata.org/) - allows the use of the DataFrame, which is essentially a powerful spreadsheet/table
    * [Sklearn](https://scikit-learn.org/stable/) - allows use of a nearest neighbors implementation, only needed if using vdMs
    * [ProDy](http://prody.csb.pitt.edu/) - used to create pdb files of vdMs, again only needed if using vdMs
* If you are using vdMs, an installation of a reduced subset of the vdM database (&lt;2 gb)

**General Usage:**

Each script is ran with the path to the config file as an argument, such as 


```
run a_script.py path/to/conf.txt
```


**Config File:**

The config file contains all of the information needed for the scripts to run.

Information in the [DEFAULT] category is used by all scripts, and each script has its own respective category.

_REQUIRED_ options need to be filled in to fit your own specific project’s needs, while _OPTIONAL_ options do NOT need to be filled in and can be left as is. The default settings in the optional options are what I have found to work the best. 

Additionally, a lot of the options in the config files are _paths_ to locations on your computer. Importantly these paths are in relation to where your scripts are located, NOT the config file. 

**Running the Scripts:**

Once the config file has been set up with all of the necessary information. The necessary commands and a brief outline of what each script does is outlined below. If you would like more information on what each script does or how to use them, that can be found **[here](https://docs.google.com/document/d/1NEq-mbIoxclpstKW4C55wvxyhdNPPFbA7jrmUidYLdk/edit?usp=sharing)**. 


```
run create_table.py conf.txt
```




* This will create a spreadsheet from the ligand sdf files provided in the _MoleculeSDFs _option, create Rosetta .params files to make them Rosetta-readable, and create .pdb files for each ligand
* If you want to have several conformers for each ligand, simply have all of the conformers for each ligand in one file and pass that to _MoleculeSDFs_
```
Manual input of atom alignments
```


* Here you will fill in the resulting “Molecule Atoms” and “Target Atoms” columns in the generated spreadsheet by adding the atom labels for each that can be found with a software such as PyMOL by clicking on the atoms.
* For example, if I wanted to align indole to Tryptophan, I’d go into PyMOL and find a substructure they share in common (such as the six-membered ring), choose corresponding atoms on that substructure, and list the atom labels .
* This would ultimately look like a “C1-C5-C7” in the “Molecule Atoms” columns and a “"CD2-CZ2-CZ3” in the “Target Atoms” column


```
run grade_conformers.py conf.txt
```


* This will align your ligands to your structural feature based off the alignments you have entered, as well as check for backbone collisions with the ligand
* This will turn the spreadsheet into a Pandas DataFrame and include some information about which conformers passed the collision check. It will save this as a .pkl file


```
run precalculate_vdm_space.py conf.txt
```


* This will use a [Resfile ](https://new.rosettacommons.org/docs/latest/rosetta_basics/file_types/resfiles)to pre-calculate all of the possible interactions in the active site
* The output of this will be three new files _VDMSpaceFileStem_-info.pkl, _VDMSpaceFileStem_-coords.pkl, and _VDMSpaceFileStem_-trees.pkl.
* This step will take up to an hour to run, but the results of it can be used several times. If you wish to change the resfile or change the settings, it will have to be ran again


```
run evaluate_conformers_on_vdm_space.py conf.txt
```


* This will use the outputted files from the previous script and give scores for each ligand, as well as store viewable .pdb files of the ligands and each of the interactions identified


```
run to_csv.py [path/to/dataframe.pkl] [path/to/new_spreadsheet.csv]
```


* If you are not comfortable using Pandas or pickle files, use this script at any point to convert a .pkl file DataFrame into an easily-readable .csv
* The VDMSpace files are not DataFrames so this will not produce anything useful for them

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import subprocess
import logging
import sisl as si
from sisl.viz import merge_plots
from sisl.viz.processors.math import normalize
import matplotlib.pyplot as plt
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import subprocess 

def get_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return None

Unitcell = """LatticeConstant 1 Ang
%block LatticeVectors
  35.000 0.000 0.000
  0.000 35.000 0.000
  0.000 0.000 35.000
%endblock LatticeVectors"""

Externalfield = """ %block ExternalElectricField
 0 0 0.1 V/ang
%endblock ExternalElectricField """

general_blocks = [

    'PAO.SoftDefault     T',
    'PAO.OldStylePolOrbs F',
    'PAO.EnergyShift  0.001 Ry',  
    'PAO.SplitNorm    0.15',      
    'PAO.BasisSize    DZP',
    '\n\n'
    #Exchange-Correlation functional
    'XC.Functional GGA',
    'XC.authors    PBE', 

    'MeshCutoff 200.0 Ry',
    'kgrid.Cutoff 0 bohr',

    #Structure relaxation
    'MD.TypeOfRun          CG',
    'MD.NumCGSteps         100',
    'MD.MaxForceTol        0.1 eV/Ang',
    'MD.MaxStressTol       0.1 GPa',
    'MD.TargetPressure     0.0 GPa',

    #Convergence of SCF
    'MaxSCFIterations    100',

    #SCF.mixer.method linear

    'SCF.Mixer.Weight    0.1',    # percentage new 
    'SCF.Mix density',
    'SCF.Mixer.History   3',       # num of steps taken to next iter 2-6
    'SCF.DM.Tolerance    1.0d-3',

    'DM.UseSaveDM           T',
    'SolutionMethod    diagon',
    'SaveRho                T',
    'writeDenchar           T',
    'TS.HS.Save             T'
#ElectroicTemperature 25 meV  # useful for metal cases to help convergence
#LongOutput T                 # for debugging
]
def generate_chemical_species_label(smiles):
    """
    Generate ChemicalSpeciesLabel block from a SMILES string

    :param smiles: SMILES representation of the molecule
    :return: Formatted ChemicalSpeciesLabel block
    """
    # Create RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Get unique atoms
    unique_atoms = {}
    atom_labels = []

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        symbol = atom.GetSymbol()

        if atomic_num not in unique_atoms:
            unique_atoms[atomic_num] = symbol

    # Generate ChemicalSpeciesLabel block
    label_block = ["%block ChemicalSpeciesLabel"]

    for index, (atomic_num, symbol) in enumerate(unique_atoms.items(), 1):
        label_line = f"{index}    {atomic_num} {symbol}"
        label_block.append(label_line)

    label_block.append("%endblock ChemicalSpeciesLabel")

    return "\n".join(label_block)

#The following is for coordinates 
def generate_chemical_coordinates(smiles, cell=[40, 40, 40], z_coordinate=None):
    """
    Generate AtomicCoordinatesAndAtomicSpecies block with an option for relaxed or fixed coordinates.

    :param smiles: SMILES string of the molecule
    :param cell: Cell dimensions [x, y, z]
    :param z_coordinate: Fixed z-coordinate for all atoms (if None, use relaxed coordinates)
    :return: AtomicCoordinatesAndAtomicSpecies block as a string
    """
    # Create RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Get unique atoms with their atomic numbers
    unique_atoms = {}
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        symbol = atom.GetSymbol()

        if atomic_num not in unique_atoms:
            unique_atoms[atomic_num] = symbol

    # Generate species indices for ChemicalSpeciesLabel
    species_indices = {}
    for index, (atomic_num, symbol) in enumerate(unique_atoms.items(), 1):
        species_indices[symbol] = index

    # Get atom coordinates
    positions = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Center the molecule in the cell
    mol_center = np.mean(positions, axis=0)
    cell_center = np.array(cell) / 2

    # Shift coordinates to cell center
    centered_positions = positions - mol_center + cell_center

    # Generate AtomicCoordinatesAndAtomicSpecies block
    coord_block = ["%block AtomicCoordinatesAndAtomicSpecies"]

    for pos, symbol in zip(centered_positions, symbols):
        # Use fixed z-coordinate if provided, otherwise use relaxed z-coordinate
        z = z_coordinate if z_coordinate is not None else pos[2]
        coord_line = f"  {pos[0]:.10f}  {pos[1]:.10f}  {z:.10f}    {species_indices[symbol]}"
        coord_block.append(coord_line)

    coord_block.append("%endblock AtomicCoordinatesAndAtomicSpecies")

    return "\n".join(coord_block)

# Create subfolders based on the DataFrame column
for index, row in df_continue.iterrows():
    # create folders and subfolders
    subfolder_name = row['Label']
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    #os.environ['SIESTA_SCRATCH'] = subfolder_path
    fdf_file_path = os.path.join(subfolder_path, f"{subfolder_name}.fdf")
    output_file_path = os.path.join(subfolder_path, "out")  # Define output path

    logging.info(f"Creating subfolder: {subfolder_path}")
    os.makedirs(subfolder_path, exist_ok=True)

    # dump pseudopotenital file 
    command = f'cp C.psml N.psml S.psml H.psml O.psml Cl.psml Br.psml {subfolder_path}/'
    subprocess.run(command, shell=True, check=True)  

    # get number of atoms from SMILES 
    smiles = row['skeleton_big']
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # get number of species then
    if mol is not None:
        mol_with_h = Chem.AddHs(mol)
        species_set = set()
        for atom in mol_with_h.GetAtoms():
            species_set.add(atom.GetSymbol())

        # The number of distinct species is the size of the set
        num_species = len(species_set)

    # fill up the script
    with open(fdf_file_path, 'w') as f:
            f.write("SystemName  HT_batch \n"),
            f.write(f"SystemLabel {subfolder_name}\n")
            f.write(f"NumberOfAtoms {num_atoms} \n")
            f.write(f"NumberOfSpecies {num_species} \n")
            f.write('\n\n')
            f.write(generate_chemical_species_label(smiles))
            f.write('\n\n')
            f.write(f'AtomicCoordinatesFormat  Ang \n')
            f.write(generate_chemical_coordinates(smiles))
            f.write('\n\n')
            f.write(Unitcell)
            f.write('\n\n')
            f.write("\n".join(general_blocks))

        # Modify paths to be relative to the subfolder
    relative_fdf_file = os.path.basename(fdf_file_path)
    relative_output_file = os.path.basename(output_file_path)

    command = f"mpirun -np 24 siesta < {relative_fdf_file} > {relative_output_file}"
    subprocess.run(command, sh)

# Create subfolders based on the DataFrame column
for index, row in df_continue.iterrows():
    # create folders and subfolders
    subfolder_name = row['Label']
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    fdf_file_path = os.path.join(subfolder_path, f"{subfolder_name}.fdf")
    output_file_path = os.path.join(subfolder_path, "out")  # Define output path

    logging.info(f"Creating subfolder: {subfolder_path}")
    os.makedirs(subfolder_path, exist_ok=True)

    # dump pseudopotenital file 
    command = f'cp C.psml N.psml S.psml H.psml O.psml Cl.psml Br.psml {subfolder_path}/'
    subprocess.run(command, shell=True, check=True)  

    # get number of atoms from SMILES 
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # get number of species then
    if mol is not None:
        mol_with_h = Chem.AddHs(mol)
        species_set = set()
        for atom in mol_with_h.GetAtoms():
            species_set.add(atom.GetSymbol())

        # The number of distinct species is the size of the set
        num_species = len(species_set)

    # fill up the script
    with open(fdf_file_path, 'w') as f:
            f.write("SystemName  HT_batch \n"),
            f.write(f"SystemLabel {subfolder_name}\n")
            f.write(f"NumberOfAtoms {num_atoms} \n")
            f.write(f"NumberOfSpecies {num_species} \n")
            f.write(f'NetCharge 1 \n')
            f.write('\n\n')
            f.write(generate_chemical_species_label(smiles))
            f.write('\n\n')
            f.write(f'AtomicCoordinatesFormat  Ang \n')
            f.write(generate_chemical_coordinates(smiles))
            f.write('\n\n')
            f.write(Unitcell)
            f.write('\n\n')
            f.write("\n".join(general_blocks))

        # Modify paths to be relative to the subfolder
    relative_fdf_file = os.path.basename(fdf_file_path)
    relative_output_file = os.path.basename(output_file_path)

    command = f"mpirun -np 22 siesta < {relative_fdf_file} > {relative_output_file}"
    subprocess.run(command, shell=True, check=True, cwd=subfolder_path)


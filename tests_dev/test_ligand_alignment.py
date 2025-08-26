#!/usr/bin/env python3
"""
Test script for improved ligand alignment functionality.
Tests the enhanced align_pdb_to_smiles_via_substructure_match function.
"""

import logging
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the improved function
sys.path.append('.')
from chai_lab.data.sources.pdb import align_pdb_to_smiles_via_substructure_match

def test_ligand_alignment():
    """Test the improved ligand alignment function."""
    
    # Test SMILES (example ligand)
    test_smiles = "CC(=O)N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc1ccc(-c2cc[nH]n2)c2ccccc12)C(C)(C)C"
    
    # Test PDB path (use your actual PDB file path)
    pdb_path = Path("./example_ligand.pdb")
    
    if not pdb_path.exists():
        logger.warning(f"PDB file not found: {pdb_path}")
        logger.info("Creating a test PDB file from SMILES...")
        
        # Create a test PDB file from SMILES
        create_test_pdb_from_smiles(test_smiles, pdb_path)
    
    logger.info("=== Testing Enhanced Ligand Alignment ===")
    logger.info(f"SMILES: {test_smiles}")
    logger.info(f"PDB file: {pdb_path}")
    
    # Test the alignment function
    try:
        aligned_mol, atom_mapping = align_pdb_to_smiles_via_substructure_match(
            smiles_string=test_smiles,
            pdb_path=pdb_path,
            ligand_chain_id='A'
        )
        
        if aligned_mol is not None and atom_mapping is not None:
            logger.info("✅ Alignment successful!")
            logger.info(f"Aligned molecule: {aligned_mol.GetNumAtoms()} atoms, {aligned_mol.GetNumBonds()} bonds")
            logger.info(f"Atom mapping: {len(atom_mapping)} mappings")
            
            # Analyze the mapping
            analyze_atom_mapping(aligned_mol, atom_mapping, test_smiles)
            
            # Test coordinate quality
            test_coordinate_quality(aligned_mol, atom_mapping)
            
        else:
            logger.error("❌ Alignment failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during alignment: {e}")

def create_test_pdb_from_smiles(smiles, output_path):
    """Create a test PDB file from SMILES for testing purposes."""
    from rdkit.Chem import AllChem
    
    # Generate 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Remove hydrogens for testing "coordinates only" scenario
    mol = Chem.RemoveHs(mol)
    
    # Write to PDB file
    writer = Chem.PDBWriter(str(output_path))
    writer.write(mol)
    writer.close()
    
    logger.info(f"Created test PDB file: {output_path}")

def analyze_atom_mapping(aligned_mol, atom_mapping, smiles):
    """Analyze the quality of atom mapping."""
    logger.info("=== Atom Mapping Analysis ===")
    
    # Create SMILES molecule for comparison
    smiles_mol = Chem.MolFromSmiles(smiles)
    smiles_mol = Chem.RemoveHs(smiles_mol)
    
    # Check element consistency
    element_matches = 0
    total_mappings = len(atom_mapping)
    
    for smiles_idx, pdb_idx in atom_mapping.items():
        if smiles_idx < smiles_mol.GetNumAtoms() and pdb_idx < aligned_mol.GetNumAtoms():
            smiles_element = smiles_mol.GetAtomWithIdx(smiles_idx).GetSymbol()
            pdb_element = aligned_mol.GetAtomWithIdx(pdb_idx).GetSymbol()
            
            if smiles_element == pdb_element:
                element_matches += 1
            else:
                logger.warning(f"Element mismatch: SMILES atom {smiles_idx} ({smiles_element}) -> PDB atom {pdb_idx} ({pdb_element})")
    
    element_accuracy = element_matches / total_mappings * 100 if total_mappings > 0 else 0
    logger.info(f"Element accuracy: {element_matches}/{total_mappings} ({element_accuracy:.1f}%)")
    
    # Count elements
    element_counts = {}
    for atom in aligned_mol.GetAtoms():
        element = atom.GetSymbol()
        element_counts[element] = element_counts.get(element, 0) + 1
    
    logger.info(f"Element distribution: {element_counts}")

def test_coordinate_quality(aligned_mol, atom_mapping):
    """Test the quality of coordinates."""
    logger.info("=== Coordinate Quality Test ===")
    
    # Get conformer
    if aligned_mol.GetNumConformers() > 0:
        conf = aligned_mol.GetConformer()
        
        # Calculate some basic geometric properties
        coords = []
        for i in range(aligned_mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        
        coords = np.array(coords)
        
        # Calculate center of mass
        center = np.mean(coords, axis=0)
        
        # Calculate distances from center
        distances = np.linalg.norm(coords - center, axis=1)
        
        logger.info(f"Molecule center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        logger.info(f"Average distance from center: {np.mean(distances):.2f}Å")
        logger.info(f"Max distance from center: {np.max(distances):.2f}Å")
        logger.info(f"Coordinate range: X=[{np.min(coords[:, 0]):.2f}, {np.max(coords[:, 0]):.2f}], "
                   f"Y=[{np.min(coords[:, 1]):.2f}, {np.max(coords[:, 1]):.2f}], "
                   f"Z=[{np.min(coords[:, 2]):.2f}, {np.max(coords[:, 2]):.2f}]")
    else:
        logger.warning("No conformer found in aligned molecule")

def test_different_smiles():
    """Test with different types of SMILES."""
    test_cases = [
        # Simple organic molecules
        ("Benzene", "c1ccccc1"),
        ("Ethanol", "CCO"),
        ("Acetone", "CC(=O)C"),
        
        # More complex molecules
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        
        # Molecules with different elements
        ("Sulfur compound", "CS(=O)(=O)C"),
        ("Phosphorus compound", "P(=O)(O)(O)O"),
    ]
    
    for name, smiles in test_cases:
        logger.info(f"\n=== Testing {name}: {smiles} ===")
        
        # Create temporary PDB file
        temp_pdb = Path(f"temp_{name.lower().replace(' ', '_')}.pdb")
        
        try:
            create_test_pdb_from_smiles(smiles, temp_pdb)
            
            aligned_mol, atom_mapping = align_pdb_to_smiles_via_substructure_match(
                smiles_string=smiles,
                pdb_path=temp_pdb,
                ligand_chain_id='A'
            )
            
            if aligned_mol is not None and atom_mapping is not None:
                logger.info(f"✅ {name} alignment successful: {len(atom_mapping)} mappings")
            else:
                logger.warning(f"❌ {name} alignment failed")
                
        except Exception as e:
            logger.error(f"❌ Error testing {name}: {e}")
        
        finally:
            # Clean up temporary file
            if temp_pdb.exists():
                temp_pdb.unlink()

if __name__ == "__main__":
    logger.info("Starting ligand alignment tests...")
    
    # Test main function
    test_ligand_alignment()
    
    # Test different SMILES
    logger.info("\n" + "="*50)
    logger.info("Testing different SMILES patterns...")
    test_different_smiles()
    
    logger.info("\n" + "="*50)
    logger.info("All tests completed!") 
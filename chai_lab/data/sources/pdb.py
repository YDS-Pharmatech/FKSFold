"""
Chai's num_atoms = atom_single_mask.shape (from feature_context, load_chains_from_raw):
protein: backbone + side chain atoms
ligand: all atoms
H is NOT included in num_atoms

SMILES Alignment Policy:
- SMILES coordinates are left empty by default
- Use align_pdb_to_smiles_via_substructure_match() for precise graph isomorphism alignment (but bond is needed)
  - rdkit AssignBondOrdersFromTemplate isn't accurate for PDBs without bond information
- Removed old element-based matching algorithms to avoid incorrect mappings
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D
from Bio.PDB import PDBParser
import numpy as np

from chai_lab.data.parsing.structure.residue import ConformerData
from chai_lab.data.sources.rdkit import RefConformerGenerator
from chai_lab.data.dataset.structure.all_atom_structure_context import AllAtomStructureContext
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.data.residue_constants import residue_atoms

logger = logging.getLogger(__name__)


def align_pdb_to_smiles_via_substructure_match(smiles_string, pdb_path, ligand_chain_id='E'):
    """
    Align PDB ligand to SMILES using RDKit substructure matching (graph isomorphism).
    
    This function handles PDB ligands that may only have coordinates without explicit 
    bond connectivity information. It uses RDKit to guess bonds based on inter-atomic 
    distances and then aligns with SMILES template.
    
    Args:
        smiles_string: SMILES string of the ligand
        pdb_path: Path to PDB file
        ligand_chain_id: Chain ID containing the ligand
        
    Returns:
        tuple: (aligned_mol, atom_mapping) where atom_mapping is dict {smiles_idx: pdb_idx}
        Returns (None, None) if alignment fails
    """
    try:
        # Generate molecule from SMILES
        smiles_mol = Chem.MolFromSmiles(smiles_string)
        if smiles_mol is None:
            logger.warning(f"Invalid SMILES: {smiles_string}")
            return None, None
        
        # Step 1: Read PDB ligand with minimal processing to preserve coordinates
        # Use removeHs=False and sanitize=False to keep original structure
        pdb_mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        if pdb_mol is None:
            logger.warning(f"Failed to read PDB file: {pdb_path}")
            return None, None
        
        # Step 2: Let RDKit guess bonds based on inter-atomic distances
        # This creates a "naked graph" with elements + single bonds
        try:
            # Sanitize only to adjust hydrogens and guess connectivity
            Chem.SanitizeMol(pdb_mol, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            logger.info(f"PDB molecule after sanitization: {pdb_mol.GetNumAtoms()} atoms, {pdb_mol.GetNumBonds()} bonds")
        except Exception as e:
            logger.warning(f"Failed to sanitize PDB molecule: {e}")
            # Try minimal sanitization
            try:
                Chem.SanitizeMol(pdb_mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
            except Exception as e2:
                logger.warning(f"Minimal sanitization also failed: {e2}")
                return None, None
        
        # Step 3: Handle hydrogen consistency between SMILES and PDB
        # Strategy: make both molecules have same hydrogen treatment
        original_pdb_mol = pdb_mol
        original_smiles_mol = smiles_mol
        
        # First try: Remove hydrogens from both
        try:
            pdb_mol_no_h = Chem.RemoveHs(pdb_mol)
            smiles_mol_no_h = Chem.RemoveHs(smiles_mol)
            
            logger.info(f"After removing H - PDB: {pdb_mol_no_h.GetNumAtoms()} atoms, SMILES: {smiles_mol_no_h.GetNumAtoms()} atoms")
            
            # Use SMILES as template to assign bond orders to PDB structure
            aligned_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol_no_h, pdb_mol_no_h)
            
            # Get atom mapping using substructure match
            atom_mapping = smiles_mol_no_h.GetSubstructMatch(aligned_mol, useChirality=True)
            if not atom_mapping:
                # Try without chirality constraint
                atom_mapping = smiles_mol_no_h.GetSubstructMatch(aligned_mol, useChirality=False)
            
            if atom_mapping:
                logger.info(f"Successfully aligned without hydrogens: {len(atom_mapping)} atom mappings")
                # Convert to dict for easier use: {smiles_idx: pdb_idx}
                mapping_dict = {i: atom_mapping[i] for i in range(len(atom_mapping))}
                
                # Ensure hybridization is correct
                for atom in aligned_mol.GetAtoms():
                    atom.UpdatePropertyCache(strict=False)
                
                return aligned_mol, mapping_dict
                
        except Exception as e:
            logger.warning(f"Failed to align without hydrogens: {e}")
        
        # Second try: Add hydrogens to both
        try:
            pdb_mol_with_h = Chem.AddHs(pdb_mol, addCoords=True)
            smiles_mol_with_h = Chem.AddHs(smiles_mol, addCoords=False)
            
            logger.info(f"After adding H - PDB: {pdb_mol_with_h.GetNumAtoms()} atoms, SMILES: {smiles_mol_with_h.GetNumAtoms()} atoms")
            
            # Use SMILES as template to assign bond orders to PDB structure
            aligned_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol_with_h, pdb_mol_with_h)
            
            # Get atom mapping using substructure match
            atom_mapping = smiles_mol_with_h.GetSubstructMatch(aligned_mol, useChirality=True)
            if not atom_mapping:
                # Try without chirality constraint
                atom_mapping = smiles_mol_with_h.GetSubstructMatch(aligned_mol, useChirality=False)
            
            if atom_mapping:
                logger.info(f"Successfully aligned with hydrogens: {len(atom_mapping)} atom mappings")
                # Convert to dict for easier use: {smiles_idx: pdb_idx}
                mapping_dict = {i: atom_mapping[i] for i in range(len(atom_mapping))}
                
                # Ensure hybridization is correct
                for atom in aligned_mol.GetAtoms():
                    atom.UpdatePropertyCache(strict=False)
                
                return aligned_mol, mapping_dict
                
        except Exception as e:
            logger.warning(f"Failed to align with hydrogens: {e}")
        
        # Third try: Use original molecules as-is
        try:
            logger.info(f"Original molecules - PDB: {original_pdb_mol.GetNumAtoms()} atoms, SMILES: {original_smiles_mol.GetNumAtoms()} atoms")
            
            # Use SMILES as template to assign bond orders to PDB structure
            aligned_mol = AllChem.AssignBondOrdersFromTemplate(original_smiles_mol, original_pdb_mol)
            
            # Get atom mapping using substructure match
            atom_mapping = original_smiles_mol.GetSubstructMatch(aligned_mol, useChirality=True)
            if not atom_mapping:
                # Try without chirality constraint
                atom_mapping = original_smiles_mol.GetSubstructMatch(aligned_mol, useChirality=False)
            
            if atom_mapping:
                logger.info(f"Successfully aligned with original molecules: {len(atom_mapping)} atom mappings")
                # Convert to dict for easier use: {smiles_idx: pdb_idx}
                mapping_dict = {i: atom_mapping[i] for i in range(len(atom_mapping))}
                
                # Ensure hybridization is correct
                for atom in aligned_mol.GetAtoms():
                    atom.UpdatePropertyCache(strict=False)
                
                return aligned_mol, mapping_dict
                
        except Exception as e:
            logger.warning(f"Failed to align with original molecules: {e}")
        
        logger.warning("No substructure match found between SMILES and PDB with any hydrogen treatment")
        return None, None
        
    except Exception as e:
        logger.error(f"Error in substructure matching: {e}")
        return None, None


def align_pdb_to_smiles(smiles_string, pdb_path):
    ref_mol = Chem.MolFromSmiles(smiles_string)
    pdb_mol = Chem.MolFromPDBFile(str(pdb_path))
    
    # Use SMILES as template to transfer bond order information to PDB structure
    aligned_mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, pdb_mol)

    # Ensure hybridization is correct
    for atom in aligned_mol.GetAtoms():
        atom.UpdatePropertyCache(strict=False)

    return aligned_mol


class PDBConformerGenerator(RefConformerGenerator):
    def __init__(self, pdb_path: Path):
        super().__init__()
        self.pdb_path = pdb_path
        
    def generate(self, smiles: str) -> ConformerData:
        """Generate conformer data with empty coordinates by default.
        
        Note: SMILES coordinates are left empty by default.
        Use align_pdb_to_smiles_via_substructure_match() separately if PDB alignment is needed.
        """
        # Generate molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid smiles {smiles}"

        mol_with_hs = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        params.randomSeed = 123
        params.useChirality = True
        params.maxAttempts = 10_000
        params.useRandomCoords = True

        # Generate standard ETKDG conformer without PDB alignment
        AllChem.EmbedMultipleConfs(mol_with_hs, numConfs=1, params=params)
        mol_etkdg = Chem.RemoveHs(mol_with_hs)

        element_counter: dict = defaultdict(int)
        for atom in mol_etkdg.GetAtoms():
            elem = atom.GetSymbol()
            element_counter[elem] += 1
            atom.SetProp("name", elem + str(element_counter[elem]))

        # mol_aligned = align_pdb_to_smiles(smiles, self.pdb_path)
        mol_aligned = align_pdb_to_smiles_via_substructure_match(smiles, self.pdb_path)
        
        if mol_aligned is None:
            print(f"Warning: Failed to align PDB to SMILES, falling back to ETKDG")
            retval = self._load_ref_conformer_from_rdkit(mol_etkdg)
            retval.atom_names = [a.upper() for a in retval.atom_names]
            return retval
        
        # Copy PDB coordinates to ETKDG-generated molecule
        conf_id = mol_etkdg.AddConformer(Chem.Conformer(mol_etkdg.GetNumAtoms()), assignId=True)
        conf = mol_etkdg.GetConformer(conf_id)
        
        # Copy coordinates from aligned_mol
        for i in range(mol_etkdg.GetNumAtoms()):
            pos = mol_aligned.GetConformer().GetAtomPosition(i)
            conf.SetAtomPosition(i, pos)
        
        # Use ETKDG molecule with PDB coordinates
        retval = self._load_ref_conformer_from_rdkit(mol_etkdg)
        retval.atom_names = [a.upper() for a in retval.atom_names]
        return retval


def load_pdb_coordinates(
    pdb_path: Path,
    structure_context: AllAtomStructureContext,
    device: torch.device,
    center_coords: bool = True,
    add_noise: float = 0.0,
    copy_coords_only: bool = False,
    ligand_smiles: Optional[str] = None,
    align_smiles: bool = False,
) -> torch.Tensor | None:
    """
    Load coordinates from PDB for structure context.
    
    Args:
        pdb_path: Path to PDB file
        structure_context: Structure context to match
        device: Device to put tensors on
        center_coords: Whether to center coordinates
        add_noise: Amount of noise to add
        copy_coords_only: If True, only copy coordinates without processing
        ligand_smiles: Optional SMILES string for ligand alignment
        align_smiles: If True, use graph isomorphism matching for precise SMILES-PDB alignment
    """
    logger.info(f"Loading PDB coordinates from {pdb_path}")
    
    if align_smiles and ligand_smiles:
        logger.info(f"Graph isomorphism alignment enabled for SMILES: {ligand_smiles}")
    elif align_smiles and not ligand_smiles:
        logger.warning("align_smiles=True but no ligand_smiles provided, skipping graph isomorphism alignment")
    else:
        logger.info("Graph isomorphism alignment disabled (align_smiles=False)")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    
    # Create extended residue mapping for non-standard residues
    from chai_lab.data.residue_constants import residue_atoms
    standard_residues = set(residue_atoms.keys())
    
    # Extended non-standard residue mapping
    extended_residue_map = {
        'MSE': 'MET', 'CSO': 'CYS', 'SEP': 'SER', 'PTR': 'TYR', 'TPO': 'THR',
        'HYP': 'PRO', 'ASX': 'ASP', 'GLX': 'GLU', 'PCA': 'GLU', 'KCX': 'LYS',
        '2AS': 'ASP', '3AH': 'HIS', '5HP': 'GLU', 'ACL': 'ARG', 'AIB': 'ALA',
        'ALM': 'ALA', 'ALO': 'THR', 'ALY': 'LYS', 'ARM': 'ARG', 'ASA': 'ASP',
        'ASB': 'ASP', 'ASK': 'ASP', 'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA',
        'BCS': 'CYS', 'BHD': 'ASP', 'BMT': 'THR', 'BNN': 'ALA', 'BUC': 'CYS',
        'BUG': 'LEU', 'C5C': 'CYS', 'C6C': 'CYS', 'CCS': 'CYS', 'CEA': 'CYS',
        'CHG': 'ALA', 'CLE': 'LEU', 'CME': 'CYS', 'CSD': 'ALA', 'CSP': 'CYS',
        'CSS': 'CYS', 'CSW': 'CYS', 'CXM': 'MET', 'CY1': 'CYS', 'CY3': 'CYS',
        'CYG': 'CYS', 'CYM': 'CYS', 'CYQ': 'CYS', 'DAH': 'PHE', 'DAL': 'ALA',
        'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'GLU', 'DGN': 'GLN',
        'DHA': 'ALA', 'DHI': 'HIS', 'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'LEU',
        'DLY': 'LYS', 'DNP': 'ALA', 'DPN': 'PHE', 'DPR': 'PRO', 'DSN': 'SER',
        'DSP': 'ASP', 'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL',
        'EFC': 'CYS', 'FLA': 'ALA', 'FME': 'MET', 'GGL': 'GLU', 'GLZ': 'GLY',
        'GMA': 'GLU', 'GSC': 'GLY', 'HAC': 'ALA', 'HAR': 'ARG', 'HIC': 'HIS',
        'HIP': 'HIS', 'HMR': 'ARG', 'HPQ': 'PHE', 'HSD': 'HIS', 'HSE': 'HIS',
        'HSP': 'HIS', 'HTR': 'TRP', 'IIL': 'ILE', 'IYR': 'TYR', 'LLY': 'LYS',
        'LTR': 'TRP', 'LYM': 'LYS', 'LYZ': 'LYS', 'MAA': 'ALA', 'MEN': 'ASN',
        'MHS': 'HIS', 'MIS': 'SER', 'MLE': 'LEU', 'MPQ': 'GLY', 'MSA': 'GLY',
        'MVA': 'VAL', 'NEM': 'HIS', 'NEP': 'HIS', 'NLE': 'LEU', 'NLN': 'LEU',
        'NLP': 'LEU', 'NMC': 'GLY', 'OAS': 'SER', 'OCS': 'CYS', 'OMT': 'MET',
        'PAQ': 'TYR', 'PEC': 'CYS', 'PHI': 'PHE', 'PHL': 'PHE', 'PR3': 'CYS',
        'PRR': 'ALA', 'SAC': 'SER', 'SAR': 'GLY', 'SCH': 'CYS', 'SCS': 'CYS',
        'SCY': 'CYS', 'SEL': 'SER', 'SET': 'SER', 'SHC': 'CYS', 'SHR': 'LYS',
        'SOC': 'CYS', 'STY': 'TYR', 'SVA': 'SER', 'TIH': 'ALA', 'TPL': 'TRP',
        'TPQ': 'ALA', 'TRG': 'LYS', 'TRO': 'TRP', 'TYB': 'TYR', 'TYQ': 'TYR',
        'TYS': 'TYR', 'TYY': 'TYR', 'AGM': 'ARG', 'GL3': 'GLY', 'SMC': 'CYS',
        'CGU': 'GLU', 'CSX': 'CYS',
    }
    
    # Extract PDB atoms with multiple indexing structures
    pdb_atoms = {}
    pdb_atoms_by_chain_res = defaultdict(list)  # Index by chain and residue
    pdb_atoms_by_atom_name = defaultdict(list)  # Index by atom name
    residue_name_mapping = {}  # Record actual PDB residue names
    chain_residue_offsets = {}  # Record residue number offsets for each chain
    
    # Extract residue types and atom information from PDB
    pdb_residues = {}
    pdb_chains = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            pdb_chains[chain_id] = []
            
            # Calculate residue offset for this chain
            residues = list(chain)
            if residues:
                first_res_id = residues[0].get_id()[1]
                expected_first_res_id = 1
                offset = first_res_id - expected_first_res_id
                chain_residue_offsets[chain_id] = offset
                logger.info(f"Chain {chain_id} residue offset: {offset} (starts from {first_res_id})")
            
            for residue in chain:
                res_id = residue.get_id()[1]
                res_name = residue.get_resname()
                
                # Build residue name mapping
                if res_name not in residue_name_mapping:
                    if res_name in standard_residues:
                        residue_name_mapping[res_name] = res_name
                    elif res_name in extended_residue_map:
                        residue_name_mapping[res_name] = extended_residue_map[res_name]
                        logger.info(f"Mapping non-standard residue {res_name} to {extended_residue_map[res_name]}")
                    else:
                        residue_name_mapping[res_name] = "UNK"
                        logger.warning(f"Unknown residue type {res_name}, treating as UNK")
                
                atoms = {}
                for atom in residue:
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    element = atom.element if hasattr(atom, 'element') else _guess_element_from_name(atom_name)
                    atoms[atom_name] = {
                        'coord': coord,
                        'element': element
                    }
                    
                    # Build multiple indexing structures
                    original_key = (chain_id, res_id, res_name, atom_name)
                    pdb_atoms[original_key] = coord
                    
                    # Multiple indices for search
                    pdb_atoms_by_chain_res[(chain_id, res_id)].append((res_name, atom_name, coord))
                    pdb_atoms_by_atom_name[atom_name].append((chain_id, res_id, res_name, coord))
                    
                    # Create mapping for standard residue names
                    if res_name in residue_name_mapping and residue_name_mapping[res_name] != "UNK":
                        mapped_res_name = residue_name_mapping[res_name]
                        mapped_key = (chain_id, res_id, mapped_res_name, atom_name)
                        pdb_atoms[mapped_key] = coord
                
                residue_key = (chain_id, res_id, res_name)
                pdb_residues[residue_key] = atoms
                pdb_chains[chain_id].append((res_id, res_name))
    
    # Log chain information
    num_atoms = structure_context.num_atoms
    logger.info(f"Structure context num_atoms: {num_atoms}")
    
    model_atom_mask = structure_context.atom_exists_mask.to(device)
    logger.info(f"model_atom_mask shape: {model_atom_mask.shape}")
    
    unique_residue_types = set()
    for (_, _, res_name) in pdb_residues.keys():
        unique_residue_types.add(res_name)
    logger.info(f"Found {len(unique_residue_types)} unique residue types in PDB")
    
    # Count residues and atoms per chain
    chain_stats = []
    for chain_id, residues in pdb_chains.items():
        num_residues = len(residues)
        chain_num_atoms = sum(len(pdb_residues[(chain_id, res_id, res_name)]) for res_id, res_name in residues)
        chain_stats.append((chain_id, num_residues, chain_num_atoms))
    logger.info(f"PDB chain sizes (residues/atoms): {chain_stats}")
    
    # Get expected asym sizes
    expected_protein_sizes = []
    expected_ligand_sizes = []
    
    for asym_id in torch.unique(structure_context.token_asym_id):
        asym_mask = structure_context.token_asym_id == asym_id
        asym_tokens = asym_mask.sum().item()
        
        asym_atom_mask = torch.isin(structure_context.atom_token_index, torch.where(asym_mask)[0])
        asym_atoms = asym_atom_mask.sum().item()
        
        entity_type = structure_context.token_entity_type[asym_mask][0].item()
        
        if entity_type == EntityType.PROTEIN.value:
            expected_protein_sizes.append((asym_id.item(), asym_tokens, asym_atoms))
        elif entity_type == EntityType.LIGAND.value:
            expected_ligand_sizes.append((asym_id.item(), asym_tokens, asym_atoms))
    
    logger.info(f"Expected asym sizes (residues/atoms): protein={expected_protein_sizes}, ligand={expected_ligand_sizes}")
    
    # Initialize coordinates tensor
    structure_coords = torch.zeros((1, num_atoms, 3), device=device)
    
    # Strategy statistics
    strategy_stats = {
        "offset_exact": 0,
        "smart_match": 0,
        "chain_flexible": 0,
        "strategy_6_graph_isomorphism": 0,  # Graph isomorphism matching for SMILES
    }
    
    strategy_names = {
        "offset_exact": "策略1a: 偏移精确匹配",
        "smart_match": "策略1b: 智能匹配",
        "chain_flexible": "策略2: 链灵活匹配", 
        "strategy_6_graph_isomorphism": "策略6: 图同构匹配"
    }
    
    chain_assignment_stats = {}
    correct_chain_count = 0
    wrong_chain_count = 0
    not_found_count = 0
    
    # Match protein chains first
    protein_chain_mapping = {}
    
    for chain_id, num_residues, chain_num_atoms in chain_stats:
        for asym_id, expected_residues, expected_atoms in expected_protein_sizes:
            if (asym_id not in protein_chain_mapping.values() and 
                num_residues == expected_residues and chain_num_atoms == expected_atoms):
                protein_chain_mapping[chain_id] = asym_id
                logger.info(f"Matched protein asym_id {asym_id} (res {expected_residues}, atoms {expected_atoms}) to chain {chain_id} (res {num_residues}, atoms {chain_num_atoms})")
                break
    
    # Match ligand chains
    ligand_chain_mapping = {}
    
    for chain_id, num_residues, chain_num_atoms in chain_stats:
        if chain_id not in protein_chain_mapping:
            for asym_id, expected_residues, expected_atoms in expected_ligand_sizes:
                if (asym_id not in ligand_chain_mapping.values() and 
                    chain_num_atoms == expected_atoms):
                    ligand_chain_mapping[chain_id] = asym_id
                    logger.info(f"Matched ligand asym_id {asym_id} (res {expected_residues}, atoms {expected_atoms}) to chain {chain_id} (res {num_residues}, atoms {chain_num_atoms})")
                    break
    
    all_chain_mapping = {**protein_chain_mapping, **ligand_chain_mapping}
    
    # Process ligand atoms with graph isomorphism first
    ligand_processed_atoms = set()
    
    # First pass: Handle all ligand atoms with graph isomorphism matching
    for asym_id in torch.unique(structure_context.token_asym_id):
        asym_mask = structure_context.token_asym_id == asym_id
        entity_type = structure_context.token_entity_type[asym_mask][0].item()
        
        if entity_type == EntityType.LIGAND.value and align_smiles and ligand_smiles is not None:
            # Collect all ligand atoms for this asym_id
            ligand_token_mask = structure_context.token_asym_id == asym_id
            ligand_atom_mask = torch.isin(structure_context.atom_token_index, torch.where(ligand_token_mask)[0])
            ligand_atom_indices = torch.where(ligand_atom_mask)[0]
            
            # Find the ligand chain
            ligand_chain_id = None
            for chain_id, mapped_asym_id in ligand_chain_mapping.items():
                if mapped_asym_id == asym_id.item():
                    ligand_chain_id = chain_id
                    break
            
            if ligand_chain_id:
                logger.info(f"Processing ligand asym_id {asym_id.item()} using ONLY graph isomorphism alignment in chain {ligand_chain_id}")
                
                # Use graph isomorphism matching
                aligned_mol, atom_mapping = align_pdb_to_smiles_via_substructure_match(
                    smiles_string=ligand_smiles,
                    pdb_path=pdb_path,
                    ligand_chain_id=ligand_chain_id
                )
                
                if aligned_mol is not None and atom_mapping is not None:
                    logger.info(f"Graph isomorphism alignment successful for ligand asym_id {asym_id.item()} with {len(atom_mapping)} atom mappings")
                    
                    # Apply the mapping to all ligand atoms
                    conf = aligned_mol.GetConformer()
                    
                    for i, ligand_atom_idx in enumerate(ligand_atom_indices):
                        if i in atom_mapping:
                            pdb_atom_idx = atom_mapping[i]
                            pos = conf.GetAtomPosition(pdb_atom_idx)
                            coord = torch.tensor([pos.x, pos.y, pos.z], device=device, dtype=torch.float32)
                            structure_coords[0, ligand_atom_idx] = coord
                            strategy_stats["strategy_6_graph_isomorphism"] += 1
                            ligand_processed_atoms.add(ligand_atom_idx.item())
                            
                            # Update chain assignment statistics
                            expected_chain = ligand_chain_id
                            if asym_id.item() not in chain_assignment_stats:
                                chain_assignment_stats[asym_id.item()] = {"correct": 0, "wrong": 0, "not_found": 0, "expected_chain": expected_chain}
                            
                            chain_assignment_stats[asym_id.item()]["correct"] += 1
                            correct_chain_count += 1
                        else:
                            # Atom not in mapping - mark as not found
                            if asym_id.item() not in chain_assignment_stats:
                                chain_assignment_stats[asym_id.item()] = {"correct": 0, "wrong": 0, "not_found": 0, "expected_chain": ligand_chain_id}
                            
                            chain_assignment_stats[asym_id.item()]["not_found"] += 1
                            not_found_count += 1
                            ligand_processed_atoms.add(ligand_atom_idx.item())  # Mark as processed to skip in second pass
                else:
                    logger.warning(f"Graph isomorphism alignment failed for ligand asym_id {asym_id.item()}, skipping all ligand atoms")
                    # Mark all ligand atoms as not found
                    for ligand_atom_idx in ligand_atom_indices:
                        if asym_id.item() not in chain_assignment_stats:
                            chain_assignment_stats[asym_id.item()] = {"correct": 0, "wrong": 0, "not_found": 0, "expected_chain": ligand_chain_id}
                        
                        chain_assignment_stats[asym_id.item()]["not_found"] += 1
                        not_found_count += 1
                        ligand_processed_atoms.add(ligand_atom_idx.item())
            else:
                logger.warning(f"No chain mapping found for ligand asym_id {asym_id.item()}, skipping all ligand atoms")
                # Mark all ligand atoms as not found
                for ligand_atom_idx in ligand_atom_indices:
                    if asym_id.item() not in chain_assignment_stats:
                        chain_assignment_stats[asym_id.item()] = {"correct": 0, "wrong": 0, "not_found": 0, "expected_chain": None}
                    
                    chain_assignment_stats[asym_id.item()]["not_found"] += 1
                    not_found_count += 1
                    ligand_processed_atoms.add(ligand_atom_idx.item())
    
    # Second pass: Handle non-ligand atoms with traditional strategies
    for atom_idx in range(num_atoms):
        if not model_atom_mask[atom_idx]:
            continue
        
        # Skip atoms that were already processed in first pass (ligand atoms)
        if atom_idx in ligand_processed_atoms:
            continue
            
        token_idx = structure_context.atom_token_index[atom_idx].item()
        asym_id = structure_context.token_asym_id[token_idx].item()
        residue_idx = structure_context.token_residue_index[token_idx].item()
        entity_type = structure_context.token_entity_type[token_idx].item()
        
        atom_name = structure_context.atom_ref_name[atom_idx]
        expected_element = structure_context.atom_ref_element[atom_idx].item()
        
        # Get expected chain mapping
        expected_chain = None
        for chain_id, mapped_asym_id in all_chain_mapping.items():
            if mapped_asym_id == asym_id:
                expected_chain = chain_id
                break
        
        # Default to chain A if no mapping found
        if expected_chain is None:
            expected_chain = 'A'
        
        found_coord = None
        found_chain = None
        used_strategy = None
        matched_via = None
        
        # Strategy 1: Exact matching - prioritize smart matched chain with residue offset
        if found_coord is None:
            # First try exact matching with known residue offset
            chain_offset = chain_residue_offsets.get(expected_chain, 0)
            target_res_id_with_offset = residue_idx + 1 + chain_offset
            
            # Try offset exact matching
            for res_name in list(standard_residues) + list(residue_name_mapping.keys()):
                atom_key = (expected_chain, target_res_id_with_offset, res_name, atom_name)
                if atom_key in pdb_atoms:
                    found_coord = torch.tensor(pdb_atoms[atom_key], device=device, dtype=torch.float32)
                    found_chain = expected_chain
                    used_strategy = "offset_exact"
                    matched_via = f"offset_exact_{res_name}_offset{chain_offset}"
                    break
            
            # If offset matching fails, try small range search
            if found_coord is None:
                for res_id_offset in range(-2, 3):  # Small search range
                    target_res_id = target_res_id_with_offset + res_id_offset
                    
                    # Try smart matched chain first
                    for res_name in list(standard_residues) + list(residue_name_mapping.keys()):
                        atom_key = (expected_chain, target_res_id, res_name, atom_name)
                        if atom_key in pdb_atoms:
                            found_coord = torch.tensor(pdb_atoms[atom_key], device=device, dtype=torch.float32)
                            found_chain = expected_chain
                            used_strategy = "smart_match"
                            matched_via = f"smart_match_{res_name}_totaloffset{chain_offset + res_id_offset}"
                            break
                    if found_coord is not None:
                        break
        
        # Strategy 2: Chain flexible matching - if smart matching fails, try all chains
        if found_coord is None:
            # Try all chains sorted by size
            for chain_id, num_residues, chain_num_atoms in chain_stats:
                if found_coord is not None:
                    break
                
                for res_id_offset in range(-5, 6):  # Larger search range
                    target_res_id = residue_idx + 1 + res_id_offset
                    
                    for res_name in list(standard_residues) + list(residue_name_mapping.keys()):
                        atom_key = (chain_id, target_res_id, res_name, atom_name)
                        if atom_key in pdb_atoms:
                            found_coord = torch.tensor(pdb_atoms[atom_key], device=device, dtype=torch.float32)
                            found_chain = chain_id
                            used_strategy = "chain_flexible"
                            matched_via = f"chain_flexible_{res_name}_offset{res_id_offset}"
                            break
                    if found_coord is not None:
                        break
        

        
        # Update coordinates and statistics for non-ligand atoms
        if found_coord is not None:
            structure_coords[0, atom_idx] = found_coord
            
            # Record strategy statistics
            if matched_via:
                # Extract strategy type
                if matched_via.startswith('offset_exact'):
                    strategy_key = "offset_exact"
                elif matched_via.startswith('smart_match'):
                    strategy_key = "smart_match"
                elif matched_via.startswith('chain_flexible'):
                    strategy_key = "chain_flexible"
                else:
                    strategy_key = matched_via.split('_')[0] if '_' in matched_via else matched_via
                
                strategy_stats[strategy_key] += 1
            elif used_strategy:
                strategy_stats[used_strategy] += 1
            
            # Update chain assignment statistics
            expected_chain = None
            for chain_id, mapped_asym_id in all_chain_mapping.items():
                if mapped_asym_id == asym_id:
                    expected_chain = chain_id
                    break
            
            if expected_chain:
                if asym_id not in chain_assignment_stats:
                    chain_assignment_stats[asym_id] = {"correct": 0, "wrong": 0, "not_found": 0, "expected_chain": expected_chain}
                
                if found_chain == expected_chain:
                    chain_assignment_stats[asym_id]["correct"] += 1
                    correct_chain_count += 1
                else:
                    chain_assignment_stats[asym_id]["wrong"] += 1
                    wrong_chain_count += 1
        else:
            if asym_id not in chain_assignment_stats:
                expected_chain = None
                for chain_id, mapped_asym_id in all_chain_mapping.items():
                    if mapped_asym_id == asym_id:
                        expected_chain = chain_id
                        break
                chain_assignment_stats[asym_id] = {"correct": 0, "wrong": 0, "not_found": 0, "expected_chain": expected_chain}
            
            chain_assignment_stats[asym_id]["not_found"] += 1
            not_found_count += 1
    
    # Log detailed statistics
    logger.info("=== Chain Assignment Statistics ===")
    for asym_id, stats in chain_assignment_stats.items():
        logger.info(f"Asym ID {asym_id} (expected chain {stats['expected_chain']}):")
        logger.info(f"  Correct chain: {stats['correct']}")
        logger.info(f"  Wrong chain: {stats['wrong']}")
        logger.info(f"  Not found: {stats['not_found']}")
    
    total_atoms = correct_chain_count + wrong_chain_count + not_found_count
    logger.info(f"Total atoms - Correct: {correct_chain_count}, Wrong chain: {wrong_chain_count}, Not found: {not_found_count}")
    
    # Log strategy usage
    logger.info("=== Matching Strategy Usage ===")
    total_matched = sum(strategy_stats.values())
    
    for strategy_key, count in sorted(strategy_stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = count / total_matched * 100 if total_matched > 0 else 0
            strategy_name = strategy_names.get(strategy_key, strategy_key)
            logger.info(f"{strategy_name}: {count} atoms ({percentage:.1f}%)")
    
    # Calculate success rates
    if total_atoms > 0:
        success_rate = (correct_chain_count / total_atoms) * 100
        chain_accuracy = (correct_chain_count / (correct_chain_count + wrong_chain_count)) * 100 if (correct_chain_count + wrong_chain_count) > 0 else 0
        
        logger.info(f"PDB coordinate alignment success rate: {success_rate:.2f}%")
        logger.info(f"Chain assignment accuracy: {chain_accuracy:.2f}%")
    
    if copy_coords_only:
        logger.info("Copy coordinates only mode: skipping centering, noise, and other processing")
        logger.info("=== Copy Coordinates Only Mode - Quality Check ===")
        
        ref_coords = structure_context.atom_ref_pos.to(device)
        
        # Debug information
        logger.info(f"structure_coords shape: {structure_coords.shape}")
        logger.info(f"structure_coords[0] shape: {structure_coords[0].shape}")
        logger.info(f"ref_coords shape: {ref_coords.shape}")
        logger.info(f"model_atom_mask shape: {model_atom_mask.shape}")
        
        # Ensure consistent shapes
        if structure_coords[0].shape[0] != ref_coords.shape[0]:
            logger.error(f"Shape mismatch: structure_coords[0] has {structure_coords[0].shape[0]} atoms, but ref_coords has {ref_coords.shape[0]} atoms")
            logger.error("This indicates a bug in coordinate assignment. Returning None.")
            return None
        
        # Only compare atoms that have coordinates set (non-zero)
        coords_set_mask = (structure_coords[0].abs().sum(dim=-1) > 0)
        valid_mask = model_atom_mask & coords_set_mask
        
        if valid_mask.sum() > 0:
            coord_diff = (structure_coords[0] - ref_coords).abs().sum(dim=-1)
            valid_diff = coord_diff[valid_mask]
            mean_diff = valid_diff.mean().item()
            max_diff = valid_diff.max().item()
            std_diff = valid_diff.std().item()
            
            logger.info("Coordinate difference between PDB and reference (before any processing):")
            logger.info(f"  Mean difference: {mean_diff:.2f}Å")
            logger.info(f"  Max difference: {max_diff:.2f}Å")
            logger.info(f"  Std difference: {std_diff:.2f}Å")
            logger.info(f"  Atoms with coordinates: {valid_mask.sum().item()}/{model_atom_mask.sum().item()}")
            logger.info("Large coordinate differences are expected in copy-only mode due to different coordinate frames")
        else:
            logger.warning("No atoms have coordinates set!")
        
        logger.info(f"Returning structure_coords with shape: {structure_coords.shape}")
        return structure_coords
    
    # Continue with normal processing (centering, noise, etc.)
    if center_coords:
        atom_mask_expanded = model_atom_mask.unsqueeze(-1)  
        valid_coords = structure_coords * atom_mask_expanded
        
        total_mass = model_atom_mask.sum()
        if total_mass > 0:
            center_of_mass = valid_coords.sum(dim=1, keepdim=True) / total_mass
            structure_coords = structure_coords - center_of_mass
    
    if add_noise > 0:
        noise = torch.randn_like(structure_coords) * add_noise
        atom_mask_expanded = model_atom_mask.unsqueeze(0).unsqueeze(-1)
        structure_coords = structure_coords + noise * atom_mask_expanded
    
    return structure_coords


def _calculate_residue_offset(chain_id: str, pdb_chains: dict) -> int:
    """Calculate the residue offset for a given chain."""
    residues = pdb_chains.get(chain_id, [])
    if not residues:
        return 0
    
    # Get the minimum residue ID in this chain
    min_res_id = min(res_id for res_id, _ in residues)
    
    # Offset is typically (min_res_id - 1) to make it 0-based
    offset = min_res_id - 1
    
    return offset


def _guess_element_from_name(atom_name: str) -> str:
    """Guess element from atom name."""
    if not atom_name:
        return 'C'
    
    # Remove digits and take the first character(s)
    clean_name = ''.join(c for c in atom_name if c.isalpha())
    if not clean_name:
        return 'C'
    
    # Common element prefixes
    if clean_name.startswith(('CL', 'Cl')):
        return 'Cl'
    elif clean_name.startswith(('BR', 'Br')):
        return 'Br'
    elif clean_name[0] in 'HCNOFPS':
        return clean_name[0]
    else:
        return 'C'  # Default to carbon
# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""
CIF coordinates loading functionality for partial diffusion.

This module provides functions to load coordinates from CIF files that are
model predictions, where the atom order is guaranteed to match the internal
structure context order exactly.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import gemmi
import numpy as np

from chai_lab.data.dataset.structure.all_atom_structure_context import AllAtomStructureContext

logger = logging.getLogger(__name__)


def load_cif_coordinates(
    cif_path: Path,
    structure_context: AllAtomStructureContext,
    device: torch.device,
    center_coords: bool = True,
    add_noise: float = 0.0,
    ignore_chains: Optional[list[str]] = None,
    target_chain_id: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Load coordinates from a CIF file.
    
    Args:
        cif_path: Path to the CIF file
        structure_context: Structure context to match atom order
        device: Device to load coordinates on
        center_coords: Whether to center coordinates at origin
        add_noise: Amount of noise to add to coordinates
        ignore_chains: Chain IDs to ignore
        target_chain_id: Specific chain ID to load (if None, load all)
        
    Returns:
        Tensor of coordinates with shape (n_atoms, 3) or None if failed
    """
    logger.info(f"Loading CIF coordinates from {cif_path}")
    
    if not cif_path.exists():
        logger.error(f"CIF file not found: {cif_path}")
        return None
    
    try:
        # Load structure using gemmi
        structure = gemmi.read_structure(str(cif_path))
        
        if len(structure) == 0:
            logger.error("CIF file contains no models")
            return None
            
        model = structure[0]
        logger.info(f"Loaded CIF structure with {len(model)} chains")
        
        # Extract coordinates from all chains
        coordinates = []
        atom_info = []
        
        for chain in model:
            # Skip chains if specified
            if ignore_chains and chain.name in ignore_chains:
                logger.info(f"Skipping chain {chain.name} (in ignore list)")
                continue
                
            # Filter by target chain if specified
            if target_chain_id is not None and chain.name != target_chain_id:
                continue
                
            logger.info(f"Processing chain {chain.name}: {len(chain)} residues")
            
            for residue in chain:
                for atom in residue:
                    coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    atom_info.append((chain.name, residue.name, residue.seqid.num, atom.name))
        
        if len(coordinates) == 0:
            logger.error("No coordinates found in CIF file")
            return None
            
        # Convert to numpy array
        coords = np.array(coordinates, dtype=np.float32)
        logger.info(f"Extracted {len(coords)} coordinates from CIF")
        
        # Get expected number of atoms from structure context
        expected_atoms = structure_context.num_atoms
        logger.info(f"Expected {expected_atoms} atoms from structure context")
        
        if len(coords) != expected_atoms:
            logger.warning(f"CIF has {len(coords)} atoms, but structure context expects {expected_atoms}")
            # Try to match by truncating or padding
            if len(coords) > expected_atoms:
                logger.info(f"Truncating coordinates to match expected size")
                coords = coords[:expected_atoms]
            else:
                logger.warning(f"CIF has fewer atoms than expected, returning None")
                return None
        
        # Center coordinates if requested
        if center_coords:
            coords = coords - coords.mean(axis=0)
            logger.info("Centered coordinates at origin")
        
        # Add noise if requested
        if add_noise > 0:
            noise = np.random.normal(0, add_noise, coords.shape)
            coords = coords + noise
            logger.info(f"Added Gaussian noise with scale {add_noise}")
        
        # Convert to torch tensor
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        
        logger.info(f"Successfully loaded CIF coordinates: {coords_tensor.shape}")
        logger.info(f"Coordinate range: x=[{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}], "
                   f"y=[{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}], "
                   f"z=[{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
        
        return coords_tensor
        
    except Exception as e:
        logger.error(f"Error loading CIF coordinates: {e}")
        return None


def load_cif_coordinates_for_structure_context(
    cif_path: Path,
    structure_context: AllAtomStructureContext,
    device: torch.device,
    ignore_chains: Optional[list[str]] = None,
    add_noise: float = 0.1,
) -> Optional[torch.Tensor]:
    """
    Load CIF coordinates for structure context with PDB-compatible interface.
    
    This function provides a compatible interface with the PDB loading function
    for easy integration with existing code.
    
    Args:
        cif_path: Path to the CIF file
        structure_context: Structure context to match
        device: Device to load coordinates on
        ignore_chains: Chain IDs to ignore
        add_noise: Amount of noise to add
        
    Returns:
        Tensor of coordinates or None if failed
    """
    return load_cif_coordinates(
        cif_path=cif_path,
        structure_context=structure_context,
        device=device,
        center_coords=True,
        add_noise=add_noise,
        ignore_chains=ignore_chains,
    )

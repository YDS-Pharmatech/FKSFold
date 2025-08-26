from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional
import torch
from torch import Tensor
from einops import einsum

import chai_lab.ranking.ptm as ptm
from chai_lab.ranking.frames import get_frames_and_mask


class ScoringFunction(Protocol):
    """Protocol definition for custom scoring functions"""
    
    def calculate_score(self, 
                       particle_pos: torch.Tensor,
                       device: torch.device,
                       **kwargs) -> float:
        """
        Calculate score for a particle position
        
        Args:
            particle_pos: Particle position tensor
            device: Computing device
            **kwargs: Additional arguments
            
        Returns:
            Score as float
        """
        ...


class InterfacePTMScoringFunction:
    """Scoring function using Interface PTM"""
    
    def __init__(self, 
                 confidence_head,
                 inputs: Dict[str, Any],
                 token_single_initial_repr: Tensor,
                 token_single_trunk_repr: Tensor,
                 token_pair_trunk_repr: Tensor,
                 token_single_mask: Tensor,
                 atom_single_mask: Tensor,
                 token_reference_atom_index: Tensor,
                 atom_token_indices: Tensor,
                 atom_within_token_index: Tensor,
                 model_size: int):
        self.confidence_head = confidence_head
        self.inputs = inputs
        self.token_single_initial_repr = token_single_initial_repr
        self.token_single_trunk_repr = token_single_trunk_repr
        self.token_pair_trunk_repr = token_pair_trunk_repr
        self.token_single_mask = token_single_mask
        self.atom_single_mask = atom_single_mask
        self.token_reference_atom_index = token_reference_atom_index
        self.atom_token_indices = atom_token_indices
        self.atom_within_token_index = atom_within_token_index
        self.model_size = model_size
    
    def calculate_score(self, 
                       particle_pos: torch.Tensor,
                       device: torch.device,
                       **kwargs) -> float:
        """Calculate Interface PTM score"""
        try:
            # Run confidence model
            confidence_output = self.confidence_head.forward(
                move_to_device=device,
                token_single_input_repr=self.token_single_initial_repr,
                token_single_trunk_repr=self.token_single_trunk_repr,
                token_pair_trunk_repr=self.token_pair_trunk_repr,
                token_single_mask=self.token_single_mask,
                atom_single_mask=self.atom_single_mask,
                atom_coords=particle_pos,
                token_reference_atom_index=self.token_reference_atom_index,
                atom_token_index=self.atom_token_indices,
                atom_within_token_index=self.atom_within_token_index,
                crop_size=self.model_size,
            )
            
            # Get frames and masks
            _, valid_frames_mask = get_frames_and_mask(
                particle_pos,
                self.inputs["token_asym_id"].to(device),
                self.inputs["token_residue_index"].to(device),
                self.inputs["token_backbone_frame_mask"].to(device),
                self.inputs["token_centre_atom_index"].to(device),
                self.inputs["token_exists_mask"].to(device),
                self.atom_single_mask,
                self.inputs["token_backbone_frame_index"].to(device),
                self.atom_token_indices,
            )
            
            # Calculate bin centers
            bin_centers = self._bin_centers(0.0, 32.0, 64).to(device)
            
            # Calculate interface PTM
            interface_ptm_score = ptm.interface_ptm(
                pae_logits=confidence_output[0].float(),
                token_exists_mask=self.token_single_mask.to(device),
                valid_frames_mask=valid_frames_mask.to(device),
                bin_centers=bin_centers,
                token_asym_id=self.inputs["token_asym_id"].to(device),
            )
            
            return interface_ptm_score.mean().item()
            
        except Exception as e:
            print(f"Error calculating Interface PTM score: {e}")
            return 0.0
    
    def _bin_centers(self, min_bin: float, max_bin: float, no_bins: int) -> Tensor:
        """Helper function to create bin centers"""
        return torch.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]


class PLDDTScoringFunction:
    """Scoring function using pLDDT"""
    
    def __init__(self, 
                 confidence_head,
                 inputs: Dict[str, Any],
                 token_single_initial_repr: Tensor,
                 token_single_trunk_repr: Tensor,
                 token_pair_trunk_repr: Tensor,
                 token_single_mask: Tensor,
                 atom_single_mask: Tensor,
                 token_reference_atom_index: Tensor,
                 atom_token_indices: Tensor,
                 atom_within_token_index: Tensor,
                 model_size: int):
        self.confidence_head = confidence_head
        self.inputs = inputs
        self.token_single_initial_repr = token_single_initial_repr
        self.token_single_trunk_repr = token_single_trunk_repr
        self.token_pair_trunk_repr = token_pair_trunk_repr
        self.token_single_mask = token_single_mask
        self.atom_single_mask = atom_single_mask
        self.token_reference_atom_index = token_reference_atom_index
        self.atom_token_indices = atom_token_indices
        self.atom_within_token_index = atom_within_token_index
        self.model_size = model_size
    
    def calculate_score(self, 
                       particle_pos: torch.Tensor,
                       device: torch.device,
                       **kwargs) -> float:
        """Calculate pLDDT score"""
        try:
            # Run confidence model
            confidence_output = self.confidence_head.forward(
                move_to_device=device,
                token_single_input_repr=self.token_single_initial_repr,
                token_single_trunk_repr=self.token_single_trunk_repr,
                token_pair_trunk_repr=self.token_pair_trunk_repr,
                token_single_mask=self.token_single_mask,
                atom_single_mask=self.atom_single_mask,
                atom_coords=particle_pos,
                token_reference_atom_index=self.token_reference_atom_index,
                atom_token_index=self.atom_token_indices,
                atom_within_token_index=self.atom_within_token_index,
                crop_size=self.model_size,
            )
            
            # Calculate pLDDT
            bin_centers = self._bin_centers(0, 1, confidence_output[2].shape[-1]).to(device)
            plddt = einsum(
                confidence_output[2].float().softmax(dim=-1),
                bin_centers,
                "b a d, d -> b a"
            )
            
            return plddt.mean().item()
            
        except Exception as e:
            print(f"Error calculating pLDDT score: {e}")
            return 0.0
    
    def _bin_centers(self, min_bin: float, max_bin: float, no_bins: int) -> Tensor:
        """Helper function to create bin centers"""
        return torch.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]


class DefaultScoringFunction:
    """Default scoring function using avg_interface_ptm"""
    
    def calculate_score(self, 
                       particle_pos: torch.Tensor,
                       device: torch.device,
                       **kwargs) -> float:
        """Get particle's avg_interface_ptm score"""
        # This function expects pre-computed avg_interface_ptm
        # to be passed in via kwargs
        particle_state = kwargs.get('particle_state', None)
        if particle_state is not None and hasattr(particle_state, 'avg_interface_ptm'):
            if particle_state.avg_interface_ptm is not None:
                return particle_state.avg_interface_ptm
        
        # Return 0.0 if no avg_interface_ptm available
        return 0.0


class MeanInterfacePTMScoringFunction:
    """Use Mean Interface PTM as scoring function"""
    
    def __init__(self, 
                 confidence_head,
                 inputs: Dict[str, Any],
                 token_single_initial_repr: Tensor,
                 token_single_trunk_repr: Tensor,
                 token_pair_trunk_repr: Tensor,
                 token_single_mask: Tensor,
                 atom_single_mask: Tensor,
                 token_reference_atom_index: Tensor,
                 atom_token_indices: Tensor,
                 atom_within_token_index: Tensor,
                 model_size: int):
        self.confidence_head = confidence_head
        self.inputs = inputs
        self.token_single_initial_repr = token_single_initial_repr
        self.token_single_trunk_repr = token_single_trunk_repr
        self.token_pair_trunk_repr = token_pair_trunk_repr
        self.token_single_mask = token_single_mask
        self.atom_single_mask = atom_single_mask
        self.token_reference_atom_index = token_reference_atom_index
        self.atom_token_indices = atom_token_indices
        self.atom_within_token_index = atom_within_token_index
        self.model_size = model_size
    
    def calculate_score(self, 
                       particle_pos: torch.Tensor,
                       device: torch.device,
                       **kwargs) -> float:
        """Calculate Mean Interface PTM score"""
        try:
            # Run confidence model
            confidence_output = self.confidence_head.forward(
                move_to_device=device,
                token_single_input_repr=self.token_single_initial_repr,
                token_single_trunk_repr=self.token_single_trunk_repr,
                token_pair_trunk_repr=self.token_pair_trunk_repr,
                token_single_mask=self.token_single_mask,
                atom_single_mask=self.atom_single_mask,
                atom_coords=particle_pos,
                token_reference_atom_index=self.token_reference_atom_index,
                atom_token_index=self.atom_token_indices,
                atom_within_token_index=self.atom_within_token_index,
                crop_size=self.model_size,
            )
            
            # Get frames and masks
            _, valid_frames_mask = get_frames_and_mask(
                particle_pos,
                self.inputs["token_asym_id"].to(device),
                self.inputs["token_residue_index"].to(device),
                self.inputs["token_backbone_frame_mask"].to(device),
                self.inputs["token_centre_atom_index"].to(device),
                self.inputs["token_exists_mask"].to(device),
                self.atom_single_mask,
                self.inputs["token_backbone_frame_index"].to(device),
                self.atom_token_indices,
            )
            
            # Calculate bin centers
            bin_centers = self._bin_centers(0.0, 32.0, 64).to(device)
            
            # Calculate mean interface PTM
            mean_interface_ptm_score = ptm.mean_interface_ptm(
                pae_logits=confidence_output[0].float(),
                token_exists_mask=self.token_single_mask.to(device),
                valid_frames_mask=valid_frames_mask.to(device),
                bin_centers=bin_centers,
                token_asym_id=self.inputs["token_asym_id"].to(device),
            )
            
            return mean_interface_ptm_score.mean().item()
            
        except Exception as e:
            print(f"Error calculating Mean Interface PTM score: {e}")
            return 0.0
    
    def _bin_centers(self, min_bin: float, max_bin: float, no_bins: int) -> Tensor:
        """Helper function to create bin centers"""
        return torch.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]


class ProteinMeanInterfacePTMScoringFunction:
    """Use Protein Mean Interface PTM as scoring function"""
    def __init__(self, 
                 confidence_head,
                 inputs: Dict[str, Any],
                 token_single_initial_repr: Tensor,
                 token_single_trunk_repr: Tensor,
                 token_pair_trunk_repr: Tensor,
                 token_single_mask: Tensor,
                 atom_single_mask: Tensor,
                 token_reference_atom_index: Tensor,
                 atom_token_indices: Tensor,
                 atom_within_token_index: Tensor,
                 model_size: int):
        self.confidence_head = confidence_head
        self.inputs = inputs
        self.token_single_initial_repr = token_single_initial_repr
        self.token_single_trunk_repr = token_single_trunk_repr
        self.token_pair_trunk_repr = token_pair_trunk_repr
        self.token_single_mask = token_single_mask
        self.atom_single_mask = atom_single_mask
        self.token_reference_atom_index = token_reference_atom_index
        self.atom_token_indices = atom_token_indices
        self.atom_within_token_index = atom_within_token_index
        self.model_size = model_size
    
    def calculate_score(self, 
                       particle_pos: torch.Tensor,
                       device: torch.device,
                       **kwargs) -> float:
        """Calculate Protein Mean Interface PTM score"""
        try:
            # Run confidence model
            confidence_output = self.confidence_head.forward(
                move_to_device=device,
                token_single_input_repr=self.token_single_initial_repr,
                token_single_trunk_repr=self.token_single_trunk_repr,
                token_pair_trunk_repr=self.token_pair_trunk_repr,
                token_single_mask=self.token_single_mask,
                atom_single_mask=self.atom_single_mask,
                atom_coords=particle_pos,
                token_reference_atom_index=self.token_reference_atom_index,
                atom_token_index=self.atom_token_indices,
                atom_within_token_index=self.atom_within_token_index,
                crop_size=self.model_size,
            )
            
            # Get frames and masks
            _, valid_frames_mask = get_frames_and_mask(
                particle_pos,
                self.inputs["token_asym_id"].to(device),
                self.inputs["token_residue_index"].to(device),
                self.inputs["token_backbone_frame_mask"].to(device),
                self.inputs["token_centre_atom_index"].to(device),
                self.inputs["token_exists_mask"].to(device),
                self.atom_single_mask,
                self.inputs["token_backbone_frame_index"].to(device),
                self.atom_token_indices,
            )
            
            # Calculate bin centers
            bin_centers = self._bin_centers(0.0, 32.0, 64).to(device)
            
            # Get token_entity_type if exists
            token_entity_type = self.inputs.get("token_entity_type", None)
            if token_entity_type is not None:
                token_entity_type = token_entity_type.to(device)
            
            # Calculate protein mean interface PTM
            protein_mean_interface_ptm_score = ptm.protein_mean_interface_ptm(
                pae_logits=confidence_output[0].float(),
                token_exists_mask=self.token_single_mask.to(device),
                valid_frames_mask=valid_frames_mask.to(device),
                bin_centers=bin_centers,
                token_asym_id=self.inputs["token_asym_id"].to(device),
                token_entity_type=token_entity_type,
            )
            
            # If result is None, return 0.0
            if protein_mean_interface_ptm_score is None:
                return 0.0
                
            return protein_mean_interface_ptm_score.mean().item()
            
        except Exception as e:
            print(f"Error calculating Protein Mean Interface PTM score: {e}")
            return 0.0
    
    def _bin_centers(self, min_bin: float, max_bin: float, no_bins: int) -> Tensor:
        """Helper function to create bin centers"""
        return torch.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]

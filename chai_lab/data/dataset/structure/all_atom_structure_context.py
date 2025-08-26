# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import logging
from dataclasses import asdict, dataclass
from functools import cached_property, partial
from typing import Any

import torch
from torch import Tensor

from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.utils.tensor_utils import (
    batch_tensorcode_to_string,
    cdist,
    tensorcode_to_string,
)
from chai_lab.utils.typing import Bool, Float, Int, UInt8, typecheck

logger = logging.getLogger(__name__)


@typecheck
@dataclass
class AllAtomStructureContext:
    # token-level
    token_residue_type: Int[Tensor, "n_tokens"]
    token_residue_index: Int[Tensor, "n_tokens"]
    token_index: Int[Tensor, "n_tokens"]
    token_centre_atom_index: Int[Tensor, "n_tokens"]
    token_ref_atom_index: Int[Tensor, "n_tokens"]
    token_exists_mask: Bool[Tensor, "n_tokens"]
    token_backbone_frame_mask: Bool[Tensor, "n_tokens"]
    token_backbone_frame_index: Int[Tensor, "n_tokens 3"]
    token_asym_id: Int[Tensor, "n_tokens"]
    token_entity_id: Int[Tensor, "n_tokens"]
    token_sym_id: Int[Tensor, "n_tokens"]
    token_entity_type: Int[Tensor, "n_tokens"]
    token_residue_name: UInt8[Tensor, "n_tokens 8"]
    token_b_factor_or_plddt: Float[Tensor, "n_tokens"]
    # atom-level
    atom_token_index: Int[Tensor, "n_atoms"]
    atom_within_token_index: Int[Tensor, "n_atoms"]  # consistent atom ordering
    atom_ref_pos: Float[Tensor, "n_atoms 3"]
    atom_ref_mask: Bool[Tensor, "n_atoms"]
    atom_ref_element: Int[Tensor, "n_atoms"]
    atom_ref_charge: Int[Tensor, "n_atoms"]
    atom_ref_name: list[str]
    atom_ref_name_chars: Int[Tensor, "n_atoms 4"]
    atom_ref_space_uid: Int[Tensor, "n_atoms"]
    atom_is_not_padding_mask: Bool[Tensor, "n_atoms"]
    # supervision only
    atom_gt_coords: Float[Tensor, "n_atoms 3"]
    atom_exists_mask: Bool[Tensor, "n_atoms"]
    # structure-level
    pdb_id: UInt8[Tensor, "n_tokens 32"]
    # source_pdb_chain_id corresponds to auth_asym_id in pdb
    # can be the same for two different asym_id values
    # (we split protein and ligand for example)
    source_pdb_chain_id: UInt8[Tensor, "n_tokens 4"]
    # subchain_id is label_asym_id in pdb
    # it is assigned by the PDB and separates different
    # chemical entities (protein, ligand)
    # should be a 1-1 mapping to asym_id
    subchain_id: UInt8[Tensor, "n_tokens 4"]
    resolution: Float[Tensor, "1"]
    is_distillation: Bool[Tensor, "1"]
    # symmetric atom swap indices
    symmetries: Int[Tensor, "n_atoms n_symmetries"]
    # atom-wise bond feature; corresponding lists of atoms that are covalently bound
    atom_covalent_bond_indices: tuple[Int[Tensor, "n_bonds"], Int[Tensor, "n_bonds"]]
    # PDB initialization fields (with default values, must come last)
    atom_init_coords: Float[Tensor, "n_atoms 3"] | None = None  # PDB initialization coordinates
    use_pdb_init: bool = False  # Whether to use PDB initialization
    pdb_init_noise_scale: float = 0.1  # Noise scale for PDB initialization
    pdb_source_path: str | None = None  # PDB source file path

    def __post_init__(self):
        # Resolved residues filter should eliminate PDBs with missing residues, but that
        # we can still have atom_exists mask set to False at every position if we have a
        # bad crop so we log examples with no valid coordinates
        if self.num_atoms > 0 and not torch.any(self.atom_exists_mask):
            pdb_id = tensorcode_to_string(self.pdb_id[0])
            logger.error(f"No valid coordinates found in any atoms for {pdb_id}")

        # Check that atom and token masks are compatible. Anywhere that the atom mask is
        # true, the token mask should also be true
        if self.num_atoms > 0 and not torch.all(
            self.token_exists_mask[self.atom_token_index][self.atom_exists_mask]
        ):
            pdb_id = tensorcode_to_string(self.pdb_id[0])
            logger.error(f"Incompatible masks for {pdb_id}")

        # Check that bonds are specified in atom space
        assert torch.all(self.atom_covalent_bond_indices[0] < self.num_atoms)
        assert torch.all(self.atom_covalent_bond_indices[1] < self.num_atoms)
        
        # Debug prints for atom_gt_coords and atom_ref_pos consistency
        print(f"[DEBUG] AllAtomStructureContext shape check:")
        print(f"  atom_gt_coords shape: {self.atom_gt_coords.shape}")
        print(f"  atom_ref_pos shape: {self.atom_ref_pos.shape}")
        print(f"  atom_exists_mask shape: {self.atom_exists_mask.shape}")
        print(f"  atom_ref_mask shape: {self.atom_ref_mask.shape}")
        
        # Check if shapes are consistent
        assert self.atom_gt_coords.shape == self.atom_ref_pos.shape, \
            f"Shape mismatch: atom_gt_coords {self.atom_gt_coords.shape} vs atom_ref_pos {self.atom_ref_pos.shape}"
        
        # Check numerical consistency for existing atoms
        if self.num_atoms > 0:
            existing_mask = self.atom_exists_mask & self.atom_ref_mask
            if existing_mask.any():
                gt_coords_existing = self.atom_gt_coords[existing_mask]
                ref_pos_existing = self.atom_ref_pos[existing_mask]
                coord_diff = torch.norm(gt_coords_existing - ref_pos_existing, dim=-1)
                max_diff = coord_diff.max().item()
                mean_diff = coord_diff.mean().item()
                print(f"  Coordinate differences for existing atoms:")
                print(f"    Max difference: {max_diff:.6f}")
                print(f"    Mean difference: {mean_diff:.6f}")
                print(f"    Num existing atoms: {existing_mask.sum().item()}/{self.num_atoms}")
                
                # Check if coordinates are identical (within tolerance)
                if max_diff < 1e-5:
                    print(f"  ✓ atom_gt_coords and atom_ref_pos are nearly identical")
                elif max_diff < 1.0:
                    print(f"  ⚠ atom_gt_coords and atom_ref_pos have small differences")
                else:
                    print(f"  ⚠ atom_gt_coords and atom_ref_pos have significant differences")
        
        # PDB initialization debug info
        if self.use_pdb_init and self.atom_init_coords is not None:
            print(f"[DEBUG] PDB initialization:")
            print(f"  atom_init_coords shape: {self.atom_init_coords.shape}")
            print(f"  PDB source: {self.pdb_source_path}")
            print(f"  Noise scale: {self.pdb_init_noise_scale}")
            
            # Check consistency with other coordinate fields
            init_coords_cpu = self.atom_init_coords.cpu()
            gt_coords_cpu = self.atom_gt_coords.cpu()
            ref_pos_cpu = self.atom_ref_pos.cpu()
            
            init_vs_gt_diff = torch.norm(init_coords_cpu - gt_coords_cpu, dim=-1)
            init_vs_ref_diff = torch.norm(init_coords_cpu - ref_pos_cpu, dim=-1)
            print(f"  Init vs GT max diff: {init_vs_gt_diff.max().item():.6f}")
            print(f"  Init vs Ref max diff: {init_vs_ref_diff.max().item():.6f}")
    
    def get_initial_coords_for_diffusion(
        self, 
        sigma: float, 
        device: torch.device,
        add_noise: bool = True,
        particle_idx: int = 0,
        target_num_atoms: int | None = None
    ) -> Float[Tensor, "1 n_atoms 3"]:
        """Get initialization coordinates for diffusion, can be PDB coords or random noise"""
        if self.use_pdb_init and self.atom_init_coords is not None:
            init_coords = self.atom_init_coords.to(device).unsqueeze(0)  # Add batch dimension
            if add_noise and self.pdb_init_noise_scale > 0:
                # Add different noise for different particles
                noise_scale = self.pdb_init_noise_scale * (1.0 + 0.3 * particle_idx)
                noise = noise_scale * torch.randn_like(init_coords)
                init_coords = init_coords + noise
                print(f"[DEBUG] PDB init for particle {particle_idx}: noise_scale={noise_scale:.4f}")
            else:
                print(f"[DEBUG] PDB init for particle {particle_idx}: no noise")
            
            # Handle padding if target_num_atoms is specified
            if target_num_atoms is not None and target_num_atoms != init_coords.shape[1]:
                print(f"[DEBUG] Padding PDB coords from {init_coords.shape[1]} to {target_num_atoms} atoms")
                batch_size = init_coords.shape[0]
                padded_coords = torch.zeros(batch_size, target_num_atoms, 3, device=device)
                
                # Copy existing coordinates to the beginning
                current_num_atoms = init_coords.shape[1]
                if current_num_atoms <= target_num_atoms:
                    padded_coords[:, :current_num_atoms] = init_coords
                    # Fill the rest with random noise
                    padded_coords[:, current_num_atoms:] = sigma * torch.randn(
                        batch_size, target_num_atoms - current_num_atoms, 3, device=device
                    )
                else:
                    # Truncate if current is larger (shouldn't happen normally)
                    padded_coords = init_coords[:, :target_num_atoms]
                    print(f"[DEBUG] Warning: Truncated coordinates from {current_num_atoms} to {target_num_atoms}")
                
                init_coords = padded_coords
                print(f"[DEBUG] Final padded coords shape: {init_coords.shape}")
            
            return init_coords
        else:
            # Use random noise initialization
            num_atoms = target_num_atoms if target_num_atoms is not None else self.num_atoms
            random_coords = sigma * torch.randn(1, num_atoms, 3, device=device)
            print(f"[DEBUG] Random init for particle {particle_idx}: sigma={sigma:.4f}, shape={random_coords.shape}")
            return random_coords
    
    def set_pdb_initialization(
        self, 
        pdb_coords: Float[Tensor, "n_atoms 3"],
        noise_scale: float = 0.1,
        source_path: str | None = None
    ):
        """Set PDB initialization coordinates"""
        assert pdb_coords.shape == self.atom_gt_coords.shape, \
            f"PDB coords shape {pdb_coords.shape} doesn't match atom_gt_coords {self.atom_gt_coords.shape}"
        
        # 确保坐标保存在CPU上，与其他字段保持一致
        self.atom_init_coords = pdb_coords.cpu().clone()
        self.use_pdb_init = True  
        self.pdb_init_noise_scale = noise_scale
        self.pdb_source_path = source_path
        
        print(f"[DEBUG] Set PDB initialization:")
        print(f"  Source: {source_path}")
        print(f"  Shape: {pdb_coords.shape}")
        print(f"  Noise scale: {noise_scale}")
        print(f"  PDB coords device: {pdb_coords.device}")
        print(f"  Stored init coords device: {self.atom_init_coords.device}")
        
        # Check consistency with existing coordinates
        if self.atom_exists_mask.any():
            existing_mask = self.atom_exists_mask
            # All tensors should be on CPU now
            pdb_coords_cpu = self.atom_init_coords  # Already on CPU
            gt_coords_cpu = self.atom_gt_coords.cpu()
            ref_pos_cpu = self.atom_ref_pos.cpu()
            
            pdb_vs_gt_diff = torch.norm(pdb_coords_cpu[existing_mask] - gt_coords_cpu[existing_mask], dim=-1)
            pdb_vs_ref_diff = torch.norm(pdb_coords_cpu[existing_mask] - ref_pos_cpu[existing_mask], dim=-1)
            print(f"  PDB vs GT max diff: {pdb_vs_gt_diff.max().item():.6f}")
            print(f"  PDB vs Ref max diff: {pdb_vs_ref_diff.max().item():.6f}")

    @cached_property
    def residue_names(self) -> list[str]:
        return batch_tensorcode_to_string(self.token_residue_name)

    @typecheck
    def index_select(self, idxs: Int[Tensor, "n"]) -> "AllAtomStructureContext":
        """
        Selects a subset of the data in the context, reindexing the tokens and atoms in
        the new context (i.e. the new context will be indexed from 0).

        Parameters
        ----------
        idxs : Int[Tensor, "n"]
            The indices of the tokens to select.

        Returns
        -------
        AllAtomStructureContext
            A new context with the selected tokens and atoms.
        """
        assert ((idxs >= 0) & (idxs < self.num_tokens)).all()

        # get atoms to keep
        selected_atom_index = torch.where(
            (self.atom_token_index == idxs[..., None]).any(dim=0)
        )[0]

        # rebuild token index and atom-token index
        token_index = torch.arange(len(idxs))

        atom_token_index, selected_atom_idx = torch.where(
            self.atom_token_index == idxs[..., None]
        )

        def _reselect_atom_indices(
            prior_atom_index: Int[Tensor, "n_tokens"],
        ) -> Int[Tensor, "n_tokens_new"]:
            mask = torch.zeros(self.num_atoms, dtype=torch.bool)
            mask[prior_atom_index] = True
            selected_mask = mask[selected_atom_idx]
            return torch.where(selected_mask)[0]

        token_centre_atom_index = _reselect_atom_indices(self.token_centre_atom_index)
        token_ref_atom_index = _reselect_atom_indices(self.token_ref_atom_index)

        atom_covalent_bond_indices = None
        if self.atom_covalent_bond_indices is not None:
            left_idx, right_idx = self.atom_covalent_bond_indices
            atom_pairs = torch.zeros(self.num_atoms, self.num_atoms, dtype=torch.bool)
            atom_pairs[left_idx, right_idx] = True
            selected_atom_pairs = atom_pairs[selected_atom_idx][:, selected_atom_idx]
            new_left, new_right = torch.where(selected_atom_pairs)
            atom_covalent_bond_indices = new_left, new_right

        token_backbone_frame_atom_index = torch.stack(
            [
                _reselect_atom_indices(x)
                for x in torch.unbind(self.token_backbone_frame_index, dim=-1)
            ],
            dim=-1,
        )

        return AllAtomStructureContext(
            # token-level
            token_residue_type=self.token_residue_type[idxs],
            token_residue_index=self.token_residue_index[idxs],
            token_index=token_index,
            token_centre_atom_index=token_centre_atom_index,
            token_ref_atom_index=token_ref_atom_index,
            token_exists_mask=self.token_exists_mask[idxs],
            token_backbone_frame_mask=self.token_backbone_frame_mask[idxs],
            token_backbone_frame_index=token_backbone_frame_atom_index,
            token_asym_id=self.token_asym_id[idxs],
            token_entity_id=self.token_entity_id[idxs],
            token_sym_id=self.token_sym_id[idxs],
            token_entity_type=self.token_entity_type[idxs],
            token_residue_name=self.token_residue_name[idxs],
            token_b_factor_or_plddt=self.token_b_factor_or_plddt[idxs],
            # atom-level
            atom_token_index=atom_token_index,
            atom_within_token_index=self.atom_within_token_index[selected_atom_index],
            atom_ref_pos=self.atom_ref_pos[selected_atom_index],
            atom_ref_mask=self.atom_ref_mask[selected_atom_index],
            atom_ref_element=self.atom_ref_element[selected_atom_index],
            atom_ref_charge=self.atom_ref_charge[selected_atom_index],
            atom_ref_name=[self.atom_ref_name[i] for i in selected_atom_index],
            atom_ref_name_chars=self.atom_ref_name_chars[selected_atom_index],
            atom_ref_space_uid=self.atom_ref_space_uid[selected_atom_index],
            atom_is_not_padding_mask=self.atom_is_not_padding_mask[selected_atom_index],
            # supervision-only
            atom_gt_coords=self.atom_gt_coords[selected_atom_index],
            atom_exists_mask=self.atom_exists_mask[selected_atom_index],
            # PDB initialization fields
            atom_init_coords=self.atom_init_coords[selected_atom_index] if self.atom_init_coords is not None else None,
            use_pdb_init=self.use_pdb_init,
            pdb_init_noise_scale=self.pdb_init_noise_scale,
            pdb_source_path=self.pdb_source_path,
            # structure-level
            pdb_id=self.pdb_id[idxs],
            source_pdb_chain_id=self.source_pdb_chain_id[idxs],
            subchain_id=self.subchain_id[idxs],
            resolution=self.resolution,
            is_distillation=self.is_distillation,
            symmetries=self.symmetries[selected_atom_index],
            atom_covalent_bond_indices=atom_covalent_bond_indices,
        )

    def report_bonds(self) -> None:
        """Log information about covalent bonds."""
        for i, (atom_a, atom_b) in enumerate(zip(*self.atom_covalent_bond_indices)):
            tok_a = self.atom_token_index[atom_a]
            tok_b = self.atom_token_index[atom_b]
            asym_a = self.token_asym_id[tok_a]
            asym_b = self.token_asym_id[tok_b]
            res_idx_a = self.token_residue_index[tok_a]
            res_idx_b = self.token_residue_index[tok_b]
            resname_a = tensorcode_to_string(self.token_residue_name[tok_a])
            resname_b = tensorcode_to_string(self.token_residue_name[tok_b])
            logger.info(
                f"Bond {i} (asym res_idx resname): {asym_a} {res_idx_a} {resname_a} <> {asym_b} {res_idx_b} {resname_b}"
            )

    @typecheck
    def _infer_CO_bonds_within_glycan(
        self,
        atom_idx: int,
        allowed_elements: list[int] | None = None,
    ) -> Bool[Tensor, "{self.num_atoms}"]:
        """Return mask for atoms that atom_idx might bond to based on distances.

        If exclude_polymers is True, then always return no bonds for polymer entities
        """
        tok = self.atom_token_index[atom_idx]
        res = self.token_residue_index[tok]
        asym = self.token_asym_id[tok]

        if self.token_entity_type[tok].item() != EntityType.MANUAL_GLYCAN.value:
            return torch.zeros(self.num_atoms, dtype=torch.bool)

        mask = (
            (self.atom_residue_index == res)
            & (self.atom_asym_id == asym)
            & self.atom_exists_mask
        )

        # This field contains reference conformers for each residue
        # Pairwise distances are therefore valid within each residue
        distances = cdist(self.atom_gt_coords)
        assert distances.shape == (self.num_atoms, self.num_atoms)
        distances[torch.arange(self.num_atoms), torch.arange(self.num_atoms)] = (
            torch.inf
        )

        is_allowed_element = (
            torch.isin(
                self.atom_ref_element, test_elements=torch.tensor(allowed_elements)
            )
            if allowed_elements is not None
            else torch.ones_like(mask)
        )
        # Canonical bond length for C-O is 1.43 angstroms; add a bit of headroom
        bond_candidates = (distances[atom_idx] < 1.5) & mask & is_allowed_element
        return bond_candidates

    def drop_glycan_leaving_atoms_inplace(self) -> None:
        """Drop OH groups that leave upon bond formation by setting atom_exists_mask."""
        # For each of the bonds, identify the atoms within bond radius and guess which are leaving
        oxygen = 8
        for i, (atom_a, atom_b) in enumerate(zip(*self.atom_covalent_bond_indices)):
            # Find the C-O bonds
            [bond_candidates_b] = torch.where(
                self._infer_CO_bonds_within_glycan(
                    atom_b.item(), allowed_elements=[oxygen]
                )
            )
            # Filter to bonds that link to terminal atoms
            # NOTE do not specify element here
            bonds_b = [
                candidate
                for candidate in bond_candidates_b.tolist()
                if (self._infer_CO_bonds_within_glycan(candidate).sum() == 1)
            ]
            # If there are multiple such bonds, we can't infer which to drop
            if len(bonds_b) == 1:
                [b_bond] = bonds_b
                self.atom_exists_mask[b_bond] = False
                logger.info(
                    f"Bond {i} right: Dropping latter atom in bond {self.atom_residue_index[atom_b]} {self.atom_ref_name[atom_b]} -> {self.atom_residue_index[b_bond]} {self.atom_ref_name[b_bond]}"
                )
                continue  # Only identify one leaving atom per bond

            # Repeat the above for atom_a if we didn't find anything for atom B
            [bond_candidates_a] = torch.where(
                self._infer_CO_bonds_within_glycan(
                    atom_a.item(), allowed_elements=[oxygen]
                )
            )
            # Filter to bonds that link to terminal atoms
            bonds_a = [
                candidate
                for candidate in bond_candidates_a.tolist()
                if (self._infer_CO_bonds_within_glycan(candidate).sum() == 1)
            ]
            # If there are multiple such bonds, we can't infer which to drop
            if len(bonds_a) == 1:
                [a_bond] = bonds_a
                self.atom_exists_mask[a_bond] = False
                logger.info(
                    f"Bond {i} left: Dropping latter atom in bond {self.atom_residue_index[atom_a]} {self.atom_ref_element[atom_a]} -> {self.atom_residue_index[a_bond]} {self.atom_ref_element[a_bond]}"
                )

    def pad(
        self,
        n_tokens: int,
        n_atoms: int,
    ) -> "AllAtomStructureContext":
        assert n_tokens >= self.num_tokens
        pad_tokens_func = partial(_pad_func, pad_size=n_tokens - self.num_tokens)

        assert n_atoms >= self.num_atoms
        pad_atoms_func = partial(_pad_func, pad_size=n_atoms - self.num_atoms)

        return AllAtomStructureContext(
            # token-level
            token_residue_type=pad_tokens_func(self.token_residue_type),
            token_residue_index=pad_tokens_func(self.token_residue_index),
            token_index=pad_tokens_func(self.token_index),
            token_centre_atom_index=pad_tokens_func(self.token_centre_atom_index),
            token_ref_atom_index=pad_tokens_func(self.token_ref_atom_index),
            token_exists_mask=pad_tokens_func(self.token_exists_mask),
            token_backbone_frame_mask=pad_tokens_func(self.token_backbone_frame_mask),
            token_backbone_frame_index=torch.cat(
                [
                    pad_tokens_func(self.token_backbone_frame_index[..., i]).unsqueeze(
                        -1
                    )
                    for i in range(3)
                ],
                dim=-1,
            ),
            token_asym_id=pad_tokens_func(self.token_asym_id),
            token_entity_id=pad_tokens_func(self.token_entity_id),
            token_sym_id=pad_tokens_func(self.token_sym_id),
            token_entity_type=pad_tokens_func(self.token_entity_type),
            token_residue_name=pad_tokens_func(self.token_residue_name),
            token_b_factor_or_plddt=pad_tokens_func(self.token_b_factor_or_plddt),
            # atom-level
            atom_token_index=pad_atoms_func(self.atom_token_index),
            atom_within_token_index=pad_atoms_func(self.atom_within_token_index),
            atom_ref_pos=pad_atoms_func(self.atom_ref_pos),
            atom_ref_mask=pad_atoms_func(self.atom_ref_mask),
            atom_ref_element=pad_atoms_func(self.atom_ref_element),
            atom_ref_charge=pad_atoms_func(self.atom_ref_charge),
            atom_ref_name=self.atom_ref_name,
            atom_ref_name_chars=pad_atoms_func(self.atom_ref_name_chars),
            atom_ref_space_uid=pad_atoms_func(self.atom_ref_space_uid, pad_value=-1),
            atom_is_not_padding_mask=pad_atoms_func(self.atom_is_not_padding_mask),
            # supervision-only
            atom_gt_coords=pad_atoms_func(self.atom_gt_coords),
            atom_exists_mask=pad_atoms_func(self.atom_exists_mask),
            # PDB initialization fields
            atom_init_coords=pad_atoms_func(self.atom_init_coords) if self.atom_init_coords is not None else None,
            use_pdb_init=self.use_pdb_init,
            pdb_init_noise_scale=self.pdb_init_noise_scale,
            pdb_source_path=self.pdb_source_path,
            # structure-level
            pdb_id=pad_tokens_func(self.pdb_id),
            source_pdb_chain_id=pad_tokens_func(self.source_pdb_chain_id),
            subchain_id=pad_tokens_func(self.subchain_id),
            resolution=self.resolution,
            is_distillation=self.is_distillation,
            symmetries=pad_atoms_func(self.symmetries, pad_value=-1),
            atom_covalent_bond_indices=self.atom_covalent_bond_indices,
        )

    @typecheck
    @classmethod
    def merge(
        cls,
        contexts: list["AllAtomStructureContext"],
    ) -> "AllAtomStructureContext":
        # indexes:
        token_offsets = _exclusive_cum_lengths([x.token_residue_type for x in contexts])
        atom_offsets = _exclusive_cum_lengths([x.atom_token_index for x in contexts])

        atom_token_index = torch.cat(
            [x.atom_token_index + count for x, count in zip(contexts, token_offsets)]
        )

        token_centre_atom_index = torch.cat(
            [
                x.token_centre_atom_index + count
                for x, count in zip(contexts, atom_offsets)
            ]
        )
        token_ref_atom_index = torch.cat(
            [x.token_ref_atom_index + count for x, count in zip(contexts, atom_offsets)]
        )
        token_backbone_frame_index = torch.cat(
            [
                x.token_backbone_frame_index + count
                for x, count in zip(contexts, token_offsets)
            ]
        )

        n_tokens = sum(x.num_tokens for x in contexts)
        token_index = torch.arange(n_tokens, dtype=torch.int)

        # Merge and offset bond indices, which are indexed by *token*
        atom_covalent_bond_indices_manual_a = []
        atom_covalent_bond_indices_manual_b = []
        for ctx, count in zip(contexts, atom_offsets):
            if ctx.atom_covalent_bond_indices is None:
                continue
            a, b = ctx.atom_covalent_bond_indices
            atom_covalent_bond_indices_manual_a.append(a + count)
            atom_covalent_bond_indices_manual_b.append(b + count)
        assert len(atom_covalent_bond_indices_manual_a) == len(
            atom_covalent_bond_indices_manual_b
        )
        atom_covalent_bond_indices = (
            (
                torch.concatenate(atom_covalent_bond_indices_manual_a),
                torch.concatenate(atom_covalent_bond_indices_manual_b),
            )
            if atom_covalent_bond_indices_manual_a
            else (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )
        )

        # re-index the reference space from 0..n_tokens-1.
        zero_indexed_ref_uids = [
            torch.unique_consecutive(x.atom_ref_space_uid, return_inverse=True)[1]
            for x in contexts
        ]

        ref_space_uids_offsets = _exclusive_cum_lengths(
            [x.atom_ref_space_uid for x in contexts]
        )
        atom_ref_space_uid = torch.cat(
            [
                x + count
                for x, count in zip(zero_indexed_ref_uids, ref_space_uids_offsets)
            ],
        )

        # pad symmetric permutations to have same length
        max_symms = max(x.symmetries.shape[-1] for x in contexts)
        padded_symms = [
            torch.nn.functional.pad(
                x.symmetries, (0, max_symms - x.symmetries.shape[-1]), value=-1
            )
            for x in contexts
        ]
        # offset symmetries by number of atoms in each chain
        symm_mask = torch.cat([x >= 0 for x in padded_symms])
        symmetries = torch.cat(padded_symms)
        symmetries = symmetries.masked_fill(~symm_mask, -1)

        return cls(
            # token-level
            token_residue_type=torch.cat([x.token_residue_type for x in contexts]),
            token_residue_index=torch.cat([x.token_residue_index for x in contexts]),
            token_index=token_index,
            token_centre_atom_index=token_centre_atom_index,
            token_ref_atom_index=token_ref_atom_index,
            token_exists_mask=torch.cat([x.token_exists_mask for x in contexts]),
            token_backbone_frame_mask=torch.cat(
                [x.token_backbone_frame_mask for x in contexts]
            ),
            token_backbone_frame_index=token_backbone_frame_index,
            token_asym_id=torch.cat([x.token_asym_id for x in contexts]),
            token_entity_id=torch.cat([x.token_entity_id for x in contexts]),
            token_sym_id=torch.cat([x.token_sym_id for x in contexts]),
            token_entity_type=torch.cat([x.token_entity_type for x in contexts]),
            token_residue_name=torch.cat([x.token_residue_name for x in contexts]),
            token_b_factor_or_plddt=torch.cat(
                [x.token_b_factor_or_plddt for x in contexts]
            ),
            # atom-level
            atom_token_index=atom_token_index,
            atom_within_token_index=torch.cat(
                [x.atom_within_token_index for x in contexts]
            ),
            atom_ref_pos=torch.cat([x.atom_ref_pos for x in contexts]),
            atom_ref_mask=torch.cat([x.atom_ref_mask for x in contexts]),
            atom_ref_element=torch.cat([x.atom_ref_element for x in contexts]),
            atom_ref_charge=torch.cat([x.atom_ref_charge for x in contexts]),
            atom_ref_name=[x for context in contexts for x in context.atom_ref_name],
            atom_ref_name_chars=torch.cat([x.atom_ref_name_chars for x in contexts]),
            atom_ref_space_uid=atom_ref_space_uid,
            atom_is_not_padding_mask=torch.cat(
                [x.atom_is_not_padding_mask for x in contexts]
            ),
            # supervision only
            atom_gt_coords=torch.cat([x.atom_gt_coords for x in contexts]),
            atom_exists_mask=torch.cat([x.atom_exists_mask for x in contexts]),
            # PDB initialization fields - merge from first context
            atom_init_coords=torch.cat([x.atom_init_coords for x in contexts]) if contexts[0].atom_init_coords is not None else None,
            use_pdb_init=any(x.use_pdb_init for x in contexts),
            pdb_init_noise_scale=contexts[0].pdb_init_noise_scale,  # Use first context's value
            pdb_source_path=contexts[0].pdb_source_path,  # Use first context's value
            # structure-level
            pdb_id=torch.cat([x.pdb_id for x in contexts]),
            source_pdb_chain_id=torch.cat([x.source_pdb_chain_id for x in contexts]),
            subchain_id=torch.cat([x.subchain_id for x in contexts]),
            resolution=torch.max(
                torch.stack([x.resolution for x in contexts]), 0
            ).values,
            is_distillation=torch.max(
                torch.stack([x.is_distillation for x in contexts]), 0
            ).values,
            symmetries=symmetries,
            atom_covalent_bond_indices=atom_covalent_bond_indices,
        )

    def to(self, device: torch.device | str) -> "AllAtomStructureContext":
        dict_: dict[str, Any] = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in asdict(self).items()
        }
        return AllAtomStructureContext(**dict_)

    @property
    def num_tokens(self) -> int:
        (n_tokens,) = self.token_index.shape
        return n_tokens

    @property
    def num_atoms(self) -> int:
        (n_atoms,) = self.atom_token_index.shape
        return n_atoms

    @property
    def atom_residue_index(self) -> Int[Tensor, "n_atoms"]:
        return self.token_residue_index[self.atom_token_index]

    @property
    def atom_asym_id(self) -> Int[Tensor, "n_atoms"]:
        return self.token_asym_id[self.atom_token_index]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)


def _pad_func(x: Tensor, pad_size: int, pad_value: float | None = None) -> Tensor:
    sizes = [0, 0] * (x.ndim - 1) + [0, pad_size]
    return torch.nn.functional.pad(x, sizes, value=pad_value)


def _exclusive_cum_lengths(tensors: list[Int[Tensor, "n"]]):
    lengths = torch.tensor([t.shape[0] for t in tensors])
    cum_lengths = torch.cumsum(lengths, dim=0).roll(1, 0)
    cum_lengths[0] = 0
    return cum_lengths

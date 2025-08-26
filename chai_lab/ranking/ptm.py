# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor

from chai_lab.ranking.utils import expectation, get_chain_masks_and_asyms
from chai_lab.utils.tensor_utils import und
from chai_lab.utils.typing import Bool, Float, Int, typecheck
from chai_lab.data.parsing.structure.entity_type import EntityType

from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG, DEV


@typecheck
@dataclass
class PTMScores:
    """
    complex_ptm: pTM score of the complex
    interface_ptm: ipTM score of the complex
    actif_ptm: improved interface pTM score using contact probabilities http://arxiv.org/abs/2412.15970v1
    mean_interface_ptm: mean of all inter-chain pTM scores
    protein_mean_interface_ptm: mean of symmetrized inter-chain pTM scores
    per_chain_ptm: pTM score for each chain in the complex
    per_chain_pair_iptm: ipTM score for each chain pair in the complex
    """

    complex_ptm: Float[Tensor, "..."]
    interface_ptm: Float[Tensor, "..."]
    actif_ptm: Float[Tensor, "..."]
    mean_interface_ptm: Float[Tensor, "..."]
    protein_mean_interface_ptm: Float[Tensor, "..."]
    per_chain_ptm: Float[Tensor, "... c"]
    per_chain_pair_iptm: Float[Tensor, "... c c"]


@typecheck
def tm_d0(n_tokens: Float[Tensor, "*dims"]) -> Float[Tensor, "*dims"]:
    """Compute TM-Score d0 from the number of tokens"""
    n_tokens = torch.clamp_min(n_tokens, 19)
    return 1.24 * (n_tokens - 15) ** (1.0 / 3) - 1.8


@typecheck
def _compute_ptm(
    logits: Float[Tensor, "... n n bins"],
    query_res_mask: Bool[Tensor, "... n"],
    query_has_frame_mask: Bool[Tensor, "... n"],
    key_res_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "bins"],
    pair_residue_weights: Optional[Float[Tensor, "... n n"]] = None,
) -> Float[Tensor, "..."]:
    """
    Compute predicted TM score, normalized by the number of "key" tokens, with optional contact-based weights
    """
    device = bin_centers.device
    num_key_tokens = reduce(key_res_mask, "... n -> ...", "sum").to(logits.dtype).to(device)
    
    # compute pairwise-TM normalized by the number of key tokens
    if pair_residue_weights is not None:
        d0 = rearrange(tm_d0(num_key_tokens), "... -> ... 1 1").to(device)
    else:
        d0 = rearrange(tm_d0(num_key_tokens), "... -> ... 1").to(device)
    if DEBUG > 1: cprint(d0)

    bin_weights: Float[Tensor, "bins"] = 1. / (1 + (bin_centers / d0) ** 2)
    # btm has shape (b,bins). Need to broadcast with probs of shape (b,n,n,bins)
    bin_weights = rearrange(bin_weights, "... bins -> ... 1 1 bins")
    
    # determine key-query pairs with valid logits
    valid_pairs = und(
        query_has_frame_mask & query_res_mask, key_res_mask, "... i, ... j -> ... i j"
    )
    
    # compute per-pair expected TM scores
    expected_pair_tm = expectation(logits, bin_weights)

    if pair_residue_weights is not None:
        # (New) Use contact-based weights if provided
        qk_weights = valid_pairs.float() * pair_residue_weights.to(device)
    else:
        # normalized scores by the number of key tokens
        num_key_tokens = rearrange(num_key_tokens, "... -> ... 1 1")
        qk_weights = valid_pairs.float() / torch.clamp_min(num_key_tokens, 1)
        # (b i j) -> (b i) 
        
    # Normalize weights
    qk_weights = qk_weights / torch.clamp_min(qk_weights.sum(-1, keepdim=True), 1e-8)
    
    # Compute weighted TM score
    query_key_tm = torch.sum(qk_weights * expected_pair_tm, dim=-1)
    # want to select the row with the most optimistic logits
    # and compute TM for this rows predicted alignment
    return torch.max(query_key_tm, dim=-1)[0]


@typecheck
def complex_ptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
) -> Float[Tensor, "..."]:
    """Compute pTM score of the complex"""
    return _compute_ptm(
        logits=pae_logits,
        query_res_mask=token_exists_mask,
        query_has_frame_mask=valid_frames_mask,
        key_res_mask=token_exists_mask,
        bin_centers=bin_centers,
    )


@typecheck
def interface_ptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
) -> Float[Tensor, "..."]:
    """Compute Interface pTM score

    ipTM is the max TM score over chains c \in C, restricting
    to interactions between c and C - {c}.
    """
    query_res_mask, _ = get_chain_masks_and_asyms(
        asym_id=token_asym_id, mask=token_exists_mask
    )

    per_chain_ptm = _compute_ptm(
        logits=rearrange(pae_logits, "... i j n_bins -> ... 1 i j n_bins"),
        query_res_mask=query_res_mask,
        query_has_frame_mask=rearrange(valid_frames_mask, "... n -> ... 1 n"),
        key_res_mask=~query_res_mask & rearrange(token_exists_mask, "... n -> ... 1 n"),
        bin_centers=bin_centers,
    )

    return torch.max(per_chain_ptm, dim=-1)[0]


@typecheck
def per_chain_pair_iptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
    batched=False,
) -> tuple[Float[Tensor, "... n_chains n_chains"], Int[Tensor, "n_chains"]]:
    """Compute pairwise pTM score for each chain pair"""
    chain_mask, asyms = get_chain_masks_and_asyms(
        asym_id=token_asym_id, mask=token_exists_mask
    )
    c = asyms.numel()
    size = 32 * chain_mask.numel() ** 2 * c**2

    batched = batched and size < 2**32

    if not batched:
        # in the interest of saving memory we compute this in a for-loop
        results = []
        for i in range(c):
            result = _compute_ptm(
                logits=rearrange(pae_logits, "... i j n_bins -> ... 1 i j n_bins"),
                query_res_mask=repeat(chain_mask[..., i, :], "... n -> ... k n", k=c),
                query_has_frame_mask=rearrange(valid_frames_mask, "... n -> ... 1 n"),
                key_res_mask=chain_mask,
                bin_centers=bin_centers,
            )
            results.append(result)
        return torch.stack(results, dim=-2), asyms  # b, query_chain, key_chain
    else:
        # compute batched
        query_mask = repeat(chain_mask, "... c n -> ... c k n", k=c)
        key_mask = repeat(chain_mask, "... c n -> ... k c n", k=c)
        result = _compute_ptm(
            logits=rearrange(pae_logits, "... i j n_bins -> ... 1 1 i j n_bins"),
            query_res_mask=query_mask,
            query_has_frame_mask=rearrange(valid_frames_mask, "... n -> ... 1 1 n"),
            key_res_mask=key_mask,
            bin_centers=bin_centers,
        )
        return result, asyms


@typecheck
def per_chain_ptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
) -> tuple[Float[Tensor, "... n_chains"], Int[Tensor, "n_chains"]]:
    """Computes pTM for each chain in the input"""
    chain_mask, unique_asyms = get_chain_masks_and_asyms(
        asym_id=token_asym_id, mask=token_exists_mask
    )
    per_chain_ptm = _compute_ptm(
        logits=rearrange(pae_logits, "... i j n_bins -> ... 1 i j n_bins"),
        query_res_mask=chain_mask,
        query_has_frame_mask=rearrange(valid_frames_mask, "... n -> ... 1 n"),
        key_res_mask=chain_mask,
        bin_centers=bin_centers,
    )
    return per_chain_ptm, unique_asyms


@typecheck
def actif_ptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
    pde_logits: Optional[Float[Tensor, "... n n n_bins"]] = None,
) -> Float[Tensor, "..."]:
    """Compute actual interface pTM score using contact probabilities
    
    Args:
        pae_logits: Predicted aligned error logits
        token_exists_mask: Mask for valid tokens
        valid_frames_mask: Mask for valid frames
        bin_centers: Bin centers for PAE calculation
        token_asym_id: Chain IDs for each token
        pde_logits: Predicted distance error logits
    """
    # Get chain masks
    query_res_mask, _ = get_chain_masks_and_asyms(
        asym_id=token_asym_id, mask=token_exists_mask
    )

    # Create contact-based weights from pde_logits if provided, checkout colabfold
    contact_weights = None
    if pde_logits is not None:
        # Convert pde_logits to contact probabilities
        tau = 0.1
        contact_probs = (pde_logits / tau).softmax(dim=-1)  # [1, 768, 768, 64]

        # Only consider inter-chain contacts
        token_asym_id = token_asym_id.to(pde_logits.device)
        pair_mask = ~(token_asym_id[..., None] == token_asym_id[..., None, :])
        
        # Ensure pair_mask has the right shape for broadcasting
        if len(pair_mask.shape) < len(contact_probs.shape):
            for _ in range(len(contact_probs.shape) - len(pair_mask.shape)):
                pair_mask = pair_mask.unsqueeze(-1)
        
        # Calculate contact weights - sum over the last dimension (bins)
        contact_weights = (contact_probs * pair_mask).sum(dim=-1)
        
        if DEBUG > 1: 
            cprint(f"contact_probs shape: {contact_probs.shape}")
            cprint(f"pair_mask shape: {pair_mask.shape}")
            cprint(f"contact_weights shape: {contact_weights.shape}")
            cprint(f"contact_weights sum: {contact_weights.sum()}")
    else:
        warnings.warn("pde_logits is None, actif_ptm will not be computed")

    return _compute_ptm(
        logits=pae_logits,
        query_res_mask=query_res_mask,
        query_has_frame_mask=valid_frames_mask,
        key_res_mask=token_exists_mask,
        bin_centers=bin_centers,
        pair_residue_weights=contact_weights
    )


@typecheck
def mean_interface_ptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
    batched=False,
    precomputed_pair_iptm: Optional[tuple[Float[Tensor, "... n_chains n_chains"], Int[Tensor, "n_chains"]]] = None,
) -> Float[Tensor, "..."]:
    """Compute mean interface pTM score excluding intra-chain scores
    
    This function computes the mean of all inter-chain pTM scores,
    excluding the diagonal elements (intra-chain scores).
    
    Args:
        pae_logits: Predicted aligned error logits
        token_exists_mask: Mask for valid tokens  
        valid_frames_mask: Mask for valid frames
        bin_centers: Bin centers for PAE calculation
        token_asym_id: Chain IDs for each token
        batched: Whether to use batched computation
        precomputed_pair_iptm: Optional precomputed pair_iptm result
        
    Returns:
        Mean interface pTM score across all chain pairs
    """
    # Use precomputed result if available, otherwise compute
    if precomputed_pair_iptm is not None:
        pair_iptm_matrix, asyms = precomputed_pair_iptm
    else:
        pair_iptm_matrix, asyms = per_chain_pair_iptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
            batched=batched,
        )
    
    # Create mask to exclude diagonal elements (intra-chain scores)
    n_chains = asyms.numel()
    device = pair_iptm_matrix.device
    
    # Create off-diagonal mask
    off_diagonal_mask = ~torch.eye(n_chains, dtype=torch.bool, device=device)
    
    # Apply mask to extract only inter-chain scores
    inter_chain_scores = pair_iptm_matrix[..., off_diagonal_mask]
    
    # Compute mean of inter-chain scores
    mean_score = torch.mean(inter_chain_scores, dim=-1)
    
    return mean_score


@typecheck
def protein_mean_interface_ptm(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
    token_entity_type: Optional[Int[Tensor, "... n"]] = None,
    batched=False,
    precomputed_pair_iptm: Optional[tuple[Float[Tensor, "... n_chains n_chains"], Int[Tensor, "n_chains"]]] = None,
) -> Float[Tensor, "..."]:
    """Compute mean interface pTM score for protein-protein interactions only
    
    This function computes the mean of inter-chain pTM scores between 
    protein chains only, excluding interactions with ligands, DNA, RNA, etc.
    If token_entity_type is None, falls back to mean_interface_ptm.
    
    Args:
        pae_logits: Predicted aligned error logits
        token_exists_mask: Mask for valid tokens  
        valid_frames_mask: Mask for valid frames
        bin_centers: Bin centers for PAE calculation
        token_asym_id: Chain IDs for each token
        token_entity_type: Entity type for each token (optional)
        batched: Whether to use batched computation
        precomputed_pair_iptm: Optional precomputed pair_iptm result
        
    Returns:
        Mean interface pTM score between protein chains only, or mean interface pTM if entity type is unavailable
    """
    # Use precomputed result if available, otherwise compute
    if precomputed_pair_iptm is not None:
        pair_iptm_matrix, asyms = precomputed_pair_iptm
    else:
        pair_iptm_matrix, asyms = per_chain_pair_iptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
            batched=batched,
        )
    
    n_chains = asyms.numel()
    device = pair_iptm_matrix.device
    
    # If entity type information is available, use it to identify protein chains
    if token_entity_type is not None:
        # Get chain masks for each asym_id
        chain_masks, chain_asyms = get_chain_masks_and_asyms(
            asym_id=token_asym_id, mask=token_exists_mask
        )
        
        # Identify which chains are proteins
        protein_chain_mask = torch.zeros(n_chains, dtype=torch.bool, device=device)
        for i, asym_id in enumerate(chain_asyms):
            # Get tokens for this chain
            chain_token_mask = (token_asym_id == asym_id) & token_exists_mask
            # Check if any token in this chain is a protein
            if chain_token_mask.any():
                chain_entity_types = token_entity_type[chain_token_mask]
                is_protein = (chain_entity_types == EntityType.PROTEIN.value).any()
                protein_chain_mask[i] = is_protein
        
        # Create mask for protein-protein interactions
        protein_pair_mask = protein_chain_mask.unsqueeze(-1) & protein_chain_mask.unsqueeze(-2)
        # Exclude diagonal (intra-chain interactions)
        off_diagonal_mask = ~torch.eye(n_chains, dtype=torch.bool, device=device)
        interaction_mask = protein_pair_mask & off_diagonal_mask
        
        # Extract protein-protein interaction scores
        if interaction_mask.any():
            inter_protein_scores = pair_iptm_matrix[..., interaction_mask]
            mean_score = torch.mean(inter_protein_scores, dim=-1)
        else:
            # No protein-protein interactions found
            mean_score = torch.zeros_like(pair_iptm_matrix[..., 0, 0])
    else:
        # Fallback: use first two chains (as originally requested)
        if DEBUG: tprint(f"pp iptm Fallback to use first two chains")
        if n_chains >= 2:
            # Extract scores for chain 0 -> chain 1 and chain 1 -> chain 0
            protein_protein_scores = torch.stack([
                pair_iptm_matrix[..., 0, 1],  # chain 0 to chain 1
                pair_iptm_matrix[..., 1, 0],  # chain 1 to chain 0
            ], dim=-1)
            
            # Compute mean of protein-protein interaction scores
            mean_score = torch.mean(protein_protein_scores, dim=-1)
        else:
            # If less than 2 chains, return zero score
            mean_score = torch.zeros_like(pair_iptm_matrix[..., 0, 0])
    
    return mean_score


@typecheck
def get_scores(
    pae_logits: Float[Tensor, "... n n n_bins"],
    token_exists_mask: Bool[Tensor, "... n"],
    valid_frames_mask: Bool[Tensor, "... n"],
    bin_centers: Float[Tensor, "n_bins"],
    token_asym_id: Int[Tensor, "... n"],
    pde_logits: Optional[Float[Tensor, "... n n n_bins"]] = None,
    token_entity_type: Optional[Int[Tensor, "... n"]] = None,
) -> PTMScores:
    # Compute per_chain_pair_iptm once and reuse
    pair_iptm_result = per_chain_pair_iptm(
        pae_logits=pae_logits,
        token_exists_mask=token_exists_mask,
        valid_frames_mask=valid_frames_mask,
        bin_centers=bin_centers,
        token_asym_id=token_asym_id,
    )
    
    return PTMScores(
        complex_ptm=complex_ptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
        ),
        interface_ptm=interface_ptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
        ),
        actif_ptm=actif_ptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
            pde_logits=pde_logits,
        ),
        mean_interface_ptm=mean_interface_ptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
            precomputed_pair_iptm=pair_iptm_result,
        ),
        protein_mean_interface_ptm=protein_mean_interface_ptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
            token_entity_type=token_entity_type,
            precomputed_pair_iptm=pair_iptm_result,
        ),
        per_chain_pair_iptm=pair_iptm_result[0],
        per_chain_ptm=per_chain_ptm(
            pae_logits=pae_logits,
            token_exists_mask=token_exists_mask,
            valid_frames_mask=valid_frames_mask,
            bin_centers=bin_centers,
            token_asym_id=token_asym_id,
        )[0],
    )

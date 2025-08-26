# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import logging
import math
from pathlib import Path
import warnings
from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np
import torch
import torch.export
from einops import einsum, rearrange
from torch import Tensor
from tqdm import tqdm

from chai_lab.data.collate.collate import Collate
from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
from chai_lab.data.dataset.all_atom_feature_context import (
    AllAtomFeatureContext,
)
from chai_lab.data.dataset.msas.utils import (
    subsample_and_reorder_msa_feats_n_mask,
)
from chai_lab.data.features.feature_factory import FeatureFactory
from chai_lab.data.features.feature_type import FeatureType
from chai_lab.data.features.generators.atom_element import AtomElementOneHot
from chai_lab.data.features.generators.atom_name import AtomNameOneHot
from chai_lab.data.features.generators.base import EncodingType
from chai_lab.data.features.generators.blocked_atom_pair_distances import (
    BlockedAtomPairDistances,
    BlockedAtomPairDistogram,
)
from chai_lab.data.features.generators.docking import DockingRestraintGenerator
from chai_lab.data.features.generators.esm_generator import ESMEmbeddings
from chai_lab.data.features.generators.identity import Identity
from chai_lab.data.features.generators.is_cropped_chain import ChainIsCropped
from chai_lab.data.features.generators.missing_chain_contact import MissingChainContact
from chai_lab.data.features.generators.msa import (
    IsPairedMSAGenerator,
    MSADataSourceGenerator,
    MSADeletionMeanGenerator,
    MSADeletionValueGenerator,
    MSAFeatureGenerator,
    MSAHasDeletionGenerator,
    MSAProfileGenerator,
)
from chai_lab.data.features.generators.ref_pos import RefPos
from chai_lab.data.features.generators.relative_chain import RelativeChain
from chai_lab.data.features.generators.relative_entity import RelativeEntity
from chai_lab.data.features.generators.relative_sep import RelativeSequenceSeparation
from chai_lab.data.features.generators.relative_token import RelativeTokenSeparation
from chai_lab.data.features.generators.residue_type import ResidueType
from chai_lab.data.features.generators.structure_metadata import (
    IsDistillation,
    TokenBFactor,
    TokenPLDDT,
)
from chai_lab.data.features.generators.templates import (
    TemplateDistogramGenerator,
    TemplateMaskGenerator,
    TemplateResTypeGenerator,
    TemplateUnitVectorGenerator,
)
from chai_lab.data.features.generators.token_bond import TokenBondRestraint
from chai_lab.data.features.generators.token_dist_restraint import (
    TokenDistanceRestraint,
)
from chai_lab.data.features.generators.token_pair_pocket_restraint import (
    TokenPairPocketRestraint,
)
from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif
from chai_lab.data.parsing.structure.entity_type import EntityType, get_entity_type_name
from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule

from chai_lab.model.utils import center_random_augmentation
from chai_lab.ranking.frames import get_frames_and_mask
from chai_lab.ranking.rank import SampleRanking, get_scores, rank
from chai_lab.utils.plot import plot_msa
from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self

from chai_lab.chai1 import (
    ModuleWrapper,
    StructureCandidates,  # plddt, pae, pde
    make_all_atom_feature_context,
    load_exported,
    raise_if_too_many_tokens,
    raise_if_too_many_templates,
    raise_if_msa_too_deep,
)
import chai_lab.ranking.ptm as ptm

from chai_lab_extension.steering.base import PotentialType
from chai_lab_extension.steering.particle_filter import ParticleFilter
from chai_lab_extension.steering.scoring import *

from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG

_component_cache: dict[str, ModuleWrapper] = {}

@contextmanager
def _component_moved_to(
    comp_key: str, device: torch.device
) -> Generator[ModuleWrapper, None, None]:
    # Transiently moves module to provided device, then moves to CPU.
    # Much faster than reloading module from disk.
    if comp_key not in _component_cache:
        _component_cache[comp_key] = load_exported(comp_key, device)

    component = _component_cache[comp_key]
    component.jit_module.to(device)
    yield component
    component.jit_module.to("cpu")


# %%
# Create feature factory

feature_generators = dict(
    RelativeSequenceSeparation=RelativeSequenceSeparation(sep_bins=None),
    RelativeTokenSeparation=RelativeTokenSeparation(r_max=32),
    RelativeEntity=RelativeEntity(),
    RelativeChain=RelativeChain(),
    ResidueType=ResidueType(
        # min_corrupt_prob=0.0,
        # max_corrupt_prob=0.0,
        num_res_ty=32,
        key="token_residue_type",
    ),
    ESMEmbeddings=ESMEmbeddings(),  # TODO: this can probably be the identity
    BlockedAtomPairDistogram=BlockedAtomPairDistogram(),
    InverseSquaredBlockedAtomPairDistances=BlockedAtomPairDistances(
        transform="inverse_squared",
        encoding_ty=EncodingType.IDENTITY,
    ),
    AtomRefPos=RefPos(),
    AtomRefCharge=Identity(
        key="inputs/atom_ref_charge",
        ty=FeatureType.ATOM,
        dim=1,
        can_mask=False,
    ),
    AtomRefMask=Identity(
        key="inputs/atom_ref_mask",
        ty=FeatureType.ATOM,
        dim=1,
        can_mask=False,
    ),
    AtomRefElement=AtomElementOneHot(max_atomic_num=128),
    AtomNameOneHot=AtomNameOneHot(),
    TemplateMask=TemplateMaskGenerator(),
    TemplateUnitVector=TemplateUnitVectorGenerator(),
    TemplateResType=TemplateResTypeGenerator(),
    TemplateDistogram=TemplateDistogramGenerator(),

    TokenDistanceRestraint=TokenDistanceRestraint(
        include_probability=1.0,
        size=0.33,
        min_dist=6.0,
        max_dist=30.0,
        num_rbf_radii=6,
    ),
    DockingConstraintGenerator=DockingRestraintGenerator(
        include_probability=0.0,
        structure_dropout_prob=0.75,
        chain_dropout_prob=0.75,
    ),
    TokenPairPocketRestraint=TokenPairPocketRestraint(
        size=0.33,
        include_probability=1.0,
        min_dist=6.0,
        max_dist=20.0,
        coord_noise=0.0,
        num_rbf_radii=6,
    ),
    MSAProfile=MSAProfileGenerator(),
    MSADeletionMean=MSADeletionMeanGenerator(),
    IsDistillation=IsDistillation(),
    TokenBFactor=TokenBFactor(include_prob=0.0),
    TokenPLDDT=TokenPLDDT(include_prob=0.0),
    ChainIsCropped=ChainIsCropped(),
    MissingChainContact=MissingChainContact(contact_threshold=6.0),
    MSAOneHot=MSAFeatureGenerator(),
    MSAHasDeletion=MSAHasDeletionGenerator(),
    MSADeletionValue=MSADeletionValueGenerator(),
    IsPairedMSA=IsPairedMSAGenerator(),
    MSADataSource=MSADataSourceGenerator(),
)
feature_factory = FeatureFactory(feature_generators)

# %%
# Config
from chai_lab.model.diffusion_config import DiffusionConfig

# %%
# Inference logic
@torch.no_grad()
def run_inference(
    fasta_file: Path,
    *,
    output_dir: Path,
    # Configuration for ESM, MSA, constraints, and templates
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    # Parameters controlling how we do inference
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    num_trunk_samples: int = 1,
    # Custom DiffusionConfig class (NEW)
    diffusion_config: DiffusionConfig | None = None,
    # ligand pdb conformer
    pdb_conformer_path: Path | None = None,
    # Diffusion inference time scaling
    num_particles: int = 2,
    resampling_interval: int = 5,
    lambda_weight: float = 10.0,
    potential_type: str = "vanilla",
    fk_sigma_threshold: float = 1.0,
    # Custom scoring for steering
    steering_score_type: str = "mean_interface_ptm",  # "interface_ptm", "plddt", "mean_interface_ptm", "protein_mean_interface_ptm", or "default"
    # fks visualization
    enable_visualization: bool = False,
    # trajectory recording
    enable_trajectory_recording: bool = False,
    trajectory_save_coordinates: bool = True,
    trajectory_compute_plddt: bool = False,
    trajectory_extra_save_interval: Optional[int] = None,
    # Misc
    seed: int | None = None,
    device: str | None = None,
    low_memory: bool = True,
    **kwargs
) -> StructureCandidates:
    assert num_trunk_samples > 0 and num_diffn_samples > 0
    if output_dir.exists():
        assert not any(
            output_dir.iterdir()
        ), f"Output directory {output_dir} is not empty."

    torch_device = torch.device(device if device is not None else "cuda:0")

    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        templates_path=template_hits_path,
        esm_device=torch_device,
        pdb_conformer_path=pdb_conformer_path,
    )

    all_candidates: list[StructureCandidates] = []
    for trunk_idx in range(num_trunk_samples):
        logging.info(f"Trunk sample {trunk_idx + 1}/{num_trunk_samples}")
        cand = run_folding_on_context(
            feature_context,
            output_dir=(
                output_dir / f"trunk_{trunk_idx}"
                if num_trunk_samples > 1
                else output_dir
            ),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=num_diffn_samples,
            recycle_msa_subsample=recycle_msa_subsample,
            # Custom DiffusionConfig class (NEW)
            diffusion_config=diffusion_config,
            # diffusion inference time scaling
            num_particles=num_particles,
            resampling_interval=resampling_interval,
            lambda_weight=lambda_weight,
            potential_type=potential_type,
            fk_sigma_threshold=fk_sigma_threshold,
            steering_score_type=steering_score_type,
            enable_visualization=enable_visualization,
            # trajectory recording
            enable_trajectory_recording=enable_trajectory_recording,
            trajectory_save_coordinates=trajectory_save_coordinates,
            trajectory_compute_plddt=trajectory_compute_plddt,
            trajectory_extra_save_interval=trajectory_extra_save_interval,
            # misc
            seed=seed + trunk_idx if seed is not None else None,
            device=torch_device,
            low_memory=low_memory,
            **kwargs
        )
        all_candidates.append(cand)
    return StructureCandidates.concat(all_candidates)


def _bin_centers(min_bin: float, max_bin: float, no_bins: int) -> Tensor:
    return torch.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]


@torch.no_grad()
def run_folding_on_context(
    feature_context: AllAtomFeatureContext,
    *,
    output_dir: Path,
    # expose some params for easy tweaking
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    # Custom DiffusionConfig class (NEW)
    diffusion_config: DiffusionConfig | None = None,
    # diffusion inference time scaling
    num_particles: int = 2,
    resampling_interval: int = 5,
    lambda_weight: float = 10.0,
    potential_type: str = "vanilla",
    fk_sigma_threshold: float = 1.0,
    steering_score_type: str = "interface_ptm",
    # fks visualization
    enable_visualization: bool = False,
    # trajectory recording
    enable_trajectory_recording: bool = False,
    trajectory_save_coordinates: bool = True,
    trajectory_compute_plddt: bool = False,
    trajectory_extra_save_interval: Optional[int] = None,
    # all diffusion samples come from the same trunk
    num_diffn_samples: int = 5,
    # misc
    seed: int | None = None,
    device: torch.device | None = None,
    low_memory: bool,
    **kwargs,
) -> StructureCandidates:
    """
    Function for in-depth explorations.
    User completely controls folding inputs.
    """

    if kwargs.get("save_intermediate", False):
        warnings.warn("Saving intermediate results is deprecated and needs to be reimplemented.", DeprecationWarning)

    # Set seed
    if seed is not None:
        set_seed([seed])

    if device is None:
        device = torch.device("cuda:0")

    # Clear memory
    torch.cuda.empty_cache()

    ##
    ## Setup DiffusionConfig
    ##
    
    # Use custom DiffusionConfig if provided, otherwise use default
    if diffusion_config is None:
        diffusion_config = DiffusionConfig()
        logging.info("Using default DiffusionConfig")
    else:
        logging.info("Using custom DiffusionConfig")
        logging.info(f"  - S_churn: {diffusion_config.S_churn}")
        logging.info(f"  - S_tmax: {diffusion_config.S_tmax}")
        logging.info(f"  - S_tmin: {diffusion_config.S_tmin}")
        logging.info(f"  - S_noise: {diffusion_config.S_noise}")
        logging.info(f"  - sigma_data: {diffusion_config.sigma_data}")
        logging.info(f"  - second_order: {diffusion_config.second_order}")

    ##
    ## Validate inputs
    ##

    n_actual_tokens = feature_context.structure_context.num_tokens
    raise_if_too_many_tokens(n_actual_tokens)
    raise_if_too_many_templates(feature_context.template_context.num_templates)
    raise_if_msa_too_deep(feature_context.msa_context.depth)
    # NOTE: profile MSA used only for statistics; no depth check
    feature_context.structure_context.report_bonds()

    ##
    ## Prepare batch
    ##

    # Collate inputs into batch
    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )

    feature_contexts = [feature_context]
    batch_size = len(feature_contexts)
    batch = collator(feature_contexts)

    if not low_memory:
        batch = move_data_to_device(batch, device=device)

    # Get features and inputs from batch
    features = {name: feature for name, feature in batch["features"].items()}
    inputs = batch["inputs"]
    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    token_reference_atom_index = inputs["token_ref_atom_index"]
    atom_within_token_index = inputs["atom_within_token_index"]
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]

    ##
    ## Load exported models
    ##

    _, _, model_size = msa_mask.shape
    assert model_size in AVAILABLE_MODEL_SIZES

    confidence_head = load_exported("confidence_head.pt", device)  # this is a hack

    ##
    ## Run the features through the feature embedder
    ##

    with _component_moved_to("feature_embedding.pt", device) as feature_embedding:
        embedded_features = feature_embedding.forward(
            crop_size=model_size,
            move_to_device=device,
            return_on_cpu=low_memory,
            **features,
        )

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
        "TOKEN_PAIR"
    ].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
        "ATOM"
    ].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
        embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    )
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    ##
    ## Bond feature generator
    ## Separate from other feature embeddings due to export limitations
    ##

    bond_ft_gen = TokenBondRestraint()
    bond_ft = bond_ft_gen.generate(batch=batch).data
    with _component_moved_to("bond_loss_input_proj.pt", device) as bond_loss_input_proj:
        trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
            return_on_cpu=low_memory,
            move_to_device=device,
            crop_size=model_size,
            input=bond_ft,
        ).chunk(2, dim=-1)
    token_pair_input_feats += trunk_bond_feat
    token_pair_structure_input_feats += structure_bond_feat

    ##
    ## Run the inputs through the token input embedder
    ##

    with _component_moved_to("token_embedder.pt", device) as token_input_embedder:
        token_input_embedder_outputs: tuple[Tensor, ...] = token_input_embedder.forward(
            return_on_cpu=low_memory,
            move_to_device=device,
            token_single_input_feats=token_single_input_feats,
            token_pair_input_feats=token_pair_input_feats,
            atom_single_input_feats=atom_single_input_feats,
            block_atom_pair_feat=block_atom_pair_input_feats,
            block_atom_pair_mask=block_atom_pair_mask,
            block_indices_h=block_indices_h,
            block_indices_w=block_indices_w,
            atom_single_mask=atom_single_mask,
            atom_token_indices=atom_token_indices,
            crop_size=model_size,
        )
    token_single_initial_repr, token_single_structure_input, token_pair_initial_repr = (
        token_input_embedder_outputs
    )

    ##
    ## Run the input representations through the trunk
    ##

    # Recycle the representations by feeding the output back into the trunk as input for
    # the subsequent recycle
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr
    for _ in tqdm(range(num_trunk_recycles), desc="Trunk recycles"):
        subsampled_msa_input_feats, subsampled_msa_mask = None, None
        if recycle_msa_subsample > 0:
            subsampled_msa_input_feats, subsampled_msa_mask = (
                subsample_and_reorder_msa_feats_n_mask(
                    msa_input_feats,
                    msa_mask,
                )
            )

        with _component_moved_to("trunk.pt", device) as trunk:
            (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
                move_to_device=device,
                token_single_trunk_initial_repr=token_single_initial_repr,
                token_pair_trunk_initial_repr=token_pair_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,  # recycled
                token_pair_trunk_repr=token_pair_trunk_repr,  # recycled
                msa_input_feats=(
                    subsampled_msa_input_feats
                    if subsampled_msa_input_feats is not None
                    else msa_input_feats
                ),
                msa_mask=(
                    subsampled_msa_mask if subsampled_msa_mask is not None else msa_mask
                ),
                template_input_feats=template_input_feats,
                template_input_masks=template_input_masks,
                token_single_mask=token_single_mask,
                token_pair_mask=token_pair_mask,
                crop_size=model_size,
            )
    # We won't be using the trunk anymore; remove it from memory
    torch.cuda.empty_cache()

    ##
    ## Denoise the trunk representation by passing it through the diffusion module
    ##

    atom_single_mask = atom_single_mask.to(device)

    static_diffusion_inputs = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_block_pair_mask=block_atom_pair_mask,
        token_single_mask=token_single_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_token_indices=atom_token_indices,
    )
    static_diffusion_inputs = move_data_to_device(
        static_diffusion_inputs, device=device
    )

    def _denoise(atom_pos: Tensor, sigma: Tensor, ds: int) -> Tensor:
        # Check input shape to determine if rearrangement is needed
        if atom_pos.shape[0] == batch_size * ds:
            # Original code case: multiple samples combined together
            atom_noised_coords = rearrange(
                atom_pos, "(b ds) ... -> b ds ...", ds=ds
            ).contiguous()
        else:
            # Particle filter case: each particle has only one sample
            # No need to rearrange, just add a dimension
            atom_noised_coords = atom_pos.unsqueeze(1).contiguous()
            # Now shape is [batch_size, 1, num_atoms, 3]
        
        # Adjust noise_sigma shape to match atom_noised_coords
        actual_ds = atom_noised_coords.shape[1]  # Could be ds or 1
        
        # Fix: Handle sigma shape correctly
        # noise_sigma = repeat(sigma, " -> b ds", b=batch_size, ds=actual_ds)
        if sigma.dim() == 0:  # Scalar
            noise_sigma = torch.full((batch_size, actual_ds), sigma.item(), device=device)
        elif sigma.dim() == 1:  # 1D tensor
            # Ensure sigma length is 1 or matches actual_ds
            if len(sigma) == 1:
                noise_sigma = torch.full((batch_size, actual_ds), sigma[0].item(), device=device)
            else:
                # Assume sigma already has correct length, just expand batch dimension
                noise_sigma = sigma.unsqueeze(0).expand(batch_size, -1)
        else:
            # Assume already correct shape
            noise_sigma = sigma
        
        # Call diffusion model
        with _component_moved_to("diffusion_module.pt", device) as diffusion_module:
            denoised = diffusion_module.forward(
                atom_noised_coords=atom_noised_coords.float(),
                noise_sigma=noise_sigma.float(),
                crop_size=model_size,
                **static_diffusion_inputs,
            )
        
        # If particle filter case, remove extra dimension
        if atom_pos.shape[0] != batch_size * ds:
            denoised = denoised.squeeze(1)
        
        return denoised

    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=diffusion_config.S_tmax,
        s_min=4e-4,
        p=7.0,
        sigma_data=diffusion_config.sigma_data,
    )
    
    # Generate full sigmas schedule first
    full_sigmas = inference_noise_schedule.get_schedule(
        device=device, num_timesteps=num_diffn_timesteps
    )
    
    sigmas = full_sigmas
    effective_timesteps = num_diffn_timesteps
    logging.info(f"Full diffusion: using all {num_diffn_timesteps} timesteps")
    logging.info(f"Starting sigma: {sigmas[0]:.6f}, ending sigma: {sigmas[-1]:.6f}")
    
    # Generate gammas based on the adjusted sigmas
    gammas = torch.where(
        (sigmas >= diffusion_config.S_tmin) & (sigmas <= diffusion_config.S_tmax),
        min(diffusion_config.S_churn / effective_timesteps, math.sqrt(2) - 1),
        0.0,
    )
    
    # tprint('Sigmas and Gammas')
    # cprint(sigmas, gammas)
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

    # Initialize particle filter with visualization settings
    vis_output_dir = output_dir / "visualization" if enable_visualization else None
    if enable_visualization and vis_output_dir:
        vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom scoring function for steering
    scoring_function = None
    if steering_score_type == "mean_interface_ptm":
        scoring_function = MeanInterfacePTMScoringFunction(
            confidence_head=confidence_head,
            inputs=inputs,
            token_single_initial_repr=token_single_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            token_single_mask=token_single_mask,
            atom_single_mask=atom_single_mask,
            token_reference_atom_index=token_reference_atom_index,
            atom_token_indices=atom_token_indices,
            atom_within_token_index=atom_within_token_index,
            model_size=model_size,
        )
    elif steering_score_type == "interface_ptm":
        scoring_function = InterfacePTMScoringFunction(
            confidence_head=confidence_head,
            inputs=inputs,
            token_single_initial_repr=token_single_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            token_single_mask=token_single_mask,
            atom_single_mask=atom_single_mask,
            token_reference_atom_index=token_reference_atom_index,
            atom_token_indices=atom_token_indices,
            atom_within_token_index=atom_within_token_index,
            model_size=model_size,
        )
    elif steering_score_type == "plddt":
        scoring_function = PLDDTScoringFunction(
            confidence_head=confidence_head,
            inputs=inputs,
            token_single_initial_repr=token_single_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            token_single_mask=token_single_mask,
            atom_single_mask=atom_single_mask,
            token_reference_atom_index=token_reference_atom_index,
            atom_token_indices=atom_token_indices,
            atom_within_token_index=atom_within_token_index,
            model_size=model_size,
        )
    elif steering_score_type == "protein_mean_interface_ptm":
        scoring_function = ProteinMeanInterfacePTMScoringFunction(
            confidence_head=confidence_head,
            inputs=inputs,
            token_single_initial_repr=token_single_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            token_single_mask=token_single_mask,
            atom_single_mask=atom_single_mask,
            token_reference_atom_index=token_reference_atom_index,
            atom_token_indices=atom_token_indices,
            atom_within_token_index=atom_within_token_index,
            model_size=model_size,
        )
    elif steering_score_type == "default":
        scoring_function = DefaultScoringFunction()  # Use default avg_interface_ptm
    else:
        raise ValueError(f"Unknown steering_score_type: {steering_score_type}. Must be one of: 'interface_ptm', 'plddt', 'mean_interface_ptm', 'protein_mean_interface_ptm', 'default'")
        
    logging.info(f"Using {steering_score_type} as steering score function")
        
    particle_filter = ParticleFilter(
        num_particles=num_particles,
        scoring_function=scoring_function,
        resampling_interval=resampling_interval,
        lambda_weight=lambda_weight,
        potential_type=PotentialType.from_str(potential_type),
        fk_sigma_threshold=fk_sigma_threshold,
        enable_visualization=enable_visualization,
        visualization_output_dir=vis_output_dir,
        device=device,
    )
    
    # Initial atom positions, batch_size x num_diffn_samples in single tensor
    # random seed matters in: init atom_pos and generate noise
    _, num_atoms = atom_single_mask.shape
    cprint(atom_single_mask.shape)
    
    particle_filter.initialize_particles(
        batch_size=batch_size,
        num_atoms=num_atoms,
        sigma=sigmas[0],  # Use the first sigma of the adjusted schedule
        device=device,
        structure_context=feature_context.structure_context,
        base_seed=seed
    )
    
    # Setup trajectory recorder if enabled
    if enable_trajectory_recording:
        from chai_lab_extension.steering.trajector_streamer import DiffusionTrajectoryRecorder
        
        trajectory_output_dir = output_dir / "trajectory_recording"
        
        # Setup confidence context for pLDDT computation
        confidence_context = {}
        if trajectory_compute_plddt:
            confidence_context = {
                'inputs': inputs,
                'token_single_initial_repr': token_single_initial_repr,
                'token_single_trunk_repr': token_single_trunk_repr,
                'token_pair_trunk_repr': token_pair_trunk_repr,
                'token_single_mask': token_single_mask,
                'atom_single_mask': atom_single_mask,
                'token_reference_atom_index': token_reference_atom_index,
                'atom_token_indices': atom_token_indices,
                'atom_within_token_index': atom_within_token_index,
                'model_size': model_size
            }
        
        trajectory_recorder = DiffusionTrajectoryRecorder(
            output_dir=trajectory_output_dir,
            save_coordinates=trajectory_save_coordinates,
            compute_plddt=trajectory_compute_plddt,
            extra_save_interval=trajectory_extra_save_interval,
            confidence_head=confidence_head if trajectory_compute_plddt else None,
            confidence_context=confidence_context
        )
        
        # Set trajectory recorder to particle filter
        particle_filter.set_trajectory_recorder(trajectory_recorder)
        logging.info(f"Trajectory recorder enabled. Output: {trajectory_output_dir}")
        logging.info(f"  - Save coordinates: {trajectory_save_coordinates}")
        logging.info(f"  - Compute pLDDT: {trajectory_compute_plddt}")
        logging.info(f"  - Extra save interval: {trajectory_extra_save_interval}")
    else:
        trajectory_recorder = None
    
    # Main Diffusion Loop
    for step_idx, (sigma_curr, sigma_next, gamma_curr) in tqdm(
        enumerate(sigmas_and_gammas),
        desc="Diffusion steps",
        total=len(sigmas_and_gammas)
    ):
        # Debug: Print diffusion loop state at first step
        # if step_idx == 0:
        #     logging.info(f"[DEBUG] Starting diffusion loop:")
        #     logging.info(f"  Total steps: {len(sigmas_and_gammas)}")
        #     logging.info(f"  batch_size: {batch_size}")
        #     logging.info(f"  num_atoms: {num_atoms}")
        #     logging.info(f"  atom_single_mask shape: {atom_single_mask.shape}")
        #     logging.info(f"  atom_single_mask sum: {atom_single_mask.sum().item()}")
        #     logging.info(f"  Current sigma: {sigma_curr.item():.6f}")
        #     logging.info(f"  Next sigma: {sigma_next.item():.6f}")
        #     logging.info(f"  Gamma: {gamma_curr.item():.6f}")
        #     logging.info(f"  Number of particles: {len(particle_filter.particles)}")
        #     for i, p in enumerate(particle_filter.particles):
        #         logging.info(f"    Particle {i}: atom_pos shape {p.atom_pos.shape}")
        
        # Calculate actual step index for logging purposes
        actual_step_idx = step_idx
        
        # Process each particle
        for particle_idx, particle in enumerate(particle_filter.particles):
            # NOTE: this is important to keep each candidate independent!!!
            if seed is not None: torch.manual_seed(seed + actual_step_idx*100 + particle_idx)

            # Center coords
            atom_pos_candidate = center_random_augmentation(
                particle.atom_pos.clone(),
                atom_single_mask=atom_single_mask,
            )

            # NOTE: Alg 2 probability comes from this paper Nvidia EDM, arxiv: 2206.00364
            # Alg 2. lines 4-6
            noise = diffusion_config.S_noise * torch.randn(
                atom_pos_candidate.shape, device=atom_pos_candidate.device
            )
                        
            sigma_hat = sigma_curr + gamma_curr * sigma_curr
            atom_pos_noise = (sigma_hat**2 - sigma_curr**2).clamp_min(1e-6).sqrt()
            atom_pos_hat = atom_pos_candidate + noise * atom_pos_noise

            # Lines 7-8
            denoised_pos = _denoise(
                atom_pos=atom_pos_hat,
                sigma=sigma_hat,
                ds=1,  # fk steering can track single sample only
            )
            d_i = (atom_pos_hat - denoised_pos) / sigma_hat
            atom_pos_candidate = atom_pos_hat + (sigma_next - sigma_hat) * d_i

            # Lines 9-11
            if sigma_next != 0 and diffusion_config.second_order:  # second order update
                denoised_pos = _denoise(
                    atom_pos_candidate,
                    sigma=sigma_next,
                    ds=1,
                )
                d_i_prime = (atom_pos_candidate - denoised_pos) / sigma_next
                atom_pos_candidate = atom_pos_candidate + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2)

            # Update particle state
            particle.atom_pos = atom_pos_candidate.detach()
        
        # Record extra saves for trajectory recorder
        if (trajectory_recorder and 
            trajectory_extra_save_interval and 
            step_idx % trajectory_extra_save_interval == 0):
            try:
                # Synchronize CUDA before recording to avoid device-side assertions
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                trajectory_recorder.record_step(
                    actual_step_idx, sigma_next.item(), particle_filter.particles, 
                    is_extra_save=True
                )
            except Exception as e:
                logging.warning(f"Failed to record extra save at step {actual_step_idx}: {e}")
            
        # Resampling logic
        if particle_filter.should_resample(step_idx, sigma_next):
            logging.info(f"Resampling at step {actual_step_idx} (local step {step_idx}) using {steering_score_type} scoring function")
            # Special handling: if using default scoring, need to compute avg_interface_ptm first
            if steering_score_type == "default":
                for particle in particle_filter.particles:
                    temp_plddt, ptm_scores, _ = calculate_final_confidence_scores(
                        particle.atom_pos, device, confidence_head, inputs,
                        token_single_initial_repr, token_single_trunk_repr,
                        token_pair_trunk_repr, token_single_mask,
                        atom_single_mask, token_reference_atom_index,
                        atom_token_indices, atom_within_token_index,
                        model_size
                    )
                    # Only update avg_interface_ptm, used for DefaultScoringFunction
                    particle.avg_interface_ptm = ptm_scores.interface_ptm.mean().item()

            # Perform resampling (using custom scoring function)
            particle_filter.resample(step_idx)

            torch.cuda.empty_cache()

    # Finalize trajectory recorder
    if trajectory_recorder:
        try:
            # Synchronize CUDA before final recording
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Record final state
            trajectory_recorder.record_step(
                len(sigmas_and_gammas),
                sigmas[-1].item(),
                particle_filter.particles,
            )
            # Finalize and save all trajectory data
            trajectory_recorder.finalize()
            logging.info("Trajectory recording completed and finalized")
        except Exception as e:
            logging.error(f"Failed to finalize trajectory recording: {e}")
    
    if enable_visualization:
        particle_filter.visualizer.record_step(
            len(sigmas_and_gammas),
            sigmas[-1].item(),
            particle_filter.particles,
        )
        particle_filter.visualizer.plot_score_trajectories()
        particle_filter.visualizer.plot_resampling_flow()
        particle_filter.visualizer.save_trajectory_data()
        particle_filter.visualizer.save_trajectory_data_as_csv()
    
    # Use the best particle for final output
    best_particle = particle_filter.get_best_particle()

    # Get the final atom positions for output
    # NOTE: it's highly recommended to use the num_particles >= num_diffn_samples or use num_trunk_samples instead of num_diffn_samples
    # atom_pos = torch.cat([p.atom_pos for p in particle_filter.particles[:num_diffn_samples]], dim=0)
    # Instead of repeating the best particle, run additional diffusion steps
    # to generate more diverse samples from the best particles
    if num_particles < num_diffn_samples:
        # Start with the available particles
        atom_pos_list = [p.atom_pos for p in particle_filter.particles]
        
        # We need to generate (num_diffn_samples - num_particles) more samples
        additional_samples_needed = num_diffn_samples - num_particles
        
        # Use the best particle as a starting point for additional samples
        best_pos = best_particle.atom_pos
        
        # Run a few more diffusion steps from the best particle with different noise seeds
        # to generate diverse additional samples
        for i in range(additional_samples_needed):
            if seed is not None:
                # Use a different seed for each additional sample
                torch.manual_seed(seed + 10000 + i)
            
            # Add some noise to create a new starting point
            # Use a small sigma value to stay close to the good solution
            # but still generate diversity
            new_pos = best_pos + 0.1 * torch.randn_like(best_pos)
            
            # Optional: Run a few more diffusion steps to refine this sample
            # This is a simplified version - you could run more steps if needed
            current_pos = new_pos
            for j in range(min(5, resampling_interval)):
                # Apply a simplified diffusion step
                noise = 0.05 * torch.randn_like(current_pos)
                denoised = _denoise(
                    current_pos + noise, 
                    torch.tensor(0.1, device=device),
                    1
                )
                current_pos = denoised
            
            atom_pos_list.append(current_pos)
        
        # Combine all samples
        atom_pos = torch.cat(atom_pos_list, dim=0)
    else:
        # If we have enough particles, just take the first num_diffn_samples
        atom_pos = torch.cat([p.atom_pos for p in particle_filter.particles[:num_diffn_samples]], dim=0)
    
    if DEBUG:
        cprint(num_particles < num_diffn_samples)
        cprint(atom_pos.shape)

    # Make sure we have exactly num_diffn_samples
    assert atom_pos.shape[0] == num_diffn_samples
    # Make sure we have exactly num_diffn_samples
    assert atom_pos.shape[0] == num_diffn_samples

    temp_plddt = best_particle.plddt

    # We won't be running diffusion anymore
    del static_diffusion_inputs
    torch.cuda.empty_cache()

    ##
    ## Run the confidence model
    ##

    confidence_outputs: list[tuple[Tensor, ...]] = []
    for ds in range(num_diffn_samples):
        _, _, confidence_output = calculate_final_confidence_scores(
            atom_pos[ds : ds + 1], device, confidence_head, inputs,
            token_single_initial_repr, token_single_trunk_repr,
            token_pair_trunk_repr, token_single_mask,
            atom_single_mask, token_reference_atom_index,
            atom_token_indices, atom_within_token_index,
            model_size
        )
        confidence_outputs.append(confidence_output)

    pae_logits, pde_logits, plddt_logits, pae_scores, pde_scores, plddt_scores_atom, plddt_scores = process_confidence_outputs(
        confidence_outputs, token_single_mask, atom_single_mask, atom_token_indices
    )

    assert atom_pos.shape[0] == num_diffn_samples
    assert pae_logits.shape[0] == num_diffn_samples

    inputs = move_data_to_device(inputs, torch.device("cpu"))
    atom_pos = atom_pos.cpu()
    plddt_logits = plddt_logits.cpu()
    pae_logits = pae_logits.cpu()

    # Plot coverage of tokens by MSA, save plot
    output_dir.mkdir(parents=True, exist_ok=True)

    if feature_context.msa_context.mask.any():
        msa_plot_path = plot_msa(
            input_tokens=feature_context.structure_context.token_residue_type,
            msa_tokens=feature_context.msa_context.tokens,
            out_fname=output_dir / "msa_depth.pdf",
        )
    else:
        msa_plot_path = None

    cif_paths: list[Path] = []
    ranking_data: list[SampleRanking] = []

    for idx in range(num_diffn_samples):
        ##
        ## Compute ranking scores
        ##

        _, valid_frames_mask = get_frames_and_mask(
            atom_pos[idx : idx + 1],
            inputs["token_asym_id"],
            inputs["token_residue_index"],
            inputs["token_backbone_frame_mask"],
            inputs["token_centre_atom_index"],
            inputs["token_exists_mask"],
            inputs["atom_exists_mask"],
            inputs["token_backbone_frame_index"],
            inputs["atom_token_index"],
        )

        ranking_outputs: SampleRanking = rank(
            atom_pos[idx : idx + 1],
            atom_mask=inputs["atom_exists_mask"],
            atom_token_index=inputs["atom_token_index"],
            token_exists_mask=inputs["token_exists_mask"],
            token_asym_id=inputs["token_asym_id"],
            token_entity_type=inputs["token_entity_type"],
            token_valid_frames_mask=valid_frames_mask,
            lddt_logits=plddt_logits[idx : idx + 1],
            lddt_bin_centers=_bin_centers(0, 1, plddt_logits.shape[-1]).to(
                plddt_logits.device
            ),
            pae_logits=pae_logits[idx : idx + 1],
            pae_bin_centers=_bin_centers(0.0, 32.0, 64).to(pae_logits.device),
            pde_logits=pde_logits[idx : idx + 1],
        )

        if DEBUG: cprint(ranking_outputs)
        ranking_data.append(ranking_outputs)

        ##
        ## Write output files
        ##

        cif_out_path = output_dir.joinpath(f"pred.model_idx_{idx}.cif")
        aggregate_score = ranking_outputs.aggregate_score.item()
        print(f"Score={aggregate_score:.4f}, writing output to {cif_out_path}")

        # use 0-100 scale for pLDDT in pdb outputs
        scaled_plddt_scores_per_atom = 100 * plddt_scores_atom[idx : idx + 1]

        save_to_cif(
            coords=atom_pos[idx : idx + 1],
            bfactors=scaled_plddt_scores_per_atom,
            output_batch=inputs,
            write_path=cif_out_path,
            # Set asym names to be A, B, C, ...
            asym_entity_names={
                i: get_chain_letter(i)
                for i in range(1, len(feature_context.chains) + 1)
            },
        )
        cif_paths.append(cif_out_path)

        scores_out_path = output_dir.joinpath(f"scores.model_idx_{idx}.npz")
        scores_dict = get_scores(
            ranking_outputs,
            pae_scores=pae_scores[idx],
            pde_scores=pde_scores[idx]
        )
        if DEBUG: cprint(scores_dict, c='red')
        np.savez(scores_out_path, **scores_dict)

    return StructureCandidates(
        cif_paths=cif_paths,
        ranking_data=ranking_data,
        msa_coverage_plot_path=msa_plot_path,
        pae=pae_scores,
        pde=pde_scores,
        plddt=plddt_scores,
    )


def calculate_final_confidence_scores(particle_pos, device, confidence_head, inputs, 
                                    token_single_initial_repr, token_single_trunk_repr, 
                                    token_pair_trunk_repr, token_single_mask, 
                                    atom_single_mask, token_reference_atom_index,
                                    atom_token_indices, atom_within_token_index, 
                                    model_size):
    """Calculate final confidence scores for output including plddt and ptm scores"""
    # Run confidence model
    confidence_output = confidence_head.forward(
        move_to_device=device,
        token_single_input_repr=token_single_initial_repr,
        token_single_trunk_repr=token_single_trunk_repr,
        token_pair_trunk_repr=token_pair_trunk_repr,
        token_single_mask=token_single_mask,
        atom_single_mask=atom_single_mask,
        atom_coords=particle_pos,
        token_reference_atom_index=token_reference_atom_index,
        atom_token_index=atom_token_indices,
        atom_within_token_index=atom_within_token_index,
        crop_size=model_size,
    )
    
    # Calculate plddt scores
    plddt = einsum(
        confidence_output[2].float().softmax(dim=-1),
        _bin_centers(0, 1, confidence_output[2].shape[-1]).to(device),
        "b a d, d -> b a"
    )
    
    # Get frames and mask
    _, valid_frames_mask = get_frames_and_mask(
        particle_pos,
        inputs["token_asym_id"].to(device),
        inputs["token_residue_index"].to(device),
        inputs["token_backbone_frame_mask"].to(device),
        inputs["token_centre_atom_index"].to(device),
        inputs["token_exists_mask"].to(device),
        atom_single_mask,
        inputs["token_backbone_frame_index"].to(device),
        atom_token_indices,
    )
    
    # Calculate ptm scores
    ptm_scores = ptm.get_scores(
        pae_logits=confidence_output[0].float(),
        token_exists_mask=token_single_mask.to(device),
        valid_frames_mask=valid_frames_mask.to(device),
        bin_centers=_bin_centers(0.0, 32.0, 64).to(device),
        token_asym_id=inputs["token_asym_id"].to(device),
        pde_logits=confidence_output[1].float(),
        token_entity_type=inputs["token_entity_type"].to(device),
    )
    
    return plddt.detach(), ptm_scores, confidence_output


def process_confidence_outputs(confidence_outputs, token_single_mask, atom_single_mask, atom_token_indices):
    """
    PAE: token level
      pae_logits: [num_diffn_samples, num_tokens, num_tokens, 64]
      masked_pae_logits = pae_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :] -> [num_diffn_samples, num_tokens (valid), num_tokens (valid), 64]
      pae_scores: [num_diffn_samples, num_tokens (valid), num_tokens (valid)]

    PDE: same shape as PAE

    plddt: atom level
      plddt_logits: (num_diffn_samples, total_atoms, n_bins_plddt)
      plddt_scores_atom: (num_diffn_samples, total_atoms)
      plddt_scores: (num_diffn_samples, num_tokens (valid)) -> to token level
    """
    # Merge outputs from all samples
    pae_logits, pde_logits, plddt_logits = [
        torch.cat(single_sample, dim=0)
        for single_sample in zip(*confidence_outputs, strict=True)
    ]
    
    def softmax_einsum_and_cpu(logits, bin_mean, pattern):
        res = einsum(
            logits.float().softmax(dim=-1), bin_mean.to(logits.device), pattern
        )
        return res.to(device="cpu")
    
    # Calculate token mask
    token_mask_1d = rearrange(token_single_mask, "1 b -> b")
    
    # Calculate PAE scores
    pae_scores = softmax_einsum_and_cpu(
        pae_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :],
        _bin_centers(0.0, 32.0, 64),
        "b n1 n2 d, d -> b n1 n2",
    )
    
    # Calculate PDE scores
    pde_scores = softmax_einsum_and_cpu(
        pde_logits[:, token_mask_1d, :, :][:, :, token_mask_1d, :],
        _bin_centers(0.0, 32.0, 64),
        "b n1 n2 d, d -> b n1 n2",
    )
    
    # Calculate atom-level pLDDT scores
    plddt_scores_atom = softmax_einsum_and_cpu(
        plddt_logits,
        _bin_centers(0, 1, plddt_logits.shape[-1]),
        "b a d, d -> b a",
    )
    
    # Convert to token-level pLDDT
    [mask] = atom_single_mask.cpu()
    [indices] = atom_token_indices.cpu()
    
    def avg_per_token_1d(x):
        n = torch.bincount(indices[mask], weights=x[mask])
        d = torch.bincount(indices[mask]).clamp(min=1)
        return n / d
    
    plddt_scores = torch.stack([avg_per_token_1d(x) for x in plddt_scores_atom])
    
    return pae_logits, pde_logits, plddt_logits, pae_scores, pde_scores, plddt_scores_atom, plddt_scores
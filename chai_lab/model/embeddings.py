# NOTE: not official code, just for reverse-engineering

import torch
import torch.nn as nn

from chai_lab.data.collate.collate import Collate
from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
from chai_lab.data.dataset.all_atom_feature_context import (
    MAX_MSA_DEPTH,
    MAX_NUM_TEMPLATES,
    AllAtomFeatureContext,
)
from chai_lab.data.dataset.constraints.restraint_context import (
    RestraintContext,
    load_manual_restraints_for_chai1,
)
from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
from chai_lab.data.dataset.embeddings.esm import get_esm_embedding_context
from chai_lab.data.dataset.inference_dataset import load_chains_from_raw, read_inputs
from chai_lab.data.dataset.msas.colabfold import generate_colabfold_msas
from chai_lab.data.dataset.msas.load import get_msa_contexts
from chai_lab.data.dataset.msas.msa_context import MSAContext
from chai_lab.data.dataset.structure.all_atom_structure_context import (
    AllAtomStructureContext,
)
from chai_lab.data.dataset.structure.bond_utils import (
    get_atom_covalent_bond_pairs_from_constraints,
)
from chai_lab.data.dataset.templates.context import TemplateContext
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
from chai_lab.data.io.cif_utils import save_to_cif
from chai_lab.data.parsing.restraints import parse_pairwise_table
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
from chai_lab.model.utils import center_random_augmentation
from chai_lab.ranking.frames import get_frames_and_mask
from chai_lab.ranking.rank import SampleRanking, get_scores, rank
from chai_lab.utils.paths import chai1_component
from chai_lab.utils.plot import plot_msa
from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self
from chai_lab.utils.typing import Float, typecheck


class FeatureEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ATOM embeddings
        self.atom_embeddings = nn.ModuleDict({
            'AtomNameOneHot': AtomNameOneHot(),
            'AtomRefCharge': Identity(
                key="inputs/atom_ref_charge",
                ty=FeatureType.ATOM,
                dim=1,
                can_mask=False,
            ),
            'AtomRefElement': AtomElementOneHot(max_atomic_num=128),
            'AtomRefMask': Identity(
                key="inputs/atom_ref_mask",
                ty=FeatureType.ATOM,
                dim=1,
                can_mask=False,
            ),
            'AtomRefPos': RefPos()
        })
        
        # ATOM_PAIR embeddings
        self.atom_pair_embeddings = nn.ModuleDict({
            'BlockedAtomPairDistogram': BlockedAtomPairDistogram(),
            'InverseSquaredBlockedAtomPairDistances': BlockedAtomPairDistances(
                transform="inverse_squared",
                encoding_ty=EncodingType.IDENTITY,
            )
        })
        
        # TOKEN embeddings 
        self.token_embeddings = nn.ModuleDict({
            'ChainIsCropped': ChainIsCropped(),
            'ESMEmbeddings': ESMEmbeddings(),
            'IsDistillation': IsDistillation(),
            'MSADeletionMean': MSADeletionMeanGenerator(),
            'MSAProfile': MSAProfileGenerator(),
            'MissingChainContact': MissingChainContact(contact_threshold=6.0),
            'ResidueType': ResidueType(
                min_corrupt_prob=0.0,
                max_corrupt_prob=0.0,
                num_res_ty=32,
                key="token_residue_type",
            ),
            'TokenBFactor': TokenBFactor(include_prob=0.0),
            'TokenPLDDT': TokenPLDDT(include_prob=0.0)
        })
        
        # TOKEN_PAIR embeddings
        self.token_pair_embeddings = nn.ModuleDict({
            'DockingConstraintGenerator': DockingRestraintGenerator(
                include_probability=0.0,
                structure_dropout_prob=0.75,
                chain_dropout_prob=0.75,
            ),
            'RelativeChain': RelativeChain(),
            'RelativeEntity': RelativeEntity(),
            'RelativeSequenceSeparation': RelativeSequenceSeparation(sep_bins=None),
            'RelativeTokenSeparation': RelativeTokenSeparation(r_max=32),
            'TokenDistanceRestraint': TokenDistanceRestraint(
                include_probability=1.0,
                size=0.33,
                min_dist=6.0,
                max_dist=30.0,
                num_rbf_radii=6,
            ),
            'TokenPairPocketRestraint': TokenPairPocketRestraint(
                size=0.33,
                include_probability=1.0,
                min_dist=6.0,
                max_dist=20.0,
                coord_noise=0.0,
                num_rbf_radii=6,
            )
        })
        
        # MSA embeddings
        self.msa_embeddings = nn.ModuleDict({
            'IsPairedMSA': IsPairedMSAGenerator(),
            'MSADataSource': MSADataSourceGenerator(),
            'MSADeletionValue': MSADeletionValueGenerator(),
            'MSAHasDeletion': MSAHasDeletionGenerator(),
            'MSAOneHot': MSAFeatureGenerator()
        })
        
        # TEMPLATES embeddings
        self.template_embeddings = nn.ModuleDict({
            'TemplateDistogram': TemplateDistogramGenerator(),
            'TemplateMask': TemplateMaskGenerator(),
            'TemplateResType': TemplateResTypeGenerator(),
            'TemplateUnitVector': TemplateUnitVectorGenerator()
        })
        
        # Input projections
        self.input_projs = nn.ModuleDict({
            'ATOM': nn.Linear(395, 256),
            'ATOM_PAIR': nn.Linear(14, 32),
            'TOKEN': nn.Linear(2638, 384),
            'TOKEN_PAIR': nn.Linear(163, 512),
            'MSA': nn.Linear(42, 64),
            'TEMPLATES': nn.Linear(76, 64)
        })

    def forward(self, x: dict) -> dict:
        output = {}
        
        embedding_groups = {
            'ATOM': self.atom_embeddings,
            'ATOM_PAIR': self.atom_pair_embeddings,
            'TOKEN': self.token_embeddings,
            'TOKEN_PAIR': self.token_pair_embeddings,
            'MSA': self.msa_embeddings,
            'TEMPLATES': self.template_embeddings
        }
        
        for group_name, embedding_dict in embedding_groups.items():
            group_features = {}
            for key, embedding_layer in embedding_dict.items():
                if key in x:
                    group_features[key] = embedding_layer(x[key])
            
            if group_features:
                combined_features = torch.cat(list(group_features.values()), dim=-1)
                output[group_name] = self.input_projs[group_name](combined_features)
        
        return output


class BondLossInputProj(nn.Module):
    """Projects bond features into trunk and structure components"""
    def __init__(self):
        super().__init__()
        # Based on usage, output needs to be split into two halves (trunk_bond_feat and structure_bond_feat)
        # So output dimension is 512*2=1024
        self.weight = nn.Parameter(torch.zeros(512, 1024))
        self.bias = nn.Parameter(torch.zeros(1024))
        
    def forward(self, x):
        # Project and split into trunk and structure components
        projected = torch.matmul(x, self.weight) + self.bias
        return projected  # Will be chunked(2, dim=-1) when used
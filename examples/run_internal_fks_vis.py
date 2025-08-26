# NOTE: run this from the root directory of the repo

import logging
import shutil
from pathlib import Path

# from chai_lab.chai1 import run_inference
from chai_lab.chai1_internal_hack import run_inference  # [1] 
from boring_utils.utils import cprint, tprint

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

fasta_name = "8cyo"
fasta_context = """
>protein|8cyo-protein
MKKGHHHHHHGAISLISALVRAHVDSNPAMTSLDYSRFQANPDYQMSGDDTQHIQQFYDLLTGSMEIIRGWAEKIPGFADLPKADQDLLFESAFLELFVLRLAYRSNPVEGKLIFCNGVVLHRLQCVRGFGEWIDSIVEFSSNLQNMNIDISAFSCIAALAMVTERHGLKEPKRVEELQNKIVNTLKDHVTFNNGGLNRPNYLSKLLGKLPELRTLCTQGLQRIFYLKLEDLVPPPAIIDKLFLDTLPF
>ligand|8cyo-ligand
c1cc(c(cc1OCC(=O)NCCS)Cl)Cl
""".strip()

# Inference expects an empty directory; enforce this
# fasta_path = Path(f"examples/{fasta_name}.fasta")
fasta_path = Path(f"examples/Temp_{fasta_name}.fasta")
fasta_path.write_text(fasta_context)


output_dir = Path(f"outputs/output_inference_time_scaling_{fasta_name}_trajectory")
if output_dir.exists():
    logging.warning(f"Removing old output directory: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

# if you want to use ft steering:
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # constraint_path="./path_to_contact.restraints",
    num_trunk_recycles=1,
    num_diffn_timesteps=20,
    num_particles=2,        # number of diffusion paths
    resampling_interval=5,  # diffusion path length
    # num_diffn_timesteps=100,
    # num_particles=5,        # number of diffusion paths
    # resampling_interval=10,  # diffusion path length
    lambda_weight=10.0,     # lower this to, say 2.0, to make it more random
    # potential_type="diff",  # "diff" or "max" or "vanilla"
    potential_type="max",  # "diff" or "max" or "vanilla"
    num_diffn_samples=1,
    num_trunk_samples=1,  # new features in chai1 v0.6
    fk_sigma_threshold=float("inf"),  # disable fk sigma threshold
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    low_memory=True,
    # NOTE: new params
    enable_visualization=True,
)

# the default no inference time scaling version:
# set num_particles=1, but this is slower than the chai1.py for the extra iptm calculation
# candidates = run_inference(
#     fasta_file=fasta_path,
#     output_dir=output_dir,
#     # constraint_path="./path_to_contact.restraints",
#     num_trunk_recycles=1,
#     num_diffn_timesteps=10,
#     num_particles=1,
#     resampling_interval=1,
#     lambda_weight=100.0,
#     num_diffn_samples=1,
#     seed=42,
#     device="cuda:0",
#     use_esm_embeddings=True,
#     low_memory=True,
# )

# if you want to select the best particle every step:
# set resampling_interval=1 and lambda_weight=100.0
# candidates = run_inference(
#     fasta_file=fasta_path,
#     output_dir=output_dir,
#     # constraint_path="./path_to_contact.restraints",
#     num_trunk_recycles=1,
#     num_diffn_timesteps=10,
#     num_particles=2,
#     resampling_interval=1,
#     lambda_weight=100.0,
#     num_diffn_samples=1,
#     seed=42,
#     device="cuda:0",
#     use_esm_embeddings=True,
#     low_memory=True,
# )

# cprint(candidates)
# cif_paths = candidates.cif_paths
# agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

# Load pTM, ipTM, pLDDTs and clash scores for sample 2
# scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))

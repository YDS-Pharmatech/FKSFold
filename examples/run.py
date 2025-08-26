# NOTE: run this from the root directory of the repo

import logging
import shutil
from pathlib import Path

import numpy as np
from chai_lab.chai1 import run_inference
from boring_utils.utils import cprint, tprint

logging.basicConfig(level=logging.INFO)  # control verbosity

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

# Example given below, just modify it

fasta_name = "example"
fasta_context = """
>protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()

# fasta_name = "8cyo"
# fasta_context = """
# >protein|8cyo-protein
# MKKGHHHHHHGAISLISALVRAHVDSNPAMTSLDYSRFQANPDYQMSGDDTQHIQQFYDLLTGSMEIIRGWAEKIPGFADLPKADQDLLFESAFLELFVLRLAYRSNPVEGKLIFCNGVVLHRLQCVRGFGEWIDSIVEFSSNLQNMNIDISAFSCIAALAMVTERHGLKEPKRVEELQNKIVNTLKDHVTFNNGGLNRPNYLSKLLGKLPELRTLCTQGLQRIFYLKLEDLVPPPAIIDKLFLDTLPF
# >ligand|8cyo-ligand
# c1cc(c(cc1OCC(=O)NCCS)Cl)Cl
# """.strip()

fasta_path = Path(f"examples/Temp_{fasta_name}.fasta")
fasta_path.write_text(fasta_context)

# Inference expects an empty directory; enforce this
output_dir = Path(f"outputs/output_basic_{fasta_name}")
if output_dir.exists():
    logging.warning(f"Removing old output directory: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

# # default setup
# candidates = run_inference(
#     fasta_file=fasta_path,
#     output_dir=output_dir,
#     # 'default' setup
#     num_trunk_recycles=3,
#     num_diffn_timesteps=200,
#     num_trunk_samples=1,  # new features in chai1 v0.6
#     seed=42,
#     device="cuda:0",
#     use_esm_embeddings=True,
# )

# light setup
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    num_trunk_recycles=1,
    num_diffn_timesteps=10,
    num_diffn_samples=1,
    num_trunk_samples=1,  # new features in chai1 v0.6
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    low_memory=True,
)

# cif_paths = candidates.cif_paths
# agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

# # Load pTM, ipTM, pLDDTs and clash scores for sample 0
# scores = np.load(output_dir.joinpath("scores.model_idx_0.npz"))

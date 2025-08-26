# NOTE: run this from the root directory of the repo
# Example script for trajectory recording during diffusion process

import logging
import shutil
import gzip
import re
import os
from pathlib import Path

# Set environment variables for better CUDA error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# from chai_lab.chai1 import run_inference
from chai_lab.chai1_internal_hack import run_inference  # Enhanced version with trajectory recording
from boring_utils.utils import cprint, tprint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

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

# Inference expects an empty directory; enforce this
fasta_path = Path(f"examples/Temp_{fasta_name}.fasta")
fasta_path.write_text(fasta_context)

output_dir = Path(f"outputs/output_trajectory_recording_{fasta_name}")
if output_dir.exists():
    logging.warning(f"Removing old output directory: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

print("🚀 Running inference with trajectory recording enabled...")
print(f"📁 Output directory: {output_dir}")
print(f"📊 Trajectory data will be saved to: {output_dir}/trajectory_recording/")
print("📊 pLDDT computation is enabled")

# Run inference with trajectory recording enabled
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    
    # Basic inference parameters
    num_trunk_recycles=1,
    num_diffn_timesteps=100,        # Reduced steps for testing
    num_particles=2,               # Reduced particles for testing
    resampling_interval=5,         # Resample every 5 steps
    lambda_weight=10.0,
    potential_type="vanilla",      # Can also try "diff" or "max"
    num_diffn_samples=1,
    num_trunk_samples=1,
    fk_sigma_threshold=float("inf"),  # Disable fk sigma threshold
    seed=42,
    device="cuda:0",
    use_esm_embeddings=True,
    low_memory=True,
    
    # Visualization (existing feature)
    enable_visualization=True,
    
    # 🎯 NEW: Trajectory recording parameters
    enable_trajectory_recording=True,        # Enable trajectory recording
    trajectory_save_coordinates=True,        # Save atom coordinates
    trajectory_compute_plddt=True,           # Compute and save pLDDT scores
    trajectory_extra_save_interval=3,       # Save extra coordinates every 3 steps
)


def print_structure_candidates_summary(candidates):
    """Print a formatted summary of structure candidates without pae/pde/plddt data"""
    tprint("Structure Prediction Results Summary")
    
    print(f"Number of candidates: {len(candidates.cif_paths)}")
    print(f"Output files generated: {len(candidates.cif_paths)} CIF files")
    
    if candidates.msa_coverage_plot_path:
        print(f"MSA coverage plot: {candidates.msa_coverage_plot_path}")
    else:
        print("MSA coverage plot: Not available")
    
    print("\nCandidate Details:")
    print("-" * 50)
    
    for idx, (cif_path, ranking) in enumerate(zip(candidates.cif_paths, candidates.ranking_data)):
        print(f"\nCandidate {idx + 1}:")
        print(f"  CIF file: {cif_path}")
        print(f"  Aggregate score: {ranking.aggregate_score.item():.4f}")
        
        # PTM scores
        print(f"  Complex PTM: {ranking.ptm_scores.complex_ptm.item():.4f}")
        print(f"  Interface PTM: {ranking.ptm_scores.interface_ptm.item():.4f}")
        print(f"  Mean interface PTM: {ranking.ptm_scores.mean_interface_ptm.item():.4f}")
        print(f"  Protein mean interface PTM: {ranking.ptm_scores.protein_mean_interface_ptm.item():.4f}")
        
        # Chain information
        asym_ids = ranking.asym_ids.tolist()
        print(f"  Chain IDs: {asym_ids}")
        
        # Per-chain PTM scores - handle tensor dimensions properly
        per_chain_ptm = ranking.ptm_scores.per_chain_ptm.flatten().tolist()
        for chain_idx, asym_id in enumerate(asym_ids):
            if chain_idx < len(per_chain_ptm):
                ptm_value = per_chain_ptm[chain_idx]
                print(f"    Chain {asym_id} PTM: {ptm_value:.4f}")
        
        # Clash information
        print(f"  Total clashes: {ranking.clash_scores.total_clashes.item()}")
        print(f"  Inter-chain clashes: {ranking.clash_scores.total_inter_chain_clashes.item()}")
        print(f"  Has inter-chain clashes: {ranking.clash_scores.has_inter_chain_clashes.item()}")
        
        # PLDDT summary (without full tensor)
        print(f"  Complex PLDDT: {ranking.plddt_scores.complex_plddt.item():.4f}")
        per_chain_plddt = ranking.plddt_scores.per_chain_plddt.flatten().tolist()
        for chain_idx, asym_id in enumerate(asym_ids):
            if chain_idx < len(per_chain_plddt):
                plddt_value = per_chain_plddt[chain_idx]
                print(f"    Chain {asym_id} PLDDT: {plddt_value:.4f}")
    
    print("\n" + "=" * 80)


# Print results
print_structure_candidates_summary(candidates)

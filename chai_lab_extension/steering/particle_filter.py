import logging
from pathlib import Path
from typing import Optional

import torch

from chai_lab_extension.steering.base import PotentialType, ParticleState
from chai_lab_extension.steering.vis_mixin import VisualizationMixin
from chai_lab_extension.steering.scoring import ScoringFunction
from chai_lab_extension.steering.trajector_streamer import DiffusionTrajectoryRecorder
from chai_lab.data.sources.pdb import load_pdb_coordinates

from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG, DEV


class ParticleFilter:
    def __init__(
        self,
        num_particles: int,
        scoring_function: ScoringFunction,
        resampling_interval: int = 5,
        lambda_weight: float = 10.0,
        potential_type: PotentialType = PotentialType.VANILLA,
        fk_sigma_threshold: float = 1.0,
        enable_visualization: bool = False,
        visualization_output_dir: Optional[Path] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.enable_visualization = enable_visualization
        if enable_visualization:
            self.visualizer = VisualizationMixin(output_dir=visualization_output_dir)
        
        self.num_particles = num_particles
        self.resampling_interval = resampling_interval
        self.lambda_weight = lambda_weight
        self.potential_type = potential_type
        self.restraint_sigma_threshold = fk_sigma_threshold
        self.device = device
        self.particles: list[ParticleState] = []
        self.scoring_function = scoring_function
        
        # Trajectory recorder for coordinate and pLDDT tracking
        self.trajectory_recorder: Optional[DiffusionTrajectoryRecorder] = None
    
    def set_trajectory_recorder(self, recorder: DiffusionTrajectoryRecorder) -> None:
        """Set trajectory recorder for coordinate and pLDDT tracking"""
        self.trajectory_recorder = recorder
        # Integrate with visualization if enabled
        if self.enable_visualization:
            recorder.set_visualization_mixin(self.visualizer)
        
    def initialize_particles(
        self, 
        batch_size: int, 
        num_atoms: int, 
        sigma: float, 
        device: torch.device,
        pdb_path: Optional[Path] = None,
        structure_context = None,
        use_pdb_for_all_particles: bool = False,
        pdb_noise_scale: float = 0.1,
        base_seed: Optional[int] = None
    ):
        """
        Initialize particles with different random noise or PDB coordinates.
        NOTE: num_atoms could be 23 * 768 = 17664
        """

        self.particles = []
        logging.info(f"Initializing particles with num_atoms: {num_atoms}")
        print(f"[DEBUG] ParticleFilter initialization:")
        print(f"  batch_size: {batch_size}, num_atoms: {num_atoms}")
        print(f"  pdb_path: {pdb_path}")
        print(f"  use_pdb_for_all_particles: {use_pdb_for_all_particles}")
        print(f"  pdb_noise_scale: {pdb_noise_scale}")
        
        # Prefer to use PDB initialization coordinates from structure_context if available
        if structure_context is not None and structure_context.use_pdb_init and structure_context.atom_init_coords is not None:
            print(f"[DEBUG] Using PDB coords from structure_context:")
            print(f"  atom_init_coords shape: {structure_context.atom_init_coords.shape}")
            print(f"  PDB source: {structure_context.pdb_source_path}")
            print(f"  Configured noise scale: {structure_context.pdb_init_noise_scale}")
            
            for particle_idx in range(self.num_particles):
                # Set a different random seed for each particle
                if base_seed is not None:
                    torch.manual_seed(base_seed + particle_idx * 1000)
                
                # Get initial coordinates from structure_context
                initial_atom_pos = structure_context.get_initial_coords_for_diffusion(
                    sigma=sigma,
                    device=device,
                    add_noise=(particle_idx > 0 or structure_context.pdb_init_noise_scale > 0),
                    particle_idx=particle_idx,
                    target_num_atoms=num_atoms  # Pass target number of atoms to ensure dimension match
                )
                
                particle = ParticleState(atom_pos=initial_atom_pos)
                self.particles.append(particle)
                
                print(f"[DEBUG] Particle {particle_idx}: shape {initial_atom_pos.shape}")
        
        # If no initialization coordinates from structure_context, fallback to original logic
        else:
            # Try to load coordinates from PDB file
            pdb_coords = None
            if pdb_path is not None and structure_context is not None:
                print(f"[DEBUG] Fallback: loading PDB coords from file {pdb_path}")
                
                # Load PDB coordinates (returns coordinates matching structure_context dimensions)
                structure_pdb_coords = load_pdb_coordinates(
                    pdb_path=pdb_path,
                    structure_context=structure_context,
                    device=device,
                    center_coords=True,
                    add_noise=0.0,
                    target_num_atoms=num_atoms
                )
                
                if structure_pdb_coords is not None:
                    logging.info(f"Successfully loaded PDB coordinates from {pdb_path}")
                    logging.info(f"Structure PDB coordinates shape: {structure_pdb_coords.shape}")
                    
                    # load_pdb_coordinates now automatically handles dimension matching, use returned coordinates directly
                    pdb_coords = structure_pdb_coords
                    logging.info(f"Final PDB coordinates shape: {pdb_coords.shape}")
                else:
                    logging.warning(f"Failed to load PDB coordinates from {pdb_path}, falling back to random initialization")
            
            for particle_idx in range(self.num_particles):
                # Set a different random seed for each particle
                if base_seed is not None:
                    torch.manual_seed(base_seed + particle_idx * 1000)
                
                if pdb_coords is not None and (use_pdb_for_all_particles or particle_idx == 0):
                    # Use PDB coordinates for initialization (first particle or all particles)
                    initial_atom_pos = pdb_coords.clone()
                    
                    # Add different levels of noise for different particles
                    if particle_idx > 0 or pdb_noise_scale > 0:
                        noise_scale = pdb_noise_scale * (1.0 + 0.5 * particle_idx)  # Increasing noise
                        noise = noise_scale * torch.randn_like(initial_atom_pos)
                        initial_atom_pos += noise
                        
                    logging.info(f"Initialized particle {particle_idx} with PDB coordinates (noise scale: {noise_scale if particle_idx > 0 or pdb_noise_scale > 0 else 0.0})")
                    logging.info(f"Particle {particle_idx} atom_pos shape: {initial_atom_pos.shape}")
                else:
                    # Use random noise for initialization (default)
                    initial_atom_pos = sigma * torch.randn(
                        batch_size, num_atoms, 3, device=device
                    )
                    logging.info(f"Initialized particle {particle_idx} with random noise (sigma: {sigma}, seed offset: {particle_idx * 1000 if base_seed else 'None'})")
                    logging.info(f"Particle {particle_idx} atom_pos shape: {initial_atom_pos.shape}")
                
                particle = ParticleState(atom_pos=initial_atom_pos)
                self.particles.append(particle)
            
        if self.enable_visualization:
            self.visualizer.initialize_trajectories(self.num_particles)
            self.visualizer.record_step(0, sigma.item(), self.particles)
            
    def calculate_custom_score(self, particle: ParticleState) -> float:
        """Calculate score using custom scoring function"""
        return self.scoring_function.calculate_score(
            particle_pos=particle.atom_pos,
            device=self.device,
            particle_state=particle  # Pass particle_state for use in DefaultScoringFunction
        )

    def should_resample(self, step_idx: int, sigma_next: float) -> bool:
        return (step_idx > 0 and 
                step_idx % self.resampling_interval == 0 and 
                sigma_next < self.restraint_sigma_threshold)
    
    def resample(self, step_idx: int) -> None:
        """Resample particles based on their scores"""
        # Calculate custom scores for all particles
        custom_scores = []
        for particle in self.particles:
            score = self.calculate_custom_score(particle)
            custom_scores.append(score)
            # Also update the custom_score attribute of the particle
            particle.custom_score = score
        
        if not custom_scores:
            logging.warning(f"No custom scores found, skipping resampling")
            return

        if DEBUG: cprint(f"Resampling at step {step_idx} with custom scores: {custom_scores}")

        # Get current scores (always use custom scores since we always have a scoring function)
        current_scores = torch.tensor(custom_scores, device=self.device)
        
        # Get historical scores (if exists)
        historical_scores = torch.tensor([
            p.historical_score if p.historical_score is not None else 0.0
            for p in self.particles
        ], device=self.device)

        # Calculate weights based on different potential types
        if self.potential_type == PotentialType.VANILLA:
            # Vanilla method
            weights = torch.exp(self.lambda_weight * current_scores)
        elif self.potential_type == PotentialType.DIFF:
            # Diff method - make sure diffs are not all 0
            diffs = current_scores - historical_scores
            weights = torch.exp(self.lambda_weight * diffs + 1e-6)
        elif self.potential_type == PotentialType.MAX:
            # Max method
            weights = torch.exp(self.lambda_weight * torch.max(current_scores, historical_scores))
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")
        weights = weights.clamp(min=1e-6)
        
        # Calculate sampling probabilities
        probs = weights / weights.sum()
        
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logging.warning(f"Invalid probabilities detected: {probs}")
            probs = torch.ones_like(probs) / len(probs)
        
        # Multinomial sampling with lambda weight
        indices = torch.multinomial(probs, num_samples=self.num_particles, replacement=True)
        
        resampling_mapping = {i: indices[i].item() for i in range(self.num_particles)}
        
        # Record resampling snapshot BEFORE resampling
        if self.trajectory_recorder:
            self.trajectory_recorder.record_resampling_snapshot(step_idx, self.particles, resampling_mapping)
        
        if self.enable_visualization:
            self.visualizer.record_resampling(step_idx, resampling_mapping)
        
        # Create new particle list based on "indices" sampling
        new_particles = []
        for idx in indices:
            # Create a new particle with copied state
            new_particle = ParticleState(
                atom_pos=self.particles[idx].atom_pos.clone(),
                custom_score=self.particles[idx].custom_score,
                historical_score=self.particles[idx].custom_score,  # update historical score to current custom score
                # Copy other fields if they exist
                plddt=self.particles[idx].plddt.clone() if self.particles[idx].plddt is not None else None,
                interface_ptm=self.particles[idx].interface_ptm.clone() if self.particles[idx].interface_ptm is not None else None,
                avg_interface_ptm=self.particles[idx].avg_interface_ptm,
            )
            new_particles.append(new_particle)
            
        self.particles = new_particles

    def get_best_particle(self) -> ParticleState:
        """Return the particle with highest score"""
        def get_score(particle: ParticleState) -> float:
            if particle.custom_score is not None:
                return particle.custom_score
            else:
                return 0.0
        
        best_idx = max(range(len(self.particles)), key=lambda i: get_score(self.particles[i]))
        return self.particles[best_idx]
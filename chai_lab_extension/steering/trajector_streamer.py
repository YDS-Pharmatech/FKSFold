from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import torch
from torch import Tensor
import logging
import numpy as np

from .base import TrajectoryPoint, ParticleTrajectory, ParticleState


class DiffusionTrajectoryRecorder:
    """Diffusion trajectory recorder with coordinate and pLDDT saving capabilities"""
    
    def __init__(self,
                 output_dir: Path,
                 save_coordinates: bool = True,
                 compute_plddt: bool = False,
                 extra_save_interval: Optional[int] = None,
                 confidence_head = None,
                 confidence_context: Optional[Dict] = None):
        """
        Initialize diffusion trajectory recorder
        
        Args:
            output_dir: Directory to save trajectory data
            save_coordinates: Whether to save coordinates
            compute_plddt: Whether to compute and save pLDDT scores
            extra_save_interval: Interval for extra coordinate saves (e.g., every 3 steps)
            confidence_head: Model component for confidence calculation
            confidence_context: Context data needed for confidence calculation
        """
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_coordinates = save_coordinates
        self.compute_plddt = compute_plddt
        self.extra_save_interval = extra_save_interval
        
        # pLDDT computation setup
        self.confidence_head = confidence_head
        self.confidence_context = confidence_context or {}
        
        # Data storage
        self.trajectories: Dict[int, ParticleTrajectory] = {}
        self.resampling_events: List[Dict] = []
        
        # Integration with visualization
        self.vis_mixin = None
        
        logging.info(f"DiffusionTrajectoryRecorder initialized: {self.output_dir}")
    
    def set_visualization_mixin(self, vis_mixin) -> None:
        """Integrate with visualization system"""
        self.vis_mixin = vis_mixin
        
    def record_step(self, 
                   step_idx: int, 
                   sigma: float, 
                   particles: List[ParticleState],
                   is_extra_save: bool = False) -> None:
        """Record current step state"""
        
        for i, particle in enumerate(particles):
            # Decide whether to save coordinates
            save_coords = self.save_coordinates and (
                is_extra_save or 
                (self.extra_save_interval and step_idx % self.extra_save_interval == 0)
            )
            
            # Compute pLDDT if requested and coordinates are being saved
            plddt_scores = None
            confidence_scores = None
            
            if self.compute_plddt and save_coords and self.confidence_head:
                try:
                    plddt_scores, confidence_scores = self._compute_confidence_scores(particle.atom_pos)
                except Exception as e:
                    logging.warning(f"Failed to compute confidence scores at step {step_idx}: {e}")
            
            # Create trajectory point - handle potential CUDA errors
            try:
                # Clone coordinates safely
                atom_pos_clone = None
                if save_coords and particle.atom_pos is not None:
                    # Ensure tensor is on CPU before cloning to avoid CUDA errors
                    if particle.atom_pos.is_cuda:
                        # Check for CUDA errors before cloning
                        torch.cuda.synchronize()
                        atom_pos_clone = particle.atom_pos.detach().cpu().clone()
                    else:
                        atom_pos_clone = particle.atom_pos.clone()
                
                point = TrajectoryPoint(
                    step=step_idx,
                    sigma=sigma,
                    score=particle.custom_score,
                    atom_pos=atom_pos_clone,
                    plddt_scores=plddt_scores,
                    confidence_scores=confidence_scores,
                    is_extra_saved_point=is_extra_save
                )
                
                # Store trajectory point
                if i not in self.trajectories:
                    self.trajectories[i] = ParticleTrajectory(particle_id=i)
                
                self.trajectories[i].points.append(point)
                
            except Exception as e:
                logging.error(f"Failed to record step {step_idx} for particle {i}: {e}")
                # Continue with other particles even if one fails
                continue
        
        # Integrate with visualization mixin
        if self.vis_mixin:
            try:
                self.vis_mixin.record_step(step_idx, sigma, particles)
            except Exception as e:
                logging.warning(f"Visualization recording failed at step {step_idx}: {e}")
    
    def record_resampling_snapshot(self,
                                  step_idx: int,
                                  particles: List[ParticleState],
                                  resampling_mapping: Dict[int, int]) -> None:
        """Record snapshot before resampling - this is triggered by resample events"""
        
        for i, particle in enumerate(particles):
            # Always save coordinates for resampling points if coordinate saving is enabled
            plddt_scores = None
            confidence_scores = None
            
            if self.compute_plddt and self.confidence_head:
                try:
                    plddt_scores, confidence_scores = self._compute_confidence_scores(particle.atom_pos)
                except Exception as e:
                    logging.warning(f"Failed to compute confidence scores for resampling at step {step_idx}: {e}")
            
            try:
                # Clone coordinates safely for resampling points
                atom_pos_clone = None
                if self.save_coordinates and particle.atom_pos is not None:
                    if particle.atom_pos.is_cuda:
                        torch.cuda.synchronize()
                        atom_pos_clone = particle.atom_pos.detach().cpu().clone()
                    else:
                        atom_pos_clone = particle.atom_pos.clone()
                
                # Create resampling point
                resampling_point = TrajectoryPoint(
                    step=step_idx,
                    sigma=0.0,  # Resampling doesn't involve sigma
                    score=particle.custom_score,
                    atom_pos=atom_pos_clone,
                    plddt_scores=plddt_scores,
                    confidence_scores=confidence_scores,
                    is_resampling_point=True
                )
                
                # Store
                if i not in self.trajectories:
                    self.trajectories[i] = ParticleTrajectory(particle_id=i)
                
                self.trajectories[i].points.append(resampling_point)
                
            except Exception as e:
                logging.error(f"Failed to record resampling snapshot for particle {i} at step {step_idx}: {e}")
                continue
        
        # Record resampling event
        self.resampling_events.append({
            'step': step_idx,
            'mapping': resampling_mapping
        })
        
        # Integrate with visualization mixin
        if self.vis_mixin:
            try:
                self.vis_mixin.record_resampling(step_idx, resampling_mapping)
            except Exception as e:
                logging.warning(f"Visualization resampling recording failed at step {step_idx}: {e}")
        
        logging.info(f"Recorded resampling snapshot at step {step_idx}: {resampling_mapping}")
    
    def _compute_confidence_scores(self, atom_pos: Tensor) -> tuple[Optional[Tensor], Optional[Dict[str, float]]]:
        """Compute pLDDT and other confidence scores with error handling"""
        if not self.confidence_head:
            return None, None
        
        try:
            # Import the confidence calculation function
            from chai_lab.chai1_internal_hack import calculate_final_confidence_scores
            
            # Ensure atom_pos is on the correct device and has valid data
            if atom_pos.is_cuda:
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            # Check for NaN or infinite values
            if torch.isnan(atom_pos).any() or torch.isinf(atom_pos).any():
                logging.warning("Invalid coordinates detected (NaN or inf), skipping confidence calculation")
                return None, None
            
            # Validate tensor shape
            if atom_pos.dim() != 3 or atom_pos.shape[-1] != 3:
                logging.warning(f"Invalid atom_pos shape: {atom_pos.shape}, expected [batch, atoms, 3]")
                return None, None
            
            # Add batch dimension if needed
            if atom_pos.shape[0] != 1:
                atom_pos_batch = atom_pos.unsqueeze(0)
            else:
                atom_pos_batch = atom_pos
            
            # Move to CPU for confidence calculation to avoid CUDA issues
            atom_pos_cpu = atom_pos_batch.detach().cpu()
            device = atom_pos.device
            
            plddt, ptm_scores, _ = calculate_final_confidence_scores(
                atom_pos_cpu.to(device),
                device=device,
                confidence_head=self.confidence_head,
                **self.confidence_context
            )
            
            # Extract confidence scores safely
            confidence_scores = {}
            try:
                if hasattr(ptm_scores, 'interface_ptm') and ptm_scores.interface_ptm is not None:
                    confidence_scores['interface_ptm'] = float(ptm_scores.interface_ptm.mean().item())
                if hasattr(ptm_scores, 'complex_ptm') and ptm_scores.complex_ptm is not None:
                    confidence_scores['complex_ptm'] = float(ptm_scores.complex_ptm.mean().item())
                if hasattr(ptm_scores, 'mean_interface_ptm') and ptm_scores.mean_interface_ptm is not None:
                    confidence_scores['mean_interface_ptm'] = float(ptm_scores.mean_interface_ptm.mean().item())
            except Exception as e:
                logging.warning(f"Failed to extract PTM scores: {e}")
            
            # Move pLDDT to CPU and remove batch dimension
            if plddt is not None:
                plddt_cpu = plddt.detach().cpu()
                if plddt_cpu.dim() > 1:
                    plddt_cpu = plddt_cpu.squeeze(0)
                return plddt_cpu, confidence_scores
            else:
                return None, confidence_scores
            
        except Exception as e:
            logging.error(f"Error computing confidence scores: {e}")
            return None, None
    
    def save_trajectories(self) -> None:
        """Save trajectory data to files"""
        
        # Save coordinates data
        if self.save_coordinates:
            coords_dir = self.output_dir / "coordinates"
            coords_dir.mkdir(exist_ok=True)
            
            for particle_id, trajectory in self.trajectories.items():
                # Save coordinates for points that have them
                coord_points = trajectory.get_points_with_coordinates()
                if coord_points:
                    coords_data = {
                        'particle_id': particle_id,
                        'coordinates': [],
                        'metadata': []
                    }
                    
                    for point in coord_points:
                        if point.atom_pos is not None:
                            # Ensure coordinates are numpy arrays
                            if isinstance(point.atom_pos, torch.Tensor):
                                coords_np = point.atom_pos.detach().cpu().numpy()
                            else:
                                coords_np = np.array(point.atom_pos)
                            
                            coords_data['coordinates'].append(coords_np)
                            coords_data['metadata'].append({
                                'step': point.step,
                                'sigma': point.sigma,
                                'score': point.score,
                                'is_resampling_point': point.is_resampling_point,
                                'is_extra_saved_point': point.is_extra_saved_point,
                                'confidence_scores': point.confidence_scores
                            })
                    
                    # Save as npz file
                    if coords_data['coordinates']:
                        coord_file = coords_dir / f"particle_{particle_id}_coords.npz"
                        np.savez(coord_file, 
                                coordinates=np.array(coords_data['coordinates']),
                                metadata=coords_data['metadata'])
        
        # Save pLDDT data
        if self.compute_plddt:
            plddt_dir = self.output_dir / "plddt"
            plddt_dir.mkdir(exist_ok=True)
            
            for particle_id, trajectory in self.trajectories.items():
                plddt_data = []
                metadata = []
                
                for point in trajectory.points:
                    if point.plddt_scores is not None:
                        # Ensure pLDDT scores are numpy arrays
                        if isinstance(point.plddt_scores, torch.Tensor):
                            plddt_np = point.plddt_scores.detach().cpu().numpy()
                        else:
                            plddt_np = np.array(point.plddt_scores)
                        
                        plddt_data.append(plddt_np)
                        metadata.append({
                            'step': point.step,
                            'sigma': point.sigma,
                            'score': point.score,
                            'is_resampling_point': point.is_resampling_point,
                            'is_extra_saved_point': point.is_extra_saved_point
                        })
                
                if plddt_data:
                    plddt_file = plddt_dir / f"particle_{particle_id}_plddt.npz"
                    np.savez(plddt_file, 
                            plddt_scores=np.array(plddt_data),
                            metadata=metadata)
        
        # Save resampling events
        if self.resampling_events:
            resampling_file = self.output_dir / "resampling_events.json"
            with open(resampling_file, 'w') as f:
                json.dump(self.resampling_events, f, indent=2)
        
        # Save trajectory summary
        summary = self.get_trajectory_summary()
        summary_file = self.output_dir / "trajectory_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Saved trajectories for {len(self.trajectories)} particles to {self.output_dir}")
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of recorded trajectories"""
        total_points = sum(len(traj.points) for traj in self.trajectories.values())
        resampling_points = sum(len(traj.get_resampling_points()) for traj in self.trajectories.values())
        extra_saved_points = sum(len(traj.get_extra_saved_points()) for traj in self.trajectories.values())
        coord_points = sum(len(traj.get_points_with_coordinates()) for traj in self.trajectories.values())
        
        return {
            'num_particles': len(self.trajectories),
            'total_points': total_points,
            'resampling_points': resampling_points,
            'extra_saved_points': extra_saved_points,
            'points_with_coordinates': coord_points,
            'num_resampling_events': len(self.resampling_events),
            'save_coordinates': self.save_coordinates,
            'compute_plddt': self.compute_plddt,
            'extra_save_interval': self.extra_save_interval,
            'output_directory': str(self.output_dir)
        }
    
    def finalize(self) -> None:
        """Finalize recording and save all data"""
        try:
            self.save_trajectories()
            
            summary = self.get_trajectory_summary()
            logging.info(f"DiffusionTrajectoryRecorder finalized. "
                        f"Total points: {summary['total_points']}, "
                        f"Resampling points: {summary['resampling_points']}, "
                        f"Extra saved points: {summary['extra_saved_points']}")
        except Exception as e:
            logging.error(f"Failed to finalize trajectory recording: {e}")

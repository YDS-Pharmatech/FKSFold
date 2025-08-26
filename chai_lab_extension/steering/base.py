from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from torch import Tensor
from pathlib import Path


class PotentialType(Enum):
    VANILLA = "vanilla"  # vanilla method
    DIFF = "diff"  # diff method
    MAX = "max"  # max method
    
    @classmethod
    def from_str(cls, value: str) -> "PotentialType":
        """Convert string to PotentialType"""
        value = value.lower()
        if value == "default":
            value = "vanilla"
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"Invalid potential type: {value}. Must be one of {valid_values}")


@dataclass 
class ParticleState:
    atom_pos: Tensor
    custom_score: float | None = None  # Custom score for steering
    historical_score: float | None = None  # Historical score for DIFF and MAX potential types
    plddt: Tensor | None = None  # Only needed for final calculation
    interface_ptm: Tensor | None = None  # Only needed for final calculation
    avg_interface_ptm: float | None = None  # Only needed for DefaultScoringFunction


@dataclass
class TrajectoryPoint:
    """Record state at a specific time point during diffusion process"""
    step: int
    sigma: float
    score: Optional[float] = None
    
    # Enhanced fields for coordinate and confidence tracking
    atom_pos: Optional[Tensor] = None  # Coordinates [num_atoms, 3]
    plddt_scores: Optional[Tensor] = None  # pLDDT scores per atom
    confidence_scores: Optional[Dict[str, float]] = None  # PTM, interface PTM etc
    
    # Metadata flags
    is_resampling_point: bool = False
    is_extra_saved_point: bool = False
    
    def has_coordinates(self) -> bool:
        """Check if coordinates are available"""
        return self.atom_pos is not None


@dataclass
class ParticleTrajectory:
    """Record complete trajectory of a single particle"""
    particle_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)
    resampled_from: List[int] = field(default_factory=list)  # Record resampling history
    
    def get_resampling_points(self) -> List[TrajectoryPoint]:
        """Get all resampling points in this trajectory"""
        return [point for point in self.points if point.is_resampling_point]
    
    def get_extra_saved_points(self) -> List[TrajectoryPoint]:
        """Get all extra saved points in this trajectory"""
        return [point for point in self.points if point.is_extra_saved_point]
    
    def get_points_with_coordinates(self) -> List[TrajectoryPoint]:
        """Get all points that have coordinate data"""
        return [point for point in self.points if point.has_coordinates()]
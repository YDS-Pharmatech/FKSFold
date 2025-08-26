from typing import Optional, List, Dict, Tuple
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from chai_lab_extension.steering.base import ParticleTrajectory, TrajectoryPoint, ParticleState
from boring_utils.helpers import DEBUG


class VisualizationMixin:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.trajectories: Dict[int, ParticleTrajectory] = {}
        self.score_history: Dict[int, List[Tuple[int, float]]] = {}
        self.resampling_events: List[Tuple[int, Dict[int, int]]] = []  # [(step, {new_id: source_id})]
        
        # Experiment-specific data storage
        self.experiment_data: Dict[str, Dict] = {}  # {experiment_name: {score_history, resampling_events, trajectories}}
    
    def initialize_trajectories(self, num_particles: int, reset_score_history: bool = True):
        """Initialize all particle trajectories"""
        self.trajectories = {
            i: ParticleTrajectory(particle_id=i) 
            for i in range(num_particles)
        }
        if reset_score_history:
            self.score_history = {i: [] for i in range(num_particles)}
    
    def record_step(self, step_idx: int, sigma: float, particles: List[ParticleState]):
        """Record current step state"""
        for i, particle in enumerate(particles):
            # Record score (use custom score if available)
            if particle.custom_score is not None:
                self.score_history[i].append((step_idx, particle.custom_score))
            else:
                if step_idx > 0: print(f"particle {i} has no custom score at step {step_idx}, set to 0.0")
                self.score_history[i].append((step_idx, 0.0))
                
            # Create trajectory point
            point = TrajectoryPoint(
                step=step_idx,
                sigma=sigma,
                score=particle.custom_score if particle.custom_score is not None else 0.0
            )
                
            self.trajectories[i].points.append(point)
    
    def record_resampling(self, step_idx: int, old_to_new_mapping: Dict[int, int]):
        """Record resampling event"""
        # Record resampling history
        self.resampling_events.append((step_idx, old_to_new_mapping))
        
        # Update resampling source for each particle
        for new_idx, old_idx in old_to_new_mapping.items():
            self.trajectories[new_idx].resampled_from.append(old_idx)
    
    def load_from_csv(self, score_csv_path: str, resampling_csv_path: str, experiment_name: str = "default"):
        """Load trajectories and resampling data from CSV files
        
        Args:
            score_csv_path: Path to score trajectory CSV file
            resampling_csv_path: Path to resampling events CSV file
            experiment_name: Name of the experiment for labeling
        """
        # Initialize experiment data storage
        if experiment_name not in self.experiment_data:
            self.experiment_data[experiment_name] = {
                'score_history': {},
                'resampling_events': [],
                'trajectories': {}
            }
        
        # Load score trajectory data
        try:
            score_df = pd.read_csv(score_csv_path)
            if DEBUG:
                print(f"Loading score data for {experiment_name}: {len(score_df)} rows")
            
            # Convert to internal data structure
            experiment_score_history = {}
            for _, row in score_df.iterrows():
                pid = int(row["particle_id"])
                if pid not in experiment_score_history:
                    experiment_score_history[pid] = []
                experiment_score_history[pid].append((int(row["step"]), float(row["score"])))
            
            # Ensure all particle scores are sorted by step
            for pid in experiment_score_history:
                experiment_score_history[pid].sort(key=lambda x: x[0])

            # Store in experiment data
            self.experiment_data[experiment_name]['score_history'] = experiment_score_history
                
            if DEBUG:
                print(f"Processing score data for {experiment_name}: {len(experiment_score_history)} particles")
                for pid, scores in experiment_score_history.items():
                    print(f"  Particle {pid}: {len(scores)} data points")
            
        except Exception as e:
            print(f"Error loading score data for {experiment_name}: {e}")
        
        # Load resampling event data
        try:
            resampling_df = pd.read_csv(resampling_csv_path)
            if DEBUG:
                print(f"Loading resampling data for {experiment_name}: {len(resampling_df)} rows")
            
            # Convert to internal data structure
            event_dict = {}
            for _, row in resampling_df.iterrows():
                event_id = int(row["event_id"])
                step = int(row["step"])
                new_pid = int(row["new_particle"])
                source_pid = int(row["source_particle"])
                
                if event_id not in event_dict:
                    event_dict[event_id] = (step, {})
                
                event_dict[event_id][1][new_pid] = source_pid
            
            experiment_resampling_events = [(step, mapping) for _, (step, mapping) in sorted(event_dict.items())]
            
            # Store in experiment data
            self.experiment_data[experiment_name]['resampling_events'] = experiment_resampling_events
            
            if DEBUG:
                print(f"Processing resampling events for {experiment_name}: {len(experiment_resampling_events)} events")
                for i, (step, mapping) in enumerate(experiment_resampling_events):
                    print(f"  Event {i} (Step {step}): {mapping}")
                    
        except Exception as e:
            print(f"Error loading resampling data for {experiment_name}: {e}")
            
        # Create virtual trajectory points for each particle (for visualization only, no actual coordinates)
        experiment_trajectories = {}
        for pid in experiment_score_history:
            experiment_trajectories[pid] = ParticleTrajectory(particle_id=pid)
            for step, score in experiment_score_history[pid]:
                point = TrajectoryPoint(step=step, sigma=0.0, score=score)
                experiment_trajectories[pid].points.append(point)
            # Ensure trajectory points are sorted by step
            experiment_trajectories[pid].points.sort(key=lambda p: p.step)
        
        # Store trajectories in experiment data
        self.experiment_data[experiment_name]['trajectories'] = experiment_trajectories
        
        # For backward compatibility, also update the main data structures with the latest experiment
        self.score_history = experiment_score_history
        self.resampling_events = experiment_resampling_events
        self.trajectories = experiment_trajectories

    def plot_score_trajectories(self, title: str = "Particle score trajectories") -> Figure:
        """Plot all particle score trajectories with segments based on resampling events"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate maximum step value
        max_step = 0
        for pid, scores in self.score_history.items():
            if scores:
                max_step = max(max_step, max(step for step, _ in scores))
        
        # Initialize segments
        next_segment_id = 0
        segments = {}  # Store segment ID -> list of data points
        segment_particles = {}  # Store segment ID -> particle ID
        segment_sources = {}  # Store segment ID -> source segment ID/particle ID
        particle_to_segment = {}  # Store (step, particle_id) -> segment ID
        
        # Debug output of available data
        print(f"Score history: {len(self.score_history)} particles, max step: {max_step}")
        for pid, scores in self.score_history.items():
            print(f"  Particle {pid}: {len(scores)} points, steps {min(s for s, _ in scores)}-{max(s for s, _ in scores)}")
        print(f"Resampling events: {len(self.resampling_events)}")
        for i, (step, mapping) in enumerate(self.resampling_events):
            print(f"  Event {i}: Step {step}, Mapping {mapping}")
        
        # Sort resampling events (by step)
        resampling_steps = sorted(set(step for step, _ in self.resampling_events))
        
        # Create initial segments (from step 0 to first resampling)
        for pid in self.score_history.keys():
            segment_id = f"segment_{next_segment_id}"
            next_segment_id += 1
            segments[segment_id] = []
            segment_particles[segment_id] = pid
            segment_sources[segment_id] = f"init_{pid}"
            
            # Record segment ID for this particle in initial phase
            if not resampling_steps:
                # If no resampling events, map entire time period
                for step in range(0, max_step + 1):
                    particle_to_segment[(step, pid)] = segment_id
            else:
                # Map until first resampling
                first_resample = resampling_steps[0]
                for step in range(0, first_resample + 1):
                    particle_to_segment[(step, pid)] = segment_id
        
        print(f"Initial segments: {len(segments)}")
        
        # Process each resampling event
        for i, resample_step in enumerate(resampling_steps):
            print(f"Processing resampling at step {resample_step} (event {i+1}/{len(resampling_steps)})")
            
            # Get resampling mapping for this step {new_pid: source_pid}
            mapping = {}
            for event_step, event_mapping in self.resampling_events:
                if event_step == resample_step:
                    mapping = event_mapping
                    break
            
            if not mapping:
                print(f"  No mapping found for step {resample_step}!")
                continue
            
            print(f"  Mapping: {mapping}")
            
            # Calculate time period end
            if i + 1 < len(resampling_steps):
                next_resample = resampling_steps[i+1]
            else:
                next_resample = max_step
            
            print(f"  Time range: {resample_step} to {next_resample}")
            
            # Create new segment ID for each particle after resampling
            for new_pid, source_pid in mapping.items():
                # Find current segment ID of source particle (before resampling)
                source_segment = None
                for step in range(resample_step-1, -1, -1):
                    if (step, source_pid) in particle_to_segment:
                        source_segment = particle_to_segment[(step, source_pid)]
                        break
                
                if not source_segment:
                    print(f"  Warning: Could not find source segment for particle {source_pid} at step {resample_step-1}")
                    continue
                
                # Always create new segment regardless of whether new and old particle IDs are same
                segment_id = f"segment_{next_segment_id}"
                next_segment_id += 1
                segments[segment_id] = []
                segment_particles[segment_id] = new_pid
                segment_sources[segment_id] = source_segment
                
                # Map segment ID for this particle during this time period
                for step in range(resample_step, next_resample + 1):
                    particle_to_segment[(step, new_pid)] = segment_id
                
                print(f"  Created {segment_id} for particle {new_pid} from {source_segment}")
        
        print(f"Total segments after resampling: {len(segments)}")
        
        # Collect data points for each segment
        for pid, scores in self.score_history.items():
            for step, score in scores:
                # Find segment ID corresponding to this (step, pid)
                segment_id = particle_to_segment.get((step, pid))
                if segment_id:
                    segments[segment_id].append((step, score))
                else:
                    print(f"  Warning: No segment found for particle {pid} at step {step}")
        
        # Sort data points in each segment by step
        for segment_id in segments:
            segments[segment_id].sort(key=lambda x: x[0])
        
        # Plot each segment and print info
        print("Segments with data:")
        segment_connections = {}  # Store connections between segments {segment_id: previous_segment_id}
        all_plot_data = {}  # Store all segment data {segment_id: (steps, scores)}

        # First collect all segment data and build connection relationships
        for segment_id, points in segments.items():
            if points:  # Ensure there are data points
                steps, scores = zip(*points)
                all_plot_data[segment_id] = (steps, scores)
                
                # Record connection relationship
                if segment_sources[segment_id].startswith("segment_"):
                    segment_connections[segment_id] = segment_sources[segment_id]
                
                # Print segment info
                start_step = min(steps)
                end_step = max(steps)
                particle_id = segment_particles[segment_id]
                source = segment_sources[segment_id]
                print(f"  {segment_id}: Particle {particle_id}, From {source}, Steps {start_step}-{end_step}, Points: {len(points)}")
            else:
                print(f"  {segment_id}: NO DATA for Particle {segment_particles[segment_id]}")

        # Group segments by particle ID to assign different colors for plotting
        particle_segments = {}
        for segment_id, particle_id in segment_particles.items():
            if particle_id not in particle_segments:
                particle_segments[particle_id] = []
            if segment_id in all_plot_data:
                particle_segments[particle_id].append(segment_id)

        # Assign a color to each particle
        colors = plt.cm.tab10(np.linspace(0, 1, len(particle_segments)))
        particle_colors = {pid: colors[i] for i, pid in enumerate(particle_segments.keys())}

        # Plot main segments
        for segment_id, (steps, scores) in all_plot_data.items():
            particle_id = segment_particles[segment_id]
            color = particle_colors[particle_id]
            
            # Plot segment points and lines
            ax.plot(steps, scores, '-o', color=color)
            
            # If there's a source segment, add connecting line
            if segment_id in segment_connections:
                source_id = segment_connections[segment_id]
                if source_id in all_plot_data:
                    # Get last point of source segment and first point of current segment
                    source_steps, source_scores = all_plot_data[source_id]
                    source_last_x, source_last_y = source_steps[-1], source_scores[-1]
                    current_first_x, current_first_y = steps[0], scores[0]
                    
                    # Plot connecting line (dashed)
                    ax.plot([source_last_x, current_first_x], [source_last_y, current_first_y], 
                            '--', color=color, alpha=0.5)

        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys())
        # Add legend item for each particle ID
        legend_handles = []
        legend_labels = []
        for pid, color in particle_colors.items():
            # Create Line2D object as legend item
            legend_handle = plt.Line2D([0], [0], color=color, marker='o', linestyle='-')
            legend_handles.append(legend_handle)
            legend_labels.append(f"Particle {pid}")

        # Only show particle ID legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels)
        
        ax.set_xlabel("Diffusion steps")
        ax.set_ylabel("Steering score")
        
        # Calculate min score and set y-axis lower limit
        all_scores = []
        for _, (steps, scores) in all_plot_data.items():
            all_scores.extend(scores)

        if all_scores:
            # Filter out invalid values (inf, nan, zero)
            valid_scores = [score for score in all_scores if np.isfinite(score) and score != 0]
            
            if valid_scores:
                min_score = min(valid_scores)
                max_score = max(valid_scores)
                # Set lower limit with some margin
                score_range = max_score - min_score
                if score_range > 0:
                    bottom_limit = min_score - 0.2 * score_range
                else:
                    bottom_limit = min_score - 0.1  # Small margin if all scores are the same
                
                print(f"min score: {min_score}, max score: {max_score}, bottom limit: {bottom_limit}")
                
                # Additional safety check
                if np.isfinite(bottom_limit):
                    ax.set_ylim(bottom=bottom_limit)
                else:
                    ax.set_ylim(bottom=0.0)
            else:
                # If no valid data after filtering, use default
                print("No valid scores found, using default y-axis range")
                ax.set_ylim(bottom=0.0, top=1.0)
        else:
            # If no data, use default
            print("No score data found, using default y-axis range")
            ax.set_ylim(bottom=0.0, top=1.0)
        
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if self.output_dir:
            save_path = self.output_dir / "segmented_score_trajectories.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_score_trajectories_backup(self, title: str = "particle score trajectories") -> Figure:
        """Plot all particle score trajectories"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for particle_id, scores in self.score_history.items():
            if not scores:
                continue
                
            steps, values = zip(*scores)
            ax.plot(steps, values, '-o', label=f"particle {particle_id}")
            
            # Mark resampling points
            resampling_steps = [event[0] for event in self.resampling_events]
            for step in resampling_steps:
                if step in steps:
                    ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Diffusion steps")
        ax.set_ylabel("Steering score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if self.output_dir:
            save_path = self.output_dir / "score_trajectories.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_resampling_flow(self, title: str = "particle resampling flow") -> Figure:
        """Plot the resampling flow as a directed graph"""
        if not self.resampling_events:
            return None
            
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Create nodes for each particle
        for particle_id in self.trajectories:
            G.add_node(f"P{particle_id}")
        
        # Add resampling edges
        for step_idx, mapping in self.resampling_events:
            for new_idx, old_idx in mapping.items():
                if new_idx != old_idx:  # Only show edges where particle actually changed
                    G.add_edge(f"P{old_idx}", f"P{new_idx}", step=step_idx)
        
        # Draw graph
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        
        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos)
        
        # Draw edge labels (steps)
        edge_labels = {(u, v): f"Step {d['step']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        ax.set_title(title)
        plt.axis('off')
        
        if self.output_dir:
            save_path = self.output_dir / "resampling_flow.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def save_trajectory_data(self):
        """Save trajectory data as numpy files for later analysis"""
        if not self.output_dir:
            print("no output directory")
            return
        
        # Save score history
        scores_data = {
            f"particle_{pid}": np.array(scores) 
            for pid, scores in self.score_history.items() if scores
        }
        
        if scores_data:
            np.savez(self.output_dir / "score_trajectories.npz", **scores_data)
        
        # Save resampling events
        resampling_data = {
            f"event_{i}": {
                "step": event[0],
                "mapping": list(event[1].items())
            }
            for i, event in enumerate(self.resampling_events)
        }
        
        if resampling_data:
            import json
            with open(self.output_dir / "resampling_events.json", "w") as f:
                json.dump(resampling_data, f, indent=2)

    def save_trajectory_data_as_csv(self):
        """Save trajectory data as CSV format"""
        if not self.output_dir:
            print("no output directory")
            return
        
        # Save score history as CSV
        all_scores = []
        for particle_id, scores in self.score_history.items():
            for step, score in scores:
                all_scores.append({
                    "particle_id": particle_id,
                    "step": step,
                    "score": score
                })
        
        if all_scores:
            df = pd.DataFrame(all_scores)
            print(df)
            df.to_csv(self.output_dir / "score_trajectories.csv", index=False)
        
        # Save resampling events as CSV
        all_resamplings = []
        for event_idx, (step, mapping) in enumerate(self.resampling_events):
            for new_idx, old_idx in mapping.items():
                all_resamplings.append({
                    "event_id": event_idx,
                    "step": step,
                    "new_particle": new_idx,
                    "source_particle": old_idx
                })
        
        if all_resamplings:
            df = pd.DataFrame(all_resamplings)
            df.to_csv(self.output_dir / "resampling_events.csv", index=False)

    def plot_multiple_experiments(self, title: str = "Multiple experiment score trajectories") -> Figure:
        """Plot multiple experiments on the same graph with different colors and labels
        
        Returns:
            Figure: matplotlib figure with all experiments plotted
        """
        if len(self.experiment_data) < 2:
            print("Need at least 2 experiments to use plot_multiple_experiments")
            return self.plot_score_trajectories(title)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define custom colors for different experiments - purple and grey
        custom_colors = ['grey', 'purple', 'orange', 'green', 'pink']
        experiment_colors = custom_colors[:len(self.experiment_data)]
        # experiment_colors = plt.cm.Set1(np.linspace(0, 1, len(self.experiment_data)))
        
        # Calculate global maximum step value across all experiments
        max_step = 0
        for exp_name, exp_data in self.experiment_data.items():
            for pid, scores in exp_data['score_history'].items():
                if scores:
                    max_step = max(max_step, max(step for step, _ in scores))
        
        print(f"Plotting {len(self.experiment_data)} experiments, max step: {max_step}")
        
        # Process each experiment
        for exp_idx, (exp_name, exp_data) in enumerate(self.experiment_data.items()):
            print(f"Processing experiment: {exp_name}")
            
            # Get experiment color
            exp_color = experiment_colors[exp_idx]
            
            # Initialize segments for this experiment
            next_segment_id = 0
            segments = {}  # Store segment ID -> list of data points
            segment_particles = {}  # Store segment ID -> particle ID
            segment_sources = {}  # Store segment ID -> source segment ID/particle ID
            particle_to_segment = {}  # Store (step, particle_id) -> segment ID
            
            score_history = exp_data['score_history']
            resampling_events = exp_data['resampling_events']
            
            # Sort resampling events (by step)
            resampling_steps = sorted(set(step for step, _ in resampling_events))
            
            # Create initial segments (from step 0 to first resampling)
            for pid in score_history.keys():
                segment_id = f"{exp_name}_segment_{next_segment_id}"
                next_segment_id += 1
                segments[segment_id] = []
                segment_particles[segment_id] = pid
                segment_sources[segment_id] = f"{exp_name}_init_{pid}"
                
                # Record segment ID for this particle in initial phase
                if not resampling_steps:
                    # If no resampling events, map entire time period
                    for step in range(0, max_step + 1):
                        particle_to_segment[(step, pid)] = segment_id
                else:
                    # Map until first resampling
                    first_resample = resampling_steps[0]
                    for step in range(0, first_resample + 1):
                        particle_to_segment[(step, pid)] = segment_id
            
            # Process each resampling event for this experiment
            for i, resample_step in enumerate(resampling_steps):
                # Get resampling mapping for this step {new_pid: source_pid}
                mapping = {}
                for event_step, event_mapping in resampling_events:
                    if event_step == resample_step:
                        mapping = event_mapping
                        break
                
                if not mapping:
                    continue
                
                # Calculate time period end
                if i + 1 < len(resampling_steps):
                    next_resample = resampling_steps[i+1]
                else:
                    next_resample = max_step
                
                # Create new segment ID for each particle after resampling
                for new_pid, source_pid in mapping.items():
                    # Find current segment ID of source particle (before resampling)
                    source_segment = None
                    for step in range(resample_step-1, -1, -1):
                        if (step, source_pid) in particle_to_segment:
                            source_segment = particle_to_segment[(step, source_pid)]
                            break
                    
                    if not source_segment:
                        continue
                    
                    # Always create new segment regardless of whether new and old particle IDs are same
                    segment_id = f"{exp_name}_segment_{next_segment_id}"
                    next_segment_id += 1
                    segments[segment_id] = []
                    segment_particles[segment_id] = new_pid
                    segment_sources[segment_id] = source_segment
                    
                    # Map segment ID for this particle during this time period
                    for step in range(resample_step, next_resample + 1):
                        particle_to_segment[(step, new_pid)] = segment_id
            
            # Collect data points for each segment
            for pid, scores in score_history.items():
                for step, score in scores:
                    # Find segment ID corresponding to this (step, pid)
                    segment_id = particle_to_segment.get((step, pid))
                    if segment_id:
                        segments[segment_id].append((step, score))
            
            # Sort data points in each segment by step
            for segment_id in segments:
                segments[segment_id].sort(key=lambda x: x[0])
            
            # Plot segments for this experiment
            all_plot_data = {}  # Store all segment data {segment_id: (steps, scores)}
            
            # Collect all segment data
            for segment_id, points in segments.items():
                if points:  # Ensure there are data points
                    steps, scores = zip(*points)
                    all_plot_data[segment_id] = (steps, scores)
            
            # Plot main segments for this experiment
            for segment_id, (steps, scores) in all_plot_data.items():
                # Plot segment points and lines with experiment color - make lines thicker
                ax.plot(steps, scores, '-o', color=exp_color, alpha=0.7, markersize=4, linewidth=2.5)
                
                # Add connecting lines between segments if needed
                if segment_sources[segment_id].startswith(f"{exp_name}_segment_"):
                    source_id = segment_sources[segment_id]
                    if source_id in all_plot_data:
                        # Get last point of source segment and first point of current segment
                        source_steps, source_scores = all_plot_data[source_id]
                        source_last_x, source_last_y = source_steps[-1], source_scores[-1]
                        current_first_x, current_first_y = steps[0], scores[0]
                        
                        # Plot connecting line (dashed) - make lines thicker
                        ax.plot([source_last_x, current_first_x], [source_last_y, current_first_y], 
                                '--', color=exp_color, alpha=0.5, linewidth=2.0)
        
        # Create legend for experiments
        legend_handles = []
        legend_labels = []
        for exp_idx, exp_name in enumerate(self.experiment_data.keys()):
            exp_color = experiment_colors[exp_idx]
            # Create Line2D object as legend item - make lines thicker
            legend_handle = plt.Line2D([0], [0], color=exp_color, marker='o', linestyle='-', markersize=8, linewidth=3)
            legend_handles.append(legend_handle)
            legend_labels.append(f"{exp_name}")

        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=12)
        
        ax.set_xlabel("Diffusion steps", fontsize=14)
        ax.set_ylabel("Steering score", fontsize=14)
        
        # Calculate global min/max scores across all experiments
        all_scores = []
        for exp_data in self.experiment_data.values():
            for pid, scores in exp_data['score_history'].items():
                all_scores.extend([score for _, score in scores])

        if all_scores:
            # Filter out invalid values (inf, nan, zero)
            valid_scores = [score for score in all_scores if np.isfinite(score) and score != 0]
            
            if valid_scores:
                min_score = min(valid_scores)
                max_score = max(valid_scores)
                # Set lower limit with some margin
                score_range = max_score - min_score
                if score_range > 0:
                    bottom_limit = min_score - 0.2 * score_range
                else:
                    bottom_limit = min_score - 0.1  # Small margin if all scores are the same
                
                print(f"Global min score: {min_score}, max score: {max_score}, bottom limit: {bottom_limit}")
                
                # Additional safety check
                if np.isfinite(bottom_limit):
                    ax.set_ylim(bottom=bottom_limit)
                else:
                    ax.set_ylim(bottom=0.0)
            else:
                # If no valid data after filtering, use default
                print("No valid scores found, using default y-axis range")
                ax.set_ylim(bottom=0.0, top=1.0)
        else:
            # If no data, use default
            print("No score data found, using default y-axis range")
            ax.set_ylim(bottom=0.0, top=1.0)
        
        ax.set_title(title, fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Make tick labels bigger too
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        if self.output_dir:
            save_path = self.output_dir / "multiple_experiments_score_trajectories.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def test_plotting():
    """Example test for plot_score_trajectories function"""
    # Create temp directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # score_df = pd.read_csv("examples/fks_path/score_trajectories.csv")
        # resampling_df = pd.read_csv("examples/fks_path/resampling_events.csv")
        
        # Create visualization object
        viz = VisualizationMixin(output_dir=temp_path)
        
        # Load first experiment
        viz.load_from_csv(
            score_csv_path=str("examples/fks_path/score_trajectories.csv"),
            resampling_csv_path=str("examples/fks_path/resampling_events.csv"),
            experiment_name="experiment_1"
        )
        
        # Load second experiment (using same data for demonstration)
        viz.load_from_csv(
            score_csv_path=str("examples/fks_path/score_trajectories.csv"),
            resampling_csv_path=str("examples/fks_path/resampling_events.csv"),
            experiment_name="experiment_2"
        )
        
        # Draw single experiment plot
        fig1 = viz.plot_score_trajectories()
        
        # Save single experiment image
        output_path1 = Path("./temp_test_plot_single.png")
        fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
        print(f"Single experiment visualization saved to {output_path1.absolute()}")
        
        # Draw multiple experiments plot
        fig2 = viz.plot_multiple_experiments()
        
        # Save multiple experiments image
        output_path2 = Path("./temp_test_plot_multiple.png")
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Multiple experiments visualization saved to {output_path2.absolute()}")
        
        # Return true if test completed successfully
        return os.path.exists(output_path1) and os.path.exists(output_path2)
        
    return False


if __name__ == "__main__":
    test_plotting()

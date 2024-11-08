from dataclasses import dataclass
from typing import Optional, Tuple, List
import random
import time

from core.grid import Grid
from core.pattern import Pattern, PatternType
from .cost_function import CostFunction, CostWeights
from .temperature import TemperatureSchedule, TemperatureParameters
from .move_generator import MoveGenerator, MoveType
from output.formatter import OutputFormatter, ColorMapper
from .cluster import ClusterOptimizer, ClusterStage

@dataclass
class AnnealingStats:
    """Statistics from the annealing process."""
    initial_cost: float
    final_cost: float
    best_cost: float
    total_iterations: int
    accepted_moves: int
    rejected_moves: int
    execution_time: float
    temperature_stages: int

class PatternOptimizer:
    """
    Implements simulated annealing optimization for pattern layout.
    Attempts to find optimal arrangement of patterns while minimizing
    total patterns and especially single-pixel patterns.
    """

    def __init__(self,
                 cost_weights: Optional[CostWeights] = None,
                 temp_params: Optional[TemperatureParameters] = None,
                 color_mapper = None):
        """Initialize optimizer with optional custom parameters."""
        self.cost_function = CostFunction(weights=cost_weights)
        self.temperature_schedule = TemperatureSchedule(params=temp_params)
        self.color_mapper = color_mapper if color_mapper is not None else ColorMapper()
        self.stagnation_threshold = 3  # Stages without improvement
        self.cluster_optimizer = None
        self.in_cluster_mode = False

    def optimize(self, grid: Grid) -> Tuple[Grid, AnnealingStats]:
        """
        Run simulated annealing optimization on the grid.
        Returns optimized grid and statistics.
        """
        start_time = time.time()
        # Calculate initial state
        initial_grid = self._copy_grid(grid)  # Preserve the initial state for reporting
        current_grid = self._copy_grid(grid)  # Use a copy for optimization
        current_cost = self.cost_function.calculate(current_grid)
        best_grid = self._copy_grid(current_grid)  # Initialize best grid
        best_cost = current_cost

        # Save initial state visualization
        formatter = OutputFormatter(current_grid, self.color_mapper)
        initial_viz = formatter.to_visualizer(pixel_size=20)
        with open('results/snapshots/stage_0_initial.png', 'wb') as f:
            f.write(initial_viz)

        print("\n[1/6] Starting optimization:")
        print(f"Initial state - Patterns: {len(grid)} | Cost: {current_cost:.2f}")
        pattern_types = grid.get_pattern_count()
        print(f"Pattern distribution - Singles: {pattern_types.get(PatternType.SINGLE, 0)} | "
              f"Horizontal: {pattern_types.get(PatternType.HORIZONTAL, 0)} | "
              f"Vertical: {pattern_types.get(PatternType.VERTICAL, 0)}\n")

        # Sample moves for parameter estimation
        sample_deltas = self._sample_cost_deltas(current_grid, 10)
        avg_delta = sum(abs(d) for d in sample_deltas) / len(sample_deltas)
        min_delta = min(abs(d) for d in sample_deltas)

        print("[2/6] Parameter estimation:")
        print(f"Average cost delta: {avg_delta:.2f}")
        print(f"Minimum cost delta: {min_delta:.2f}\n")

        # Calculate temperature schedule
        start_temp = self.temperature_schedule.calculate_start_temp(current_cost, avg_delta)
        end_temp = self.temperature_schedule.calculate_end_temp(start_temp, min_delta)

        print("[3/6] Temperature schedule:")
        print(f"Start temperature: {start_temp:.2f}")
        print(f"End temperature: {end_temp:.2f}\n")

        # Calculate iterations per temperature
        problem_size = len(current_grid)
        iterations_per_temp = self.temperature_schedule.calculate_iterations(problem_size)

        print("[4/6] Optimization parameters:")
        print(f"Problem size: {problem_size} patterns")
        print(f"Iterations per temperature: {iterations_per_temp}\n")

        # Initialize statistics
        stats = AnnealingStats(initial_cost=current_cost, final_cost=0.0, best_cost=best_cost,
                               total_iterations=0, accepted_moves=0, rejected_moves=0,
                               execution_time=0.0, temperature_stages=0)
        # Initialize stagnation tracking
        stagnation_counter = 0
        previous_best_cost = float('inf')

        print("[5/6] Starting annealing process...")

        move_type_counts = {move_type: 0 for move_type in MoveType}

        # ** Generate temperature schedule once and store it in a list **
        temperature_schedule = list(self.temperature_schedule.generate_schedule(start_temp, end_temp))
        total_temps = len(temperature_schedule)
        snapshot_interval = max(1, total_temps // 30)  # Take 8 snapshots during process

        # Initialize stage count and total stages
        stage_count = 0
        total_stages = len(temperature_schedule)

        # Main annealing loop
        for temp_idx, temp in enumerate(temperature_schedule):
            stage_count += 1
            stats.temperature_stages += 1
            # Stagnation check
            if abs(current_cost - previous_best_cost) / previous_best_cost < 0.1:
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_threshold:
                    if not self.in_cluster_mode:
                        # Start cluster optimization
                        self.in_cluster_mode = True
                        current_grid, current_cost, cluster_count = self._try_cluster_optimization(
                            current_grid, temp, iterations_per_temp, stage_count)

                        # Print stage info with cluster indicator
                        elapsed_time = time.time() - start_time
                        print(f"Stage {stage_count}/{total_stages} [{elapsed_time:.1f}s] | "
                              f"Temp: {temp:.4f} | "
                              f"Total Cost: {current_cost:.2f} | "
                              f"Pattern Count: {len(current_grid)} | "
                              f"Accept Rate: -- % | "
                              f"Clusters = {cluster_count}")

                        # Force reset stagnation counter after cluster operation
                        stagnation_counter = 0
                        self.in_cluster_mode = False
                        continue  # Skip to next iteration
            else:
                stagnation_counter = 0

            previous_best_cost = current_cost

            # Temperature stage optimization
            temp_ratio = (temp - end_temp) / (start_temp - end_temp)
            move_gen = MoveGenerator(temperature_ratio=temp_ratio)
            accepted_moves_stage = 0
            rejected_moves_stage = 0

            for _ in range(iterations_per_temp):
                stats.total_iterations += 1
                move = move_gen.generate_move(current_grid)
                if not move:
                    continue

                # Track move types
                move_type_counts[move.move_type] += 1
                move.apply(current_grid)
                cost_delta = self.cost_function.calculate_delta(
                    current_grid, move.patterns_removed, move.patterns_added, move.affected_positions)
                new_cost = current_cost + cost_delta
                accept_prob = self.temperature_schedule.acceptance_probability(cost_delta, temp)
                if random.random() < accept_prob:
                    current_cost = new_cost
                    stats.accepted_moves += 1  # Cumulative count
                    accepted_moves_stage += 1  # Per-stage count
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_grid = self._copy_grid(current_grid)
                else:
                    move.undo(current_grid)
                    stats.rejected_moves += 1  # Cumulative count
                    rejected_moves_stage += 1  # Per-stage count

            # Calculate acceptance rate for the stage
            total_moves_stage = accepted_moves_stage + rejected_moves_stage
            acceptance_rate = accepted_moves_stage / total_moves_stage if total_moves_stage > 0 else 0

            # Print stats for the temperature stage
            elapsed_time = time.time() - start_time

            print(f"Stage {stage_count}/{total_stages} [{elapsed_time:.1f}s] | "
                  f"Temp: {temp:.4f} | "
                  f"Total Cost: {current_cost:.2f} | "
                  f"Pattern Count: {len(current_grid)} | "
                  f"Accept Rate: {acceptance_rate:.2%} | "
                  f"Cluster = {'Yes' if self.in_cluster_mode else 'No'}")

            # Save snapshot at all stages
            if temp_idx % snapshot_interval == 0:
                formatter = OutputFormatter(current_grid, self.color_mapper)
                stage_viz = formatter.to_visualizer(pixel_size=20)
                stage_num = (temp_idx // snapshot_interval) + 1
                with open(f'results/snapshots/stage_{stage_num}.png', 'wb') as f:
                    f.write(stage_viz)

        # Final statistics
        stats.final_cost = current_cost
        stats.best_cost = best_cost
        stats.execution_time = time.time() - start_time

        # Save final state visualization
        formatter = OutputFormatter(best_grid, self.color_mapper)
        final_viz = formatter.to_visualizer(pixel_size=20)
        with open('results/snapshots/stage_final.png', 'wb') as f:
            f.write(final_viz)

        print("\n[6/6] Optimization complete!")
        print(f"Total iterations: {stats.total_iterations}")
        print(f"Execution time: {stats.execution_time:.2f} seconds")
        print(f"Final best cost: {stats.best_cost:.2f}")
        print(f"Initial patterns: {len(initial_grid)} -> Final patterns: {len(best_grid)}")

        final_pattern_types = best_grid.get_pattern_count()
        print(f"Final pattern distribution:")
        print(f"  Singles: {final_pattern_types.get(PatternType.SINGLE, 0)}")
        print(f"  Horizontal: {final_pattern_types.get(PatternType.HORIZONTAL, 0)}")
        print(f"  Vertical: {final_pattern_types.get(PatternType.VERTICAL, 0)}")
        print(f"Acceptance rate: {(stats.accepted_moves / stats.total_iterations) * 100:.1f}%\n")

        return best_grid, stats

    def _sample_cost_deltas(self, grid: Grid, num_samples: int) -> List[float]:
        """
        Sample cost changes from random moves to estimate parameters.
        """
        deltas = []
        move_gen = MoveGenerator()
        original_cost = self.cost_function.calculate(grid)

        for _ in range(num_samples):
            # Try to generate and apply a move
            move = move_gen.generate_move(grid)
            if not move:
                continue

            # Calculate cost change
            move.apply(grid)
            delta = self.cost_function.calculate_delta(
                grid, move.patterns_removed, move.patterns_added, move.affected_positions)
            deltas.append(delta)

            # Undo the move
            move.undo(grid)

        return deltas or [original_cost * 0.1]  # Fallback if no moves possible

    def _copy_grid(self, grid: Grid) -> Grid:
        """Create a deep copy of a grid and its patterns."""
        new_grid = Grid(width=grid.width, height=grid.height)
        for pattern in grid.patterns:
            new_pattern = Pattern(
                position=pattern.position,
                color=pattern.color,
                pattern_type=pattern.pattern_type,
                size=pattern.size
            )
            new_grid.add_pattern(new_pattern)
        return new_grid

    def _try_cluster_optimization(self, grid: Grid, temperature: float,
                                  iterations_per_temp: int, current_stage: int) -> Tuple[Grid, float, int]:
        """Execute a cluster optimization stage when stagnation is detected"""
        self.cluster_optimizer = ClusterOptimizer(grid, self.cost_function)
        cluster_count = len(self.cluster_optimizer.find_clusters())

        # Create and run cluster stage
        cluster_stage = ClusterStage(grid, self.cost_function, self.cluster_optimizer)
        result = cluster_stage.run_cluster_stage(temperature, iterations_per_temp)

        # Save snapshot after cluster optimization
        formatter = OutputFormatter(grid, self.color_mapper)
        stage_viz = formatter.to_visualizer(pixel_size=20)
        with open(f'results/snapshots/stage_{current_stage}.png', 'wb') as f:
            f.write(stage_viz)

        return grid, result.final_cost, cluster_count

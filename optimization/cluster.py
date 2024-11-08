from collections import deque
from dataclasses import dataclass
from typing import Set, List, Optional, Dict, Tuple
from core.types import PatternType, Position
from core.pattern import Pattern
from core.grid import Grid
from .cost_function import CostFunction
from .move_generator import MoveGenerator, Move
import random
import math
import time

@dataclass
class ClusterStageResult:
    """Tracks results from a cluster optimization stage"""
    clusters_processed: int
    clusters_improved: int
    initial_cost: float
    final_cost: float
    execution_time: float
    pattern_count: int

@dataclass
class Cluster:
    """Represents a group of related patterns"""
    patterns: Set[Pattern]
    boundary_patterns: Set[Pattern]  # Patterns that connect to other clusters
    center_of_mass: Tuple[float, float]  # Approximate cluster center

    @property
    def size(self) -> int:
        return len(self.patterns)

    def get_pattern_distribution(self) -> Dict[PatternType, int]:
        """Count patterns by type in cluster"""
        distribution = {
            PatternType.SINGLE: 0,
            PatternType.HORIZONTAL: 0,
            PatternType.VERTICAL: 0
        }
        for pattern in self.patterns:
            distribution[pattern.pattern_type] += 1
        return distribution

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get cluster bounding box (min_x, min_y, max_x, max_y)"""
        min_x = min(p.min_x for p in self.patterns)
        min_y = min(p.min_y for p in self.patterns)
        max_x = max(p.max_x for p in self.patterns)
        max_y = max(p.max_y for p in self.patterns)
        return (min_x, min_y, max_x, max_y)

    def get_affected_positions(self) -> Set[Position]:
        """Get all positions covered by cluster patterns"""
        positions = set()
        for pattern in self.patterns:
            positions.update(pattern.get_pixels())
        return positions


class ClusterStage:
    """Handles cluster-specific optimization stages"""
    def __init__(self, grid: Grid, cost_function, cluster_optimizer):
        self.grid = grid
        self.cost_function = cost_function
        self.cluster_optimizer = cluster_optimizer

    def run_cluster_stage(self, temperature: float, iterations_per_temp: int) -> ClusterStageResult:
        """Execute a complete cluster optimization stage"""
        start_time = time.time()
        clusters = self.cluster_optimizer.find_clusters()
        initial_cost = self.cost_function.calculate(self.grid)
        clusters_improved = 0
        # Quietly process clusters without printing
        iterations_per_cluster = iterations_per_temp if clusters else 0
        for cluster in clusters:
            improved_patterns = self.cluster_optimizer.optimize_cluster(
                cluster, temperature, iterations=iterations_per_cluster)
            if improved_patterns:
                for pattern in cluster.patterns:
                    self.grid.remove_pattern(pattern)
                for pattern in improved_patterns:
                    self.grid.add_pattern(pattern)
                clusters_improved += 1

        return ClusterStageResult(clusters_processed=len(clusters),
                                  clusters_improved=clusters_improved,
                                  initial_cost=initial_cost,
                                  final_cost=self.cost_function.calculate(self.grid),
                                  execution_time=time.time() - start_time,
                                  pattern_count=len(self.grid))


class ClusterOptimizer:
    def __init__(self, grid: Grid, cost_function: CostFunction):
        self.grid = grid
        self.cost_function = cost_function
        self.clusters: List[Cluster] = []
        self.move_generator = MoveGenerator()

    def find_clusters(self) -> List[Cluster]:
        """Find and split natural pattern clusters in the grid"""
        visited = set()
        initial_clusters = []  # First find large natural clusters

        # Debug information
        total_patterns = len(self.grid.patterns)
        colors_present = set(p.color for p in self.grid.patterns)
        # First pass: find natural clusters
        for pattern in self.grid.patterns:
            if pattern in visited:
                continue

            cluster_patterns, boundary_patterns = self._grow_cluster(pattern, visited)
            if cluster_patterns:
                # Calculate center of mass
                total_x = sum(p.min_x for p in cluster_patterns)
                total_y = sum(p.min_y for p in cluster_patterns)
                size = len(cluster_patterns)
                center = (total_x / size, total_y / size)

                initial_clusters.append(Cluster(
                    patterns=cluster_patterns,
                    boundary_patterns=boundary_patterns,
                    center_of_mass=center
                ))

        # Second pass: split large clusters
        final_clusters = []
        for cluster in initial_clusters:
            if len(cluster.patterns) > 10:  # If cluster is large enough to split
                sub_clusters = self._split_large_cluster(cluster)
                final_clusters.extend(sub_clusters)
            else:
                final_clusters.append(cluster)

        # Sort clusters by size (largest first)
        self.clusters = sorted(final_clusters, key=lambda c: c.size, reverse=True)
        return self.clusters

    def _grow_cluster(self, start_pattern: Pattern, visited: Set[Pattern]) -> Tuple[Set[Pattern], Set[Pattern]]:
        """
        Grow cluster from seed pattern using pattern type compatibility rules.
        Returns (cluster_patterns, boundary_patterns)
        """
        cluster = set()
        boundary = set()
        queue = deque([start_pattern])

        # Initialize cluster bounds
        min_x = start_pattern.min_x
        min_y = start_pattern.min_y
        max_x = start_pattern.max_x
        max_y = start_pattern.max_y

        while queue:
            pattern = queue.popleft()
            if pattern in visited:
                continue
            visited.add(pattern)
            cluster.add(pattern)

            # Update cluster bounds
            min_x = min(min_x, pattern.min_x)
            min_y = min(min_y, pattern.min_y)
            max_x = max(max_x, pattern.max_x)
            max_y = max(max_y, pattern.max_y)

            # Get adjacent patterns
            neighbors = []
            for neighbor in self.grid.get_adjacent_patterns(pattern):
                if neighbor in visited:
                    continue
                if self._is_cluster_compatible(pattern, neighbor):
                    # Predict new bounds if neighbor is added
                    new_min_x = min(min_x, neighbor.min_x)
                    new_min_y = min(min_y, neighbor.min_y)
                    new_max_x = max(max_x, neighbor.max_x)
                    new_max_y = max(max_y, neighbor.max_y)

                    # Calculate new width and height
                    new_width = new_max_x - new_min_x + 1
                    new_height = new_max_y - new_min_y + 1

                    # Calculate squareness metric (absolute difference)
                    squareness = abs(new_width - new_height)

                    neighbors.append((squareness, neighbor))
                else:
                    boundary.add(neighbor)

            # Sort neighbors to prioritize squareness
            neighbors.sort(key=lambda x: x[0])

            # Add neighbors to the queue
            for _, neighbor in neighbors:
                queue.append(neighbor)

        return cluster, boundary

    def _is_cluster_compatible(self, pattern1: Pattern, pattern2: Pattern) -> bool:
        """
        Determine if patterns should belong to same cluster.
        Patterns are compatible if they share the same color, as this allows
        for potential pattern type optimizations within the cluster.
        """
        return pattern1.color == pattern2.color

    def optimize_cluster(self, cluster: Cluster, temperature: float, iterations: int = 100) -> Optional[Set[Pattern]]:
        """
        Attempt to optimize patterns within a cluster.
        Returns improved pattern set if found, None otherwise.
        """
        # Create temporary grid with just cluster patterns for move generation
        temp_grid = Grid(width=self.grid.width, height=self.grid.height)
        for pattern in cluster.patterns:
            temp_grid.add_pattern(pattern)

        # Track best configuration
        best_patterns = cluster.patterns.copy()
        best_cost = self._calculate_cluster_cost(cluster)
        current_patterns = cluster.patterns.copy()
        current_cost = best_cost

        # Optimization loop
        accepted_moves = 0

        for _ in range(iterations):
            # Generate move within cluster
            move = self._generate_cluster_move(temp_grid, cluster)
            if not move:
                continue

            # Apply move to temporary grid
            move.apply(temp_grid)

            # Calculate cost change
            new_cost = self._calculate_cluster_cost(Cluster(patterns=set(temp_grid.patterns),
                                                            boundary_patterns=cluster.boundary_patterns,
                                                            center_of_mass=cluster.center_of_mass))
            cost_delta = new_cost - current_cost

            # Accept or reject move
            accept_prob = self._acceptance_probability(cost_delta, temperature)

            if accept_prob > random.random():
                current_cost = new_cost
                current_patterns = set(temp_grid.patterns)  # Update current patterns
                accepted_moves += 1

                # Update best if improved
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_patterns = current_patterns.copy()
            else:
                # Undo move
                move.undo(temp_grid)

        # Return improved patterns if found
        if best_cost < self._calculate_cluster_cost(cluster):
            return best_patterns
        return None

    def _generate_cluster_move(self, grid: Grid, cluster: Cluster) -> Optional[Move]:
        """Generate a move that respects cluster boundaries"""
        # Get affected positions
        cluster_positions = cluster.get_affected_positions()

        # Try to generate a valid move
        for _ in range(10):  # Limited attempts
            move = self.move_generator.generate_move(grid)
            if not move:
                continue

            # Check if move only affects cluster positions
            move_positions = set()
            for pattern in move.patterns_removed | move.patterns_added:
                move_positions.update(pattern.get_pixels())

            # Only accept moves that stay within cluster
            if move_positions.issubset(cluster_positions):
                return move

        return None

    def _calculate_cluster_cost(self, cluster: Cluster) -> float:
        """Calculate cost for patterns within cluster"""
        # Create temporary grid with cluster patterns
        temp_grid = Grid(width=self.grid.width, height=self.grid.height)
        for pattern in cluster.patterns:
            temp_grid.add_pattern(pattern)

        return self.cost_function.calculate(temp_grid)

    def _acceptance_probability(self, cost_delta: float, temperature: float) -> float:
        """Calculate probability of accepting a move"""
        if cost_delta <= 0:  # Better solution
            return 1.0
        return math.exp(-cost_delta / temperature)

    def get_cluster_stats(self) -> List[Dict]:
        """Get statistics for all clusters"""
        stats = []
        for cluster in self.clusters:
            distribution = cluster.get_pattern_distribution()
            bounds = cluster.get_bounds()
            stats.append({
                'size': cluster.size,
                'pattern_distribution': distribution,
                'bounds': bounds,
                'boundary_size': len(cluster.boundary_patterns),
                'center': cluster.center_of_mass
            })
        return stats

    def _split_large_cluster(self, cluster: Cluster) -> List[Cluster]:
        """
        Split cluster into roughly equal sub-clusters, maintaining adjacency.
        Number of sub-clusters is determined by total pattern count.
        """
        total_patterns = len(cluster.patterns)

        # Calculate range for number of sub-clusters
        multiplier = random.choice([1,3, 5])
        num_clusters = random.randint(2, 4) * multiplier

        # Target size for each sub-cluster
        target_size = total_patterns // num_clusters

        remaining_patterns = cluster.patterns.copy()
        sub_clusters = []

        # Create num_clusters - 1 clusters (last one gets remainder)
        while len(sub_clusters) < num_clusters - 1 and remaining_patterns:
            # Pick a random seed pattern
            seed = random.choice(list(remaining_patterns))

            # Grow a sub-cluster targeting the calculated size
            sub_cluster_patterns = self._grow_sub_cluster(
                seed, remaining_patterns, target_size)

            # Calculate center of mass
            center = (sum(p.min_x for p in sub_cluster_patterns) / len(sub_cluster_patterns),
                      sum(p.min_y for p in sub_cluster_patterns) / len(sub_cluster_patterns))

            # Create new sub-cluster
            boundary = {p for p in cluster.boundary_patterns
                        if any(self._is_cluster_compatible(p, cp)
                               for cp in sub_cluster_patterns)}

            sub_clusters.append(Cluster(
                patterns=sub_cluster_patterns,
                boundary_patterns=boundary,
                center_of_mass=center
            ))

            # Remove used patterns
            remaining_patterns.difference_update(sub_cluster_patterns)

        # Add remaining patterns as final cluster if any exist
        if remaining_patterns:
            center = (sum(p.min_x for p in remaining_patterns) / len(remaining_patterns),
                      sum(p.min_y for p in remaining_patterns) / len(remaining_patterns))

            boundary = {p for p in cluster.boundary_patterns
                        if any(self._is_cluster_compatible(p, cp)
                               for cp in remaining_patterns)}

            sub_clusters.append(Cluster(
                patterns=remaining_patterns,
                boundary_patterns=boundary,
                center_of_mass=center
            ))

        return sub_clusters

    def _grow_sub_cluster(self, seed: Pattern, available_patterns: Set[Pattern],
                          target_size: int) -> Set[Pattern]:
        """
        Grow a sub-cluster from a seed pattern to approximately target_size.
        Uses wave-front growth to maintain adjacency and squareness.
        """
        sub_cluster = {seed}
        frontier = deque([seed])

        # Initialize cluster bounds
        min_x = seed.min_x
        min_y = seed.min_y
        max_x = seed.max_x
        max_y = seed.max_y

        while frontier and len(sub_cluster) < target_size:
            current = frontier.popleft()

            # Get all compatible neighbors from available patterns
            neighbors = []
            for neighbor in self.grid.get_adjacent_patterns(current):
                if neighbor in available_patterns and neighbor not in sub_cluster and self._is_cluster_compatible(
                        current, neighbor):
                    # Predict new bounds if neighbor is added
                    new_min_x = min(min_x, neighbor.min_x)
                    new_min_y = min(min_y, neighbor.min_y)
                    new_max_x = max(max_x, neighbor.max_x)
                    new_max_y = max(max_y, neighbor.max_y)

                    # Calculate new width and height
                    new_width = new_max_x - new_min_x + 1
                    new_height = new_max_y - new_min_y + 1

                    # Calculate squareness metric (absolute difference)
                    squareness = abs(new_width - new_height)

                    neighbors.append((squareness, neighbor))

            # Sort neighbors to prioritize squareness
            neighbors.sort(key=lambda x: x[0])

            # Add neighbors to the cluster
            for squareness, neighbor in neighbors:
                if len(sub_cluster) >= target_size:
                    break
                sub_cluster.add(neighbor)
                frontier.append(neighbor)

                # Update cluster bounds
                min_x = min(min_x, neighbor.min_x)
                min_y = min(min_y, neighbor.min_y)
                max_x = max(max_x, neighbor.max_x)
                max_y = max(max_y, neighbor.max_y)

        return sub_cluster

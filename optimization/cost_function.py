from dataclasses import dataclass
from typing import Dict, Set

from core.grid import Grid, Position
from core.types import PatternType
from core.pattern import Pattern

@dataclass
class CostWeights:
    """Weights used in the cost function calculation."""
    pattern_weight: float = 1.0
    single_penalty: float = 5.0
    transition_weight: float = 0.5


class CostFunction:
    """
    Calculates the cost (energy) of a pattern configuration.
    Lower cost indicates a better solution.
    """

    def __init__(self, weights: CostWeights = None):
        """Initialize with optional custom weights."""
        self.weights = weights or CostWeights()

    def calculate(self, grid: Grid) -> float:
        """
        Calculate the total cost of the current grid configuration.

        Components:
        1. Number of patterns (fewer is better)
        2. Number of single pixels (heavily penalized)
        3. Number of pattern transitions (fewer is better)
        """
        pattern_counts = grid.get_pattern_count()

        total_patterns = sum(pattern_counts.values())
        num_singles = pattern_counts.get(PatternType.SINGLE, 0)
        transitions = grid.calculate_transitions()

        return (total_patterns * self.weights.pattern_weight +
                num_singles * self.weights.single_penalty +
                transitions * self.weights.transition_weight)

    def calculate_delta(self, grid: Grid,
                        patterns_removed: Set[Pattern],
                        patterns_added: Set[Pattern],
                        affected_region: Set[Position]) -> float:
        """
        Efficiently calculate cost change for a local modification.
        Only recalculates costs in the affected region.

        Args:
            grid: Current grid state after modification
            patterns_removed: Patterns that were removed by the move
            patterns_added: Patterns that were added by the move
            affected_region: Set of positions that were modified

        Returns:
            Change in cost (positive means cost increased)
        """
        if not affected_region:
            return 0.0

        # Calculate pattern count changes
        pattern_count_delta = len(patterns_added) - len(patterns_removed)

        # Calculate singles changes
        old_pattern_types = self._count_pattern_types(patterns_removed)
        new_pattern_types = self._count_pattern_types(patterns_added)
        singles_delta = (new_pattern_types[PatternType.SINGLE] -
                         old_pattern_types[PatternType.SINGLE])

        # Calculate transition delta properly
        # First undo the move to get old transitions
        for pattern in patterns_added:
            grid.remove_pattern(pattern)
        for pattern in patterns_removed:
            grid.add_pattern(pattern)

        # Reapply the move to get new transitions
        for pattern in patterns_removed:
            grid.remove_pattern(pattern)
        for pattern in patterns_added:
            grid.add_pattern(pattern)

        # Calculate singles changes
        old_transitions = grid.calculate_transitions()
        new_transitions = grid.calculate_transitions()
        transition_delta = new_transitions - old_transitions

        # Calculate total cost delta
        cost_delta = (
                pattern_count_delta * self.weights.pattern_weight +
                singles_delta * self.weights.single_penalty +
                transition_delta * self.weights.transition_weight
        )

        return cost_delta

    def _count_pattern_types(self, patterns: Set[Pattern]) -> Dict[PatternType, int]:
        """Count patterns by type."""
        counts = {pattern_type: 0 for pattern_type in PatternType}
        for pattern in patterns:
            counts[pattern.pattern_type] += 1
        return counts

    def _count_transitions_for_patterns(self, patterns: Set[Pattern]) -> int:
        """Count transitions for a set of patterns."""
        transitions = 0
        seen_positions = set()

        for pattern in patterns:
            pattern_pixels = pattern.get_pixels()
            for pos in pattern_pixels:
                if pos in seen_positions:
                    continue

                x, y = pos
                # Check right and down neighbors
                for dx, dy in [(1, 0), (0, 1)]:
                    neighbor_pos = Position((x + dx, y + dy))
                    if neighbor_pos not in seen_positions:
                        transitions += 1

                seen_positions.add(pos)

        return transitions

    def _count_transitions_in_region(self, grid: Grid,
                                     min_x: int, max_x: int,
                                     min_y: int, max_y: int) -> int:
        """
        Count pattern transitions within a specific region.
        Only counts horizontal and vertical transitions between different patterns.
        """
        transitions = 0
        seen_positions = set()

        # Check each position in region
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                pos = Position((x, y))
                if pos in seen_positions:
                    continue

                pattern = grid.get_pattern_at(pos)
                if not pattern:
                    continue

                # Check right neighbor if not at edge
                if x < max_x:
                    right_pos = Position((x + 1, y))
                    right_pattern = grid.get_pattern_at(right_pos)
                    if right_pattern and right_pattern != pattern:
                        transitions += 1

                # Check bottom neighbor if not at edge
                if y < max_y:
                    bottom_pos = Position((x, y + 1))
                    bottom_pattern = grid.get_pattern_at(bottom_pos)
                    if bottom_pattern and bottom_pattern != pattern:
                        transitions += 1

                seen_positions.add(pos)

        return transitions

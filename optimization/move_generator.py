from dataclasses import dataclass
from typing import Optional, Set, Dict, Callable, List
import random

from core.grid import Grid
from core.pattern import Pattern, PatternType, PatternSize
from core.types import MoveType, Position


@dataclass
class Move:
    """Represents a potential transformation of the pattern grid."""
    move_type: MoveType
    patterns_removed: Set[Pattern]
    patterns_added: Set[Pattern]
    affected_positions: Set[Position]

    def apply(self, grid: Grid) -> None:
        """Apply this move to the grid."""
        for pattern in self.patterns_removed:
            grid.remove_pattern(pattern)
        for pattern in self.patterns_added:
            grid.add_pattern(pattern)

    def undo(self, grid: Grid) -> None:
        """Undo this move from the grid."""
        for pattern in self.patterns_added:
            grid.remove_pattern(pattern)
        for pattern in self.patterns_removed:
            grid.add_pattern(pattern)


class MoveGenerator:
    """
    Generates and manages pattern transformation moves.
    Implements the two move types: merge patterns, break patterns.
    """

    def __init__(self, temperature_ratio: float = 1.0):
        """
        Initialize with temperature ratio to adjust move weights.

        Args:
            temperature_ratio: Current temp / start temp (0-1)
        """
        self.temperature_ratio = temperature_ratio

    def generate_move(self, grid: Grid) -> Optional[Move]:
        """
        Generate a random valid move based on current state.
        Weights move selection based on temperature ratio.
        """
        # Adjust weights based on temperature
        weights = self._calculate_move_weights()
        # Select move type
        move_type = random.choices(list(MoveType), weights=[weights[t] for t in MoveType])[0]
        # Generate move of selected type
        move_generators: Dict[MoveType, Callable[[Grid], Optional[Move]]] = {
            MoveType.PATTERN_MERGE: self._generate_pattern_merge_move,
            MoveType.PATTERN_BREAK_RECOMBINE: self._generate_break_recombine_move,
        }
        return move_generators[move_type](grid)

    def _calculate_move_weights(self) -> dict[MoveType, float]:
        """
        Calculate move type weights based on temperature.
        Early (hot): Favor disruptive moves (break)
        Late (cool): Favor merging moves (merge)
        """
        t = self.temperature_ratio
        return {
            # More merging at high temperatures
            MoveType.PATTERN_MERGE: 1.0 + t* 2, # from 1.0 to 3.0
            # More recombination as temperature decreases
            MoveType.PATTERN_BREAK_RECOMBINE: 0.5 + ((1-t) * 2), # from 0.5 to 2.0
        }

    def _generate_pattern_merge_move(self, grid: Grid) -> Optional[Move]:
        """
        Generate a move that merges any two adjacent patterns of the same color.
        Works for any combination of singles and multi-pixel patterns.
        """
        # Get all patterns
        patterns = list(grid.patterns)
        if len(patterns) < 2:
            return None

        # Try random pairs until we find mergeable ones
        random.shuffle(patterns)
        for i, pattern1 in enumerate(patterns[:-1]):
            # Get potential merge candidates (adjacent patterns)
            neighbors = grid.get_mergeable_neighbors(pattern1)
            if not neighbors:
                continue

            pattern2 = random.choice(neighbors)
            if pattern1.can_merge_with(pattern2, grid):
                merged = pattern1.merge_with(pattern2, grid)
                if merged:
                    return Move(
                        move_type=MoveType.PATTERN_MERGE,
                        patterns_removed={pattern1, pattern2},
                        patterns_added={merged},
                        affected_positions=pattern1.get_pixels() | pattern2.get_pixels()
                    )
        return None

    def _generate_break_recombine_move(self, grid: Grid) -> Optional[Move]:
        """
        Generate a move that breaks a pattern if breaking it would enable adjacent patterns to merge,
        potentially reducing the total number of patterns.
        """
        patterns = list(grid.patterns)
        random.shuffle(patterns)
        for pattern in patterns:
            # Skip single pixels
            if pattern.pattern_type == PatternType.SINGLE:
                continue

            # Get adjacent patterns of the same color
            neighbors = grid.get_adjacent_patterns(pattern)
            same_color_neighbors = [n for n in neighbors if n.color == pattern.color]

            # For each neighbor
            for neighbor in same_color_neighbors:
                # Find potential merge candidates across the blocking pattern
                candidates = self._find_merge_candidates_across_blocker(grid, pattern, neighbor)
                for candidate in candidates:
                    # Calculate potential merged length
                    potential_length = self._calculate_merged_length(neighbor, candidate)
                    # If merged length would be greater than sum of current lengths
                    if potential_length > neighbor.size + candidate.size:
                        # Break the blocking pattern into singles
                        singles = self._break_into_singles(pattern)
                        return Move(
                            move_type=MoveType.PATTERN_BREAK_RECOMBINE,
                            patterns_removed={pattern},
                            patterns_added=singles,
                            affected_positions=pattern.get_pixels()
                        )

        return None

    def _find_merge_candidates_across_blocker(self, grid: Grid, blocker: Pattern, neighbor: Pattern) -> List[Pattern]:
        """
        Find patterns that could merge with neighbor if blocker is removed.
        """
        candidates = []

        # Scan in the direction of neighbor's orientation
        if neighbor.pattern_type == PatternType.HORIZONTAL:
            y = neighbor.min_y
            # Check to the left
            x = neighbor.min_x - 1
            while x >= 0:
                pos = Position((x, y))
                pattern_at_pos = grid.get_pattern_at(pos)
                if pattern_at_pos is None or pattern_at_pos == blocker:
                    x -= 1
                    continue
                elif pattern_at_pos.color == neighbor.color and pattern_at_pos != neighbor:
                    candidates.append(pattern_at_pos)
                    break
                else:
                    break
            # Check to the right
            x = neighbor.max_x + 1
            while x < grid.width:
                pos = Position((x, y))
                pattern_at_pos = grid.get_pattern_at(pos)
                if pattern_at_pos is None or pattern_at_pos == blocker:
                    x += 1
                    continue
                elif pattern_at_pos.color == neighbor.color and pattern_at_pos != neighbor:
                    candidates.append(pattern_at_pos)
                    break
                else:
                    break
        elif neighbor.pattern_type == PatternType.VERTICAL:
            x = neighbor.min_x
            # Check upwards
            y = neighbor.min_y - 1
            while y >= 0:
                pos = Position((x, y))
                pattern_at_pos = grid.get_pattern_at(pos)
                if pattern_at_pos is None or pattern_at_pos == blocker:
                    y -= 1
                    continue
                elif pattern_at_pos.color == neighbor.color and pattern_at_pos != neighbor:
                    candidates.append(pattern_at_pos)
                    break
                else:
                    break
            # Check downwards
            y = neighbor.max_y + 1
            while y < grid.height:
                pos = Position((x, y))
                pattern_at_pos = grid.get_pattern_at(pos)
                if pattern_at_pos is None or pattern_at_pos == blocker:
                    y += 1
                    continue
                elif pattern_at_pos.color == neighbor.color and pattern_at_pos != neighbor:
                    candidates.append(pattern_at_pos)
                    break
                else:
                    break

        return candidates

    def _calculate_merged_length(self, pattern1: Pattern, pattern2: Pattern) -> int:
        """Calculate the potential length if two patterns could be merged."""
        if pattern1.pattern_type == PatternType.HORIZONTAL:
            return abs(pattern2.max_x - pattern1.min_x) + 1
        else:  # VERTICAL
            return abs(pattern2.max_y - pattern1.min_y) + 1

    def _break_into_singles(self, pattern: Pattern) -> Set[Pattern]:
        """Break a pattern into single-pixel patterns."""
        singles = set()
        for pos in pattern.get_pixels():
            singles.add(Pattern(
                position=pos,
                color=pattern.color,
                pattern_type=PatternType.SINGLE,
                size=PatternSize(1)
            ))
        return singles

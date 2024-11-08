from typing import Dict, Set, Optional, List, Iterator
from dataclasses import dataclass, field
from collections import defaultdict

from .types import ColorIndex, ValidationResult, Position
from .pattern import Pattern, PatternType, PatternSize


@dataclass
class Grid:
    """
    Represents a 2D grid of patterns.
    Maintains efficient lookups for patterns by position and color.
    """
    width: int
    height: int
    patterns: Set[Pattern] = field(default_factory=set)
    _position_map: Dict[Position, Pattern] = field(default_factory=dict)
    _color_map: Dict[ColorIndex, Set[Pattern]] = field(default_factory=lambda: defaultdict(set))

    def add_pattern(self, pattern: Pattern) -> ValidationResult:
        """
        Add a pattern to the grid.
        Returns ValidationResult indicating success or reason for failure.
        """
        # Validate pattern bounds
        if not self._is_within_bounds(pattern):
            return ValidationResult.INVALID_BOUNDS

        # Check for overlaps with existing patterns
        pattern_pixels = pattern.get_pixels()
        for pos in pattern_pixels:
            if pos in self._position_map:
                return ValidationResult.INVALID_OVERLAP

        # Add pattern to all data structures
        self.patterns.add(pattern)
        for pos in pattern_pixels:
            self._position_map[pos] = pattern
        self._color_map[pattern.color].add(pattern)

        return ValidationResult.VALID

    def remove_pattern(self, pattern: Pattern) -> None:
        """Remove a pattern from the grid."""
        if pattern not in self.patterns:
            return

        # Remove from all data structures
        self.patterns.remove(pattern)
        for pos in pattern.get_pixels():
            self._position_map.pop(pos, None)
        self._color_map[pattern.color].discard(pattern)

        # Clean up empty color sets
        if not self._color_map[pattern.color]:
            del self._color_map[pattern.color]

    def get_pattern_at(self, pos: Position) -> Optional[Pattern]:
        """Get the pattern at a specific position, if any."""
        return self._position_map.get(pos)

    def get_patterns_by_color(self, color: ColorIndex) -> Set[Pattern]:
        """Get all patterns of a specific color."""
        return self._color_map.get(color, set()).copy()

    def get_adjacent_patterns(self, pattern: Pattern) -> List[Pattern]:
        """Get all patterns adjacent to the given pattern."""
        adjacent = set()
        pattern_pixels = pattern.get_pixels()

        # Check each pixel of the pattern for adjacent patterns
        for pos in pattern_pixels:
            x, y = pos
            # Check all eight directions (including diagonals)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),  # orthogonal
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:  # diagonal
                adj_pos = Position((x + dx, y + dy))
                adj_pattern = self._position_map.get(adj_pos)
                if adj_pattern and adj_pattern != pattern:
                    adjacent.add(adj_pattern)
        return list(adjacent)

    def get_mergeable_neighbors(self, pattern: Pattern) -> List[Pattern]:
        """Get all adjacent patterns that can be merged with the given pattern."""
        return [adj for adj in self.get_adjacent_patterns(pattern)
                if pattern.can_merge_with(adj)]

    def validate_pattern_placement(self, pattern: Pattern) -> ValidationResult:
        """
        Check if a pattern can be placed at its current position.
        Does not modify the grid.
        """
        if not self._is_within_bounds(pattern):
            return ValidationResult.INVALID_BOUNDS

        # Check for overlaps with existing patterns
        pattern_pixels = pattern.get_pixels()
        for pos in pattern_pixels:
            existing = self._position_map.get(pos)
            if existing and existing != pattern:
                return ValidationResult.INVALID_OVERLAP

        return ValidationResult.VALID

    def clear(self) -> None:
        """Remove all patterns from the grid."""
        self.patterns.clear()
        self._position_map.clear()
        self._color_map.clear()

    def get_singles(self) -> Iterator[Pattern]:
        """Get all single-pixel patterns in the grid."""
        return (p for p in self.patterns if p.pattern_type == PatternType.SINGLE)

    def get_pattern_count(self) -> Dict[PatternType, int]:
        """Get count of patterns by type."""
        counts = defaultdict(int)
        for pattern in self.patterns:
            counts[pattern.pattern_type] += 1
        return dict(counts)

    def _is_within_bounds_pos(self, pos: Position) -> bool:
        """Check if a position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_within_bounds(self, pattern: Pattern) -> bool:
        """Check if pattern is within grid bounds."""
        return all(self._is_within_bounds_pos(pos) for pos in pattern.get_pixels())

    def __iter__(self) -> Iterator[Pattern]:
        """Iterate over all patterns in the grid."""
        return iter(self.patterns)

    def __len__(self) -> int:
        """Get total number of patterns in the grid."""
        return len(self.patterns)

    def calculate_transitions(self) -> int:
        """
        Calculate number of pattern transitions (boundaries between patterns).
        Used in cost function calculations.
        """
        transitions = 0
        seen_positions = set()

        for pattern in self.patterns:
            pattern_pixels = pattern.get_pixels()
            for pos in pattern_pixels:
                x, y = pos
                # Check right and down neighbors
                for dx, dy in [(1, 0), (0, 1)]:
                    neighbor_pos = Position((x + dx, y + dy))
                    if neighbor_pos not in seen_positions:
                        neighbor = self._position_map.get(neighbor_pos)
                        if neighbor and neighbor != pattern:
                            transitions += 1
                seen_positions.add(pos)

        return transitions

    def validate_merge(self, pattern1: Pattern, pattern2: Pattern, merged_pattern: Pattern) -> ValidationResult:
        """
        Validate if two patterns can be merged into a new pattern.
        Handles the validation without modifying grid state.
        """
        # Get all affected positions
        merged_pixels = merged_pattern.get_pixels()
        original_pixels = pattern1.get_pixels() | pattern2.get_pixels()

        # Check bounds
        if not self._is_within_bounds(merged_pattern):
            return ValidationResult.INVALID_BOUNDS

        # Check for overlaps with patterns OTHER than the original two
        for pos in merged_pixels:
            existing = self._position_map.get(pos)
            if existing and existing not in (pattern1, pattern2):
                return ValidationResult.INVALID_OVERLAP

        return ValidationResult.VALID

    @classmethod
    def from_pixel_grid(cls, pixels: Dict[Position, ColorIndex], width: int, height: int) -> 'Grid':
        """
        Create a Grid from a pixel-by-pixel representation.
        Initial state creates all single-pixel patterns.
        """
        grid = cls(width=width, height=height)

        for pos, color in pixels.items():
            pattern = Pattern(position=Position(pos), color=color, pattern_type=PatternType.SINGLE,
                              size=PatternSize(1))
            grid.add_pattern(pattern)

        return grid
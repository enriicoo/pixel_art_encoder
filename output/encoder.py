from dataclasses import dataclass
from typing import List, Dict, Iterator

from core.grid import Grid
from core.pattern import Pattern
from core.types import ColorIndex, PatternType
from preprocessing.color_mapper import ColorMapper


@dataclass
class EncodedPattern:
    """Represents a pattern in the output encoding format."""
    x: int
    y: int
    color_index: int
    pattern_type: int  # 0=single, 1=horizontal, 2=vertical
    size: int

    def to_string(self) -> str:
        """Convert to string format: x,y,color_index,pattern_type,size"""
        return f"{self.x},{self.y},{self.color_index},{self.pattern_type},{self.size}"

    @classmethod
    def from_pattern(cls, pattern: Pattern) -> 'EncodedPattern':
        """Create encoded pattern from Pattern instance."""
        return cls(x=pattern.min_x, y=pattern.min_y, color_index=pattern.color,
                   pattern_type=pattern.pattern_type.value, size=pattern.size)


@dataclass
class ColorDefinition:
    """Represents a color definition in the output."""
    index: int
    color: str


class PatternEncoder:
    """
    Encodes optimized patterns into required output format.
    Handles both pattern encoding and color mapping.
    """

    def __init__(self, color_mapper: ColorMapper):
        """Initialize with color mapper used during processing."""
        self.color_mapper = color_mapper

    def encode_grid(self, grid: Grid) -> tuple[List[ColorDefinition], List[EncodedPattern]]:
        """
        Encode complete grid into color definitions and patterns.

        Returns:
            Tuple of (color definitions, encoded patterns)
        """
        # Get color definitions
        color_defs = self._encode_colors()

        # Encode patterns
        patterns = [EncodedPattern.from_pattern(p) for p in grid]

        # Sort patterns for consistent output
        patterns.sort(key=lambda p: (p.y, p.x, p.color_index))

        return color_defs, patterns

    def _encode_colors(self) -> List[ColorDefinition]:
        """Create color definitions from color mapper."""
        definitions = []

        # Get all mapped colors
        for index in range(len(self.color_mapper._index_to_color)):
            color = self.color_mapper.get_color(ColorIndex(index))
            if color:
                definitions.append(ColorDefinition(index=index, color=color))

        return sorted(definitions, key=lambda d: d.index)


class PatternIterator:
    """
    Iterates through patterns in a specific order for optimization.
    Provides different iteration strategies (row-wise, color-wise, etc.).
    """

    @staticmethod
    def by_position(patterns: List[Pattern]) -> Iterator[Pattern]:
        """Iterate patterns in row-major order (left-to-right, top-to-bottom)."""
        return iter(sorted(patterns, key=lambda p: (p.y, p.x)))

    @staticmethod
    def by_color(patterns: List[Pattern]) -> Iterator[Pattern]:
        """Iterate patterns grouped by color."""
        return iter(sorted(patterns, key=lambda p: (p.color, p.y, p.x)))

    @staticmethod
    def by_type(patterns: List[Pattern]) -> Iterator[Pattern]:
        """Iterate patterns grouped by type (singles first)."""

        def type_key(p: Pattern) -> tuple:
            # Order: singles, horizontals, verticals
            type_order = {
                PatternType.SINGLE: 0,
                PatternType.HORIZONTAL: 1,
                PatternType.VERTICAL: 2
            }
            return (type_order[p.pattern_type], p.min_y, p.min_x)

        return iter(sorted(patterns, key=type_key))

    @staticmethod
    def by_size(patterns: List[Pattern], reverse: bool = True) -> Iterator[Pattern]:
        """
        Iterate patterns by size.
        Default is largest first (reverse=True).
        """
        return iter(sorted(patterns,
                           key=lambda p: (p.size, p.y, p.x),
                           reverse=reverse))


class GridStatistics:
    """
    Calculates and provides statistics about pattern grid.
    Useful for optimization monitoring and results reporting.
    """

    @staticmethod
    def calculate(grid: Grid) -> Dict[str, int]:
        """Calculate various statistics about the grid."""
        stats = {
            'total_patterns': len(grid),
            'total_pixels': sum(p.size for p in grid),
            'single_pixels': sum(1 for p in grid if p.pattern_type == PatternType.SINGLE),
            'horizontal_lines': sum(1 for p in grid if p.pattern_type == PatternType.HORIZONTAL),
            'vertical_lines': sum(1 for p in grid if p.pattern_type == PatternType.VERTICAL),
            'total_colors': len({p.color for p in grid}),
            'max_pattern_size': max((p.size for p in grid), default=0),
            'transitions': grid.calculate_transitions()
        }

        # Calculate average pattern size
        if stats['total_patterns'] > 0:
            stats['avg_pattern_size'] = stats['total_pixels'] / stats['total_patterns']
        else:
            stats['avg_pattern_size'] = 0

        return stats
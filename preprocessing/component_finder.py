from dataclasses import dataclass
from typing import List, Set, Dict, Iterator, Optional
from collections import deque

from core.types import Position, ColorIndex, PatternType
from core.grid import Grid
from core.pattern import Pattern, PatternSize
from .color_mapper import ColorGroup


@dataclass
class Component:
    """
    Represents a connected component of same-colored pixels.
    """
    color: ColorIndex
    positions: Set[Position]

    @property
    def size(self) -> int:
        """Number of pixels in component."""
        return len(self.positions)

    def get_bounds(self) -> tuple[int, int, int, int]:
        """Get component bounds as (min_x, min_y, max_x, max_y)."""
        xs = [x for x, _ in self.positions]
        ys = [y for _, y in self.positions]
        return min(xs), min(ys), max(xs), max(ys)

    def is_single_line(self) -> tuple[bool, Optional[PatternType]]:
        """
        Check if component forms a single line.
        Returns (is_line, pattern_type if line).
        """
        if self.size <= 1:
            return False, None

        min_x, min_y, max_x, max_y = self.get_bounds()
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Check if all positions form a straight line
        if height == 1:  # Horizontal line
            return all(Position((x, min_y)) in self.positions
                       for x in range(min_x, max_x + 1)), PatternType.HORIZONTAL
        elif width == 1:  # Vertical line
            return all(Position((min_x, y)) in self.positions
                       for y in range(min_y, max_y + 1)), PatternType.VERTICAL

        return False, None


class ComponentFinder:
    """
    Finds connected components in pixel grid.
    Two pixels are connected if they share an edge (not diagonal).
    """

    def __init__(self, grid: Grid):
        """Initialize with grid to analyze."""
        self.grid = grid
        self._components: Dict[ColorIndex, List[Component]] = {}

    def find_components(self) -> Dict[ColorIndex, List[Component]]:
        """
        Find all connected components in the grid.
        Groups components by color.
        """
        # Reset components
        self._components.clear()

        # Track visited positions
        visited = set()

        # Process each pattern in grid
        for pattern in self.grid:
            # Skip if we've seen any part of this pattern
            if any(pos in visited for pos in pattern.get_pixels()):
                continue

            # Find component starting from this pattern
            component = self._flood_fill(pattern.position, pattern.color, visited)

            # Store component
            if pattern.color not in self._components:
                self._components[pattern.color] = []
            self._components[pattern.color].append(component)

        return self._components

    def get_components(self, color: ColorIndex) -> List[Component]:
        """Get all components of a specific color."""
        return self._components.get(color, [])

    def _flood_fill(self, start: Position, color: ColorIndex,
                    visited: Set[Position]) -> Component:
        """
        Find connected component using flood fill algorithm.
        Only connects through edges, not diagonals.
        """
        component = Component(color=color, positions=set())
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue

            # Check if position has matching pattern
            pattern = self.grid.get_pattern_at(pos)
            if not pattern or pattern.color != color:
                continue

            # Add to component
            visited.add(pos)
            component.positions.add(pos)

            # Check adjacent positions
            x, y = pos
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = Position((x + dx, y + dy))
                if (next_pos not in visited and
                        self._is_valid_position(next_pos)):
                    queue.append(next_pos)

        return component

    def _is_valid_position(self, pos: Position) -> bool:
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.grid.width and 0 <= y < self.grid.height


class InitialPatternGenerator:
    """
    Generates initial patterns from connected components.
    Uses greedy approach to create horizontal/vertical patterns where possible.
    """

    @staticmethod
    def generate_patterns(component: Component) -> List[Pattern]:
        """
        Generate patterns to cover component.
        Uses greedy approach:
        1. Try to form horizontal lines
        2. Try to form vertical lines for remaining pixels
        3. Use single pixels for anything left
        """
        patterns = []
        remaining = component.positions.copy()

        # Try horizontal lines first
        while remaining:
            longest = InitialPatternGenerator._find_longest_line(
                remaining, horizontal=True)
            if not longest:
                break

            pattern = Pattern(
                position=longest[0],
                color=component.color,
                pattern_type=PatternType.HORIZONTAL,
                size=PatternSize(len(longest))
            )
            patterns.append(pattern)
            remaining.difference_update(longest)

        # Try vertical lines for remaining pixels
        while remaining:
            longest = InitialPatternGenerator._find_longest_line(
                remaining, horizontal=False)
            if not longest:
                break

            pattern = Pattern(
                position=longest[0],
                color=component.color,
                pattern_type=PatternType.VERTICAL,
                size=PatternSize(len(longest))
            )
            patterns.append(pattern)
            remaining.difference_update(longest)

        # Create single patterns for any remaining pixels
        for pos in remaining:
            pattern = Pattern(
                position=pos,
                color=component.color,
                pattern_type=PatternType.SINGLE,
                size=PatternSize(1)
            )
            patterns.append(pattern)

        return patterns

    @staticmethod
    def _find_longest_line(positions: Set[Position],
                           horizontal: bool) -> Optional[List[Position]]:
        """
        Find longest horizontal/vertical line in remaining positions.
        Returns None if no valid lines found.
        """
        if not positions:
            return None

        # Try each position as potential start
        best_line = None
        best_length = 0

        for start in positions:
            line = [start]
            x, y = start

            # Extend line as far as possible
            i = 1
            while True:
                next_pos = Position((x + i, y)) if horizontal else Position((x, y + i))
                if next_pos not in positions:
                    break
                line.append(next_pos)
                i += 1

            if len(line) > best_length:
                best_length = len(line)
                best_line = line

        # Only return if line is longer than 1 pixel
        return best_line if best_length > 1 else None
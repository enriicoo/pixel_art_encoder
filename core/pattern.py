from dataclasses import dataclass
from typing import Set, Optional, TYPE_CHECKING
from .types import PatternType, Position, ColorIndex, PatternSize, Direction, ValidationResult

if TYPE_CHECKING:
    from .grid import Grid  # Avoid circular import at runtime

@dataclass(frozen=True)  # Added frozen=True to make Pattern hashable
class Pattern:
    """
    Represents a pattern in the pixel grid.
    A pattern can be a single pixel, horizontal line, or vertical line.
    """
    position: Position  # Using Position type instead of separate x,y
    color: ColorIndex
    pattern_type: PatternType
    size: PatternSize

    @property
    def min_x(self) -> int:
        """Minimum x coordinate of the pattern."""
        return self.position[0]

    @property
    def max_x(self) -> int:
        """Maximum x coordinate of the pattern."""
        if self.pattern_type == PatternType.HORIZONTAL:
            return self.position[0] + self.size - 1
        return self.position[0]

    @property
    def min_y(self) -> int:
        """Minimum y coordinate of the pattern."""
        return self.position[1]

    @property
    def max_y(self) -> int:
        """Maximum y coordinate of the pattern."""
        if self.pattern_type == PatternType.VERTICAL:
            return self.position[1] + self.size - 1
        return self.position[1]

    def get_pixels(self) -> Set[Position]:
        """Returns set of all pixel positions covered by this pattern."""
        pixels = set()
        if self.pattern_type == PatternType.SINGLE:
            pixels.add(self.position)
        elif self.pattern_type == PatternType.HORIZONTAL:
            pixels.update(Position((self.min_x + i, self.min_y)) for i in range(self.size))
        elif self.pattern_type == PatternType.VERTICAL:
            pixels.update(Position((self.min_x, self.min_y + i)) for i in range(self.size))
        return pixels

    def overlaps_with(self, other: 'Pattern') -> bool:
        """Check if this pattern overlaps with another pattern."""
        return bool(self.get_pixels() & other.get_pixels())

    def is_adjacent_to(self, other: 'Pattern', grid: Optional['Grid'] = None) -> Optional[Direction]:
        """
        Check if patterns are adjacent. Returns the direction of adjacency if found.
        Returns None if not adjacent.
        """
        my_pixels = self.get_pixels()
        other_pixels = other.get_pixels()
        for pixel in my_pixels:
            px, py = pixel
            for direction in Direction:
                dx, dy = direction.get_delta()
                adjacent_pos = Position((px + dx, py + dy))
                if grid is not None and not grid._is_within_bounds_pos(adjacent_pos):
                    continue
                if adjacent_pos in other_pixels:
                    # They are adjacent
                    return direction
                # If grid provided, check if there's a blocking pattern that prevents adjacency
                if grid is not None:
                    existing = grid.get_pattern_at(adjacent_pos)
                    if existing and existing not in (self, other):
                        # Blocking pattern found; adjacency in this direction is not possible
                        continue
        return None

    def can_merge_with(self, other: 'Pattern', grid: Optional['Grid'] = None) -> bool:
        """Check if this pattern can be merged with another pattern."""
        if self.color != other.color:
            return False
        # Get adjacency direction, considering grid if provided
        adj_direction = self.is_adjacent_to(other, grid)
        if not adj_direction:
            return False
        # Singles can always merge if adjacent and same color
        if self.pattern_type == PatternType.SINGLE and other.pattern_type == PatternType.SINGLE:
            return True
        # Handle merging singles with horizontal or vertical patterns
        if ((self.pattern_type == PatternType.SINGLE and other.pattern_type in [PatternType.HORIZONTAL,
                                                                                PatternType.VERTICAL]) or
            (other.pattern_type == PatternType.SINGLE and self.pattern_type in [PatternType.HORIZONTAL,
                                                                                PatternType.VERTICAL])):
            # Fixed selection of single and line patterns
            if self.pattern_type == PatternType.SINGLE:
                single, line = self, other
            else:
                single, line = other, self
            # Verify we actually got a line
            if line.pattern_type == PatternType.SINGLE:
                return False
            if line.pattern_type == PatternType.HORIZONTAL:
                # Check y alignment
                if single.min_y != line.min_y:
                    return False
                # Check if single is at either end of line
                is_left = single.min_x == line.min_x - 1
                is_right = single.min_x == line.min_x + line.size
                if is_left or is_right:
                    return True
                return False
            else:  # VERTICAL
                # Check x alignment
                if single.min_x != line.min_x:
                    return False
                # Check if single is at either end of line
                is_top = single.min_y == line.min_y - 1
                is_bottom = single.min_y == line.min_y + line.size
                if is_top or is_bottom:
                    return True
                return False
        # Same-type merge case
        if self.pattern_type == other.pattern_type:
            if self.pattern_type == PatternType.HORIZONTAL and self.min_y == other.min_y:
                is_adjacent = self.max_x + 1 == other.min_x or other.max_x + 1 == self.min_x
                return is_adjacent
            elif self.pattern_type == PatternType.VERTICAL and self.min_x == other.min_x:
                is_adjacent = self.max_y + 1 == other.min_y or other.max_y + 1 == self.min_y
                return is_adjacent
        return False

    def merge_with(self, other: 'Pattern', grid: Optional['Grid'] = None) -> Optional['Pattern']:
        """
        Attempt to merge with another pattern.
        Returns new merged pattern if possible, None if not possible.
        Now checks for potential blocking situations at endpoints.
        """
        if not self.can_merge_with(other, grid):
            return None

        # Get merge direction
        adj_direction = self.is_adjacent_to(other, grid)

        # Create potential merged pattern (existing logic)
        if self.pattern_type == PatternType.SINGLE and other.pattern_type == PatternType.SINGLE:
            # Use direction to determine orientation
            if adj_direction in [Direction.LEFT, Direction.RIGHT]:
                new_type = PatternType.HORIZONTAL
            else:
                new_type = PatternType.VERTICAL
            new_pos = Position((min(self.min_x, other.min_x), min(self.min_y, other.min_y)))
            new_pattern = Pattern(new_pos, self.color, new_type, PatternSize(2))
        elif self.pattern_type == other.pattern_type:
            # Preserve pattern type
            new_type = self.pattern_type
            if new_type == PatternType.HORIZONTAL:
                new_x = min(self.min_x, other.min_x)
                max_x = max(self.max_x, other.max_x)
                new_size = max_x - new_x + 1
                new_pos = Position((new_x, self.min_y))
            else:  # VERTICAL
                new_y = min(self.min_y, other.min_y)
                max_y = max(self.max_y, other.max_y)
                new_size = max_y - new_y + 1
                new_pos = Position((self.min_x, new_y))
            new_pattern = Pattern(new_pos, self.color, new_type, PatternSize(new_size))
        else:
            # Single + Line merge - preserve line orientation
            if self.pattern_type == PatternType.SINGLE:
                single, line = self, other
            else:
                single, line = other, self
            new_type = line.pattern_type
            if new_type == PatternType.HORIZONTAL:
                new_x = min(single.min_x, line.min_x)
                max_x = max(single.min_x, line.max_x)
                new_size = max_x - new_x + 1
                new_pos = Position((new_x, line.min_y))
            else:  # VERTICAL
                new_y = min(single.min_y, line.min_y)
                max_y = max(single.min_y, line.max_y)
                new_size = max_y - new_y + 1
                new_pos = Position((line.min_x, new_y))
            new_pattern = Pattern(new_pos, self.color, new_type, PatternSize(new_size))

        # Validate pixels
        new_pixels = new_pattern.get_pixels()
        original_pixels = self.get_pixels() | other.get_pixels()
        if new_pixels != original_pixels:
            return None

        if grid:
            # First check basic grid validation
            if grid.validate_merge(self, other, new_pattern) != ValidationResult.VALID:
                return None

            # Now check for blocking situations at endpoints
            if new_pattern.pattern_type == PatternType.VERTICAL:
                # Check bottom endpoint for potential horizontal merges it might block
                bottom_pos = Position((new_pattern.min_x, new_pattern.max_y))

                # Look for potential horizontal merges that would be blocked
                # Check positions to the left and right of the bottom point
                left_pos = Position((bottom_pos[0] - 1, bottom_pos[1]))
                right_pos = Position((bottom_pos[0] + 1, bottom_pos[1]))

                if grid._is_within_bounds_pos(left_pos) and grid._is_within_bounds_pos(right_pos):
                    left_pattern = grid.get_pattern_at(left_pos)
                    right_pattern = grid.get_pattern_at(right_pos)

                    # If both sides have patterns of the same color
                    if (left_pattern and right_pattern and
                            left_pattern.color == right_pattern.color == self.color and
                            left_pattern != self and right_pattern != self and
                            left_pattern != other and right_pattern != other):
                        # This merge would block a potential horizontal merge
                        return None

            elif new_pattern.pattern_type == PatternType.HORIZONTAL:
                # Similar check for horizontal patterns blocking vertical merges
                right_pos = Position((new_pattern.max_x, new_pattern.min_y))

                # Look for potential vertical merges that would be blocked
                top_pos = Position((right_pos[0], right_pos[1] - 1))
                bottom_pos = Position((right_pos[0], right_pos[1] + 1))

                if grid._is_within_bounds_pos(top_pos) and grid._is_within_bounds_pos(bottom_pos):
                    top_pattern = grid.get_pattern_at(top_pos)
                    bottom_pattern = grid.get_pattern_at(bottom_pos)

                    if (top_pattern and bottom_pattern and
                            top_pattern.color == bottom_pattern.color == self.color and
                            top_pattern != self and bottom_pattern != self and
                            top_pattern != other and bottom_pattern != other):
                        # This merge would block a potential vertical merge
                        return None

        return new_pattern

    def split_at(self, position: int, grid: Optional['Grid'] = None) -> tuple['Pattern', 'Pattern']:
        """
        Split pattern at given position into two patterns.
        If grid is provided, validates against grid constraints.
        """
        if position <= 0 or position >= self.size:
            raise ValueError("Invalid split position")
        if self.pattern_type == PatternType.SINGLE:
            raise ValueError("Cannot split single pixel pattern")
        # Create first part
        pattern1 = Pattern(self.position, self.color, self.pattern_type, PatternSize(position))
        # Create second part
        if self.pattern_type == PatternType.HORIZONTAL:
            x2 = self.min_x + position
            y2 = self.min_y
        else:  # VERTICAL
            x2 = self.min_x
            y2 = self.min_y + position
        pattern2 = Pattern(Position((x2, y2)), self.color, self.pattern_type, PatternSize(self.size - position))
        # Validate against grid if provided
        if grid:
            if (grid.validate_pattern_placement(pattern1) != ValidationResult.VALID or
                grid.validate_pattern_placement(pattern2) != ValidationResult.VALID):
                raise ValueError("Split patterns invalid in grid context")
        return pattern1, pattern2

from enum import Enum, auto
from typing import Tuple, NewType


# Basic pattern types
class PatternType(Enum):
    SINGLE = 0  # Single pixel
    HORIZONTAL = 1  # Horizontal line
    VERTICAL = 2  # Vertical line


# Move types for simulated annealing
class MoveType(Enum):
    PATTERN_MERGE = auto()  # Combine patterns
    PATTERN_BREAK_RECOMBINE = auto()  # Split and recombine


# Custom types for clarity
Position = NewType('Position', Tuple[int, int])
ColorIndex = NewType('ColorIndex', int)
PatternSize = NewType('PatternSize', int)


# Direction helpers
class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    def get_delta(self) -> Position:
        return Position(self.value)


# Validation types
class ValidationResult(Enum):
    VALID = auto()
    INVALID_BOUNDS = auto()
    INVALID_OVERLAP = auto()
    INVALID_COLOR = auto()
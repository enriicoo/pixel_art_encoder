from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import re

from core.types import ColorIndex


@dataclass
class ColorMap:
    """Maps between color strings and integer indices."""
    color_to_index: Dict[str, ColorIndex]
    index_to_color: Dict[ColorIndex, str]


class ColorMapper:
    """
    Manages color indexing and normalization for SVG colors.
    Only processes explicit hex color values.
    """

    def __init__(self):
        """Initialize with empty color mapping."""
        self._color_to_index: Dict[str, ColorIndex] = {}
        self._index_to_color: Dict[ColorIndex, str] = {}
        self._next_index: int = 0

    def get_or_create_index(self, color: str) -> Optional[ColorIndex]:
        """
        Get index for color, creating new mapping if needed.
        Only processes explicit hex colors, returns None for others.
        """
        # Skip if color is None or empty
        if not color:
            return None

        # Skip pattern definitions or non-hex colors
        if color.lower() in ('none', 'transparent', 'currentcolor'):
            return None

        # Only process hex colors
        if not color.startswith('#'):
            return None

        try:
            normalized = self._normalize_color(color)
            if normalized not in self._color_to_index:
                self._color_to_index[normalized] = ColorIndex(self._next_index)
                self._index_to_color[ColorIndex(self._next_index)] = normalized
                self._next_index += 1
            return self._color_to_index[normalized]
        except ValueError:
            return None

    def get_color(self, index: ColorIndex) -> Optional[str]:
        """Get color string for index, if it exists."""
        return self._index_to_color.get(index)

    def _normalize_color(self, color: str) -> str:
        """
        Normalize color to 6-digit hex format (#RRGGBB).
        Only handles explicit hex colors.
        """
        color = color.lower().strip()

        # Already normalized 6-digit hex
        if re.match(r'^#[0-9a-f]{6}$', color):
            return color

        # 3-digit hex
        if re.match(r'^#[0-9a-f]{3}$', color):
            return '#' + ''.join(c + c for c in color[1:])

        raise ValueError(f'Not a hex color: {color}')


class ColorGroup:
    """
    Groups pixels by color for connected component analysis.
    Tracks positions and provides iteration over color groups.
    """

    def __init__(self, color_mapper: ColorMapper):
        """Initialize with color mapper instance."""
        self._color_mapper = color_mapper
        self._positions: Dict[ColorIndex, List[tuple[int, int]]] = {}

    def add_pixel(self, x: int, y: int, color: str):
        """Add pixel position to appropriate color group."""
        color_index = self._color_mapper.get_or_create_index(color)
        if color_index not in self._positions:
            self._positions[color_index] = []
        self._positions[color_index].append((x, y))

    def get_positions(self, color_index: ColorIndex) -> List[tuple[int, int]]:
        """Get all pixel positions for a given color index."""
        return self._positions.get(color_index, []).copy()

    def get_color_indices(self) -> Set[ColorIndex]:
        """Get set of all color indices with pixels."""
        return set(self._positions.keys())

    def __iter__(self):
        """Iterate over (color_index, positions) pairs."""
        return iter(self._positions.items())
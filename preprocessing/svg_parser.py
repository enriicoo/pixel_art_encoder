from typing import Tuple, Optional
import xml.etree.ElementTree as ET
import re
from dataclasses import dataclass

from core.types import Position
from core.grid import Grid
from core.pattern import Pattern, PatternType, PatternSize
from .color_mapper import ColorMapper

@dataclass
class SVGDimensions:
    """Stores SVG canvas dimensions and pixel size."""
    width: int
    height: int
    pixel_size: float


@dataclass
class SVGParserOptions:
    """Configuration options for SVG parsing."""
    single_color: Optional[str] = None  # "black", "white", or hex color


class SVGParser:
    """
    SVG Parser for pixel art where all pixels are the same size.
    Automatically detects pixel size from the input.
    """

    def __init__(self, options: Optional[SVGParserOptions] = None):
        """Initialize parser with color mapper."""
        self.color_mapper = ColorMapper()
        self.options = options or SVGParserOptions()

    def parse(self, svg_content: str) -> Tuple[Grid, ColorMapper]:
        """
        Parse SVG content into Grid representation.
        Only processes rect elements with explicit hex color fills.
        Automatically detects pixel size from the first rectangle.
        """
        try:
            root = ET.fromstring(svg_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid SVG XML: {e}")

        # Get SVG namespace if present
        ns = self._get_namespace(root.tag)
        ns_dict = {'svg': ns} if ns else None

        # Find all rectangles, excluding those in defs
        rects = []
        defs_elements = root.findall('.//defs') if ns_dict is None else root.findall('.//svg:defs', ns_dict)
        defs_rect_elements = []

        # Collect all rectangles that are inside defs
        for defs in defs_elements:
            defs_rects = defs.findall('.//rect') if ns_dict is None else defs.findall('.//svg:rect', ns_dict)
            defs_rect_elements.extend(defs_rects)

        # Find all rectangles
        all_rects = root.findall('.//rect') if ns_dict is None else root.findall('.//svg:rect', ns_dict)

        # Only include rectangles that aren't in defs
        rects = [rect for rect in all_rects if rect not in defs_rect_elements]

        if not rects:
            raise ValueError("No valid rectangles found in SVG")

        # Detect pixel size from first rectangle
        try:
            pixel_size = float(rects[0].get('width', '0'))
            if pixel_size <= 0:
                raise ValueError("Invalid pixel size")
            if float(rects[0].get('height', '0')) != pixel_size:
                raise ValueError("Non-square pixels detected")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not determine pixel size: {e}")

        # Get grid dimensions based on detected pixel size
        grid_width = max(
            int(float(rect.get('x', '0')) + float(rect.get('width', '0')))
            for rect in rects
        ) // int(pixel_size) + 1

        grid_height = max(
            int(float(rect.get('y', '0')) + float(rect.get('height', '0')))
            for rect in rects
        ) // int(pixel_size) + 1

        # Create grid
        grid = Grid(width=grid_width, height=grid_height)

        # Process each rectangle
        for rect in rects:
            try:
                # Get position and color
                x = int(float(rect.get('x', '0'))) // int(pixel_size)
                y = int(float(rect.get('y', '0'))) // int(pixel_size)
                color = rect.get('fill')

                # Skip if no color or non-hex color
                if not color or not color.startswith('#'):
                    continue

                # If single_color option is set, skip non-matching colors
                if self.options.single_color and color.lower() != self.options.single_color.lower():
                    continue

                # Verify this rectangle matches the pixel size
                width = float(rect.get('width', '0'))
                height = float(rect.get('height', '0'))
                if width != pixel_size or height != pixel_size:
                    print(f"Warning: Skipping rectangle with non-standard size at ({x}, {y})")
                    continue

                # Get color index
                color_index = self.color_mapper.get_or_create_index(color)
                if color_index is None:
                    continue

                # Create and add pattern
                pattern = Pattern(position=Position((x, y)), color=color_index, pattern_type=PatternType.SINGLE,
                                  size=PatternSize(1))
                grid.add_pattern(pattern)

            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid rectangle: {e}")
                continue

        if len(grid.patterns) == 0:
            if self.options.single_color:
                raise ValueError(f"No pixels found with color {self.options.single_color}")
            else:
                raise ValueError("No valid pixels found in SVG")

        return grid, self.color_mapper

    def _get_namespace(self, tag: str) -> Optional[str]:
        """Extract namespace from element tag if present."""
        ns_match = re.match(r'\{(.+?)}', tag)
        return ns_match.group(1) if ns_match else None

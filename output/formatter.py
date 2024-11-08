from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, ClassVar
from PIL import Image, ImageDraw, ImageFont
import io

from core.grid import Grid
from core.types import ColorIndex, Position, PatternType
from preprocessing.color_mapper import ColorMapper
from .encoder import PatternEncoder, EncodedPattern, ColorDefinition, GridStatistics


@dataclass
class OptimizationResults:
    """Contains complete optimization results and statistics."""
    color_map: List[ColorDefinition]
    patterns: List[EncodedPattern]
    statistics: Dict[str, int]
    original_statistics: Optional[Dict[str, int]] = None


@dataclass
class VisualizationSnapshot:
    """Represents a snapshot of the optimization process."""
    grid: Grid
    cost: float
    iteration: int


class OutputFormatter:
    """
    Formats optimization results into various output formats.
    Supports plain text, CSV, JSON, and PNG visualization.
    """
    # Class variable to store consistent pattern type colors
    PATTERN_COLORS: ClassVar[Dict[PatternType, Tuple[int, int, int, int]]] = {
        PatternType.SINGLE: (231, 76, 60, 255),      # Soft red
        PatternType.HORIZONTAL: (46, 204, 113, 255), # Soft green
        PatternType.VERTICAL: (52, 152, 219, 255)    # Soft blue
    }

    def __init__(self, grid: Grid, color_mapper: ColorMapper,
                 original_grid: Optional[Grid] = None):
        """
        Initialize formatter with optimized grid and color mapping.

        Args:
            grid: Optimized pattern grid
            color_mapper: Color mapping used
            original_grid: Optional original grid for comparison stats
        """
        self.encoder = PatternEncoder(color_mapper)
        color_defs, patterns = self.encoder.encode_grid(grid)

        # Calculate statistics
        stats = GridStatistics.calculate(grid)
        original_stats = None
        if original_grid:
            original_stats = GridStatistics.calculate(original_grid)

        self.results = OptimizationResults(color_map=color_defs, patterns=patterns,
                                           statistics=stats, original_statistics=original_stats)

        self.grid = grid
        self.color_mapper = color_mapper

    def format_patterns_output(self) -> str:
        """Format complete pattern output including color map."""
        # Start with color map
        output_lines = ["Color Map:"]
        for color_def in self.results.color_map:
            output_lines.append(f"{color_def.index}: {color_def.color}")
        output_lines.append("")
        # Add encoded pixel data
        output_lines.append("Encoded Pixel Data:")
        for pattern in self.results.patterns:
            output_lines.append(pattern.to_string())
        return "\n".join(output_lines)

    def to_visualizer(self, pixel_size: int = 20) -> bytes:
        """Generate PNG visualization of the grid patterns with colors."""
        # Calculate dimensions
        width = self.grid.width * pixel_size
        height = self.grid.height * pixel_size

        # Create image with white background
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img, 'RGBA')

        # Draw patterns
        inner_padding = 1
        for pattern in self.grid:
            pixels = pattern.get_pixels()

            # Draw actual color fill
            color = self.color_mapper.get_color(pattern.color)
            if color:
                # Convert hex color to RGB tuple
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    rgb_color = (r, g, b, int(255//4))
                    for px, py in pixels:
                        x = px * pixel_size
                        y = py * pixel_size
                        x0 = x + inner_padding
                        y0 = y + inner_padding
                        x1 = x + pixel_size - inner_padding
                        y1 = y + pixel_size - inner_padding
                        # Draw rectangle with actual color
                        draw.rectangle(((x0, y0), (x1, y1)), fill=rgb_color)

            # Draw outline based on pattern type
            outline_color = self.PATTERN_COLORS[pattern.pattern_type]
            self._draw_pattern_outline(draw, pixels, pixel_size, outline_color)

        # Draw background
        background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        final_img = Image.alpha_composite(background, img)

        # Add this block to draw the text
        draw_final = ImageDraw.Draw(final_img)
        text = f"Patterns: {len(self.grid.patterns)}"
        font = ImageFont.load_default(size=40)
        left, top, right, bottom = draw_final.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        text_x = width - text_width - 40
        text_y = height - text_height - 40
        draw_final.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        final_img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def _draw_pattern_outline(self, draw: ImageDraw.ImageDraw,
                            pixels: set[Position], pixel_size: int,
                            color: Tuple[int, int, int, int]):
        """Draw outline around a group of pixels with specified color."""
        if not pixels:
            return

        pixel_set = set(pixels)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for x, y in pixels:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in pixel_set:
                    # Calculate line coordinates
                    x0 = x * pixel_size
                    y0 = y * pixel_size
                    x1 = x0 + pixel_size
                    y1 = y0 + pixel_size

                    if dx == -1:  # Left edge
                        draw.line([(x0, y0), (x0, y1)], fill=color, width=4)
                    elif dx == 1:  # Right edge
                        draw.line([(x1, y0), (x1, y1)], fill=color, width=4)
                    elif dy == -1:  # Top edge
                        draw.line([(x0, y0), (x1, y0)], fill=color, width=4)
                    elif dy == 1:  # Bottom edge
                        draw.line([(x0, y1), (x1, y1)], fill=color, width=4)

    def _create_single_visualization(self,
                                     color_index: Optional[ColorIndex],
                                     target_color: str,
                                     show_patterns: bool,
                                     pixel_size: int) -> bytes:
        """Create single image visualization."""
        # Calculate dimensions
        width = self.grid.width * pixel_size
        height = self.grid.height * pixel_size
        # Create image
        img = Image.new('RGB', (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Convert target color
        if target_color.lower() == 'white':
            target_rgb = (255, 255, 255)
        elif target_color.lower() == 'black':
            target_rgb = (0, 0, 0)
        else:
            # Convert hex to RGB, ensuring exactly 3 values
            r = int(target_color[1:3], 16)
            g = int(target_color[3:5], 16)
            b = int(target_color[5:7], 16)
            target_rgb = (r, g, b)
        if show_patterns:
            img = self._draw_patterns(img, draw, color_index, pixel_size)
        else:
            img = self._draw_simple(img, draw, color_index, target_rgb, pixel_size)
        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def _draw_simple(self, img: Image.Image, draw: ImageDraw.ImageDraw,
                     color_index: Optional[ColorIndex], target_rgb: Tuple[int, int, int],
                     pixel_size: int) -> Image.Image:
        """Draw simple visualization with single color."""
        for pattern in self.grid:
            if color_index is None or pattern.color == color_index:
                pixels = pattern.get_pixels()
                for px, py in pixels:
                    x = px * pixel_size
                    y = py * pixel_size
                    draw.rectangle(((x, y), (x + pixel_size - 1, y + pixel_size - 1)),fill=target_rgb)
        return img

    def _draw_patterns(self, img: Image.Image, draw: ImageDraw.ImageDraw,
                       color_index: Optional[ColorIndex],
                       pixel_size: int) -> Image.Image:
        """Draw patterns with different colors and groupings."""
        # Create a mapping from pattern kinds to outline colors
        inner_padding = pixel_size // 3  # Adjust this value as needed

        for pattern in self.grid:
            if color_index is None or pattern.color == color_index:
                pixels = pattern.get_pixels()

                # Get the actual color of the pattern, assuming it is an RGB tuple
                color_idx = pattern.color
                actual_color = self.color_mapper.get_color(color_idx)
                # Draw smaller pixels with actual color
                for px, py in pixels:
                    x = px * pixel_size
                    y = py * pixel_size
                    # Draw smaller rectangle inside the pixel grid cell
                    draw.rectangle(((x + inner_padding, y + inner_padding),
                                    (x + pixel_size - inner_padding - 1, y + pixel_size - inner_padding - 1)),
                                   fill=actual_color)

                # Get the outline color based on pattern type
                outline_color = self.PATTERN_COLORS[pattern.pattern_type]

                # Draw group outline
                self._draw_group_outline(draw, pixels, pixel_size, outline_color)
        return img

    def _draw_group_outline(self, draw: ImageDraw.ImageDraw,
                            pixels: Set[Position], pixel_size: int,
                            color: Tuple[int, int, int]):
        """Draw outline around a group of pixels."""
        if not pixels:
            return

        # Create a set of pixel positions for faster lookup
        pixel_set = set(pixels)

        # Define directions to check for edges
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # For each pixel, check if adjacent pixels are outside the group
        for x, y in pixels:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in pixel_set:
                    # Draw line along the edge
                    x0 = x * pixel_size
                    y0 = y * pixel_size
                    x1 = x0 + pixel_size
                    y1 = y0 + pixel_size

                    if dx == -1:  # Left edge
                        draw.line([(x0, y0), (x0, y1)], fill=color, width=2)
                    elif dx == 1:  # Right edge
                        draw.line([(x1, y0), (x1, y1)], fill=color, width=2)
                    elif dy == -1:  # Top edge
                        draw.line([(x0, y0), (x1, y0)], fill=color, width=2)
                    elif dy == 1:  # Bottom edge
                        draw.line([(x0, y1), (x1, y1)], fill=color, width=2)

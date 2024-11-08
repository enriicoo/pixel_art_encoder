from PIL import Image
import numpy as np
import io


def analyze_pixel_image(image_path):
    """
    Analyzes an image and returns its pixel data and a string representation
    """
    # Open and convert image to RGB
    img = Image.open(image_path)
    rgb_img = img.convert('RGBA')

    # Get dimensions
    width, height = img.size

    # Convert to numpy array for easier processing
    pixels = np.array(rgb_img)

    # Create a dictionary to store unique colors and their counts
    color_counts = {}
    pixel_data = []

    # Process each pixel
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b, a = pixels[y, x]
            if a == 0:
                color_hex = "transparent"
            else:
                color_hex = f"#{r:02x}{g:02x}{b:02x}"

            row.append(color_hex)

            if color_hex not in color_counts:
                color_counts[color_hex] = 0
            color_counts[color_hex] += 1
        pixel_data.append(row)

    # Create a compact representation of the image
    image_repr = []
    image_repr.append(f"Image size: {width}x{height}")
    image_repr.append("\nUnique colors used:")
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        if color != "transparent":
            image_repr.append(f"- {color}: {count} pixels")

    # Create a simplified ASCII visualization
    image_repr.append("\nSimplified visualization:")
    for y in range(height):
        row = ""
        for x in range(width):
            color = pixel_data[y][x]
            if color == "transparent":
                row += "."
            else:
                row += "█"
        if row.strip("."):  # Only add non-empty rows
            image_repr.append(row)

    return pixel_data, "\n".join(image_repr)


def detect_pixel_size(pixel_data):
    """
    Detect the size of a single 'art pixel' by looking for the smallest repeating block
    """
    height = len(pixel_data)
    width = len(pixel_data[0])

    def check_block_size(y, x, size):
        """Check if a block of given size is a solid color"""
        if y + size > height or x + size > width:
            return False
        color = pixel_data[y][x]
        for dy in range(size):
            for dx in range(size):
                if pixel_data[y + dy][x + dx] != color:
                    return False
        return True

    # Look for the first non-transparent pixel
    start_y, start_x = 0, 0
    for y in range(height):
        for x in range(width):
            if pixel_data[y][x] != "transparent":
                start_y, start_x = y, x
                break
        if start_y != 0 or start_x != 0:
            break

    # Try increasing block sizes until we find one that doesn't match
    pixel_size = 1
    while pixel_size < min(width, height) // 2:
        if not check_block_size(start_y, start_x, pixel_size + 1):
            break
        pixel_size += 1

    return pixel_size

def create_svg_from_pixels(pixel_data):
    """
    Creates an SVG from pixel data using detected pixel size
    """
    height = len(pixel_data)
    width = len(pixel_data[0]) if height > 0 else 0

    # Detect the size of each "art pixel"
    pixel_size = detect_pixel_size(pixel_data)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<pattern id="pixel" patternUnits="userSpaceOnUse" width="1" height="1">',
        '<rect width="1" height="1" fill="currentColor"/>',
        '</pattern>',
        '</defs>'
    ]

    # Create rectangles for each "art pixel"
    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            color = pixel_data[y][x]
            if color != "transparent":
                # Check if this is actually a full pixel (all subpixels same color)
                is_full_pixel = True
                for dy in range(pixel_size):
                    for dx in range(pixel_size):
                        if (y + dy < height and x + dx < width and
                                pixel_data[y + dy][x + dx] != color):
                            is_full_pixel = False
                            break
                    if not is_full_pixel:
                        break

                if is_full_pixel:
                    svg.append(
                        f'<rect x="{x}" y="{y}" width="{pixel_size}" '
                        f'height="{pixel_size}" fill="{color}"/>'
                    )

    svg.append('</svg>')
    return '\n'.join(svg)

def save_outputs(image_path, output_prefix="output"):
    """
    Processes an image and saves all outputs
    """
    # Analyze image and create SVG
    pixel_data, analysis = analyze_pixel_image(image_path)
    svg_output = create_svg_from_pixels(pixel_data)

    # Save analysis
    with open(f"{output_prefix}_analysis.txt", 'w', encoding='utf-8') as f:
        f.write(analysis)

    # Save SVG
    with open(f"{output_prefix}.svg", 'w', encoding='utf-8') as f:
        f.write(svg_output)

    # Save HTML display
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pixel Art Display</title>
        <style>
            body {{ display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
            svg {{ max-width: 90vw; max-height: 90vh; }}
        </style>
    </head>
    <body>
        {svg_output}
    </body>
    </html>
    """
    with open(f"{output_prefix}.html", 'w') as f:
        f.write(html_content)

    return analysis, svg_output


# Example usage
if __name__ == "__main__":
    # Specify your image path and output prefix here
    image_path = "resources/pikachu.png"
    output_prefix = "resources/pikachu"

    analysis, svg = save_outputs(image_path, output_prefix)
    print(f"\nFiles created: {output_prefix}_analysis.txt, {output_prefix}.svg, {output_prefix}.html")
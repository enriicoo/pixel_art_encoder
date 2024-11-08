from pathlib import Path
import json
from typing import Optional

from preprocessing.svg_parser import SVGParser, SVGParserOptions
from optimization.annealing import PatternOptimizer
from optimization.cost_function import CostWeights
from optimization.temperature import TemperatureParameters
from output.formatter import OutputFormatter

def optimize_svg(svg_path: str, output_dir: str, single_color: Optional[str] = "#000000"):
    """
    Process SVG file and generate optimized pattern output.

    Args:
        svg_path: Path to input SVG file
        output_dir: Directory for output files
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Parse SVG
    svg_content = Path(svg_path).read_text()
    parser_options = SVGParserOptions(single_color=single_color)
    parser = SVGParser(options=parser_options)
    initial_grid, color_mapper = parser.parse(svg_content)
    print("\nAfter parsing:")
    print(f"Color mapper state:")
    print(f"- Color to index: {color_mapper._color_to_index}")
    print(f"- Index to color: {color_mapper._index_to_color}")

    # Configure optimization
    cost_weights = CostWeights(pattern_weight=5.0, single_penalty=0.1, transition_weight=0.0)
    temp_params = TemperatureParameters(initial_accept_ratio=0.8, final_accept_ratio=0.2, alpha=0.85,
                                        min_iterations=100, max_iterations=400, iterations_multiplier=2)
    # Run optimization
    optimizer = PatternOptimizer(cost_weights=cost_weights, temp_params=temp_params, color_mapper=color_mapper)
    optimized_grid, stats = optimizer.optimize(initial_grid)
    # Format output
    formatter = OutputFormatter(grid=optimized_grid, color_mapper=optimizer.color_mapper, original_grid=initial_grid)
    # Generate encoded patterns file
    patterns_output = [pattern.to_string() for pattern in formatter.results.patterns]
    (output_path / "encoded_pixels.txt").write_text("\n".join(patterns_output))
    # Generate statistics file
    stats_path = output_path / "statistics.json"
    stats_dict = vars(stats)
    stats_path.write_text(json.dumps(stats_dict, indent=2))

if __name__ == "__main__":
    input_svg_path = "resources/pikachu.svg" # Put the path to your SVG file here
    output_path = "results/outputs" # This is where the output files will be created
    optimize_svg(input_svg_path, output_path)
    print(f"Processing complete! Check the {output_path} directory for results.")
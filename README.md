# pixel_art_encoder
Encodes a pixel art into a .txt file without losing information

After dealing a little with pixel art, I started asking myself "how to encode this information with less lines as possible?". The result was learning a whole lot about heuristic algorithms, genetic algorithms, and simulated annealing, the latter being the choice for this work.

Proposed changes:
- Refactoring 
- Add diagonals, maybe squares
- Make the performance around big blocks a little better (my main focus was on slim cases)

Features

- Converts PNG pixel art into SVG (but it's basically a XML)
- Converts SVG pixel art into optimized pattern sets
- Supports single pixel, horizontal line, and vertical line patterns
- Uses simulated annealing for optimization
- Clustering tries to approximate the local solutions to the general solution from time to time, as patterns counts stagnate
- Generates visual snapshots of each stage in the optimization process
- Provides detailed statistics and analysis

Project structure in structure.txt

Contributing
- Pull requests are welcome.

Quick Start:
1. Convert your PNG to SVG using png_to_svg.py
2. Choose input and output path in use_encoder.py

Adapt as to fulfill your needs!

<img src="https://github.com/user-attachments/assets/50597dcc-49a2-415a-8a66-906e5ed0d2d5" width="400" height="400" alt="simulated_annealing_in_action">


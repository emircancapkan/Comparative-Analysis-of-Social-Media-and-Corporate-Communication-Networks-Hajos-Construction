# Comparative Analysis of Social Media and Corporate Communication Networks: An Assessment from the Perspective of the Hajos Construction

This project analyzes network structures using Facebook social network and corporate email network data, comparing them from the perspective of Hajos construction.

## Requirements

The following Python packages are required to run the project:

- networkx
- matplotlib
- numpy
- python-louvain (community)

## Installation

1. Make sure Python 3.x is installed.

2. (Recommended) Create a new virtual environment:
   ```bash
   python3 -m venv myprojectenv
   ```

3. Activate the virtual environment:
   - For macOS/Linux:
     ```bash
     source myprojectenv/bin/activate
     ```
   - For Windows:
     ```bash
     myprojectenv\Scripts\activate
     ```

4. Install required packages:
   ```bash
   pip install networkx matplotlib numpy python-louvain
   ```

## Usage

1. Ensure that the data files (`facebook_combined.txt` and `email-Eu-core.txt`) are present in the project directory.

2. To run the analysis:
   ```bash
   python3 analyze.py
   ```

3. When the program runs, it will perform the following analyses:
   - Chromatic number analysis
   - Subgraph analysis
   - Community structure analysis
   - Comparison from Hajos construction perspective

4. Results will be saved as PNG files:
   - facebook_graph_coloring_coloring.png
   - email_graph_coloring_coloring.png
   - facebook_community_metrics.png
   - email_community_metrics.png
   - fb_vs_email_hajos_comparison.png

## Note

If the data files are not found, the program will automatically create sample graphs and perform the analyses on them.

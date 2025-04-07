# Project Overview
Web-scale graphs, such as hyperlink graphs of the internet, can contain billions of nodes and edges, making analysis computationally expensive. This project explores techniques to reduce graph size while preserving multiple graph properties, enabling more efficient analysis while maintaining accuracy.

## Implemented Methods
1. **Structural Property Summarization**
   - Groups nodes based on structural properties (communities, degree, clustering)
   - Preserves community structure and local connectivity patterns
2. **Spectral Summarization**
   - Preserves spectral properties of the graph
   - Maintains random walk behavior and graph connectivity characteristics
3. **Hybrid Summarization**
   - Two-stage approach that combines structural grouping with edge refinement
   - Balances multiple property preservation objectives
4. **Semantic-Aware Summarization**
   - Incorporates node attributes or metadata
   - Ensures semantically coherent nodes are grouped together
5. **Graph Sampling-Based Summarization**
   - Uses strategic node sampling (random walk, forest fire, snowball)
   - Produces representative subgraphs that preserve key properties
6. **Edge Sampling-Based Summarization**
   - Various edge sampling strategies (uniform, degree-biased, spanning tree, spectral)
   - Maintains connectivity while reducing edges

## Directory Structure

## Directory Structure

├── graph_properties_framework.py  # Core implementation of property-preserving summarization
├── experiment_runner.py           # Script to run experiments on different methods
├── web_graph_demo.py              # Demo script for web graph datasets
├── semantic_summarization.py      # Implementation of semantic-aware summarization
├── run_all.py                     # Main script to run all experiments
├── data/                          # Directory for datasets
├── results/                       # Directory for experiment results
├── plots/                         # Directory for visualizations
└── models/                        # Directory for saved models


## Installation

### Requirements
- Python 3.7+  
- networkx  
- numpy  
- pandas  
- matplotlib  
- scipy  
- scikit-learn  
- tqdm  
- python-louvain (community)  
- seaborn

You can install all dependencies using:
```
pip install networkx numpy pandas matplotlib scipy scikit-learn tqdm python-louvain seaborn
```

## Usage
### Running All Experiments
To run all experiments with default settings:
```
python run_all.py
```
This will:
- Download required datasets
- Run experiments with various summarization methods
- Evaluate preservation of multiple graph properties
- Generate visualizations of results

### Command-line Options
```
python run_all.py --datasets web-Stanford web-Google --reductions 0.1 0.2 0.3
```

### Additional options:
--skip_download: Skip dataset download
--methods: Specify which summarization methods to evaluate
--base_dir PATH: Base directory for the project

## Running Individual Components
### Property Evaluation Experiments
```python experiment_runner.py --dataset data/web-Stanford.txt --dataset_type snap --methods structural_community spectral hybrid --reduction 0.1 --output_dir results/web-Stanford
```
Web Graph Demo
```
python web_graph_demo.py --dataset web-Stanford --data_dir data --methods structural_community spectral hybrid semantic --reduction 0.1
```
### Semantic-Aware Summarization
```
python semantic_summarization.py --graph_file data/web-Stanford.txt --attribute_file data/web-Stanford-attributes.txt --method tfidf --clusters 50 --output_dir results/semantic
```

## Evaluation Metrics
The framework evaluates summarization techniques using multiple metrics:

1. PageRank Preservation
  - Spearman/Kendall correlation between original and summary PageRank vectors
  - Top-k node overlap for most important nodes
2. L1 Error
  - Difference between original and summary PageRank vectors
3. Centrality Preservation
  - Correlation for degree, eigenvector, and closeness centrality
  - Preservation of node importance rankings
4. Community Structure
  - Normalized Mutual Information (NMI) between original and summary communities
  - Adjusted Rand Index (ARI) for cluster similarity
5. Degree Distribution
  - KL divergence between original and summary degree distributions
  - Preservation of scale-free or other distributional properties
6. Clustering Coefficient
  - Difference in average clustering coefficient
  - Preservation of local graph density
7. Path Lengths
  - Ratio of characteristic path lengths
  - Preservation of graph diameter
8. Runtime and Compression
  - Node and edge compression ratios
  - Speedup for computational tasks like PageRank

## Datasets
The code is designed to work with SNAP and WebGraph datasets:

- web-Stanford: Web graph of Stanford.edu
- web-Google: Web graph from Google
- web-NotreDame: Web graph of Notre Dame
- web-BerkStan: Web graph of Berkeley and Stanford

The run_all.py script will automatically download these datasets from SNAP.

## Extending the Framework
To add a new summarization technique:

1. Create a new class that inherits from GraphSummarizer in graph_summarization.py
2. Implement the summarize() method with your technique
3. Add your method to the experiment runner

## Citation
If you use this code in your research, please cite:
@inproceedings{graphsummarization2025,
  title={PageRank-Preserving Graph Summarization Techniques for Web-Scale Graphs},
  author={Jiang, Evan and Jiang, Ryan and Han, Roy},
  booktitle={CS 2241 Project},
  year={2025}
}

## License
This project is licensed under the MIT License - see the LICENSE file for details.

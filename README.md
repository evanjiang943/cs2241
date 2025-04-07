# Graph Property-Preserving Summarization

A modular framework for creating and evaluating graph summaries that preserve important structural and analytical properties of the original graph, with a focus on web-scale networks.

## Overview

Large-scale graphs such as web graphs, social networks, and citation networks can be challenging to analyze due to computational constraints. This framework provides methods to create smaller summary graphs that preserve key properties of the original graph, making analysis more efficient while maintaining accuracy.

## Features

- **Multiple summarization techniques:**
  - Community-based summarization (Louvain method)
  - Spectral summarization (preserves spectral properties)
  - Customizable framework for adding new methods

- **Comprehensive property evaluation:**
  - PageRank preservation (correlation and error metrics)
  - Centrality preservation (degree, eigenvector)
  - Community structure preservation (NMI, ARI)
  - Degree distribution similarity
  - Clustering coefficient preservation
  - Path length characteristics
  - Runtime and compression metrics

- **Web-scale compatible:**
  - Memory-efficient data loading
  - Specialized algorithms for large graphs
  - Built-in support for SNAP web graph datasets

## Installation

### Using Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-summarization.git
cd graph-summarization

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate graph-sum

# Install package in development mode
pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-summarization.git
cd graph-summarization

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
import networkx as nx
from graphsum.summarizers.community import CommunityBasedSummarizer
from graphsum.evaluation.evaluator import GraphEvaluator

# Load a graph
G = nx.karate_club_graph()

# Create a summarizer
summarizer = CommunityBasedSummarizer()

# Generate a summary graph
summary = summarizer.summarize(G, reduction_factor=0.3)

# Evaluate how well properties are preserved
evaluator = GraphEvaluator(G, summary, summarizer.node_mapping, summarizer.reverse_mapping)
results = evaluator.evaluate_all()

# Print a summary of results
evaluator.print_summary()
```

## Working with Web-Scale Graphs

```bash
# List available SNAP datasets
python -m graphsum.io.snap list

# Download a dataset
python -m graphsum.io.snap download web-Stanford data/

# Run experiments
python scripts/run_experiment.py --dataset web-Stanford --methods community spectral --reductions 0.1 0.2 0.3
```

## Web Graph Demo

```bash
# Run demo on web-Stanford dataset
python examples/webgraph_demo.py --dataset web-Stanford --methods community --reduction 0.1
```

## Project Structure

```
graph_summarization/
├── graphsum/                  # Main package directory
│   ├── summarizers/           # Summarization algorithms
│   │   ├── base.py            # Base summarizer class
│   │   ├── community.py       # Community-based summarization
│   │   ├── spectral.py        # Spectral summarization
│   ├── evaluation/            # Evaluation metrics
│   │   ├── evaluator.py       # Main evaluation class
│   ├── io/                    # Input/output utilities
│   │   ├── snap.py            # SNAP dataset loader
├── scripts/                   # CLI scripts
│   ├── run_experiment.py      # Main experiment runner
├── examples/                  # Example usage
│   ├── webgraph_demo.py       # Demo on web-scale graph
├── data/                      # Dataset directory
├── results/                   # Results directory
```

## Extending the Framework

To implement a new summarization technique:

1. Create a new class that inherits from `GraphSummarizer` in `graphsum/summarizers/base.py`
2. Implement the `summarize()` method
3. Use the `GraphEvaluator` to evaluate your new method

Example:

```python
from graphsum.summarizers.base import GraphSummarizer

class MyCustomSummarizer(GraphSummarizer):
    def __init__(self, name="MyCustom"):
        super().__init__(name=name)
    
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        # Implement your summarization logic here
        # ...
        
        # Return the summary graph
        return self.summary_graph
```

## Evaluation Metrics

The `GraphEvaluator` provides comprehensive metrics to assess summary quality:

- **PageRank Preservation:**
  - Spearman/Kendall correlation
  - Top-k node overlap
  - L1 error

- **Centrality Preservation:**
  - Degree and eigenvector centrality correlation

- **Community Structure:**
  - NMI/ARI metrics comparing original and summary communities

- **Degree Distribution:**
  - KL divergence between distributions
  - Correlation of node degree values

- **Clustering and Path Metrics:**
  - Clustering coefficient comparison
  - Characteristic path length ratio
  - Diameter comparison

- **Performance:**
  - Compression ratios
  - Runtime improvements

## References

The framework implements and builds upon methods from:

- Riondato, M., & Vandin, F. (2017). Graph summarization with quality guarantees.
- Loukas, A. (2019). Graph reduction with spectral and cut guarantees.
- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances.
- Navlakha, S., Rastogi, R., & Shrivastava, N. (2008). Graph summarization with bounded error.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
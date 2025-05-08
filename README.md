# Graph Property-Preserving Summarization

This project implements and evaluates various graph summarization techniques with a focus on preserving important structural and analytical properties of the original graph. It addresses the challenge of analyzing large-scale graphs by creating compact graph summaries that maintain key properties such as community structure, centrality distributions, and spectral characteristics.

## Overview

Web-scale graphs, such as hyperlink graphs of the internet, can contain billions of nodes and edges, making analysis computationally expensive. This project explores and analyzes how well various graph reduction techniques perform on various metrics.

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

To quickly run experiments on a sample dataset:

```bash
# Run a simple experiment on a small dataset
python scripts/run.py --datasets web-NotreDame --methods collapse --compression-ratios 0.1
```

## Working with Web-Scale Graphs

```bash
# List available SNAP datasets
python -m graphsum.io.snap list

# Download a dataset
python -m graphsum.io.snap download web-Stanford data/

# Run experiments (using the three summarization methods)
python scripts/run.py --datasets web-Stanford --methods sparsifier collapse coarsener --compression-ratios 0.1 0.2

# For memory-intensive datasets, use the memory-efficient flag
python scripts/run.py \
  --datasets web-NotreDame \
  --methods collapse \
  --compression-ratios 0.01 \
  --memory-efficient
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
│   │   ├── spectral_sparsifier.py # Spectral sparsification
│   │   ├── spectral_coarsener.py  # Spectral coarsening
│   ├── evaluation/            # Evaluation metrics
│   │   ├── evaluator.py       # Main evaluation class
│   │   ├── metrics.py         # Core metrics implementation
│   ├── io/                    # Input/output utilities
│   │   ├── snap.py            # SNAP dataset loader
├── scripts/                   # CLI scripts
│   ├── run.py                 # Main experiment runner

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

The framework uses five core metrics to evaluate graph summaries:

- **Spectral Approximation Error:**
  - Measures how well the summary preserves spectral properties of the original graph
  - Quantifies the difference between the eigenvalues of the original and summary graphs

- **Community Structure Preservation:**
  - Normalized Mutual Information (NMI) comparing original and summary communities
  - Measures how well community structure is preserved in the summary

- **Distance Distortion (Average Stretch):**
  - Measures how well the summary preserves distances between nodes
  - Calculated as the average multiplicative increase in distances

- **Centrality Retention (Precision@k):**
  - Measures how well the summary preserves the most important nodes
  - Calculated as the overlap between top-k nodes in original and summary graphs

- **Compression Ratio:**
  - Quantifies the size reduction achieved by the summarization
  - Calculated as the ratio of summary size to original graph size

## References

The framework implements and builds upon methods from:

- Riondato, M., & Vandin, F. (2017). Graph summarization with quality guarantees.
- Loukas, A. (2019). Graph reduction with spectral and cut guarantees.
- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances.
- Navlakha, S., Rastogi, R., & Shrivastava, N. (2008). Graph summarization with bounded error.
- Mitliagkas, I.,et al (2015). FrogWild! -- Fast PageRank Approximations on Graph Engines.

## License

This project is licensed under the MIT License.
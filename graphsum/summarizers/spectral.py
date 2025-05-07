"""
Spectral-based graph summarization.

This module serves as a wrapper for spectral-based graph summarization techniques
that preserve spectral properties of the graph. It includes both:
- Spectral Coarsening: Node reduction via spectral clustering
- Spectral Sparsification: Edge reduction via effective resistance sampling"""

import logging

# Import the specialized implementations
from .spectral_coarsener import SpectralCoarsener
from .spectral_sparsifier import SpectralSparsifier
from .base import GraphSummarizer

logger = logging.getLogger(__name__)


class SpectralSummarizer(GraphSummarizer):
    """
    Wrapper for spectral-based graph summarization methods.
    
    Supports two distinct approaches:
    1. Spectral Coarsening: Computes the graph Laplacian and its eigenvectors, then clusters nodes
       based on their spectral embeddings to create a summary that preserves random walk behavior.
    2. Spectral Sparsification: Samples edges based on effective resistances and reweights them
       to preserve the spectral properties of the graph while reducing the number of edges.
    """
    
    def __init__(self, name="Spectral", method="coarsen", n_eigenvectors=None, epsilon=0.1):
        """
        Initialize the summarizer.
        
        Args:
            name (str): Name of the summarizer
            method (str): 'coarsen' for spectral coarsening, 'sparsify' for spectral sparsification
            n_eigenvectors (int, optional): Number of eigenvectors to use for clustering (coarsening)
            epsilon (float): Error bound for spectral approximation (sparsification)
        """
        super().__init__(name=name)
        self.method = method
        self.n_eigenvectors = n_eigenvectors
        self.epsilon = epsilon
        
        # Initialize the appropriate specialized summarizer based on the method
        if self.method == 'coarsen':
            self.summarizer = SpectralCoarsener(name=name, n_eigenvectors=n_eigenvectors)
        elif self.method == 'sparsify':
            self.summarizer = SpectralSparsifier(name=name, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'coarsen' or 'sparsify'.")
    
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        """
        Summarize graph using spectral methods.
        
        Args:
            graph (nx.Graph): The graph to summarize
            reduction_factor (float): Target size reduction factor (0-1)
            **kwargs: Additional parameters:
                method (str): 'coarsen' or 'sparsify' (overrides init)
                n_eigenvectors (int): Number of eigenvectors to use for coarsening (overrides init)
                epsilon (float): Error bound for sparsification (overrides init)
                weight (str): Edge weight attribute to use
                normalized (bool): Whether to use normalized Laplacian
                
        Returns:
            nx.Graph: The summarized graph
        """
        # Check if method has been overridden in kwargs
        method = kwargs.get('method', self.method)
        
        if method != self.method:
            # Create a new summarizer with the different method
            if method == 'coarsen':
                temp_summarizer = SpectralCoarsener(
                    name=self.name,
                    n_eigenvectors=kwargs.get('n_eigenvectors', self.n_eigenvectors)
                )
            elif method == 'sparsify':
                temp_summarizer = SpectralSparsifier(
                    name=self.name,
                    epsilon=kwargs.get('epsilon', self.epsilon)
                )
            else:
                raise ValueError(f"Unknown method: {method}. Use 'coarsen' or 'sparsify'.")
                
            # Use the temporary summarizer
            summary_graph = temp_summarizer.summarize(graph, reduction_factor, **kwargs)
            
            # Copy mappings for consistency
            self.node_mapping = temp_summarizer.node_mapping
            self.reverse_mapping = temp_summarizer.reverse_mapping
            self.summary_graph = summary_graph
            
            return summary_graph
            
        # Delegate to the specialized summarizer
        summary_graph = self.summarizer.summarize(graph, reduction_factor, **kwargs)
        
        # Copy mappings for consistency
        self.node_mapping = self.summarizer.node_mapping
        self.reverse_mapping = self.summarizer.reverse_mapping
        self.summary_graph = summary_graph
        
        return summary_graph
        
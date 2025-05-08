"""
Graph summary evaluation module.

This module provides a wrapper around the Metrics class to maintain backward compatibility
while focusing only on the five core metrics for graph summarization.
"""

import logging
import time
import pandas as pd
from .metrics import Metrics

logger = logging.getLogger(__name__)


class GraphEvaluator:
    """
    Wrapper around Metrics that evaluates how well a graph summary preserves 
    properties of the original graph using only the five core metrics for graph summarization.
    """
    
    def __init__(self, original_graph, summary_graph, node_mapping, reverse_mapping=None):
        """
        Initialize evaluator.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Maps original nodes to summary nodes
            reverse_mapping (dict): Maps summary nodes to original nodes (optional)
        """
        self.original_graph = original_graph
        self.summary_graph = summary_graph
        self.node_mapping = node_mapping
        self.reverse_mapping = reverse_mapping
        self.results = {}
        self.stats = {}
    
    def _start_timer(self):
        """Start a timer for performance measurement."""
        self._start_time = time.time()
    
    def _stop_timer(self, name):
        """
        Stop the timer and record elapsed time.
        
        Args:
            name (str): Name of the timed operation
        """
        elapsed = time.time() - self._start_time
        self.stats[name] = elapsed
        return elapsed
    
    def evaluate_all(self, top_k=100):
        """
        Run all evaluations using the five core metrics from the paper.
        
        Args:
            top_k (int): Number of top nodes for centrality retention (precision@k)
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Running paper-based evaluation of graph summary")
        self._start_timer()
        
        # Use Metrics to run all five core metrics
        metrics_results = Metrics.evaluate_all(
            self.original_graph,
            self.summary_graph,
            self.node_mapping,
            k_spectral=50,  # Use 50 eigenvalues as in the paper
            k_centrality=top_k
        )
        
        # Store results
        self.results.update(metrics_results)
        
        total_time = self._stop_timer('total_evaluation')
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        # Record runtime statistics
        for name, time_value in self.stats.items():
            self.results[f"time_{name}"] = time_value
        
        return self.results
    
    def evaluate_centrality(self, top_k=100):
        """
        Evaluate preservation of centrality using Precision@k from PaperMetrics.
        
        Args:
            top_k (int): Number of top nodes to consider
            
        Returns:
            dict: Centrality preservation metrics
        """
        logger.info("Evaluating centrality preservation")
        self._start_timer()
        
        precision = Metrics.centrality_retention(
            self.original_graph, 
            self.summary_graph, 
            self.node_mapping,
            k=top_k
        )
        
        self._stop_timer('centrality')
        
        # Store results
        results = {'precision_at_k': precision}
        self.results.update(results)
        return results
    
    def evaluate_community(self):
        """
        Evaluate preservation of community structure using NMI from PaperMetrics.
        
        Returns:
            dict: Community preservation metrics
        """
        logger.info("Evaluating community structure preservation")
        self._start_timer()
        
        nmi = Metrics.community_structure_fidelity(
            self.original_graph, 
            self.summary_graph, 
            self.node_mapping
        )
        
        self._stop_timer('community')
        
        # Store results
        results = {'community_nmi': nmi}
        self.results.update(results)
        return results
        
    def evaluate_degree_distribution(self):
        """
        Note: The paper doesn't include a specific metric for degree distribution.
        This method is kept for backward compatibility but doesn't provide metrics.
        
        Returns:
            dict: Empty results dictionary
        """
        logger.info("Degree distribution metrics not included in paper metrics")
        return {}
        
    def evaluate_clustering(self):
        """
        Note: The paper doesn't include a specific metric for clustering coefficient.
        This method is kept for backward compatibility but doesn't provide metrics.
        
        Returns:
            dict: Empty results dictionary
        """
        logger.info("Clustering metrics not included in paper metrics")
        return {}
        
    def evaluate_path_lengths(self):
        """
        Evaluate preservation of path length characteristics using Stretch from PaperMetrics.
        
        Returns:
            dict: Path length preservation metrics
        """
        logger.info("Evaluating path length preservation")
        self._start_timer()
        
        stretch = Metrics.distance_distortion(
            self.original_graph, 
            self.summary_graph, 
            self.node_mapping
        )
        
        self._stop_timer('path_lengths')
        
        # Store results
        results = {'avg_stretch': stretch}
        self.results.update(results)
        return results
        
    def evaluate_compression(self):
        """
        Evaluate compression ratio using the method from PaperMetrics.
        
        Returns:
            dict: Compression metrics
        """
        logger.info("Evaluating compression ratio")
        self._start_timer()
        
        compression = Metrics.compression_ratio(
            self.original_graph,
            self.summary_graph
        )
        
        self._stop_timer('compression')
        
        # Store results
        results = {'compression_ratio': compression}
        self.results.update(results)
        return results
        
    def to_dataframe(self):
        """
        Convert evaluation results to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: Evaluation results as a DataFrame
        """
        return pd.DataFrame([self.results])
        
    def print_summary(self):
        """
        Print a summary of the metrics evaluation results.
        """
        print("\n=== Graph Summary Evaluation (Paper Metrics) ===\n")
        
        # 1. Spectral Approximation Error
        if 'spectral_error' in self.results:
            print(f"Spectral Error: {self.results['spectral_error']:.4f}")
            
        # 2. Community Structure Fidelity (NMI)
        if 'community_nmi' in self.results:
            print(f"Community NMI: {self.results['community_nmi']:.4f}")
            
        # 3. Distance Distortion (Stretch)
        if 'avg_stretch' in self.results:
            print(f"Average Stretch: {self.results['avg_stretch']:.4f}")
            
        # 4. Centrality Retention (Precision@k)
        if 'precision_at_k' in self.results:
            print(f"Precision@k: {self.results['precision_at_k']:.4f}")
            
        # 5. Compression Ratio
        if 'compression_ratio' in self.results:
            print(f"Compression Ratio: {self.results['compression_ratio']:.4f}")
            
        # Evaluation Time
        if 'evaluation_time' in self.results:
            print(f"\nEvaluation Time: {self.results['evaluation_time']:.2f} seconds")
            
        print("\n==============================================\n")
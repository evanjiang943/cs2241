"""
Graph summary evaluation module.

This module provides functionality to evaluate how well a graph summary
preserves various properties of the original graph.
"""

import networkx as nx
import numpy as np
import pandas as pd
import logging
import time
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import community as community_louvain

logger = logging.getLogger(__name__)


class GraphEvaluator:
    """
    Evaluates how well a graph summary preserves properties of the original graph.
    """
    
    def __init__(self, original_graph, summary_graph, node_mapping, reverse_mapping):
        """
        Initialize evaluator.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Maps original nodes to summary nodes
            reverse_mapping (dict): Maps summary nodes to original nodes
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
        Run all evaluations.
        
        Args:
            top_k (int): Number of top nodes for PageRank evaluation
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Running comprehensive evaluation of graph summary")
        self._start_timer()
        
        # Run all evaluations
        self.evaluate_pagerank(top_k)
        self.evaluate_centrality()
        self.evaluate_community()
        self.evaluate_degree_distribution()
        self.evaluate_clustering()
        self.evaluate_path_lengths()
        self.evaluate_compression()
        
        total_time = self._stop_timer('total_evaluation')
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        # Record runtime statistics
        for name, time_value in self.stats.items():
            self.results[f"time_{name}"] = time_value
        
        return self.results
    
    def evaluate_pagerank(self, top_k=100):
        """
        Evaluate preservation of PageRank.
        
        Args:
            top_k (int): Number of top nodes to consider
            
        Returns:
            dict: PageRank preservation metrics
        """
        logger.info("Evaluating PageRank preservation")
        self._start_timer()
        
        # Compute PageRank for original graph
        pr_original = nx.pagerank(self.original_graph, alpha=0.85)
        
        # Compute PageRank for summary graph
        pr_summary = nx.pagerank(self.summary_graph, alpha=0.85)
        
        # Lift summary PageRank back to original nodes
        pr_lifted = {}
        for orig_node in self.original_graph.nodes():
            if orig_node in self.node_mapping:
                summary_node = self.node_mapping[orig_node]
                pr_lifted[orig_node] = pr_summary.get(summary_node, 0)
        
        # Prepare for correlation calculation
        nodes = list(self.original_graph.nodes())
        pr_orig_vec = [pr_original.get(node, 0) for node in nodes]
        pr_lifted_vec = [pr_lifted.get(node, 0) for node in nodes]
        
        # Calculate correlations
        spearman_corr, _ = stats.spearmanr(pr_orig_vec, pr_lifted_vec)
        kendall_corr, _ = stats.kendalltau(pr_orig_vec, pr_lifted_vec)
        pearson_corr, _ = stats.pearsonr(pr_orig_vec, pr_lifted_vec)
        
        # Calculate L1 error
        l1_error = sum(abs(pr_original.get(node, 0) - pr_lifted.get(node, 0)) 
                       for node in self.original_graph.nodes())
        
        # Calculate top-k overlap
        top_k_orig = sorted(pr_original.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_orig_set = set(node for node, _ in top_k_orig)
        
        top_k_lifted = sorted(pr_lifted.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_lifted_set = set(node for node, _ in top_k_lifted)
        
        jaccard = len(top_k_orig_set.intersection(top_k_lifted_set)) / len(top_k_orig_set.union(top_k_lifted_set))
        
        self._stop_timer('pagerank')
        
        # Store results
        results = {
            'pagerank_spearman': spearman_corr,
            'pagerank_kendall': kendall_corr,
            'pagerank_pearson': pearson_corr,
            'pagerank_l1_error': l1_error,
            f'pagerank_top_{top_k}_jaccard': jaccard
        }
        
        self.results.update(results)
        return results
    
    def evaluate_centrality(self):
        """
        Evaluate preservation of centrality metrics.
        
        Returns:
            dict: Centrality preservation metrics
        """
        logger.info("Evaluating centrality preservation")
        self._start_timer()
        
        # Degree centrality
        original_degree = nx.degree_centrality(self.original_graph)
        summary_degree = nx.degree_centrality(self.summary_graph)
        
        # Try eigenvector centrality
        try:
            original_eigen = nx.eigenvector_centrality(self.original_graph, max_iter=100)
            summary_eigen = nx.eigenvector_centrality(self.summary_graph, max_iter=100)
            has_eigen = True
        except:
            logger.warning("Failed to compute eigenvector centrality, skipping")
            has_eigen = False
        
        # Lift summary centralities back to original nodes
        lifted_degree = {}
        lifted_eigen = {}
        
        for orig_node in self.original_graph.nodes():
            if orig_node in self.node_mapping:
                summary_node = self.node_mapping[orig_node]
                lifted_degree[orig_node] = summary_degree.get(summary_node, 0)
                if has_eigen:
                    lifted_eigen[orig_node] = summary_eigen.get(summary_node, 0)
        
        # Calculate degree centrality correlation
        degree_orig = [original_degree.get(n, 0) for n in self.original_graph.nodes()]
        degree_lifted = [lifted_degree.get(n, 0) for n in self.original_graph.nodes()]
        degree_corr, _ = stats.spearmanr(degree_orig, degree_lifted)
        
        # Calculate eigenvector centrality correlation if available
        eigen_corr = None
        if has_eigen:
            eigen_orig = [original_eigen.get(n, 0) for n in self.original_graph.nodes()]
            eigen_lifted = [lifted_eigen.get(n, 0) for n in self.original_graph.nodes()]
            eigen_corr, _ = stats.spearmanr(eigen_orig, eigen_lifted)
        
        self._stop_timer('centrality')
        
        # Store results
        results = {'degree_centrality_corr': degree_corr}
        if eigen_corr is not None:
            results['eigenvector_centrality_corr'] = eigen_corr
        
        self.results.update(results)
        return results
    
    def evaluate_community(self):
        """
        Evaluate preservation of community structure.
        
        Returns:
            dict: Community preservation metrics
        """
        logger.info("Evaluating community structure preservation")
        self._start_timer()
        
        # Handle large graphs
        if self.original_graph.number_of_nodes() > 10000:
            logger.warning("Large graph detected, using simplified community evaluation")
            return self._evaluate_community_simplified()
        
        # Detect communities in original graph
        communities = community_louvain.best_partition(self.original_graph)
        
        # Map each original node to its community
        orig_community_list = [communities.get(node, 0) for node in self.original_graph.nodes()]
        
        # For each summary node, find the most common community among its constituent nodes
        summary_to_orig_community = {}
        
        for summary_node, orig_nodes in self.reverse_mapping.items():
            if orig_nodes:
                community_counts = {}
                for orig_node in orig_nodes:
                    if orig_node in communities:
                        comm_id = communities[orig_node]
                        community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
                
                # Assign the most common community
                if community_counts:
                    most_common = max(community_counts.items(), key=lambda x: x[1])[0]
                    summary_to_orig_community[summary_node] = most_common
        
        # Lift summary communities back to original nodes
        lifted_communities = {}
        for orig_node in self.original_graph.nodes():
            if orig_node in self.node_mapping:
                summary_node = self.node_mapping[orig_node]
                lifted_communities[orig_node] = summary_to_orig_community.get(summary_node, 0)
        
        # Prepare community assignment vectors
        lifted_community_list = [lifted_communities.get(node, 0) for node in self.original_graph.nodes()]
        
        # Calculate NMI and ARI
        nmi = normalized_mutual_info_score(orig_community_list, lifted_community_list)
        ari = adjusted_rand_score(orig_community_list, lifted_community_list)
        
        self._stop_timer('community')
        
        # Store results
        results = {
            'community_nmi': nmi,
            'community_ari': ari
        }
        
        self.results.update(results)
        return results
    
    def _evaluate_community_simplified(self):
        """
        Simplified community evaluation for large graphs.
        
        Uses a sampling approach to estimate community preservation metrics.
        
        Returns:
            dict: Community preservation metrics
        """
        # Sample nodes from the original graph
        sample_size = min(5000, self.original_graph.number_of_nodes())
        sampled_nodes = np.random.choice(list(self.original_graph.nodes()), size=sample_size, replace=False)
        
        # Create subgraph with sampled nodes
        sampled_graph = self.original_graph.subgraph(sampled_nodes)
        
        # Detect communities in sampled graph
        communities = community_louvain.best_partition(sampled_graph)
        
        # Map nodes to communities
        orig_community_list = [communities.get(node, 0) for node in sampled_nodes]
        
        # For each sampled node, get its community via the summary
        lifted_communities = {}
        for orig_node in sampled_nodes:
            if orig_node in self.node_mapping:
                summary_node = self.node_mapping[orig_node]
                # Get all original nodes in this summary node
                summary_members = self.reverse_mapping.get(summary_node, [])
                # Find which ones are in our sample
                sampled_members = [n for n in summary_members if n in communities]
                if sampled_members:
                    # Count communities among sampled members
                    comm_counts = {}
                    for n in sampled_members:
                        c = communities.get(n, 0)
                        comm_counts[c] = comm_counts.get(c, 0) + 1
                    # Assign most common
                    most_common = max(comm_counts.items(), key=lambda x: x[1])[0]
                    lifted_communities[orig_node] = most_common
        
        # Prepare lifted community list
        lifted_community_list = [lifted_communities.get(node, 0) for node in sampled_nodes]
        
        # Calculate metrics only for nodes that have both original and lifted communities
        valid_indices = [i for i, node in enumerate(sampled_nodes) if node in lifted_communities]
        
        if valid_indices:
            orig_valid = [orig_community_list[i] for i in valid_indices]
            lifted_valid = [lifted_community_list[i] for i in valid_indices]
            
            nmi = normalized_mutual_info_score(orig_valid, lifted_valid)
            ari = adjusted_rand_score(orig_valid, lifted_valid)
        else:
            nmi = 0
            ari = 0
        
        # Store results
        results = {
            'community_nmi_sampled': nmi,
            'community_ari_sampled': ari,
            'community_sample_size': len(valid_indices)
        }
        
        self.results.update(results)
        return results
    
    def evaluate_degree_distribution(self):
        """
        Evaluate preservation of degree distribution.
        
        Returns:
            dict: Degree distribution preservation metrics
        """
        logger.info("Evaluating degree distribution preservation")
        self._start_timer()
        
        # Get degree distributions
        original_degrees = dict(self.original_graph.degree())
        summary_degrees = dict(self.summary_graph.degree())
        
        # Lift summary degrees to original nodes
        lifted_degrees = {}
        for orig_node in self.original_graph.nodes():
            if orig_node in self.node_mapping:
                summary_node = self.node_mapping[orig_node]
                # Scale the degree by the relative size of the communities
                if summary_node in summary_degrees:
                    orig_size = self.original_graph.number_of_nodes()
                    summary_size = self.summary_graph.number_of_nodes()
                    scaling_factor = orig_size / summary_size if summary_size > 0 else 1
                    lifted_degrees[orig_node] = summary_degrees[summary_node] * scaling_factor
        
        # Create binned degree distributions for comparison
        max_orig_degree = max(original_degrees.values()) if original_degrees else 0
        
        # For very large graphs, use logarithmic binning
        if max_orig_degree > 1000:
            bins = np.logspace(0, np.log10(max_orig_degree + 1), 50)
        else:
            bins = np.linspace(0, max_orig_degree + 1, 50)
        
        # Count original degrees
        orig_hist, _ = np.histogram([d for d in original_degrees.values()], bins=bins)
        orig_hist = orig_hist / orig_hist.sum() if orig_hist.sum() > 0 else orig_hist
        
        # Count lifted degrees
        lifted_hist, _ = np.histogram([d for d in lifted_degrees.values()], bins=bins)
        lifted_hist = lifted_hist / lifted_hist.sum() if lifted_hist.sum() > 0 else lifted_hist
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        orig_hist += epsilon
        lifted_hist += epsilon
        
        # Renormalize
        orig_hist = orig_hist / orig_hist.sum()
        lifted_hist = lifted_hist / lifted_hist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(orig_hist * np.log(orig_hist / lifted_hist))
        
        # Calculate correlation between original and lifted degree values
        orig_degree_values = [original_degrees.get(n, 0) for n in self.original_graph.nodes()]
        lifted_degree_values = [lifted_degrees.get(n, 0) for n in self.original_graph.nodes()]
        
        degree_correlation, _ = stats.spearmanr(orig_degree_values, lifted_degree_values)
        
        self._stop_timer('degree_distribution')
        
        # Store results
        results = {
            'degree_distribution_kl': kl_div,
            'degree_correlation': degree_correlation
        }
        
        self.results.update(results)
        return results
    
    def evaluate_clustering(self):
        """
        Evaluate preservation of clustering coefficient.
        
        Returns:
            dict: Clustering coefficient preservation metrics
        """
        logger.info("Evaluating clustering coefficient preservation")
        self._start_timer()
        
        # For large graphs, sample nodes
        if self.original_graph.number_of_nodes() > 10000:
            sample_size = 5000
            sampled_nodes = np.random.choice(list(self.original_graph.nodes()), 
                                          size=min(sample_size, self.original_graph.number_of_nodes()),
                                          replace=False)
            
            # Calculate clustering for sampled nodes
            original_clustering = nx.clustering(self.original_graph, nodes=sampled_nodes)
            original_avg_clustering = sum(original_clustering.values()) / len(original_clustering)
        else:
            # Calculate average clustering coefficient
            original_avg_clustering = nx.average_clustering(self.original_graph)
        
        summary_avg_clustering = nx.average_clustering(self.summary_graph)
        
        # Calculate absolute error
        clustering_error = abs(original_avg_clustering - summary_avg_clustering)
        
        self._stop_timer('clustering')
        
        # Store results
        results = {
            'clustering_coefficient_original': original_avg_clustering,
            'clustering_coefficient_summary': summary_avg_clustering,
            'clustering_coefficient_error': clustering_error
        }
        
        self.results.update(results)
        return results
    
    def evaluate_path_lengths(self):
        """
        Evaluate preservation of path length characteristics.
        
        Returns:
            dict: Path length preservation metrics
        """
        logger.info("Evaluating path length preservation")
        self._start_timer()
        
        results = {}
        
        # For large graphs, estimate using sampling
        if self.original_graph.number_of_nodes() > 1000:
            # Sample node pairs for path length estimation
            sample_size = min(500, self.original_graph.number_of_nodes())
            sampled_nodes = np.random.choice(
                list(self.original_graph.nodes()), 
                size=sample_size, 
                replace=False
            )
            
            # Calculate paths between sampled nodes
            original_paths = []
            for i, u in enumerate(sampled_nodes):
                for v in sampled_nodes[i+1:]:
                    try:
                        path_length = nx.shortest_path_length(self.original_graph, u, v)
                        original_paths.append(path_length)
                    except nx.NetworkXNoPath:
                        continue
            
            # Get corresponding summary nodes
            summary_nodes = set()
            for node in sampled_nodes:
                if node in self.node_mapping:
                    summary_nodes.add(self.node_mapping[node])
            
            summary_paths = []
            summary_nodes = list(summary_nodes)
            for i, u in enumerate(summary_nodes):
                for v in summary_nodes[i+1:]:
                    try:
                        path_length = nx.shortest_path_length(self.summary_graph, u, v)
                        summary_paths.append(path_length)
                    except nx.NetworkXNoPath:
                        continue
            
            if original_paths and summary_paths:
                original_avg_path = np.mean(original_paths)
                summary_avg_path = np.mean(summary_paths)
                
                results['path_length_original_sampled'] = original_avg_path
                results['path_length_summary_sampled'] = summary_avg_path
                results['path_length_ratio_sampled'] = summary_avg_path / original_avg_path if original_avg_path > 0 else float('inf')
        else:
            # For smaller graphs, try to calculate exact average path length
            try:
                # Check if graphs are connected
                if nx.is_connected(self.original_graph) and nx.is_connected(self.summary_graph):
                    original_avg_path = nx.average_shortest_path_length(self.original_graph)
                    summary_avg_path = nx.average_shortest_path_length(self.summary_graph)
                    
                    results['path_length_original'] = original_avg_path
                    results['path_length_summary'] = summary_avg_path
                    results['path_length_ratio'] = summary_avg_path / original_avg_path if original_avg_path > 0 else float('inf')
                else:
                    logger.warning("Graphs are not connected, skipping exact path length calculation")
            except Exception as e:
                logger.warning(f"Error calculating path lengths: {e}")
        
        # Try to calculate diameter
        try:
            if self.original_graph.number_of_nodes() <= 1000 and nx.is_connected(self.original_graph):
                original_diameter = nx.diameter(self.original_graph)
                if nx.is_connected(self.summary_graph):
                    summary_diameter = nx.diameter(self.summary_graph)
                    results['diameter_original'] = original_diameter
                    results['diameter_summary'] = summary_diameter
                    results['diameter_ratio'] = summary_diameter / original_diameter if original_diameter > 0 else float('inf')
        except Exception as e:
            logger.warning(f"Error calculating diameters: {e}")
        
        self._stop_timer('path_lengths')
        
        self.results.update(results)
        return results
    
    def evaluate_compression(self):
        """
        Evaluate compression ratios and runtime improvements.
        
        Returns:
            dict: Compression and runtime metrics
        """
        logger.info("Evaluating compression and runtime metrics")
        self._start_timer()
        
        # Calculate compression ratios
        orig_nodes = self.original_graph.number_of_nodes()
        orig_edges = self.original_graph.number_of_edges()
        
        summary_nodes = self.summary_graph.number_of_nodes()
        summary_edges = self.summary_graph.number_of_edges()
        
        node_compression = summary_nodes / orig_nodes if orig_nodes > 0 else 1
        edge_compression = summary_edges / orig_edges if orig_edges > 0 else 1
        
        # Measure PageRank runtime on both graphs
        start_time = time.time()
        _ = nx.pagerank(self.original_graph)
        original_runtime = time.time() - start_time
        
        start_time = time.time()
        _ = nx.pagerank(self.summary_graph)
        summary_runtime = time.time() - start_time
        
        runtime_ratio = summary_runtime / original_runtime if original_runtime > 0 else 1
        speedup = 1 / runtime_ratio if runtime_ratio > 0 else float('inf')
        
        self._stop_timer('compression')
        
        # Store results
        results = {
            'node_compression_ratio': node_compression,
            'edge_compression_ratio': edge_compression,
            'pagerank_runtime_original': original_runtime,
            'pagerank_runtime_summary': summary_runtime,
            'pagerank_runtime_ratio': runtime_ratio,
            'pagerank_speedup': speedup
        }
        
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
        """Print a summary of the evaluation results."""
        print("\nEvaluation Summary:")
        print(f"Original graph: {self.original_graph.number_of_nodes()} nodes, {self.original_graph.number_of_edges()} edges")
        print(f"Summary graph: {self.summary_graph.number_of_nodes()} nodes, {self.summary_graph.number_of_edges()} edges")
        
        # Compression
        node_ratio = self.results.get('node_compression_ratio', 0)
        edge_ratio = self.results.get('edge_compression_ratio', 0)
        print(f"Compression: {node_ratio:.3f} node ratio, {edge_ratio:.3f} edge ratio")
        
        # PageRank
        pr_corr = self.results.get('pagerank_spearman', 0)
        pr_l1 = self.results.get('pagerank_l1_error', 0)
        print(f"PageRank preservation: {pr_corr:.3f} correlation, {pr_l1:.6f} L1 error")
        
        # Community structure
        nmi = self.results.get('community_nmi', 0)
        ari = self.results.get('community_ari', 0)
        print(f"Community preservation: {nmi:.3f} NMI, {ari:.3f} ARI")
        
        # Speedup
        speedup = self.results.get('pagerank_speedup', 0)
        print(f"Performance: {speedup:.2f}x PageRank speedup")
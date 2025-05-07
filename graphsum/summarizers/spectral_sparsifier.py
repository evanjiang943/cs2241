"""Spectral sparsification-based graph summarization.

This module implements graph summarization techniques that preserve 
spectral properties of the graph via sparsification (edge reduction).
"""

import networkx as nx
import numpy as np
import logging
import scipy.sparse as sp

from .base import GraphSummarizer

logger = logging.getLogger(__name__)


class SpectralSparsifier(GraphSummarizer):
    """
    Summarizes graphs using spectral sparsification.
    
    Implements edge sampling based on effective resistances to create a sparser graph
    that approximates the original graph's Laplacian quadratic form. 
    This preserves spectral properties while reducing the number of edges.
    """
    
    def __init__(self, name="Sparsifier", epsilon=0.1):
        """
        Initialize the sparsifier.
        
        Args:
            name (str): Name of the summarizer
            epsilon (float): Error bound for spectral approximation
        """
        super().__init__(name=name)
        self.epsilon = epsilon
    
    def _compute_effective_resistances(self, graph, max_nodes=1000):
        """
        Compute effective resistances for edges.
        
        For large graphs, uses an approximation based on a subset of eigenvectors.
        
        Args:
            graph (nx.Graph): The graph to analyze
            max_nodes (int): Maximum number of nodes for exact computation
            
        Returns:
            dict: Dictionary mapping edge tuples to their effective resistances
        """
        logger.info("Computing effective resistances for edges")
        self._start_timer()
        
        n_nodes = graph.number_of_nodes()
        edge_list = list(graph.edges())
        resistances = {}
        
        # For very large graphs, use approximation
        if n_nodes > max_nodes:
            logger.info(f"Graph has {n_nodes} nodes, using spectral approximation for effective resistances")
            
            # Compute normalized Laplacian and a subset of eigenvectors
            laplacian = nx.normalized_laplacian_matrix(graph).todense()
            k = min(100, n_nodes - 1)  # Use 100 eigenvectors or fewer if graph is smaller
            
            # Compute smallest eigenvalues and corresponding eigenvectors
            try:
                eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian, k=k, which='SM')
            except Exception as e:
                logger.warning(f"Sparse eigensolver failed, using dense computation: {e}")
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                eigenvalues = eigenvalues[:k]
                eigenvectors = eigenvectors[:, :k]
            
            # Sort by eigenvalues (excluding the first one which is ~0)
            idx = np.argsort(eigenvalues)[1:]  # Skip the first one (constant vector)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Compute approximate resistances using the eigenvectors
            nodes = list(graph.nodes())
            for u, v in edge_list:
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                
                # Approximate using spectral decomposition
                resistance = 0
                for j in range(len(eigenvalues)):
                    if eigenvalues[j] > 1e-10:  # Avoid division by zero
                        diff = eigenvectors[u_idx, j] - eigenvectors[v_idx, j]
                        resistance += (diff ** 2) / eigenvalues[j]
                
                resistances[(u, v)] = resistance
                resistances[(v, u)] = resistance  # Symmetric
        else:
            # For smaller graphs, compute exact resistances using pseudoinverse
            logger.info("Computing exact effective resistances")
            laplacian = nx.laplacian_matrix(graph).todense()
            laplacian_pinv = np.linalg.pinv(laplacian)
            
            nodes = list(graph.nodes())
            for u, v in edge_list:
                i, j = nodes.index(u), nodes.index(v)
                # Effective resistance formula: r_e = L⁺[i,i] + L⁺[j,j] - 2·L⁺[i,j]
                resistance = laplacian_pinv[i, i] + laplacian_pinv[j, j] - 2 * laplacian_pinv[i, j]
                resistances[(u, v)] = float(resistance)
                resistances[(v, u)] = float(resistance)  # Symmetric
        
        comp_time = self._stop_timer('effective_resistance')
        logger.info(f"Effective resistance computation completed in {comp_time:.2f} seconds")
        
        return resistances
    
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        """
        Implement spectral sparsification by sampling edges based on effective resistances.
        
        Args:
            graph (nx.Graph): The graph to sparsify
            reduction_factor (float): Target edge reduction factor (0-1)
            **kwargs: Additional parameters:
                epsilon (float): Error bound for spectral approximation (overrides init)
                max_nodes (int): Maximum nodes for exact effective resistance computation
                
        Returns:
            nx.Graph: The sparsified graph
        """
        self._start_timer()
        
        # Parse parameters
        epsilon = kwargs.get('epsilon', self.epsilon)
        max_nodes = kwargs.get('max_nodes', 1000)
        
        # Calculate target number of edges
        original_edges = graph.number_of_edges()
        # Ensure we keep at least a spanning tree (n-1 edges)
        target_edges = max(graph.number_of_nodes() - 1, int(original_edges * reduction_factor))
        
        logger.info(f"Creating spectral sparsifier with ~{target_edges} edges (reduction={reduction_factor})")
        
        # If the graph is directed, convert to undirected for effective resistance computation
        if isinstance(graph, nx.DiGraph):
            logger.info("Converting directed graph to undirected for effective resistance computation")
            work_graph = graph.to_undirected()
            is_directed = True
        else:
            work_graph = graph
            is_directed = False
        
        # Initialize summary graph with same nodes as original
        self._init_summary(graph, graph.number_of_nodes())
        
        # Create identity mapping (keeping all nodes)
        for node in graph.nodes():
            self.node_mapping[node] = node
            self.reverse_mapping[node] = [node]
            
            # Copy node attributes
            node_attrs = {k: v for k, v in graph.nodes[node].items()}
            self.summary_graph.add_node(node, **node_attrs)
        
        # Compute effective resistances
        resistances = self._compute_effective_resistances(work_graph, max_nodes=max_nodes)
        
        # Calculate sampling probabilities proportional to weight × resistance
        probabilities = {}
        total_weight_times_resistance = 0
        
        # First pass to calculate total weight × resistance
        for u, v, data in graph.edges(data=True):
            edge_weight = data.get('weight', 1.0)
            if (u, v) in resistances:
                r_e = resistances[(u, v)]
                total_weight_times_resistance += edge_weight * r_e
        
        # Calculate probability for each edge
        for u, v, data in graph.edges(data=True):
            edge_weight = data.get('weight', 1.0)
            if (u, v) in resistances:
                r_e = resistances[(u, v)]
                # Probability proportional to weight × resistance
                if total_weight_times_resistance > 0:
                    probabilities[(u, v)] = (edge_weight * r_e) / total_weight_times_resistance
                else:
                    probabilities[(u, v)] = 1.0 / graph.number_of_edges()
        
        # Sample edges
        logger.info(f"Sampling {target_edges} edges based on effective resistances")
        
        # Convert to arrays for numpy sampling
        edges = list(probabilities.keys())
        probs = np.array([probabilities[e] for e in edges])
        
        # Ensure probabilities sum to 1
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            # Fallback to uniform if all probs are 0
            probs = np.ones(len(edges)) / len(edges)
        
        # Sample edges with replacement based on probabilities
        try:
            sampled_idx = np.random.choice(
                range(len(edges)),
                size=target_edges,
                replace=True,
                p=probs
            )
            
            # Count occurrences of each edge
            edge_counts = {}
            for idx in sampled_idx:
                edge = edges[idx]
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                
            # Add edges to summary graph with reweighted values
            for (u, v), count in edge_counts.items():
                # Get original edge data
                edge_data = graph.get_edge_data(u, v).copy()
                original_weight = edge_data.get('weight', 1.0)
                
                # Calculate new weight: original_weight * count / (probability * target_edges)
                p_e = probabilities[(u, v)]
                if p_e > 0:
                    reweighted = original_weight * count / (p_e * target_edges)
                else:
                    reweighted = original_weight
                
                # Store both weights
                edge_data['weight'] = reweighted
                edge_data['original_weight'] = original_weight
                edge_data['sampled_count'] = count
                
                # Add the edge
                self.summary_graph.add_edge(u, v, **edge_data)
        
        except Exception as e:
            logger.error(f"Error during edge sampling: {e}")
            logger.warning("Falling back to uniform edge sampling")
            
            # Fallback to uniform sampling if something goes wrong
            edge_list = list(graph.edges(data=True))
            sampled_edges = np.random.choice(len(edge_list), size=target_edges, replace=True)
            
            # Count occurrences
            edge_counts = {}
            for idx in sampled_edges:
                u, v, _ = edge_list[idx]
                edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1
            
            # Add edges with uniform reweighting
            for (u, v), count in edge_counts.items():
                edge_data = graph.get_edge_data(u, v).copy()
                original_weight = edge_data.get('weight', 1.0)
                
                # Simple reweighting by count/target
                edge_data['weight'] = original_weight * count / target_edges
                edge_data['original_weight'] = original_weight
                edge_data['sampled_count'] = count
                
                self.summary_graph.add_edge(u, v, **edge_data)
        
        summary_time = self._stop_timer('sparsification')
        logger.info(f"Sparsification completed in {summary_time:.2f} seconds")
        logger.info(f"Sparsified graph: {self.summary_graph.number_of_nodes()} nodes, {self.summary_graph.number_of_edges()} edges")
        
        return self.summary_graph

"""
Spectral-based graph summarization.

This module implements graph summarization techniques that preserve 
spectral properties of the graph.
"""

import networkx as nx
import numpy as np
import logging
import scipy.sparse as sp
from sklearn.cluster import KMeans

from .base import GraphSummarizer

logger = logging.getLogger(__name__)


class SpectralSummarizer(GraphSummarizer):
    """
    Summarizes graphs by preserving spectral properties.
    
    Computes the graph Laplacian and its eigenvectors, then clusters nodes
    based on their spectral embeddings to create a summary that preserves
    random walk behavior and structural properties.
    """
    
    def __init__(self, name="Spectral", n_eigenvectors=None):
        """
        Initialize the summarizer.
        
        Args:
            name (str): Name of the summarizer
            n_eigenvectors (int, optional): Number of eigenvectors to use for clustering
        """
        super().__init__(name=name)
        self.n_eigenvectors = n_eigenvectors
    
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        """
        Summarize graph using spectral clustering.
        
        Args:
            graph (nx.Graph): The graph to summarize
            reduction_factor (float): Target size reduction factor (0-1)
            **kwargs: Additional parameters:
                n_eigenvectors (int): Number of eigenvectors to use (overrides init)
                weight (str): Edge weight attribute to use
                normalized (bool): Whether to use normalized Laplacian
                
        Returns:
            nx.Graph: The summarized graph
        """
        self._start_timer()
        
        # Parse parameters
        n_eigenvectors = kwargs.get('n_eigenvectors', self.n_eigenvectors)
        # weight = kwargs.get('weight', None)
        normalized = kwargs.get('normalized', True)
        
        # Determine target number of nodes in summary
        n_original = graph.number_of_nodes()
        n_summary = max(2, int(n_original * reduction_factor))
        
        logger.info(f"Creating spectral summary with {n_summary} nodes (reduction={reduction_factor})")
        
        # If the graph is directed, convert to undirected for Laplacian computation
        if isinstance(graph, nx.DiGraph):
            logger.info("Converting directed graph to undirected for spectral analysis")
            graph_for_laplacian = graph.to_undirected()
        else:
            graph_for_laplacian = graph
        
        # Compute Laplacian matrix
        logger.info("Computing graph Laplacian")
        if normalized:
            laplacian = nx.normalized_laplacian_matrix(graph_for_laplacian)
        else:
            laplacian = nx.laplacian_matrix(graph_for_laplacian)
        
        # Compute eigenvectors (using sparse eigensolvers for large graphs)
        logger.info("Computing eigenvectors")
        if n_original > 5000:  # Large graph
            # For large graphs, we use sparse eigensolvers and compute only the needed eigenvectors
            k = min(n_summary * 2, n_original - 1) if n_eigenvectors is None else n_eigenvectors
            eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian, k=k, which='SM')
        else:
            # For smaller graphs, compute all eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian.todense())
        
        # Sort eigenvectors by eigenvalues (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        spectral_time = self._stop_timer('spectral_computation')
        logger.info(f"Spectral computation completed in {spectral_time:.2f} seconds")
        
        # Determine number of eigenvectors to use for clustering
        if n_eigenvectors is None:
            # Use enough eigenvectors to capture the main graph structure
            # Skip the first eigenvector (constant)
            k = min(n_summary * 2, eigenvectors.shape[1] - 1)
        else:
            # Use specified number, but ensure we don't exceed what we have
            k = min(n_eigenvectors, eigenvectors.shape[1] - 1)
        
        # Create feature matrix for clustering from the first k non-trivial eigenvectors
        features = eigenvectors[:, 1:k+1]  # Skip the first eigenvector (constant)
        
        # Cluster nodes based on spectral features
        self._start_timer()
        logger.info(f"Clustering {n_original} nodes into {n_summary} clusters using {k} eigenvectors")
        
        # Use k-means for clustering
        kmeans = KMeans(n_clusters=n_summary, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        clustering_time = self._stop_timer('clustering')
        logger.info(f"Clustering completed in {clustering_time:.2f} seconds")
        
        # Initialize summary graph
        self._start_timer()
        self._init_summary(graph, n_summary)
        
        # Create node mappings
        nodes = list(graph.nodes())
        for i, node in enumerate(nodes):
            cluster_id = int(clusters[i])
            self.node_mapping[node] = cluster_id
            
            if cluster_id not in self.reverse_mapping:
                self.reverse_mapping[cluster_id] = []
            self.reverse_mapping[cluster_id].append(node)
        
        # Add nodes to summary graph with size attributes
        for cluster_id, members in self.reverse_mapping.items():
            self.summary_graph.add_node(
                cluster_id, 
                size=len(members),
                members=len(members)
            )
        
        # Add weighted edges
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            u_summary = self.node_mapping.get(u)
            v_summary = self.node_mapping.get(v)
            
            # Skip nodes that weren't clustered
            if u_summary is None or v_summary is None:
                continue
                
            # Self-loops represent internal cluster edges
            if u_summary == v_summary:
                continue  # We'll handle internal edges separately
                
            edge_weight = data.get('weight', 1.0)
            
            if self.summary_graph.has_edge(u_summary, v_summary):
                self.summary_graph[u_summary][v_summary]['weight'] += edge_weight
                self.summary_graph[u_summary][v_summary]['count'] += 1
            else:
                self.summary_graph.add_edge(
                    u_summary, v_summary, 
                    weight=edge_weight,
                    count=1
                )
            edge_count += 1
        
        # Add internal edge counts as node attributes
        for cluster_id in self.summary_graph.nodes():
            members = self.reverse_mapping[cluster_id]
            internal_edges = graph.subgraph(members).number_of_edges()
            # Add attribute for internal edges
            self.summary_graph.nodes[cluster_id]['internal_edges'] = internal_edges
        
        # Normalize edge weights by potential connections
        for u, v, data in self.summary_graph.edges(data=True):
            u_size = len(self.reverse_mapping[u])
            v_size = len(self.reverse_mapping[v])
            max_possible_edges = u_size * v_size
            
            # Add normalization info to edge
            data['max_connections'] = max_possible_edges
            if max_possible_edges > 0:
                data['density'] = data['count'] / max_possible_edges
            else:
                data['density'] = 0
        
        summary_creation_time = self._stop_timer('summary_creation')
        logger.info(f"Summary graph created in {summary_creation_time:.2f} seconds")
        logger.info(f"Summary: {self.summary_graph.number_of_nodes()} nodes, {self.summary_graph.number_of_edges()} edges")
        
        return self.summary_graph
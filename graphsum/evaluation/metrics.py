"""Evaluation metrics for graph summarization.

This module implements the five core metrics:
1. Spectral Approximation Error
2. Community-Structure Fidelity (NMI)
3. Distance Distortion (Stretch)
4. Centrality Retention (Precision@k)
5. Compression Ratio

These metrics are designed to evaluate different aspects of graph summarization quality.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import logging
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import random
import time
import community as community_louvain

logger = logging.getLogger(__name__)


class Metrics:
    """Implements the five core metrics for evaluating graph summarization quality."""
    
    @staticmethod
    def spectral_approximation_error(original_graph, summary_graph, node_mapping=None, k=50):
        """
        Compute spectral approximation error between original and summary graphs.
        
        Measures the average relative error in the first k non-trivial eigenvalues
        of the normalized Laplacian:
        SpectralErr = (1/k) * sum_{i=2}^{k+1} |λ'_i - λ_i|/|λ_i|
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Mapping from original to summary nodes (None for sparsification)
            k (int): Number of non-trivial eigenvalues to consider
            
        Returns:
            float: The spectral approximation error
        """
        
        # Validate input graphs
        if original_graph.number_of_nodes() == 0 or summary_graph.number_of_nodes() == 0:
            logger.warning("Empty graph detected in spectral approximation error calculation")
            return float('nan')
            
        # Convert directed graphs to undirected for Laplacian computation
        if isinstance(original_graph, nx.DiGraph):
            orig_graph = original_graph.to_undirected()
        else:
            orig_graph = original_graph
            
        if isinstance(summary_graph, nx.DiGraph):
            summ_graph = summary_graph.to_undirected()
        else:
            summ_graph = summary_graph
            
        # Ensure both graphs have at least k+1 nodes
        actual_k = min(k, orig_graph.number_of_nodes() - 1, summ_graph.number_of_nodes() - 1)
        if actual_k <= 0:
            logger.warning("Insufficient nodes for spectral approximation error calculation")
            return float('nan')
            
        try:
            # For larger graphs, use sparse matrix computation with scipy.sparse.linalg
            if orig_graph.number_of_nodes() > 1000 or summ_graph.number_of_nodes() > 1000:
                try:
                    from scipy import sparse
                    from scipy.sparse.linalg import eigsh
                    
                    # Compute normalized Laplacians as sparse matrices
                    l_orig = nx.normalized_laplacian_matrix(orig_graph)
                    l_summ = nx.normalized_laplacian_matrix(summ_graph)
                    
                    # Compute only the k+1 smallest eigenvalues (more efficient)
                    # Add 1 to account for the smallest eigenvalue (0)
                    k_to_compute = min(actual_k + 1, orig_graph.number_of_nodes() - 1)
                    evals_orig = eigsh(l_orig, k=k_to_compute, which='SM', return_eigenvectors=False)
                    
                    k_to_compute = min(actual_k + 1, summ_graph.number_of_nodes() - 1)
                    evals_summ = eigsh(l_summ, k=k_to_compute, which='SM', return_eigenvectors=False)
                except Exception as sparse_error:
                    logger.warning(f"Sparse eigenvalue calculation failed: {sparse_error}. Falling back to dense method.")
                    # Fall back to dense calculation
                    l_orig = nx.normalized_laplacian_matrix(orig_graph).todense()
                    l_summ = nx.normalized_laplacian_matrix(summ_graph).todense()
                    
                    # Compute all eigenvalues
                    evals_orig = np.linalg.eigvalsh(l_orig)
                    evals_summ = np.linalg.eigvalsh(l_summ)
            else:
                # For smaller graphs, use standard dense computation
                l_orig = nx.normalized_laplacian_matrix(orig_graph).todense()
                l_summ = nx.normalized_laplacian_matrix(summ_graph).todense()
                
                # Compute all eigenvalues
                evals_orig = np.linalg.eigvalsh(l_orig)
                evals_summ = np.linalg.eigvalsh(l_summ)
            
            # Sort eigenvalues
            evals_orig = np.sort(evals_orig)
            evals_summ = np.sort(evals_summ)
            
            # Skip the first eigenvalue (which is 0 for connected graphs)
            # and take the next k values
            k_orig = min(actual_k, len(evals_orig)-1)
            k_summ = min(actual_k, len(evals_summ)-1)
            
            # Skip the first (smallest) eigenvalue, which should be close to 0
            evals_orig = evals_orig[1:k_orig+1]
            evals_summ = evals_summ[1:k_summ+1]
            
            # Take the minimum length to ensure we compare the same number of eigenvalues
            min_k = min(len(evals_orig), len(evals_summ))
            if min_k == 0:
                logger.warning("No eigenvalues to compare after filtering")
                return float('nan')
                
            evals_orig = evals_orig[:min_k]
            evals_summ = evals_summ[:min_k]
            
            # Prevent division by zero with a small epsilon
            epsilon = 1e-10
            # Compute relative error with better numerical stability
            rel_errors = np.abs(evals_summ - evals_orig) / (np.abs(evals_orig) + epsilon)
            
            # Calculate the mean relative error
            spectral_error = np.mean(rel_errors)
            
            logger.info(f"Spectral approximation error: {spectral_error:.6f}")
            return spectral_error
        
        except Exception as e:
            logger.warning(f"Error computing spectral approximation error: {e}")
            return float('nan')
    
    @staticmethod
    def community_structure_fidelity(original_graph, summary_graph, node_mapping=None,
                                  original_communities=None, summary_communities=None):
        """
        Compute the community-structure fidelity using Normalized Mutual Information (NMI).
        
        For sparsification (same nodes): Directly compare community assignments
        For coarsening (node mapping): Compare original node assignments to the 
        assignments lifted from the summary.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Mapping from original to summary nodes (None for sparsification)
            original_communities (dict): Pre-computed communities for original graph (optional)
            summary_communities (dict): Pre-computed communities for summary graph (optional)
            
        Returns:
            float: NMI score between original and summary communities
        """
        try:
            # Convert to undirected for community detection if needed
            orig_graph_for_community = original_graph.to_undirected() if isinstance(original_graph, nx.DiGraph) else original_graph
            summ_graph_for_community = summary_graph.to_undirected() if isinstance(summary_graph, nx.DiGraph) else summary_graph
            
            # Detect communities if not provided
            if original_communities is None:
                logger.info("Detecting communities in original graph")
                original_communities = community_louvain.best_partition(orig_graph_for_community, random_state=42)
                
            if summary_communities is None:
                logger.info("Detecting communities in summary graph")
                summary_communities = community_louvain.best_partition(summ_graph_for_community, random_state=42)
            
            # For sparsification (same node set)
            if node_mapping is None:
                # Get nodes that exist in both graphs
                common_nodes = set(original_graph.nodes()) & set(summary_graph.nodes())
                if not common_nodes:
                    logger.warning("No common nodes between original and summary graphs")
                    return 0.0
                
                # Extract community labels for common nodes
                orig_labels = [original_communities[node] for node in common_nodes]
                summ_labels = [summary_communities[node] for node in common_nodes]
                
                # Compute NMI
                nmi = normalized_mutual_info_score(orig_labels, summ_labels)
                return nmi
                
            # For coarsening (with node mapping)
            else:
                # Map original graph nodes to their communities
                node_to_community = {}
                for node in original_graph.nodes():
                    if node in original_communities:
                        node_to_community[node] = original_communities[node]
                
                # For each original node, get the community of its corresponding summary node
                orig_to_summ_community = {}
                for orig_node, summ_node in node_mapping.items():
                    if summ_node in summary_communities:
                        orig_to_summ_community[orig_node] = summary_communities[summ_node]
                
                # Get nodes that have communities in both mappings
                common_nodes = set(node_to_community.keys()) & set(orig_to_summ_community.keys())
                if not common_nodes:
                    logger.warning("No common nodes with community assignments")
                    return 0.0
                
                # Extract community labels
                orig_labels = [node_to_community[node] for node in common_nodes]
                summ_labels = [orig_to_summ_community[node] for node in common_nodes]
                
                # Compute NMI
                nmi = normalized_mutual_info_score(orig_labels, summ_labels)
                return nmi
                
        except Exception as e:
            logger.warning(f"Error computing community structure fidelity: {e}")
            return float('nan')
    
    @staticmethod
    def distance_distortion(original_graph, summary_graph, node_mapping=None, 
                         sample_size=1000):
        """
        Compute distance distortion (average stretch) between original and summary graphs.
        
        Stretch = (1/|S|) * sum_{(u,v) ∈ S} d'_{uv}/d_{uv}
        where d_{uv} is the shortest path distance.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Mapping from original to summary nodes (None for sparsification)
            sample_size (int): Number of node pairs to sample
            
        Returns:
            float: Average stretch factor
        """
        try:
            # Sample node pairs
            orig_nodes = list(original_graph.nodes())
            
            # For large graphs, limit the sample size
            if len(orig_nodes) > 1000:
                logger.info(f"Large graph detected with {len(orig_nodes)} nodes. Using sampling.")
                sample_size = min(sample_size, 1000)
            else:
                sample_size = min(sample_size, len(orig_nodes) * (len(orig_nodes) - 1) // 2)
            
            # Randomly sample node pairs
            node_pairs = []
            for _ in range(sample_size):
                u, v = random.sample(orig_nodes, 2)
                node_pairs.append((u, v))
            
            stretches = []
            
            # For sparsification (same node set)
            if node_mapping is None:
                for u, v in node_pairs:
                    # Check if both nodes exist in the summary graph
                    if u in summary_graph.nodes() and v in summary_graph.nodes():
                        # Check if there's a path in both graphs
                        if nx.has_path(original_graph, u, v) and nx.has_path(summary_graph, u, v):
                            # Compute shortest path lengths
                            d_orig = nx.shortest_path_length(original_graph, u, v)
                            d_summ = nx.shortest_path_length(summary_graph, u, v)
                            
                            # Compute stretch
                            if d_orig > 0:  # Avoid division by zero
                                stretch = d_summ / d_orig
                                stretches.append(stretch)
            
            # For coarsening (with node mapping)
            else:
                for u, v in node_pairs:
                    # Check if both nodes are mapped to the summary graph
                    if u in node_mapping and v in node_mapping:
                        u_summ = node_mapping[u]
                        v_summ = node_mapping[v]
                        
                        # Check if there's a path in both graphs
                        if (nx.has_path(original_graph, u, v) and 
                            nx.has_path(summary_graph, u_summ, v_summ)):
                            
                            # Compute shortest path lengths
                            d_orig = nx.shortest_path_length(original_graph, u, v)
                            d_summ = nx.shortest_path_length(summary_graph, u_summ, v_summ)
                            
                            # Compute stretch
                            if d_orig > 0:  # Avoid division by zero
                                stretch = d_summ / d_orig
                                stretches.append(stretch)
            
            if not stretches:
                logger.warning("No valid node pairs for stretch computation")
                return float('nan')
                
            # Return average stretch
            return np.mean(stretches)
            
        except Exception as e:
            logger.warning(f"Error computing distance distortion: {e}")
            return float('nan')
    
    @staticmethod
    def centrality_retention(original_graph, summary_graph, node_mapping=None, k=100):
        """
        Compute centrality retention (Precision@k) between original and summary graphs.
        
        Precision@k = |top_k(π) ∩ top_k(π')|/k
        where π and π' are the PageRank vectors of original and summary graphs.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Mapping from original to summary nodes (None for sparsification)
            k (int): Number of top nodes to consider
            
        Returns:
            float: Precision@k score
        """
        try:
            # Compute PageRank for original graph
            orig_pagerank = nx.pagerank(original_graph)
            
            # Compute PageRank for summary graph
            summ_pagerank = nx.pagerank(summary_graph)
            
            # Get top-k nodes from original graph
            top_k_orig = sorted(orig_pagerank.items(), key=lambda x: x[1], reverse=True)[:k]
            top_k_orig_nodes = {node for node, _ in top_k_orig}
            
            # For sparsification (same node set)
            if node_mapping is None:
                # Get top-k nodes from summary graph
                top_k_summ = sorted(summ_pagerank.items(), key=lambda x: x[1], reverse=True)[:k]
                top_k_summ_nodes = {node for node, _ in top_k_summ}
                
                # Compute intersection
                intersection = top_k_orig_nodes & top_k_summ_nodes
                precision_at_k = len(intersection) / k
                
                return precision_at_k
                
            # For coarsening (with node mapping)
            else:
                # Create reverse mapping from summary nodes to original nodes
                reverse_mapping = {}
                for orig_node, summ_node in node_mapping.items():
                    if summ_node not in reverse_mapping:
                        reverse_mapping[summ_node] = []
                    reverse_mapping[summ_node].append(orig_node)
                
                # Get top-k nodes from summary graph and map back to original nodes
                top_k_summ = sorted(summ_pagerank.items(), key=lambda x: x[1], reverse=True)[:k]
                
                # For each top summary node, distribute its PR score to its constituent original nodes
                orig_nodes_from_summ = set()
                for summ_node, _ in top_k_summ:
                    if summ_node in reverse_mapping:
                        orig_nodes_from_summ.update(reverse_mapping[summ_node])
                
                # Compute intersection
                intersection = top_k_orig_nodes & orig_nodes_from_summ
                precision_at_k = len(intersection) / k
                
                return precision_at_k
                
        except Exception as e:
            logger.warning(f"Error computing centrality retention: {e}")
            return float('nan')
    
    @staticmethod
    def compression_ratio(original_graph, summary_graph):
        """
        Compute compression ratio between original and summary graphs.
        
        CR = (|V'| + |E'|) / (|V| + |E|)
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            
        Returns:
            float: Compression ratio
        """
        try:
            # Count nodes and edges in original graph
            orig_nodes = original_graph.number_of_nodes()
            orig_edges = original_graph.number_of_edges()
            orig_size = orig_nodes + orig_edges
            
            # Count nodes and edges in summary graph
            summ_nodes = summary_graph.number_of_nodes()
            summ_edges = summary_graph.number_of_edges()
            summ_size = summ_nodes + summ_edges
            
            # Compute compression ratio
            compression_ratio = summ_size / orig_size
            
            return compression_ratio
            
        except Exception as e:
            logger.warning(f"Error computing compression ratio: {e}")
            return float('nan')
    
    @staticmethod
    def evaluate_all(original_graph, summary_graph, node_mapping=None, k_spectral=50, k_centrality=100):
        """
        Run all paper metrics evaluation.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Mapping from original to summary nodes (None for sparsification)
            k_spectral (int): Number of eigenvalues for spectral approximation error
            k_centrality (int): Number of top nodes for centrality retention
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        logger.info("Running metrics evaluation")
        start_time = time.time()
        
        results = {}
        
        # 1. Spectral Approximation Error
        logger.info("Computing spectral approximation error")
        spectral_err = Metrics.spectral_approximation_error(
            original_graph, summary_graph, node_mapping, k=k_spectral)
        results['spectral_error'] = spectral_err
        
        # 2. Community-Structure Fidelity
        logger.info("Computing community structure fidelity")
        nmi = Metrics.community_structure_fidelity(
            original_graph, summary_graph, node_mapping)
        results['community_nmi'] = nmi
        
        # 3. Distance Distortion (Stretch)
        logger.info("Computing distance distortion")
        stretch = Metrics.distance_distortion(
            original_graph, summary_graph, node_mapping)
        results['avg_stretch'] = stretch
        
        # 4. Centrality Retention
        logger.info("Computing centrality retention")
        precision = Metrics.centrality_retention(
            original_graph, summary_graph, node_mapping, k=k_centrality)
        results['precision_at_k'] = precision
        
        # 5. Compression Ratio
        logger.info("Computing compression ratio")
        compression = Metrics.compression_ratio(original_graph, summary_graph)
        results['compression_ratio'] = compression
        
        # Total time
        evaluation_time = time.time() - start_time
        results['evaluation_time'] = evaluation_time
        
        logger.info(f"Metrics evaluation completed in {evaluation_time:.2f} seconds")
        return results

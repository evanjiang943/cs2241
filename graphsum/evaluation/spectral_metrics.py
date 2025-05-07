"""Spectral metrics for evaluating graph summarization.

This module provides metrics specifically designed for evaluating
how well spectral properties are preserved in graph summarization.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class SpectralMetrics:
    """Evaluate how well spectral properties are preserved in a graph summary."""
    
    @staticmethod
    def spectral_approximation_error(original_graph, summary_graph, node_mapping=None,
                                   k=50, normalized=True):
        """
        Compute spectral approximation error between original and summary graphs.
        
        Measures how well the summary preserves the quadratic form of the Laplacian:
        (1-ε) x^T L_G x ≤ x^T L_G' x ≤ (1+ε) x^T L_G x
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            node_mapping (dict): Mapping from original nodes to summary nodes
            k (int): Number of eigenvectors to use
            normalized (bool): Whether to use normalized Laplacian
            
        Returns:
            dict: Metrics including upper and lower bounds on ε
        """
        # Handle directed graphs
        if isinstance(original_graph, nx.DiGraph):
            orig_graph = original_graph.to_undirected()
        else:
            orig_graph = original_graph
            
        if isinstance(summary_graph, nx.DiGraph):
            summ_graph = summary_graph.to_undirected()
        else:
            summ_graph = summary_graph
        
        # Compute Laplacians
        if normalized:
            L_orig = nx.normalized_laplacian_matrix(orig_graph).todense()
            L_summ = nx.normalized_laplacian_matrix(summ_graph).todense()
        else:
            L_orig = nx.laplacian_matrix(orig_graph).todense()
            L_summ = nx.laplacian_matrix(summ_graph).todense()
        
        # For sparsification (no node mapping - same node set)
        if node_mapping is None:
            if orig_graph.number_of_nodes() == summ_graph.number_of_nodes():
                return SpectralMetrics._compute_spectral_error_same_nodes(
                    L_orig, L_summ, k=k)
        
        # For coarsening (node mapping - different node sets)
        return SpectralMetrics._compute_spectral_error_mapped_nodes(
            L_orig, L_summ, node_mapping, k=k)
    
    @staticmethod
    def _compute_spectral_error_same_nodes(L_orig, L_summ, k=50):
        """Compute spectral error when original and summary have same nodes."""
        # Compute the k smallest eigenvalues for both Laplacians
        try:
            evals_orig, evecs_orig = np.linalg.eigh(L_orig)
            evals_summ, evecs_summ = np.linalg.eigh(L_summ)
            
            # Sort and take k smallest non-zero eigenvalues
            idx_orig = np.argsort(evals_orig)
            idx_summ = np.argsort(evals_summ)
            
            # Skip first eigenvalue (zero for connected graphs)
            k_orig = min(k+1, len(evals_orig))
            k_summ = min(k+1, len(evals_summ))
            
            evals_orig = evals_orig[idx_orig][1:k_orig]  # Skip zero eigenvalue
            evals_summ = evals_summ[idx_summ][1:k_summ]  # Skip zero eigenvalue
            
            # Cut to same length for comparison
            min_len = min(len(evals_orig), len(evals_summ))
            evals_orig = evals_orig[:min_len]
            evals_summ = evals_summ[:min_len]
            
            # Compute relative errors 
            rel_errors = np.abs(evals_summ - evals_orig) / np.maximum(evals_orig, 1e-10)
            spectral_error = np.mean(rel_errors)
            max_error = np.max(rel_errors)
            
            # Compute the correlation between eigenvalues
            corr, _ = stats.spearmanr(evals_orig, evals_summ)
            
            return {
                'spectral_error': spectral_error,
                'max_spectral_error': max_error,
                'eigenvalue_correlation': corr,
                'laplacian_spectral_error': spectral_error  # For compatibility
            }
            
        except Exception as e:
            logger.warning(f"Error computing spectral error: {e}")
            return {
                'spectral_error': float('nan'),
                'max_spectral_error': float('nan'),
                'eigenvalue_correlation': float('nan'),
                'laplacian_spectral_error': float('nan')  # For compatibility
            }
    
    @staticmethod
    def _compute_spectral_error_mapped_nodes(L_orig, L_summ, node_mapping, k=50):
        """Compute spectral error when nodes are mapped (coarsening case)."""
        try:
            # Compute eigendecomposition for both graphs
            evals_orig, evecs_orig = np.linalg.eigh(L_orig)
            evals_summ, evecs_summ = np.linalg.eigh(L_summ)
            
            # Sort by eigenvalues
            idx_orig = np.argsort(evals_orig)
            idx_summ = np.argsort(evals_summ)
            
            # Skip first eigenvalue (zero for connected graphs)
            evals_orig = evals_orig[idx_orig][1:k+1]  # Skip zero eigenvalue
            evecs_orig = evecs_orig[:, idx_orig][:, 1:k+1]  # Skip constant eigenvector
            
            evals_summ = evals_summ[idx_summ][1:k+1]  # Skip zero eigenvalue
            evecs_summ = evecs_summ[:, idx_summ][:, 1:k+1]  # Skip constant eigenvector
            
            # Cut to same length for comparison
            min_len = min(len(evals_orig), len(evals_summ))
            evals_orig = evals_orig[:min_len]
            evals_summ = evals_summ[:min_len]
            
            # Calculate relative error in eigenvalues
            rel_errors = np.abs(evals_summ - evals_orig) / np.maximum(evals_orig, 1e-10)
            spectral_error = np.mean(rel_errors)
            max_error = np.max(rel_errors)
            
            # Compute the correlation between eigenvalues
            corr, _ = stats.spearmanr(evals_orig, evals_summ)
            
            return {
                'spectral_error': spectral_error,
                'max_spectral_error': max_error,
                'eigenvalue_correlation': corr,
                'laplacian_spectral_error': spectral_error  # For compatibility
            }
            
        except Exception as e:
            logger.warning(f"Error computing spectral error with node mapping: {e}")
            return {
                'spectral_error': float('nan'),
                'max_spectral_error': float('nan'),
                'eigenvalue_correlation': float('nan'),
                'laplacian_spectral_error': float('nan')  # For compatibility
            }
    
    @staticmethod
    def effective_resistance_distortion(original_graph, summary_graph, sample_size=100,
                                     node_mapping=None):
        """
        Measure the distortion in effective resistances after summarization.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            sample_size (int): Number of node pairs to sample for comparison
            node_mapping (dict): Mapping from original to summary nodes
            
        Returns:
            dict: Metrics including average and max distortion
        """
        # Handle directed graphs
        if isinstance(original_graph, nx.DiGraph):
            orig_graph = original_graph.to_undirected()
        else:
            orig_graph = original_graph
            
        if isinstance(summary_graph, nx.DiGraph):
            summ_graph = summary_graph.to_undirected()
        else:
            summ_graph = summary_graph
        
        try:
            # Sample node pairs
            orig_nodes = list(orig_graph.nodes())
            if len(orig_nodes) < 2:
                return {
                    'resistance_distortion': float('nan'),
                    'max_resistance_distortion': float('nan')
                }
            
            # Limit sample size for large graphs
            sample_size = min(sample_size, len(orig_nodes) * (len(orig_nodes) - 1) // 2)
            
            # Compute Laplacian pseudoinverses
            L_orig = nx.laplacian_matrix(orig_graph).todense()
            L_pinv_orig = np.linalg.pinv(L_orig)
            
            L_summ = nx.laplacian_matrix(summ_graph).todense()
            L_pinv_summ = np.linalg.pinv(L_summ)
            
            # Sample node pairs
            distortions = []
            for _ in range(sample_size):
                # Sample two different nodes
                u, v = np.random.choice(orig_nodes, size=2, replace=False)
                
                # Get indices in original graph
                u_idx_orig = list(orig_graph.nodes()).index(u)
                v_idx_orig = list(orig_graph.nodes()).index(v)
                
                # Compute effective resistance in original graph
                r_orig = (L_pinv_orig[u_idx_orig, u_idx_orig] + 
                          L_pinv_orig[v_idx_orig, v_idx_orig] - 
                          2 * L_pinv_orig[u_idx_orig, v_idx_orig])
                
                # If using node mapping (coarsening)
                if node_mapping:
                    if u in node_mapping and v in node_mapping:
                        u_summ = node_mapping[u]
                        v_summ = node_mapping[v]
                        
                        # Get indices in summary graph
                        u_idx_summ = list(summ_graph.nodes()).index(u_summ)
                        v_idx_summ = list(summ_graph.nodes()).index(v_summ)
                        
                        # Compute effective resistance in summary graph
                        r_summ = (L_pinv_summ[u_idx_summ, u_idx_summ] + 
                                 L_pinv_summ[v_idx_summ, v_idx_summ] - 
                                 2 * L_pinv_summ[u_idx_summ, v_idx_summ])
                        
                        # Compute distortion
                        if r_orig > 0:
                            distortion = abs(r_summ - r_orig) / r_orig
                            distortions.append(distortion)
                else:  # Same nodes (sparsification)
                    if u in summ_graph.nodes() and v in summ_graph.nodes():
                        u_idx_summ = list(summ_graph.nodes()).index(u)
                        v_idx_summ = list(summ_graph.nodes()).index(v)
                        
                        # Compute effective resistance in summary graph
                        r_summ = (L_pinv_summ[u_idx_summ, u_idx_summ] + 
                                 L_pinv_summ[v_idx_summ, v_idx_summ] - 
                                 2 * L_pinv_summ[u_idx_summ, v_idx_summ])
                        
                        # Compute distortion
                        if r_orig > 0:
                            distortion = abs(r_summ - r_orig) / r_orig
                            distortions.append(distortion)
            
            if not distortions:
                return {
                    'resistance_distortion': float('nan'),
                    'max_resistance_distortion': float('nan')
                }
                
            return {
                'resistance_distortion': np.mean(distortions),
                'max_resistance_distortion': np.max(distortions)
            }
            
        except Exception as e:
            logger.warning(f"Error computing resistance distortion: {e}")
            return {
                'resistance_distortion': float('nan'),
                'max_resistance_distortion': float('nan')
            }
    
    @staticmethod
    def random_walk_distortion(original_graph, summary_graph, steps=3, sample_size=100,
                            node_mapping=None):
        """
        Measure how well random walk distributions are preserved.
        
        Args:
            original_graph (nx.Graph): The original graph
            summary_graph (nx.Graph): The summarized graph
            steps (int): Number of random walk steps
            sample_size (int): Number of starting nodes to sample
            node_mapping (dict): Mapping from original to summary nodes
            
        Returns:
            dict: Metrics including average KL divergence between walks
        """
        try:
            # Sample starting nodes
            orig_nodes = list(original_graph.nodes())
            if not orig_nodes:
                return {'random_walk_kl': float('nan')}
                
            sample_size = min(sample_size, len(orig_nodes))
            start_nodes = np.random.choice(orig_nodes, size=sample_size, replace=False)
            
            # Compute transition matrices
            P_orig = nx.adjacency_matrix(original_graph).todense()
            D_orig = np.diag(np.sum(P_orig, axis=1).A1)
            P_orig = np.linalg.inv(D_orig) @ P_orig  # Transition matrix
            
            P_summ = nx.adjacency_matrix(summary_graph).todense()
            D_summ = np.diag(np.sum(P_summ, axis=1).A1)
            P_summ = np.linalg.inv(D_summ) @ P_summ  # Transition matrix
            
            kl_divs = []
            
            for start_node in start_nodes:
                # Compute distribution after k steps in original graph
                start_idx_orig = list(original_graph.nodes()).index(start_node)
                dist_orig = np.zeros(len(orig_nodes))
                dist_orig[start_idx_orig] = 1.0
                
                for _ in range(steps):
                    dist_orig = dist_orig @ P_orig
                
                # For coarsening (with node mapping)
                if node_mapping:
                    if start_node in node_mapping:
                        # Map the distribution to summary nodes
                        summ_nodes = list(summary_graph.nodes())
                        start_summ = node_mapping[start_node]
                        start_idx_summ = summ_nodes.index(start_summ)
                        
                        # Compute distribution in summary graph
                        dist_summ = np.zeros(len(summ_nodes))
                        dist_summ[start_idx_summ] = 1.0
                        
                        for _ in range(steps):
                            dist_summ = dist_summ @ P_summ
                            
                        # Lift the distribution back to original nodes for comparison
                        lifted_dist = np.zeros(len(orig_nodes))
                        for i, orig_node in enumerate(orig_nodes):
                            if orig_node in node_mapping:
                                summ_node = node_mapping[orig_node]
                                if summ_node in summ_nodes:
                                    summ_idx = summ_nodes.index(summ_node)
                                    lifted_dist[i] = dist_summ[summ_idx]
                        
                        # Normalize distributions
                        dist_orig = dist_orig / np.sum(dist_orig)
                        lifted_dist = lifted_dist / np.sum(lifted_dist) if np.sum(lifted_dist) > 0 else lifted_dist
                        
                        # Compute KL divergence
                        kl = 0
                        for p, q in zip(dist_orig, lifted_dist):
                            if p > 0 and q > 0:
                                kl += p * np.log(p / q)
                                
                        kl_divs.append(kl)
                else:  # For sparsification (same nodes)
                    if start_node in summary_graph.nodes():
                        start_idx_summ = list(summary_graph.nodes()).index(start_node)
                        dist_summ = np.zeros(len(summary_graph.nodes()))
                        dist_summ[start_idx_summ] = 1.0
                        
                        for _ in range(steps):
                            dist_summ = dist_summ @ P_summ
                            
                        # Normalize distributions
                        dist_orig = dist_orig / np.sum(dist_orig)
                        dist_summ = dist_summ / np.sum(dist_summ)
                        
                        # Compute KL divergence
                        kl = 0
                        for p, q in zip(dist_orig, dist_summ):
                            if p > 0 and q > 0:
                                kl += p * np.log(p / q)
                                
                        kl_divs.append(kl)
            
            if not kl_divs:
                return {'random_walk_kl': float('nan')}
                
            return {'random_walk_kl': np.mean(kl_divs)}
            
        except Exception as e:
            logger.warning(f"Error computing random walk distortion: {e}")
            return {'random_walk_kl': float('nan')}
    
    @staticmethod
    def evaluate_all(original_graph, summary_graph, node_mapping=None, k=50):
        """Run all spectral evaluation metrics."""
        logger.info("Running spectral metrics evaluation")
        
        # This is a wrapper that calls all the metrics and combines results
        results = {}
        
        # Spectral approximation error
        approx_results = SpectralMetrics.spectral_approximation_error(
            original_graph, summary_graph, node_mapping, k=k)
        results.update(approx_results)
        
        # Effective resistance distortion (sample for large graphs)
        resist_results = SpectralMetrics.effective_resistance_distortion(
            original_graph, summary_graph, 
            sample_size=min(100, original_graph.number_of_nodes()),
            node_mapping=node_mapping)
        results.update(resist_results)
        
        # Random walk distortion
        walk_results = SpectralMetrics.random_walk_distortion(
            original_graph, summary_graph, steps=3, 
            sample_size=min(100, original_graph.number_of_nodes()),
            node_mapping=node_mapping)
        results.update(walk_results)
        
        logger.info("Spectral metrics evaluation completed")
        return results

import networkx as nx
import numpy as np
import scipy.sparse as sp
import logging
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
import random
import time
import community as community_louvain

logger = logging.getLogger(__name__)

class Metrics:
    """Implements the five core metrics for evaluating graph summarization quality."""
    
    @staticmethod
    def spectral_approximation_error(original_graph, summary_graph, node_mapping=None, k=50):
        logger.info("Starting spectral_approximation_error computation (k=%d)", k)
        try:
            orig_graph = original_graph.to_undirected() if isinstance(original_graph, nx.DiGraph) else original_graph
            summ_graph = summary_graph.to_undirected()  if isinstance(summary_graph, nx.DiGraph)    else summary_graph
            logger.info("Converted graphs to undirected for Laplacian")

            l_orig = nx.normalized_laplacian_matrix(orig_graph).todense()
            l_summ = nx.normalized_laplacian_matrix(summ_graph).todense()
            logger.info("Computed normalized Laplacian matrices")

            evals_orig = np.sort(np.linalg.eigvalsh(l_orig))
            evals_summ = np.sort(np.linalg.eigvalsh(l_summ))
            logger.info("Eigenvalues computed and sorted")

            k_orig = min(k+1, len(evals_orig)-1)
            k_summ = min(k+1, len(evals_summ)-1)
            evals_orig = evals_orig[1:k_orig+1]
            evals_summ = evals_summ[1:k_summ+1]
            min_k = min(len(evals_orig), len(evals_summ))
            evals_orig, evals_summ = evals_orig[:min_k], evals_summ[:min_k]
            logger.info("Using %d eigenvalues for error computation", min_k)

            rel_errors = np.abs(evals_summ - evals_orig) / np.maximum(evals_orig, 1e-10)
            spectral_error = float(np.mean(rel_errors))
            logger.info("Spectral approximation error: %f", spectral_error)
            return spectral_error

        except Exception as e:
            logger.info("Error in spectral_approximation_error: %s", e)
            return float('nan')
    
    @staticmethod
    def community_structure_fidelity(original_graph, summary_graph, node_mapping=None,
                                     original_communities=None, summary_communities=None):
        logger.info("Starting community_structure_fidelity computation")
        try:
            orig_ug = original_graph.to_undirected() if isinstance(original_graph, nx.DiGraph) else original_graph
            summ_ug = summary_graph.to_undirected()  if isinstance(summary_graph, nx.DiGraph)    else summary_graph

            if original_communities is None:
                logger.info("Detecting communities in original graph")
                original_communities = community_louvain.best_partition(orig_ug, random_state=42)
            if summary_communities is None:
                logger.info("Detecting communities in summary graph")
                summary_communities = community_louvain.best_partition(summ_ug, random_state=42)

            if node_mapping is None:
                logger.info("Performing NMI on same-node set (sparsification)")
                common = set(original_graph) & set(summary_graph)
                if not common:
                    logger.info("No common nodes; returning NMI=0.0")
                    return 0.0
                orig_labels = [original_communities[n] for n in common]
                summ_labels = [summary_communities[n] for n in common]
            else:
                logger.info("Performing NMI with node mapping (coarsening)")
                node_to_comm = {n: original_communities.get(n) for n in original_graph if n in original_communities}
                mapping_comm = {o: summary_communities.get(s) for o, s in node_mapping.items() if s in summary_communities}
                common = set(node_to_comm) & set(mapping_comm)
                if not common:
                    logger.info("No shared nodes after mapping; returning NMI=0.0")
                    return 0.0
                orig_labels = [node_to_comm[n] for n in common]
                summ_labels = [mapping_comm[n] for n in common]

            nmi = normalized_mutual_info_score(orig_labels, summ_labels)
            logger.info("Community-structure NMI: %f", nmi)
            return nmi

        except Exception as e:
            logger.info("Error in community_structure_fidelity: %s", e)
            return float('nan')
    
    @staticmethod
    def distance_distortion(original_graph, summary_graph, node_mapping=None, sample_size=1000):
        logger.info("Starting distance_distortion computation (sample_size=%d)", sample_size)
        try:
            orig_nodes = list(original_graph.nodes())
            if len(orig_nodes) > 1000:
                sample_size = min(sample_size, 1000)
                logger.info("Large graph, sampling down to %d pairs", sample_size)
            else:
                max_pairs = len(orig_nodes)*(len(orig_nodes)-1)//2
                sample_size = min(sample_size, max_pairs)
                logger.info("Total possible pairs %d; using sample_size %d", max_pairs, sample_size)

            pairs = [tuple(random.sample(orig_nodes, 2)) for _ in range(sample_size)]
            stretches = []

            for u, v in pairs:
                if node_mapping is None:
                    if u in summary_graph and v in summary_graph and nx.has_path(original_graph, u, v) and nx.has_path(summary_graph, u, v):
                        d_orig = nx.shortest_path_length(original_graph, u, v)
                        d_summ = nx.shortest_path_length(summary_graph, u, v)
                        if d_orig > 0:
                            stretches.append(d_summ / d_orig)
                else:
                    if u in node_mapping and v in node_mapping:
                        u_s, v_s = node_mapping[u], node_mapping[v]
                        if nx.has_path(original_graph, u, v) and nx.has_path(summary_graph, u_s, v_s):
                            d_orig = nx.shortest_path_length(original_graph, u, v)
                            d_summ = nx.shortest_path_length(summary_graph, u_s, v_s)
                            if d_orig > 0:
                                stretches.append(d_summ / d_orig)

            if not stretches:
                logger.info("No valid pairs found; returning NaN")
                return float('nan')

            avg_stretch = float(np.mean(stretches))
            logger.info("Average stretch: %f", avg_stretch)
            return avg_stretch

        except Exception as e:
            logger.info("Error in distance_distortion: %s", e)
            return float('nan')
    
    @staticmethod
    def centrality_retention(original_graph, summary_graph, node_mapping=None, k=100):
        logger.info("Starting centrality_retention computation (k=%d)", k)
        try:
            orig_pr = nx.pagerank(original_graph)
            summ_pr = nx.pagerank(summary_graph)
            logger.info("Computed PageRank for original and summary graphs")

            top_orig = {n for n, _ in sorted(orig_pr.items(), key=lambda x: x[1], reverse=True)[:k]}
            logger.info("Selected top-%d nodes from original graph", k)

            if node_mapping is None:
                top_summ = {n for n, _ in sorted(summ_pr.items(), key=lambda x: x[1], reverse=True)[:k]}
                intersect = top_orig & top_summ
            else:
                reverse_map = {}
                for o, s in node_mapping.items():
                    reverse_map.setdefault(s, []).append(o)
                top_summ = sorted(summ_pr.items(), key=lambda x: x[1], reverse=True)[:k]
                orig_from_summ = {o for s, _ in top_summ for o in reverse_map.get(s, [])}
                intersect = top_orig & orig_from_summ

            precision = len(intersect) / k
            logger.info("Precision@%d: %f", k, precision)
            return precision

        except Exception as e:
            logger.info("Error in centrality_retention: %s", e)
            return float('nan')
    
    @staticmethod
    def compression_ratio(original_graph, summary_graph):
        logger.info("Starting compression_ratio computation")
        try:
            orig_size = original_graph.number_of_nodes() + original_graph.number_of_edges()
            summ_size = summary_graph.number_of_nodes()  + summary_graph.number_of_edges()
            logger.info("Original size: %d, Summary size: %d", orig_size, summ_size)

            ratio = summ_size / orig_size
            logger.info("Compression ratio: %f", ratio)
            return ratio

        except Exception as e:
            logger.info("Error in compression_ratio: %s", e)
            return float('nan')
    
    @staticmethod
    def evaluate_all(original_graph, summary_graph, node_mapping=None, k_spectral=50, k_centrality=100):
        logger.info("Running evaluate_all on graphs")
        start = time.time()

        results = {
            'spectral_error': Metrics.spectral_approximation_error(original_graph, summary_graph, node_mapping, k_spectral),
            'community_nmi': Metrics.community_structure_fidelity(original_graph, summary_graph, node_mapping),
            'avg_stretch': Metrics.distance_distortion(original_graph, summary_graph, node_mapping),
            'precision_at_k': Metrics.centrality_retention(original_graph, summary_graph, node_mapping, k_centrality),
            'compression_ratio': Metrics.compression_ratio(original_graph, summary_graph)
        }

        elapsed = time.time() - start
        results['evaluation_time'] = elapsed
        logger.info("Completed evaluate_all in %.3f seconds", elapsed)
        return results

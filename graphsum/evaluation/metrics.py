import networkx as nx
import numpy as np
import scipy.sparse as sp
import logging
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
import random
import time
import community as community_louvain
from scipy.sparse.linalg import eigsh
from networkx.algorithms.community import asyn_lpa_communities

logger = logging.getLogger(__name__)

class Metrics:
    _cache = {}

    def _sample_graph(G, n=10000, seed=42):
        if G.number_of_nodes() <= n:
            return G
        rng = random.Random(seed)
        nodes = rng.sample(list(G.nodes()), n)
        return G.subgraph(nodes).copy()

    @staticmethod
    def spectral_approximation_error(original_graph, summary_graph, node_mapping=None, k=50, sample_size=10000):
        orig_graph = original_graph.to_undirected() if isinstance(original_graph, nx.DiGraph) else original_graph
        summ_graph = summary_graph.to_undirected()  if isinstance(summary_graph, nx.DiGraph) else summary_graph
        logger.info("Converted graphs to undirected for Laplacian")
        
        # 1) maybe sample
        def sample(G):
            return G if G.number_of_nodes() <= sample_size else Metrics._sample_graph(G, sample_size)
        G_o, G_s = sample(orig_graph), sample(summ_graph)

        # 2) cache orig eigenvalues
        key = (id(G_o), k)
        if key not in Metrics._cache:
            L_o = nx.normalized_laplacian_matrix(G_o)
            e_o, _ = eigsh(L_o, k=min(k, G_o.number_of_nodes()-2)+1, which='SM', tol=1e-3)
            Metrics._cache[key] = np.sort(e_o)[1:]
        e_o = Metrics._cache[key]

        # 3) compute summary eigenvalues
        L_s = nx.normalized_laplacian_matrix(G_s)
        e_s, _ = eigsh(L_s, k=min(k, G_s.number_of_nodes()-2)+1, which='SM', tol=1e-3)
        e_s = np.sort(e_s)[1:]

        # 4) filter and average
        m = min(len(e_o), len(e_s))
        eps = 1e-3
        mask = e_o[:m] > eps
        if not mask.any():
            return float('nan')
        rel = np.abs(e_s[:m][mask] - e_o[:m][mask]) / e_o[:m][mask]
        spectral_error = float(rel.mean())
        logger.info("Spectral approximation error: %f", spectral_error)
        return spectral_error

# class Metrics:
#     """Implements the five core metrics for evaluating graph summarization quality."""
    
    # @staticmethod
    # def spectral_approximation_error(original_graph, summary_graph, node_mapping=None, k=50):
    #     logger.info("Starting spectral_approximation_error computation (k=%d)", k)
    #     try:
    #         orig_graph = original_graph.to_undirected() if isinstance(original_graph, nx.DiGraph) else original_graph
    #         summ_graph = summary_graph.to_undirected()  if isinstance(summary_graph, nx.DiGraph)    else summary_graph
    #         logger.info("Converted graphs to undirected for Laplacian")

    #         l_orig = nx.normalized_laplacian_matrix(orig_graph)
    #         l_summ = nx.normalized_laplacian_matrix(summ_graph)
    #         logger.info("Computed normalized Laplacian matrices")

    #         # compute the k+1 smallest eigenvalues (including the zero)
    #         evals_orig, _ = eigsh(l_orig, k=k+1, which='SM', tol=1e-3)
    #         evals_summ, _ = eigsh(l_summ, k=k+1, which='SM', tol=1e-3)
    #         logger.info("Eigenvalues computed and sorted")

    #         # sort and drop the trivial zero eigenvalue
    #         evals_orig = np.sort(evals_orig)[1:]
    #         evals_summ = np.sort(evals_summ)[1:]
    #         m = min(len(evals_orig), len(evals_summ))
    #         eps = 1e-3
    #         mask = evals_orig[:m] > eps
    #         if not np.any(mask):
    #             return float('nan')
    #         rel = np.abs(evals_summ[:m][mask] - evals_orig[:m][mask]) / evals_orig[:m][mask]
    #         return float(rel.mean())

    #     except Exception as e:
    #         logger.info("Error in spectral_approximation_error: %s", e)
    #         return float('nan')
        


    # def spectral_approximation_error(original_graph, summary_graph, node_mapping=None, k=50):
    #     try:
    #         # build sparse Laplacians directly
    #         L_orig = nx.normalized_laplacian_matrix(original_graph)
    #         L_summ = nx.normalized_laplacian_matrix(summary_graph)

    #         # compute the k+1 smallest eigenvalues (including the zero)
    #         evals_orig, _ = eigsh(L_orig, k=k+1, which='SM', tol=1e-3)
    #         evals_summ, _ = eigsh(L_summ, k=k+1, which='SM', tol=1e-3)

    #         # sort and drop the trivial zero eigenvalue
    #         evals_orig = np.sort(evals_orig)[1:]
    #         evals_summ = np.sort(evals_summ)[1:]
    #         m = min(len(evals_orig), len(evals_summ))
    #         rel_errors = np.abs(evals_summ[:m] - evals_orig[:m]) / np.maximum(evals_orig[:m], 1e-10)
    #         return float(rel_errors.mean())
    #     except Exception as e:
    #         logger.warning("spectral_approximation_error failed: %s", e)
    #         return float('nan')
    @staticmethod
    def _detect_communities(G, max_louvain=10000, seed=42):
        """
        If |G| <= max_louvain, use Louvain; otherwise label‐propagation.
        Returns a dict node → community_id.
        """
        n = G.number_of_nodes()
        if n <= max_louvain:
            return community_louvain.best_partition(G, random_state=seed)
        else:
            # label-propagation: O(|V|+|E|)
            parts = asyn_lpa_communities(G, seed=seed)
            return {node: cid for cid, comm in enumerate(parts) for node in comm}
        
    @staticmethod
    def community_structure_fidelity(original_graph, summary_graph, node_mapping=None,
                                     original_communities=None, summary_communities=None):
        logger.info("Starting community_structure_fidelity computation")
        try:
            orig_ug = original_graph.to_undirected() if isinstance(original_graph, nx.DiGraph) else original_graph
            summ_ug = summary_graph.to_undirected()  if isinstance(summary_graph, nx.DiGraph)    else summary_graph

            if original_communities is None:
                logger.info("Detecting communities in original graph")
                original_communities = Metrics._detect_communities(orig_ug)
            if summary_communities is None:
                logger.info("Detecting communities in summary graph")
                summary_communities = Metrics._detect_communities(summ_ug)
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
            # Build the list of "surviving" original nodes
            if node_mapping is None:
                common_nodes = list(set(original_graph.nodes()) & set(summary_graph.nodes()))
            else:
                # only original nodes that map into an existing summary node
                common_nodes = [
                    o for o in original_graph.nodes()
                    if o in node_mapping and node_mapping[o] in summary_graph
                ]

            if len(common_nodes) < 2:
                logger.info("Too few surviving nodes (%d) for distance_distortion", len(common_nodes))
                return float('nan')

            # Sample pairs from the surviving set
            pairs = [tuple(random.sample(common_nodes, 2)) for _ in range(sample_size)]
            stretches = []

            for u, v in pairs:
                if node_mapping is None:
                    # both u,v in summary by construction
                    if nx.has_path(original_graph, u, v) and nx.has_path(summary_graph, u, v):
                        d_orig = nx.shortest_path_length(original_graph, u, v)
                        d_summ = nx.shortest_path_length(summary_graph, u, v)
                        if d_orig > 0:
                            stretches.append(d_summ / d_orig)
                else:
                    # map into summary
                    u_s, v_s = node_mapping[u], node_mapping[v]
                    if nx.has_path(original_graph, u, v) and nx.has_path(summary_graph, u_s, v_s):
                        d_orig = nx.shortest_path_length(original_graph, u, v)
                        d_summ = nx.shortest_path_length(summary_graph, u_s, v_s)
                        if d_orig > 0:
                            stretches.append(d_summ / d_orig)

            valid = len(stretches)
            logger.info("Found %d valid pairs out of %d sampled", valid, sample_size)
            if valid == 0:
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

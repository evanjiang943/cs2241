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
    """

    def __init__(self, name="Sparsifier", epsilon=0.1):
        super().__init__(name=name)
        self.epsilon = epsilon

    def _compute_effective_resistances(self, graph, max_nodes=1000):
        """
        Approximate effective resistances for edges in large graphs
        using degree-based heuristic; exact via pseudoinverse for smaller.
        """
        n = graph.number_of_nodes()
        resistances = {}
        nodes = list(graph.nodes())

        if n > max_nodes:
            logger.info("Approximating resistances via degree heuristic for %d nodes", n)
            deg = dict(graph.degree(weight=None))
            for u, v, data in graph.edges(data=True):
                w = data.get('weight', 1.0)
                r = (1.0/deg.get(u,1) + 1.0/deg.get(v,1)) / w
                resistances[(u, v)] = r
                resistances[(v, u)] = r
        else:
            logger.info("Computing exact resistances via pseudoinverse for %d nodes", n)
            L = nx.laplacian_matrix(graph).todense()
            L_pinv = np.linalg.pinv(L)
            for (u, v) in graph.edges():
                i, j = nodes.index(u), nodes.index(v)
                r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
                resistances[(u, v)] = float(r)
                resistances[(v, u)] = float(r)
        return resistances

    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        self._start_timer()

        epsilon = kwargs.get('epsilon', self.epsilon)
        max_nodes = kwargs.get('max_nodes', 1000)
        original_edges = graph.number_of_edges()
        target_edges = max(graph.number_of_nodes() - 1,
                           int(original_edges * reduction_factor))

        logger.info("Sparsifying to ~%d edges (factor=%.3f)", target_edges, reduction_factor)
        work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
        self._init_summary(graph, graph.number_of_nodes())

        # keep all nodes
        for node, data in graph.nodes(data=True):
            self.node_mapping[node] = node
            self.reverse_mapping[node] = [node]
            self.summary_graph.add_node(node, **data)

        # compute resistances
        resistances = self._compute_effective_resistances(work_graph, max_nodes)

        # build sampling probabilities
        edges = list(graph.edges())
        weights = np.array([graph[u][v].get('weight',1.0) for u,v in edges])
        res_vals = np.array([resistances.get((u,v),0.0) for u,v in edges])
        probs = weights * res_vals
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones_like(probs) / len(probs)

        logger.info("Sampling %d edges based on effective resistances", target_edges)
        # sample with replacement and count via numpy.unique
        sampled_idx = np.random.choice(len(edges), size=target_edges, replace=True, p=probs)
        unique_idx, counts = np.unique(sampled_idx, return_counts=True)

        # add sampled edges
        for idx, count in zip(unique_idx, counts):
            u, v = edges[idx]
            data = graph.get_edge_data(u, v, {}).copy()
            orig_w = data.get('weight', 1.0)
            p_e = probs[idx]
            new_w = orig_w * count / (p_e * target_edges) if p_e > 0 else orig_w
            data.update({'weight': new_w,
                         'original_weight': orig_w,
                         'sampled_count': int(count)})
            self.summary_graph.add_edge(u, v, **data)

        t = self._stop_timer('sparsification')
        logger.info("Sparsification done in %.2f s: %d nodes, %d edges", 
                    t, self.summary_graph.number_of_nodes(), 
                    self.summary_graph.number_of_edges())
        return self.summary_graph

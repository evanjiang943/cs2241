import networkx as nx
import numpy as np
import logging
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans

from .base import GraphSummarizer

logger = logging.getLogger(__name__)

class SpectralCoarsener(GraphSummarizer):
    """
    Summarizes graphs using spectral coarsening.
    
    Computes the graph Laplacian and its eigenvectors, then clusters nodes
    based on their spectral embeddings to create a coarsened summary that preserves
    random walk behavior and spectral properties.
    """
    
    def __init__(self, name="Coarsener", n_eigenvectors=None):
        """
        Initialize the coarsener.
        
        Args:
            name (str): Name of the summarizer
            n_eigenvectors (int, optional): Number of eigenvectors to use for clustering
        """
        super().__init__(name=name)
        self.n_eigenvectors = n_eigenvectors
    
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        """
        Summarize graph using spectral clustering-based coarsening.
        
        Args:
            graph (nx.Graph): The graph to summarize
            reduction_factor (float): Target size reduction factor (0-1)
            **kwargs: Additional parameters:
                n_eigenvectors (int): Number of eigenvectors to use (overrides init)
                normalized (bool): Whether to use normalized Laplacian
                weight (str): Edge weight attribute to use
                normalized (bool): Whether to use normalized Laplacian
                normalized (bool): Whether to use normalized Laplacian
        Returns:
            nx.Graph: The coarsened graph
        """
        self._start_timer()
        
        # Parse parameters
        n_eigenvectors = kwargs.get('n_eigenvectors', self.n_eigenvectors)
        normalized = kwargs.get('normalized', True)
        
        # Determine target number of nodes in summary
        n_original = graph.number_of_nodes()
        n_summary = max(2, int(n_original * reduction_factor))
        
        logger.info(f"Creating spectral coarsening with {n_summary} nodes (reduction={reduction_factor})")
        
        # If the graph is directed, convert to undirected for Laplacian computation
        if isinstance(graph, nx.DiGraph):
            logger.info("Converting directed graph to undirected for spectral analysis")
            graph_for_laplacian = graph.to_undirected()
        else:
            graph_for_laplacian = graph
        
        # Compute Laplacian matrix
        logger.info("Computing graph Laplacian")
        laplacian = (nx.normalized_laplacian_matrix(graph_for_laplacian)
                     if normalized else nx.laplacian_matrix(graph_for_laplacian))
        
        # Compute eigenvectors (using sparse eigensolvers for large graphs)
        if n_original > 5000:
            logger.info("Using sparse lobpcg eigensolver for large graph")
            lap_csr = laplacian.tocsr()
            # determine k
            k = (min(n_summary, lap_csr.shape[0] - 1)
                 if n_eigenvectors is None else min(n_eigenvectors, lap_csr.shape[0] - 1))
            init_vecs = np.random.rand(lap_csr.shape[0], k)
            eigenvalues, eigenvectors = sp.linalg.lobpcg(
                lap_csr,
                init_vecs,
                # tol=1e-2,
                maxiter=500
            )
        else:
            logger.info("Using dense eigensolver for small graph")
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian.todense())
        
        # Sort eigenvectors by eigenvalues (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        spectral_time = self._stop_timer('spectral_computation')
        logger.info(f"Spectral computation completed in {spectral_time:.2f} seconds")
        
        # Determine number of eigenvectors to use for clustering
        if n_eigenvectors is None:
            k = min(n_summary, eigenvectors.shape[1] - 1)
        else:
            k = min(n_eigenvectors, eigenvectors.shape[1] - 1)
        
        # Create feature matrix for clustering from the first k non-trivial eigenvectors
        features = eigenvectors[:, 1:k+1]  # Skip the first eigenvector (constant)
        
        # Cluster nodes based on spectral features
        self._start_timer()
        logger.info(f"Clustering {n_original} nodes into {n_summary} clusters using {k} eigenvectors")

        kmeans = MiniBatchKMeans(
            n_clusters=n_summary,
            batch_size=10_000,         # tune this (e.g. 1–5% of n)
            max_iter=50,               # fewer passes over each batch
            tol=1e-3,
            random_state=42,
            init="k-means++",          # same good initializer
            n_init=1                   # no need for multiple initializations
        )

        clusters = kmeans.fit_predict(features)

        clustering_time = self._stop_timer('clustering')
        logger.info(f"Clustering completed in {clustering_time:.2f} seconds")
        
        # Initialize summary graph
        self._start_timer()
        self._init_summary(graph, n_summary)
        
        # Create node mappings
        nodes = list(graph.nodes())
        for i, node in enumerate(nodes):
            cid = int(clusters[i])
            self.node_mapping[node] = cid
            self.reverse_mapping.setdefault(cid, []).append(node)
        
        # Add nodes to summary graph with size attributes
        for cid, members in self.reverse_mapping.items():
            self.summary_graph.add_node(cid, size=len(members), members=len(members))
        
        # Add weighted edges
        for u, v, data in graph.edges(data=True):
            cu = self.node_mapping.get(u)
            cv = self.node_mapping.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            w = data.get('weight', 1.0)
            if self.summary_graph.has_edge(cu, cv):
                self.summary_graph[cu][cv]['weight'] += w
                self.summary_graph[cu][cv]['count'] += 1
            else:
                self.summary_graph.add_edge(cu, cv, weight=w, count=1)
        
        # Add internal edge counts as node attributes
        for cid in self.summary_graph.nodes():
            members = self.reverse_mapping[cid]
            self.summary_graph.nodes[cid]['internal_edges'] = graph.subgraph(members).number_of_edges()
        
        # Normalize edge weights by potential connections
        for u, v, d in self.summary_graph.edges(data=True):
            usz, vsz = len(self.reverse_mapping[u]), len(self.reverse_mapping[v])
            max_conn = usz * vsz
            d['max_connections'] = max_conn
            d['density'] = (d['count'] / max_conn) if max_conn > 0 else 0
        
        summary_time = self._stop_timer('summary_creation')
        logger.info(f"Coarsened graph created in {summary_time:.2f} seconds")
        logger.info(f"Coarsened summary: {self.summary_graph.number_of_nodes()} nodes, {self.summary_graph.number_of_edges()} edges")
        
        return self.summary_graph

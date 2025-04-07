"""
Community-based graph summarization.

This module implements graph summarization using community detection 
algorithms like Louvain method.
"""

import networkx as nx
import numpy as np
import logging
import community as community_louvain

from .base import GraphSummarizer

logger = logging.getLogger(__name__)


class CommunityBasedSummarizer(GraphSummarizer):
    """
    Summarizes graphs by detecting and grouping communities.
    
    Uses the Louvain community detection algorithm to identify communities,
    then creates a summary graph with one node per community and weighted
    edges representing inter-community connections.
    """
    
    def __init__(self, name="CommunityBased", resolution=1.0):
        """
        Initialize the summarizer.
        
        Args:
            name (str): Name of the summarizer
            resolution (float): Resolution parameter for community detection
                                Higher values give smaller communities
        """
        super().__init__(name=name)
        self.resolution = resolution
    
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        """
        Summarize graph by detecting and grouping communities.
        
        Args:
            graph (nx.Graph): The graph to summarize
            reduction_factor (float): Target reduction (ignored in this method)
            **kwargs: Additional parameters:
                resolution (float): Community detection resolution parameter
                weight (str): Edge weight attribute to use
                
        Returns:
            nx.Graph: The summarized graph
        """
        self._start_timer()
        
        resolution = kwargs.get('resolution', self.resolution)
        weight = kwargs.get('weight', None)
        
        # Handle directed graphs by converting to undirected
        if isinstance(graph, nx.DiGraph):
            logger.info("Converting directed graph to undirected for community detection")
            graph_for_communities = graph.to_undirected()
        else:
            graph_for_communities = graph
        
        # Detect communities
        logger.info(f"Detecting communities using Louvain method (resolution={resolution})")
        communities = community_louvain.best_partition(
            graph_for_communities, 
            weight=weight,
            resolution=resolution,
            random_state=42
        )
        
        community_detection_time = self._stop_timer('community_detection')
        logger.info(f"Community detection completed in {community_detection_time:.2f} seconds")
        
        # Count communities
        unique_communities = set(communities.values())
        n_communities = len(unique_communities)
        logger.info(f"Detected {n_communities} communities")
        
        # Initialize summary graph
        self._start_timer()
        self._init_summary(graph, n_communities)
        
        # Create node mappings
        self.node_mapping = communities
        self.reverse_mapping = {}
        
        for node, comm_id in communities.items():
            if comm_id not in self.reverse_mapping:
                self.reverse_mapping[comm_id] = []
            self.reverse_mapping[comm_id].append(node)
        
        # Add nodes to summary graph
        for comm_id, members in self.reverse_mapping.items():
            # Add size attribute to track community size
            self.summary_graph.add_node(
                comm_id, 
                size=len(members),
                members=len(members)
            )
        
        # Add weighted edges between communities
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            u_comm = communities.get(u)
            v_comm = communities.get(v)
            
            # Skip nodes that weren't assigned to communities
            if u_comm is None or v_comm is None:
                continue
                
            # Self-loops in summary represent internal community edges
            if u_comm == v_comm:
                continue  # We'll handle internal edges separately
                
            edge_weight = data.get(weight, 1.0)
            
            if self.summary_graph.has_edge(u_comm, v_comm):
                self.summary_graph[u_comm][v_comm]['weight'] += edge_weight
                self.summary_graph[u_comm][v_comm]['count'] += 1
            else:
                self.summary_graph.add_edge(
                    u_comm, v_comm, 
                    weight=edge_weight,
                    count=1
                )
            edge_count += 1
        
        # Add internal edge counts as node attributes
        for comm_id in self.summary_graph.nodes():
            members = self.reverse_mapping[comm_id]
            internal_edges = graph.subgraph(members).number_of_edges()
            # Add self-loops with count of internal edges
            self.summary_graph.nodes[comm_id]['internal_edges'] = internal_edges
        
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
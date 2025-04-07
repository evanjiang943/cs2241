"""
Base class for graph summarization algorithms.
"""

import networkx as nx
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class GraphSummarizer(ABC):
    """
    Abstract base class for graph summarization algorithms.
    
    All summarization techniques should inherit from this class and implement
    the summarize() method.
    """
    
    def __init__(self, name=None):
        """
        Initialize a summarizer.
        
        Args:
            name (str, optional): Name of the summarizer
        """
        self.name = name or self.__class__.__name__
        self.original_graph = None
        self.summary_graph = None
        self.node_mapping = {}     # Maps original nodes to summary nodes
        self.reverse_mapping = {}  # Maps summary nodes to original nodes
        self.stats = {}            # Stores runtime statistics
    
    @abstractmethod
    def summarize(self, graph, reduction_factor=0.1, **kwargs):
        """
        Summarize the given graph.
        
        Args:
            graph (nx.Graph): The graph to summarize
            reduction_factor (float): Target size reduction (0-1)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            nx.Graph: The summarized graph
        """
        pass
    
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
    
    def map_node(self, original_node):
        """
        Map an original node to its summary node.
        
        Args:
            original_node: Node from the original graph
            
        Returns:
            The corresponding node in the summary graph, or None if not mapped
        """
        return self.node_mapping.get(original_node)
    
    def map_back_nodes(self, summary_node):
        """
        Map a summary node back to its original nodes.
        
        Args:
            summary_node: Node from the summary graph
            
        Returns:
            List of corresponding nodes in the original graph
        """
        return self.reverse_mapping.get(summary_node, [])
    
    def _init_summary(self, original_graph, n_summary_nodes):
        """
        Initialize a new summary graph.
        
        Args:
            original_graph (nx.Graph): The original graph
            n_summary_nodes (int): Number of nodes in the summary
            
        Returns:
            nx.Graph: An empty summary graph
        """
        self.original_graph = original_graph
        
        # Create a new graph of the same type as the original
        if isinstance(original_graph, nx.DiGraph):
            self.summary_graph = nx.DiGraph()
        else:
            self.summary_graph = nx.Graph()
        
        # Add basic graph attributes
        self.summary_graph.graph['original_nodes'] = original_graph.number_of_nodes()
        self.summary_graph.graph['original_edges'] = original_graph.number_of_edges()
        self.summary_graph.graph['reduction_factor'] = n_summary_nodes / original_graph.number_of_nodes()
        self.summary_graph.graph['summarizer'] = self.name
        
        # Reset mappings
        self.node_mapping = {}
        self.reverse_mapping = {}
        
        return self.summary_graph
    
    def __repr__(self):
        """String representation of the summarizer."""
        return f"{self.name}()"
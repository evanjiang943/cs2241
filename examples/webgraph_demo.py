#!/usr/bin/env python
"""
Web Graph Summarization Demo

This script demonstrates how to use the graph summarization framework
on web-scale graphs from SNAP, with visualizations of results.
"""

import os
import sys
import time
import logging
import argparse
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from graphsum.summarizers.community import CommunityBasedSummarizer
from graphsum.summarizers.spectral import SpectralSummarizer
from graphsum.evaluation.evaluator import GraphEvaluator
from graphsum.io.snap import download_snap_dataset, load_snap_graph


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Web Graph Summarization Demo')
    
    parser.add_argument('--dataset', type=str, default='web-Stanford',
                        help='SNAP dataset to use')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for datasets')
    
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['community'],
                       help='Summarization methods to use')
    
    parser.add_argument('--reduction', type=float, default=0.1,
                       help='Reduction factor')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of communities')
    
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for visualization')
    
    parser.add_argument('--output-dir', type=str, default='demo_results',
                       help='Directory to save results')
    
    return parser.parse_args()


def sample_graph_for_visualization(graph, sample_size=1000):
    """
    Sample a subset of the graph for visualization.
    
    Args:
        graph: NetworkX graph to sample
        sample_size: Maximum number of nodes to include
        
    Returns:
        NetworkX graph of the sampled subgraph
    """
    logger = logging.getLogger(__name__)
    
    # If graph is small enough, use it directly
    if graph.number_of_nodes() <= sample_size:
        return graph
    
    logger.info(f"Sampling {sample_size} nodes for visualization")
    
    # Start with a random node
    sample_nodes = set()
    start_node = list(graph.nodes())[0]
    
    # Use BFS to get a connected sample
    bfs_queue = [start_node]
    while bfs_queue and len(sample_nodes) < sample_size:
        node = bfs_queue.pop(0)
        if node not in sample_nodes:
            sample_nodes.add(node)
            # Add neighbors to queue
            neighbors = list(graph.neighbors(node))
            # Randomize neighbor order
            import random
            random.shuffle(neighbors)
            bfs_queue.extend(neighbors)
    
    # Create the subgraph
    sampled_graph = graph.subgraph(sample_nodes).copy()
    logger.info(f"Sampled graph: {sampled_graph.number_of_nodes()} nodes, "
               f"{sampled_graph.number_of_edges()} edges")
    
    return sampled_graph


def visualize_graph_communities(graph, communities, output_file):
    """
    Visualize communities in a graph.
    
    Args:
        graph: NetworkX graph to visualize
        communities: Dictionary mapping nodes to community IDs
        output_file: Path to save the visualization
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Visualizing graph communities")
    
    # Get unique communities
    community_ids = set(communities.values())
    
    # Create a color map
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(community_ids)))
    color_map = {comm_id: colors[i] for i, comm_id in enumerate(community_ids)}
    
    # Assign colors to nodes
    node_colors = [color_map[communities.get(node, 0)] for node in graph.nodes()]
    
    # Create plot
    plt.figure(figsize=(12, 12))
    
    # Use spring layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw graph
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=node_colors,
        with_labels=False,
        node_size=50,
        edge_color='lightgray',
        alpha=0.8
    )
    
    plt.title(f"Graph Communities ({len(community_ids)} communities)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    logger.info(f"Visualization saved to {output_file}")


def run_demo():
    """Run the web graph summarization demo."""
    logger = setup_logging()
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download and load dataset
    logger.info(f"Preparing dataset: {args.dataset}")
    dataset_path = download_snap_dataset(args.dataset, args.data_dir)
    
    logger.info(f"Loading graph from {dataset_path}")
    graph = load_snap_graph(dataset_path, memory_efficient=True)
    
    logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Run each summarization method
    for method in args.methods:
        logger.info(f"Running {method} summarization with reduction factor {args.reduction}")
        
        if method == 'community':
            summarizer = CommunityBasedSummarizer()
        elif method == 'spectral':
            summarizer = SpectralSummarizer()
        else:
            logger.error(f"Unknown method: {method}")
            continue
        
        # Perform summarization
        start_time = time.time()
        summary_graph = summarizer.summarize(graph, reduction_factor=args.reduction)
        summarization_time = time.time() - start_time
        
        logger.info(f"Summary graph: {summary_graph.number_of_nodes()} nodes, "
                   f"{summary_graph.number_of_edges()} edges")
        logger.info(f"Summarization completed in {summarization_time:.2f} seconds")
        
        # Evaluate the summary
        logger.info("Evaluating summary quality")
        evaluator = GraphEvaluator(
            graph, 
            summary_graph, 
            summarizer.node_mapping, 
            summarizer.reverse_mapping
        )
        
        results = evaluator.evaluate_all()
        
        # Print summary
        evaluator.print_summary()
        
        # Save results
        results_file = os.path.join(args.output_dir, f"{method}_results.csv")
        pd.DataFrame([results]).to_csv(results_file, index=False)
        
        # Visualize if requested
        if args.visualize:
            # Sample graph for visualization
            sample_graph = sample_graph_for_visualization(graph, args.sample_size)
            
            # Get communities for sampled nodes
            if method == 'community':
                # For community-based summarization, we already have communities
                communities = {node: summarizer.node_mapping.get(node, 0) 
                               for node in sample_graph.nodes()}
            else:
                # For other methods, use node mapping
                communities = {node: summarizer.node_mapping.get(node, 0) 
                               for node in sample_graph.nodes()}
            
            # Visualize communities
            vis_file = os.path.join(args.output_dir, f"{method}_communities.png")
            visualize_graph_communities(sample_graph, communities, vis_file)
    
    logger.info(f"Demo completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    # Import numpy here to avoid circular import
    import numpy as np
    run_demo()
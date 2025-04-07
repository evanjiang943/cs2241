#!/usr/bin/env python
"""
Graph Summarization Experiment Runner

This script runs experiments to evaluate different graph summarization techniques
on large web-scale graphs.
"""

import os
import sys
import argparse
import logging
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from graphsum.summarizers.community import CommunityBasedSummarizer
from graphsum.summarizers.spectral import SpectralSummarizer
from graphsum.evaluation.evaluator import GraphEvaluator
from graphsum.io.snap import (
    download_snap_dataset, 
    load_snap_graph, 
    list_available_datasets,
    SNAP_WEB_GRAPHS
)


# Configure logging
def setup_logging(log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Graph Summarization Experiment Runner')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='web-Stanford',
                        help='SNAP dataset name or path to edge list file')
    
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available SNAP datasets and exit')
    
    parser.add_argument('--download-only', action='store_true',
                        help='Only download the dataset, do not run experiments')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for downloading datasets')
    
    parser.add_argument('--directed', action='store_true',
                        help='Treat the graph as directed')
    
    # Method options
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['community', 'spectral'],
                       help='Summarization methods to evaluate')
    
    parser.add_argument('--reductions', type=float, nargs='+',
                       default=[0.1, 0.2, 0.3],
                       help='Reduction factors to test')
    
    # Experiment options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment (default: auto-generated)')
    
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-efficient graph loading for large graphs')
    
    return parser.parse_args()


def create_summarizer(method, **kwargs):
    """
    Create a summarizer based on the specified method.
    
    Args:
        method: Name of the summarization method
        **kwargs: Additional parameters for the summarizer
        
    Returns:
        GraphSummarizer: An instance of the appropriate summarizer
    """
    if method == 'community' or method == 'louvain':
        return CommunityBasedSummarizer(**kwargs)
    elif method == 'spectral':
        return SpectralSummarizer(**kwargs)
    else:
        raise ValueError(f"Unknown summarization method: {method}")


def run_experiment(graph, methods, reduction_factors, output_dir, experiment_name):
    """
    Run experiment with multiple summarization methods.
    
    Args:
        graph: NetworkX graph to summarize
        methods: List of summarization methods to try
        reduction_factors: List of reduction factors to test
        output_dir: Directory to save results
        experiment_name: Name for the experiment
        
    Returns:
        DataFrame with evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Log configuration
    config = {
        'graph_nodes': graph.number_of_nodes(),
        'graph_edges': graph.number_of_edges(),
        'directed': isinstance(graph, nx.DiGraph),
        'methods': methods,
        'reduction_factors': reduction_factors,
        'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Track results
    all_results = []
    
    # Run experiments for each method and reduction factor
    for method in methods:
        logger.info(f"Running experiments with method: {method}")
        
        for reduction in reduction_factors:
            logger.info(f"  Reduction factor: {reduction}")
            
            try:
                # Create summarizer
                summarizer = create_summarizer(method)
                
                # Time the summarization
                start_time = time.time()
                summary_graph = summarizer.summarize(graph, reduction_factor=reduction)
                summarization_time = time.time() - start_time
                
                logger.info(f"  Summary created with {summary_graph.number_of_nodes()} nodes "
                           f"and {summary_graph.number_of_edges()} edges "
                           f"in {summarization_time:.2f} seconds")
                
                # Evaluate the summary
                evaluator = GraphEvaluator(
                    graph, 
                    summary_graph, 
                    summarizer.node_mapping, 
                    summarizer.reverse_mapping
                )
                
                # Run all evaluations
                evaluation = evaluator.evaluate_all()
                
                # Add experiment metadata
                result = evaluation.copy()
                result['method'] = method
                result['reduction_factor'] = reduction
                result['summarization_time'] = summarization_time
                result['original_nodes'] = graph.number_of_nodes()
                result['original_edges'] = graph.number_of_edges()
                result['summary_nodes'] = summary_graph.number_of_nodes()
                result['summary_edges'] = summary_graph.number_of_edges()
                
                all_results.append(result)
                
                # Print summary
                evaluator.print_summary()
                
                # Save individual result
                result_file = os.path.join(
                    experiment_dir,
                    f"{method}_{reduction}.json"
                )
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error in experiment {method}, reduction={reduction}: {str(e)}")
                logger.exception(e)
    
    # Convert to DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save as CSV
        csv_file = os.path.join(experiment_dir, "all_results.csv")
        results_df.to_csv(csv_file, index=False)
        
        logger.info(f"All results saved to {csv_file}")
        return results_df
    else:
        logger.warning("No successful experiments to report")
        return None


def visualize_results(results_df, output_dir, experiment_name):
    """
    Create visualizations of experiment results.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save visualizations
        experiment_name: Name of the experiment
    """
    logger = logging.getLogger(__name__)
    
    if results_df is None or results_df.empty:
        logger.warning("No results to visualize")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, experiment_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(font_scale=1.2)
    
    # 1. PageRank preservation by method
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        data=results_df,
        x='method',
        y='pagerank_spearman',
        hue='reduction_factor',
        palette='viridis'
    )
    
    plt.title('PageRank Preservation by Method')
    plt.xlabel('Summarization Method')
    plt.ylabel('PageRank Spearman Correlation')
    plt.ylim(0, 1)
    plt.legend(title='Reduction Factor')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pagerank_preservation.png'), dpi=300)
    plt.close()
    
    # 2. Community structure preservation
    plt.figure(figsize=(12, 6))
    
    community_metric = 'community_nmi_sampled' if 'community_nmi_sampled' in results_df.columns else 'community_nmi'
    
    if community_metric in results_df.columns:
        ax = sns.barplot(
            data=results_df,
            x='method',
            y=community_metric,
            hue='reduction_factor',
            palette='viridis'
        )
        
        plt.title('Community Structure Preservation by Method')
        plt.xlabel('Summarization Method')
        plt.ylabel('Normalized Mutual Information (NMI)')
        plt.ylim(0, 1)
        plt.legend(title='Reduction Factor')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'community_preservation.png'), dpi=300)
        plt.close()
    
    # 3. Compression vs. preservation scatter plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    g = sns.scatterplot(
        data=results_df,
        x='node_compression_ratio',
        y='pagerank_spearman',
        hue='method',
        size='edge_compression_ratio',
        sizes=(100, 500),
        alpha=0.7
    )
    
    # Add annotations
    for i, row in results_df.iterrows():
        g.text(
            row['node_compression_ratio'] + 0.01, 
            row['pagerank_spearman'] - 0.01,
            f"{row['method']}-{row['reduction_factor']:.1f}",
            fontsize=8
        )
    
    plt.title('Compression vs. PageRank Preservation')
    plt.xlabel('Node Compression Ratio (smaller is better)')
    plt.ylabel('PageRank Correlation (higher is better)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'compression_vs_preservation.png'), dpi=300)
    plt.close()
    
    # 4. Runtime improvement
    plt.figure(figsize=(12, 6))
    
    if 'pagerank_speedup' in results_df.columns:
        ax = sns.barplot(
            data=results_df,
            x='method',
            y='pagerank_speedup',
            hue='reduction_factor',
            palette='viridis'
        )
        
        plt.title('PageRank Computation Speedup')
        plt.xlabel('Summarization Method')
        plt.ylabel('Speedup Factor (higher is better)')
        plt.legend(title='Reduction Factor')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'runtime_improvement.png'), dpi=300)
        plt.close()
    
    # 5. Comparison across multiple metrics
    metrics = [
        ('pagerank_spearman', 'PageRank Correlation'),
        ('degree_correlation', 'Degree Correlation'),
        (community_metric, 'Community NMI')
    ]
    
    plt.figure(figsize=(14, 8))
    
    # Create a metrics-focused DataFrame for easier plotting
    plot_data = []
    for i, row in results_df.iterrows():
        for metric_name, metric_label in metrics:
            if metric_name in row:
                plot_data.append({
                    'Method': row['method'],
                    'Reduction': row['reduction_factor'],
                    'Metric': metric_label,
                    'Value': row[metric_name]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    if not plot_df.empty:
        ax = sns.catplot(
            data=plot_df,
            x='Method',
            y='Value',
            hue='Reduction',
            col='Metric',
            kind='bar',
            palette='viridis',
            height=5,
            aspect=0.8,
            sharey=False
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'multi_metric_comparison.png'), dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {plots_dir}")


def main():
    """Main function to run the experiment."""
    args = parse_arguments()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        dataset_name = os.path.basename(args.dataset).split('.')[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{dataset_name}_{timestamp}"
    
    # Set up logging
    log_file = os.path.join(args.output_dir, f"{args.experiment_name}.log")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(log_file)
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # List datasets and exit if requested
    if args.list_datasets:
        list_available_datasets()
        return
    
    # Handle dataset loading
    if args.dataset in SNAP_WEB_GRAPHS:
        # Download from SNAP
        logger.info(f"Using SNAP dataset: {args.dataset}")
        dataset_path = download_snap_dataset(args.dataset, args.data_dir)
        
        if args.download_only:
            logger.info("Download completed. Exiting as requested.")
            return
        
        # Load the graph
        directed = args.directed if args.directed is not None else SNAP_WEB_GRAPHS[args.dataset]['directed']
        graph = load_snap_graph(
            dataset_path, 
            directed=directed,
            memory_efficient=args.memory_efficient
        )
    else:
        # Assume it's a path to an edge list file
        if not os.path.exists(args.dataset):
            logger.error(f"File not found: {args.dataset}")
            return
        
        logger.info(f"Loading graph from file: {args.dataset}")
        graph = load_snap_graph(
            args.dataset, 
            directed=args.directed,
            memory_efficient=args.memory_efficient
        )
    
    logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Run experiments
    results_df = run_experiment(
        graph,
        args.methods,
        args.reductions,
        args.output_dir,
        args.experiment_name
    )
    
    # Create visualizations
    if results_df is not None:
        visualize_results(results_df, args.output_dir, args.experiment_name)
    
    logger.info(f"Experiment {args.experiment_name} completed")


if __name__ == "__main__":
    main()
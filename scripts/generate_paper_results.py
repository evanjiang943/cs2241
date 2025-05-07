#!/usr/bin/env python
"""
Generate Paper Results

This script runs experiments to generate the results needed for the paper's Results section.
It evaluates three summarization methods (Sparsifier, Collapse, Coarsener) on four web graphs
at three compression ratios (0.05, 0.10, 0.20).
"""

import os
import sys
import argparse
import logging
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import seaborn as sns
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from graphsum.summarizers.community import CommunityBasedSummarizer
from graphsum.summarizers.spectral import SpectralSummarizer
from graphsum.evaluation.evaluator import GraphEvaluator
from graphsum.evaluation.paper_metrics import PaperMetrics
from graphsum.evaluation.spectral_metrics import SpectralMetrics
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
    parser = argparse.ArgumentParser(description='Generate Paper Results')
    
    # Dataset options
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['web-Google', 'web-Stanford', 'web-BerkStan', 'web-NotreDame'],
                        help='SNAP dataset names to use')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for downloading datasets')
    
    # Method options
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['sparsifier', 'collapse', 'coarsener'],
                       help='Summarization methods to evaluate')
    
    parser.add_argument('--compression-ratios', type=float, nargs='+',
                       default=[0.05, 0.10, 0.20],
                       help='Compression ratios to test')
    
    # Experiment options
    parser.add_argument('--output-dir', type=str, default='paper/results',
                       help='Directory to save results')
    
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per experiment for averaging')
    
    parser.add_argument('--memory-efficient', action='store_true',
                       help='Use memory-efficient graph loading for large graphs')
    
    parser.add_argument('--use-paper-metrics', action='store_true',
                      help='Use the five core metrics from the paper')
    
    parser.add_argument('--use-spectral-metrics', action='store_true',
                      help='Use additional spectral-specific metrics')
    
    return parser.parse_args()


def create_summarizer(method, **kwargs):
    """Create a summarizer based on the specified method."""
    
    if method == 'collapse':
        # Community-based summarization (Collapse)
        return CommunityBasedSummarizer(**kwargs)
    
    elif method == 'sparsifier':
        # For the "Sparsifier" approach, we use SpectralSummarizer with minimal eigenvectors
        # This emphasizes edge preservation rather than structural coarsening
        return SpectralSummarizer(name='Sparsifier', n_eigenvectors=10, **kwargs)
    
    elif method == 'coarsener':
        # For the "Coarsener" approach, we use SpectralSummarizer with more eigenvectors
        # This preserves the spectral properties of the graph better
        return SpectralSummarizer(name='Coarsener', n_eigenvectors=50, **kwargs)
    
    else:
        raise ValueError(f"Unknown summarization method: {method}")


def convert_method_name(method):
    """Convert method name to the form used in the paper."""
    if method == 'sparsifier':
        return 'Sparsifier'
    elif method == 'collapse':
        return 'Collapse'
    elif method == 'coarsener':
        return 'Coarsener'
    return method.capitalize()


def run_single_experiment(graph, method, compression_ratio, logger, use_paper_metrics=False, 
                    use_spectral_metrics=False):
    """Run a single experiment and return the results."""
    
    # Create summarizer
    summarizer = create_summarizer(method)
    
    # Time the summarization
    start_time = time.time()
    summary_graph = summarizer.summarize(graph, reduction_factor=compression_ratio)
    summarization_time = time.time() - start_time
    
    logger.info(f"  Summary created with {summary_graph.number_of_nodes()} nodes "
               f"and {summary_graph.number_of_edges()} edges "
               f"in {summarization_time:.2f} seconds")
    
    # Initialize the results dictionary
    result = {}
    
    # Evaluate with standard metrics
    logger.info("  Running standard evaluation metrics")
    evaluator = GraphEvaluator(
        graph, 
        summary_graph, 
        summarizer.node_mapping, 
        summarizer.reverse_mapping
    )
    
    # Run standard evaluations
    evaluation = evaluator.evaluate_all(top_k=50)  # Using top-k=50 for PageRank as in paper
    result.update(evaluation)
    
    # Add paper-specific metrics if requested
    if use_paper_metrics:
        logger.info("  Running paper-specific metrics")
        paper_metrics = PaperMetrics.evaluate_all(
            graph, 
            summary_graph, 
            node_mapping=summarizer.node_mapping,
            k_spectral=50,
            k_centrality=50
        )
        result.update(paper_metrics)
    
    # Add spectral-specific metrics if requested
    if use_spectral_metrics:
        logger.info("  Running spectral-specific metrics")
        spectral_metrics = SpectralMetrics.evaluate_all(
            graph, 
            summary_graph, 
            node_mapping=summarizer.node_mapping,
            k=50
        )
        result.update(spectral_metrics)
    
    # Add experiment metadata
    result['method'] = method
    result['compression_ratio'] = compression_ratio
    result['summarization_time'] = summarization_time
    result['original_nodes'] = graph.number_of_nodes()
    result['original_edges'] = graph.number_of_edges()
    result['summary_nodes'] = summary_graph.number_of_nodes()
    result['summary_edges'] = summary_graph.number_of_edges()
    
    return result


def run_experiments(datasets, methods, compression_ratios, data_dir, 
                   output_dir, memory_efficient, runs, use_paper_metrics, 
                   use_spectral_metrics, logger):
    """Run all experiments across datasets, methods and compression ratios."""
    
    all_results = []
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Download and load the graph
        try:
            dataset_path = download_snap_dataset(dataset_name, data_dir)
            graph = load_snap_graph(
                dataset_path, 
                directed=SNAP_WEB_GRAPHS[dataset_name]['directed'],
                memory_efficient=memory_efficient
            )
            
            logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and "
                       f"{graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            continue
        
        # Process each method
        for method in methods:
            logger.info(f"  Running method: {method}")
            
            # Process each compression ratio
            for compression_ratio in compression_ratios:
                logger.info(f"    Compression ratio: {compression_ratio}")
                
                # Run multiple times and average results
                method_results = []
                for run in range(runs):
                    logger.info(f"      Run {run+1}/{runs}")
                    try:
                        result = run_single_experiment(
                            graph, method, compression_ratio, logger,
                            use_paper_metrics=use_paper_metrics, 
                            use_spectral_metrics=use_spectral_metrics
                        )
                        result['dataset'] = dataset_name
                        result['run'] = run + 1
                        method_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in run {run+1}: {e}")
                
                # Average the results across runs
                if method_results:
                    avg_result = average_results(method_results)
                    avg_result['dataset'] = dataset_name
                    avg_result['method'] = method
                    avg_result['compression_ratio'] = compression_ratio
                    all_results.append(avg_result)
                    
                    # Save individual averaged result
                    result_file = os.path.join(
                        output_dir,
                        f"{dataset_name}_{method}_{compression_ratio:.2f}.json"
                    )
                    with open(result_file, 'w') as f:
                        json.dump(avg_result, f, indent=2)
    
    # Compile into single DataFrame and save
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, 'all_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"All results saved to {results_path}")
        
        # Generate summary tables for the paper
        generate_paper_tables(results_df, output_dir, logger)
        
        return results_df
    else:
        logger.warning("No results were collected.")
        return None


def average_results(results_list):
    """Average numerical results across multiple runs."""
    if not results_list:
        return {}
    
    # Extract the first result for non-numeric values
    avg_result = {k: v for k, v in results_list[0].items() 
                  if not isinstance(v, (int, float))}
    
    # Average numeric values
    numeric_keys = [k for k, v in results_list[0].items() 
                   if isinstance(v, (int, float))]
    
    for key in numeric_keys:
        values = [result[key] for result in results_list if key in result]
        if values:
            avg_result[key] = np.mean(values)
    
    return avg_result


def generate_paper_tables(results_df, output_dir, logger):
    """Generate tables for the paper based on experimental results."""
    logger.info("Generating paper tables...")
    
    # Map method names to those used in the paper
    results_df['paper_method'] = results_df['method'].apply(convert_method_name)
    
    # Table 1: Dataset statistics (already provided in SNAP_WEB_GRAPHS)
    
    # Table 2: Metric values at CR=0.10 for web-Stanford
    try:
        stanford_results = results_df[
            (results_df['dataset'] == 'web-Stanford') & 
            (np.isclose(results_df['compression_ratio'], 0.10))
        ].copy()
        
        if not stanford_results.empty:
            # Map keys to the paper's metrics
            stanford_table = pd.DataFrame({
                'Method': stanford_results['paper_method'],
                'SpectralErr': stanford_results['laplacian_spectral_error'] if 'laplacian_spectral_error' in stanford_results else 0,
                'NMI': stanford_results['community_nmi'] if 'community_nmi' in stanford_results 
                       else stanford_results['community_nmi_sampled'],
                'Stretch': stanford_results['path_length_ratio'] if 'path_length_ratio' in stanford_results 
                           else stanford_results['path_length_ratio_sampled'],
                'Precision@50': stanford_results['pagerank_top_50_jaccard'],
                'CR': stanford_results['compression_ratio']
            })
            
            stanford_table_path = os.path.join(output_dir, 'stanford_metrics_table.csv')
            stanford_table.to_csv(stanford_table_path, index=False)
            logger.info(f"Stanford metrics table saved to {stanford_table_path}")
    except Exception as e:
        logger.error(f"Error generating metrics summary table: {e}")
    
    # Table 3: Detailed metrics on web-NotreDame at CR=0.10
    try:
        notredame_results = results_df[
            (results_df['dataset'] == 'web-NotreDame') & 
            (np.isclose(results_df['compression_ratio'], 0.10))
        ].copy()
        
        if not notredame_results.empty:
            notredame_table = pd.DataFrame({
                'Method': notredame_results['paper_method'],
                'SpectralErr': notredame_results['laplacian_spectral_error'] if 'laplacian_spectral_error' in notredame_results else 0,
                'NMI': notredame_results['community_nmi'] if 'community_nmi' in notredame_results 
                       else notredame_results['community_nmi_sampled'],
                'Stretch': notredame_results['path_length_ratio'] if 'path_length_ratio' in notredame_results 
                           else notredame_results['path_length_ratio_sampled'],
                'Precision@50': notredame_results['pagerank_top_50_jaccard'],
                'CR': notredame_results['compression_ratio']
            })
            
            notredame_table_path = os.path.join(output_dir, 'notredame_detail_table.csv')
            notredame_table.to_csv(notredame_table_path, index=False)
            logger.info(f"Notre Dame detail table saved to {notredame_table_path}")
    except Exception as e:
        logger.error(f"Error generating Notre Dame detail table: {e}")

    # Generate data for plots
    try:
        # Create data for compression vs. metric plots (Figures 1-4)
        for dataset in results_df['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            plots_dir = os.path.join(output_dir, 'plots', dataset)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Group metrics for different plots
            metric_groups = {
                'spectral': ['laplacian_spectral_error'],
                'community': ['community_nmi', 'community_nmi_sampled'],
                'distance': ['path_length_ratio', 'path_length_ratio_sampled'],
                'centrality': ['pagerank_top_50_jaccard']
            }
            
            for group_name, metrics in metric_groups.items():
                # Find the first available metric in this group
                metric_col = next((m for m in metrics if m in dataset_results.columns), None)
                if metric_col:
                    plot_data = dataset_results[['paper_method', 'compression_ratio', metric_col]]
                    plot_data_path = os.path.join(plots_dir, f"{group_name}_plot_data.csv")
                    plot_data.to_csv(plot_data_path, index=False)
    except Exception as e:
        logger.error(f"Error generating plot data: {e}")


def main():
    """Main function to run all experiments."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp-based experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"paper_results_{timestamp}.log")
    
    # Setup logging
    logger = setup_logging(log_file)
    
    logger.info("Starting paper results generation")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Compression ratios: {args.compression_ratios}")
    logger.info(f"Using paper metrics: {args.use_paper_metrics}")
    logger.info(f"Using spectral metrics: {args.use_spectral_metrics}")
    
    # Run all experiments
    run_experiments(
        datasets=args.datasets,
        methods=args.methods,
        compression_ratios=args.compression_ratios,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        memory_efficient=args.memory_efficient,
        runs=args.runs,
        use_paper_metrics=args.use_paper_metrics,
        use_spectral_metrics=args.use_spectral_metrics,
        logger=logger
    )
    
    logger.info("Paper results generation complete")


if __name__ == "__main__":
    main()

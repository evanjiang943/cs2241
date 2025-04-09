# demo_spectral.py
import networkx as nx
from graphsum.summarizers.spectral import SpectralSummarizer
from graphsum.evaluation.evaluator import GraphEvaluator

def main():
    # Create a simple Karate Club graph
    G = nx.karate_club_graph()
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Initialize the spectral summarizer
    # You can customize n_eigenvectors, or leave it as None for an automatic choice
    summarizer = SpectralSummarizer(n_eigenvectors=None)

    # Summarize the graph using a chosen reduction factor
    # reduction_factor=0.2 => summary graph ~ 20% the number of original nodes
    summary = summarizer.summarize(G, reduction_factor=0.2)

    print(f"Summary graph: {summary.number_of_nodes()} nodes, {summary.number_of_edges()} edges")

    # Evaluate how well properties are preserved
    # We retrieve the node_mapping and reverse_mapping from the summarizer
    evaluator = GraphEvaluator(G, summary, summarizer.node_mapping, summarizer.reverse_mapping)
    results = evaluator.evaluate_all()
    evaluator.print_summary()

    print("Done! Spectral-based summary created successfully.")

if __name__ == "__main__":
    main()

# quickstart.py
import networkx as nx
import community as community_louvain

# Create a simple Karate Club graph
G = nx.karate_club_graph()
print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Detect communities using Louvain method
communities = community_louvain.best_partition(G)

# Count communities
unique_communities = set(communities.values())
print(f"Detected {len(unique_communities)} communities")

# Create a summary graph with one node per community
summary = nx.Graph()
summary.add_nodes_from(unique_communities)

# Add edges between communities
for u, v in G.edges():
    u_comm = communities[u]
    v_comm = communities[v]
    
    if u_comm != v_comm:
        if summary.has_edge(u_comm, v_comm):
            # If edge already exists, increment weight
            summary[u_comm][v_comm]['weight'] = summary[u_comm][v_comm].get('weight', 0) + 1
        else:
            # Add new edge with weight 1
            summary.add_edge(u_comm, v_comm, weight=1)

print(f"Summary graph: {summary.number_of_nodes()} nodes, {summary.number_of_edges()} edges")

# Calculate PageRank for both graphs
pr_original = nx.pagerank(G)
pr_summary = nx.pagerank(summary)

print("Done! Summary graph created successfully.")
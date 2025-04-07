"""
SNAP graph dataset loader.

This module provides functions to download and load graph datasets from the
Stanford Network Analysis Project (SNAP).
"""

import os
import sys
import logging
import tempfile
import urllib.request
import gzip
import shutil
import networkx as nx
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Dictionary of available SNAP web graphs with URLs and descriptions
SNAP_WEB_GRAPHS = {
    'web-Google': {
        'url': 'https://snap.stanford.edu/data/web-Google.txt.gz',
        'description': 'Web graph from Google',
        'nodes': 875713,
        'edges': 5105039,
        'directed': True
    },
    'web-Stanford': {
        'url': 'https://snap.stanford.edu/data/web-Stanford.txt.gz',
        'description': 'Web graph of Stanford.edu',
        'nodes': 281903,
        'edges': 2312497,
        'directed': True
    },
    'web-BerkStan': {
        'url': 'https://snap.stanford.edu/data/web-BerkStan.txt.gz',
        'description': 'Web graph of Berkeley and Stanford',
        'nodes': 685230,
        'edges': 7600595,
        'directed': True
    },
    'web-NotreDame': {
        'url': 'https://snap.stanford.edu/data/web-NotreDame.txt.gz',
        'description': 'Web graph of Notre Dame',
        'nodes': 325729,
        'edges': 1497134,
        'directed': True
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block
            tsize: Total size
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    Download a file with a progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the downloaded file
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_snap_dataset(dataset_name, output_dir, force=False):
    """
    Download a SNAP dataset if not already present.
    
    Args:
        dataset_name: Name of the SNAP dataset
        output_dir: Directory to save the dataset
        force: Whether to force download even if already exists
        
    Returns:
        str: Path to the downloaded dataset
    """
    if dataset_name not in SNAP_WEB_GRAPHS:
        raise ValueError(f"Unknown SNAP dataset: {dataset_name}. "
                        f"Available datasets: {', '.join(SNAP_WEB_GRAPHS.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths for compressed and extracted files
    gz_file = os.path.join(output_dir, f"{dataset_name}.txt.gz")
    txt_file = os.path.join(output_dir, f"{dataset_name}.txt")
    
    # Download if not exists or force is True
    if not os.path.exists(txt_file) or force:
        if not os.path.exists(gz_file) or force:
            logger.info(f"Downloading {dataset_name} from SNAP...")
            download_url(SNAP_WEB_GRAPHS[dataset_name]['url'], gz_file)
        
        # Extract the gz file
        logger.info(f"Extracting {gz_file}...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(txt_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Dataset saved to {txt_file}")
    else:
        logger.info(f"Dataset already exists at {txt_file}")
    
    return txt_file


def load_snap_graph(file_path, directed=None, create_using=None, nodetype=int, 
                   comments='#', delimiter=None, memory_efficient=False):
    """
    Load a graph from a SNAP edge list file.
    
    Args:
        file_path: Path to the edge list file
        directed: Whether to create a directed graph (overrides dataset default)
        create_using: NetworkX graph constructor (overrides directed)
        nodetype: Type of node labels
        comments: Character that marks comment lines
        delimiter: Delimiter between node IDs
        memory_efficient: If True, uses a memory-efficient approach for large graphs
        
    Returns:
        networkx.Graph: The loaded graph
    """
    # Determine if the graph should be directed
    if create_using is None:
        # Extract dataset name from file path
        basename = os.path.basename(file_path)
        dataset_name = os.path.splitext(basename)[0]
        
        # Use dataset default if directed is not specified
        if directed is None and dataset_name in SNAP_WEB_GRAPHS:
            directed = SNAP_WEB_GRAPHS[dataset_name]['directed']
        elif directed is None:
            directed = False
        
        create_using = nx.DiGraph() if directed else nx.Graph()
    
    logger.info(f"Loading graph from {file_path}")
    
    if memory_efficient:
        return _load_snap_graph_memory_efficient(
            file_path, create_using, nodetype, comments, delimiter
        )
    else:
        # Use NetworkX's built-in read_edgelist
        G = nx.read_edgelist(
            file_path, 
            create_using=create_using,
            nodetype=nodetype,
            comments=comments,
            delimiter=delimiter
        )
        
        logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G


def _load_snap_graph_memory_efficient(file_path, create_using, nodetype, comments, delimiter):
    """
    Load a large graph in a memory-efficient way by streaming the edge list.
    
    Args:
        file_path: Path to the edge list file
        create_using: NetworkX graph constructor
        nodetype: Type of node labels
        comments: Character that marks comment lines
        delimiter: Delimiter between node IDs
        
    Returns:
        networkx.Graph: The loaded graph
    """
    # Create empty graph
    G = create_using
    
    # Count lines for progress bar (skip comments)
    logger.info("Counting lines in edge list file...")
    with open(file_path, 'r') as f:
        n_lines = sum(1 for line in f if not line.startswith(comments))
    
    # Add edges in batches
    logger.info("Reading edges...")
    edge_count = 0
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=n_lines, desc="Loading edges"):
            if line.startswith(comments):
                continue
                
            # Parse the line
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(delimiter)
            if len(parts) >= 2:
                u, v = parts[0], parts[1]
                try:
                    u, v = nodetype(u), nodetype(v)
                    G.add_edge(u, v)
                    edge_count += 1
                except:
                    # Skip edges with invalid node IDs
                    pass
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    return G


def list_available_datasets():
    """List all available SNAP datasets with info."""
    print("Available SNAP Web Graph Datasets:")
    print("---------------------------------")
    for name, info in SNAP_WEB_GRAPHS.items():
        print(f"{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Nodes: {info['nodes']:,}")
        print(f"  Edges: {info['edges']:,}")
        print(f"  Directed: {info['directed']}")
        print()


if __name__ == "__main__":
    # Simple CLI for downloading datasets
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python snap.py [list|download DATASET_NAME OUTPUT_DIR]")
        sys.exit(1)
    
    if sys.argv[1] == "list":
        list_available_datasets()
    elif sys.argv[1] == "download" and len(sys.argv) >= 4:
        dataset_name = sys.argv[2]
        output_dir = sys.argv[3]
        download_snap_dataset(dataset_name, output_dir)
    else:
        print("Invalid command. Use 'list' or 'download DATASET_NAME OUTPUT_DIR'")
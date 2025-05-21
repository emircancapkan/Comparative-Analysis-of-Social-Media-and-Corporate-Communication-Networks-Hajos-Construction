import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from community import community_louvain
import itertools
import os

# datasets
def load_graph_from_file(file_path):
    """
    Loading graph from file and printing basic information
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
        
    G = nx.read_edgelist(file_path)
    print(f"{file_path} loaded:")
    print(f"- {len(G.nodes())} nodes")
    print(f"- {len(G.edges())} edges")
    return G

# Files
facebook_file = "facebook_combined.txt"
email_file = "email-Eu-core.txt"

try:
    # Load real data
    fb_graph = load_graph_from_file(facebook_file)
    email_graph = load_graph_from_file(email_file)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Creating sample graphs...")
    # Create sample graphs if files not found
    fb_graph = nx.barabasi_albert_graph(100, 5, seed=42)
    email_graph = nx.watts_strogatz_graph(100, 10, 0.3, seed=42)


def create_sample_graphs():
    # Facebook-like sample graph (100 can be changed to any number. I did it with 100)
    fb_graph = nx.barabasi_albert_graph(100, 5, seed=42)
    # Email sample graph (100 can be changed to any number. I did it with 100)
    email_graph = nx.watts_strogatz_graph(100, 10, 0.3, seed=42)
    
    return fb_graph, email_graph

# Create sample graphs
fb_graph, email_graph = create_sample_graphs()

print("Graphs loaded.")
print(f"Facebook graph: {len(fb_graph.nodes())} nodes, {len(fb_graph.edges())} edges")
print(f"Email graph: {len(email_graph.nodes())} nodes, {len(email_graph.edges())} edges")


# 1. CHROMATIC NUMBER ANALYSIS
# --------------------------

def estimate_chromatic_number(G, max_iterations=100):

    # Coloring with Welch-Powell algorithm
    coloring = nx.greedy_color(G, strategy="largest_first")
    
    # Number of colors
    chromatic_number = max(coloring.values()) + 1
    
    print(f"Estimated chromatic number: {chromatic_number}")
    
    return chromatic_number, coloring

# Estimate chromatic number for both graphs
print("\n1. CHROMATIC NUMBER ANALYSIS")
print("--------------------------")
print("For Facebook graph:")
fb_chromatic, fb_coloring = estimate_chromatic_number(fb_graph)

print("\nFor Email graph:")
email_chromatic, email_coloring = estimate_chromatic_number(email_graph)


def visualize_coloring(G, coloring, title):
    plt.figure(figsize=(10, 8))
    
    # Create color map
    unique_colors = max(coloring.values()) + 1
    color_map = plt.colormaps["tab20"].resampled(unique_colors)
    
    # Determine node colors
    node_colors = [color_map(coloring[node]) for node in G.nodes()]
    
    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, node_color=node_colors, with_labels=True, 
                   node_size=300, font_size=8, width=0.5, alpha=0.7)
    
    plt.title(f"{title} - {unique_colors} Colors")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_coloring.png")
    plt.close()

# 2. SUBGRAPH ANALYSIS
# -------------------

def find_critical_subgraphs(G, chromatic_number):
    
    communities = community_louvain.best_partition(G)
    
    # Convert communities to subgraphs
    community_subgraphs = {}
    for node, community_id in communities.items():
        if community_id not in community_subgraphs:
            community_subgraphs[community_id] = []
        community_subgraphs[community_id].append(node)
    
    # Create induced subgraph for each community
    induced_subgraphs = {comm_id: G.subgraph(nodes) for comm_id, nodes in community_subgraphs.items()}
    
    critical_subgraphs = {}
    
    # Estimate chromatic number for each subgraph
    for comm_id, subgraph in induced_subgraphs.items():
        if len(subgraph) < 3:  # Skip very small subgraphs
            continue
            
        sub_chromatic, _ = estimate_chromatic_number(subgraph)
        
        # Is subgraph's chromatic number close to main graph's?
        if sub_chromatic >= chromatic_number - 1:
            critical_subgraphs[comm_id] = subgraph
    
    return critical_subgraphs, communities

print("\n2. SUBGRAPH ANALYSIS")
print("-------------------")
print("Looking for critical subgraphs in Facebook graph...")
fb_critical_subgraphs, fb_communities = find_critical_subgraphs(fb_graph, fb_chromatic)

print("Looking for critical subgraphs in Email graph...")
email_critical_subgraphs, email_communities = find_critical_subgraphs(email_graph, email_chromatic)

print(f"Found {len(fb_critical_subgraphs)} critical subgraphs in Facebook.")
print(f"Found {len(email_critical_subgraphs)} critical subgraphs in Email.")

# Visualize largest critical subgraphs
def visualize_critical_subgraphs(G, communities, critical_subgraphs, title):
    if not critical_subgraphs:
        print(f"No critical subgraphs found for {title}.")
        return
        
    # Select largest critical subgraph
    largest_id = max(critical_subgraphs.keys(), key=lambda k: len(critical_subgraphs[k]))
    critical_nodes = list(critical_subgraphs[largest_id].nodes())
    
    # Draw full graph and critical subgraph
    plt.figure(figsize=(12, 10))
    
    # Create color map
    max_community = max(communities.values())
    color_map = plt.colormaps["tab20"].resampled(max_community + 1)
    
    # Determine node colors - highlight critical subgraph nodes
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node in critical_nodes:
            node_colors.append('red')  # Critical subgraph nodes in red
            node_sizes.append(400)     # And larger
        else:
            node_colors.append(color_map(communities[node]))
            node_sizes.append(200)
    
    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=node_sizes,
                   with_labels=True, font_size=8, width=0.5, alpha=0.7)
    
    plt.title(f"{title} - Critical Subgraph Highlighted")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_critical_subgraph.png")
    plt.close()

# 3. COMMUNITY STRUCTURE AND CRITICAL GRAPHS

def analyze_community_structure(G, communities):
    """
    Analyzes community structure and examines properties in Hajós context
    """
    # Community sizes
    community_sizes = {}
    for comm_id in set(communities.values()):
        community_sizes[comm_id] = sum(1 for v in communities.values() if v == comm_id)
    
    # Internal and external edge counts for communities
    internal_edges = {}
    external_edges = {}
    
    for comm_id in set(communities.values()):
        internal_edges[comm_id] = 0
        external_edges[comm_id] = 0
    
    for u, v in G.edges():
        if communities[u] == communities[v]:
            internal_edges[communities[u]] += 1
        else:
            external_edges[communities[u]] += 1
            external_edges[communities[v]] += 1
    
    # Community density
    density = {}
    for comm_id in set(communities.values()):
        size = community_sizes[comm_id]
        if size > 1:  # Density requires at least 2 nodes
            max_edges = size * (size - 1) / 2
            density[comm_id] = internal_edges[comm_id] / max_edges if max_edges > 0 else 0
        else:
            density[comm_id] = 0
    
    return {
        'sizes': community_sizes,
        'internal_edges': internal_edges,
        'external_edges': external_edges,
        'density': density
    }

print("\n3. COMMUNITY STRUCTURE AND CRITICAL GRAPHS")
print("------------------------------------")
print("Analyzing Facebook communities...")
fb_community_analysis = analyze_community_structure(fb_graph, fb_communities)

print("Analyzing Email communities...")
email_community_analysis = analyze_community_structure(email_graph, email_communities)

# Visualize results
def visualize_community_metrics(analysis, title):
    metrics = ['sizes', 'internal_edges', 'external_edges', 'density']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        communities = list(analysis[metric].keys())
        values = list(analysis[metric].values())
        
        # Show top 10 communities
        if len(communities) > 10:
            sorted_idx = np.argsort(values)[::-1]
            communities = [communities[i] for i in sorted_idx[:10]]
            values = [values[i] for i in sorted_idx[:10]]
        
        ax.bar(communities, values)
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Community ID")
        ax.set_ylabel(metric.replace('_', ' ').title())
    
    plt.suptitle(f"{title} Community Metrics")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_community_metrics.png")
    plt.close()

# 4. HAJOS CONSTRUCTION TEST

def test_hajos_construction(G, critical_subgraphs):

    # Compare chromatic numbers of critical subgraphs
    critical_chromatic_numbers = {}
    
    for comm_id, subgraph in critical_subgraphs.items():
        chromatic, _ = estimate_chromatic_number(subgraph)
        critical_chromatic_numbers[comm_id] = chromatic
    
    # Check if critical subgraphs contain complete subgraphs (cliques)
    clique_counts = {}
    for comm_id, subgraph in critical_subgraphs.items():
        try:
            # Approximate maximum clique size using a greedy algorithm
            approx_clique = nx.approximation.max_clique(subgraph)
            clique_counts[comm_id] = len(approx_clique)
        except Exception as e:
            print(f"Warning: Clique calculation failed (comm_id: {comm_id}): {e}")
            clique_counts[comm_id] = -1
    
    # Calculate number of triangles important for Hajós structure
    triangle_counts = {}
    for comm_id, subgraph in critical_subgraphs.items():
        try:
            triangle_counts[comm_id] = sum(1 for _ in nx.triangles(subgraph).values()) // 3
        except Exception as e:
            print(f"Warning: Triangle count calculation failed (comm_id: {comm_id}): {e}")
            triangle_counts[comm_id] = -1
    
    return {
        'critical_chromatic': critical_chromatic_numbers,
        'clique_counts': clique_counts,
        'triangle_counts': triangle_counts
    }

print("\n4. HAJÓS CONSTRUCTION TEST")
print("--------------------------")
if fb_critical_subgraphs and email_critical_subgraphs:
    print("Testing Hajós construction for Facebook graph...")
    fb_hajos_test = test_hajos_construction(fb_graph, fb_critical_subgraphs)
    print(f"Maximum chromatic number in critical subgraphs: {max(fb_hajos_test['critical_chromatic'].values(), default=0)}")
    valid_cliques = [v for v in fb_hajos_test['clique_counts'].values() if v > 0]
    if valid_cliques:
        print(f"Maximum clique size in critical subgraphs: {max(valid_cliques)}")
else:
    print("Cannot perform Hajós analysis as no critical subgraphs were found.")

if email_critical_subgraphs:
    print("\nTesting Hajós construction for Email graph...")
    email_hajos_test = test_hajos_construction(email_graph, email_critical_subgraphs)
    print(f"Maximum chromatic number in critical subgraphs: {max(email_hajos_test['critical_chromatic'].values(), default=0)}")
    valid_cliques = [v for v in email_hajos_test['clique_counts'].values() if v > 0]
    if valid_cliques:
        print(f"Maximum clique size in critical subgraphs: {max(valid_cliques)}")
else:
    print("Cannot perform Hajós test for Email as no critical subgraphs were found.")

# 5. COMPARING GRAPHS FROM HAJÓS CONSTRUCTION PERSPECTIVE


def compare_graphs_hajos_perspective(fb_results, email_results):
    """
    Compares Facebook and email graphs from 
    Hajós construction properties perspective
    """
    print("\n5. COMPARISON FROM HAJÓS CONSTRUCTION PERSPECTIVE")
    print("----------------------------------------------------------------")
    
    metrics = {
        "Chromatic Number": (fb_chromatic, email_chromatic),
        "Number of Critical Subgraphs": (len(fb_critical_subgraphs), len(email_critical_subgraphs)),
        "Average Community Density": (
            sum(fb_community_analysis["density"].values()) / len(fb_community_analysis["density"]) 
                if fb_community_analysis["density"] else 0,
            sum(email_community_analysis["density"].values()) / len(email_community_analysis["density"]) 
                if email_community_analysis["density"] else 0
        )
    }
    
    print("\nComparison of Hajós Properties between Facebook and Email Graphs:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Facebook':<20} {'Email':<20}")
    print("-" * 70)
    
    for metric, (fb_val, email_val) in metrics.items():
        print(f"{metric:<30} {fb_val:<20.4f} {email_val:<20.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize values
    normalized_fb = []
    normalized_email = []
    
    for metric, (fb_val, email_val) in metrics.items():
        max_val = max(fb_val, email_val)
        if max_val > 0:
            normalized_fb.append(fb_val / max_val)
            normalized_email.append(email_val / max_val)
        else:
            normalized_fb.append(0)
            normalized_email.append(0)
    
    plt.bar(x - width/2, normalized_fb, width, label='Facebook')
    plt.bar(x + width/2, normalized_email, width, label='Email')
    
    plt.xlabel('Metrics')
    plt.ylabel('Normalized Values')
    plt.title('Hajós Properties of Facebook and Email Graphs')
    plt.xticks(x, list(metrics.keys()), rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fb_vs_email_hajos_comparison.png")
    plt.close()
    
    # Interpret results
    print("\nInterpretation from Hajós Construction Perspective:")
    if fb_chromatic > email_chromatic:
        print("- Facebook graph has a higher chromatic number, indicating a more complex community structure.")
    elif email_chromatic > fb_chromatic:
        print("- Email graph has a higher chromatic number, indicating a more complex community structure.")
    
    if len(fb_critical_subgraphs) > len(email_critical_subgraphs):
        print("- Facebook graph has more critical subgraphs, showing a richer structure in terms of Hajós construction.")
    elif len(email_critical_subgraphs) > len(fb_critical_subgraphs):
        print("- Email graph has more critical subgraphs, showing a richer structure in terms of Hajós construction.")
    
    avg_fb_density = metrics["Average Community Density"][0]
    avg_email_density = metrics["Average Community Density"][1]
    
    if avg_fb_density > avg_email_density:
        print("- Facebook communities are denser on average, indicating stronger clustering tendency.")
    elif avg_email_density > avg_fb_density:
        print("- Email communities are denser on average, indicating stronger clustering tendency.")

# Make comparison
compare_graphs_hajos_perspective(fb_critical_subgraphs, email_critical_subgraphs)

print("\nAnalysis completed! This script analyzed your social network graphs from Hajós construction perspective.")
print("The results show the structural properties of both graphs and how they can be evaluated from Hajós construction perspective.")


if __name__ == "__main__":
    # 1. Chromatic number analysis
    print("\n1. CHROMATIC NUMBER ANALYSIS")
    print("--------------------------")
    print("For Facebook graph:")
    fb_chromatic, fb_coloring = estimate_chromatic_number(fb_graph)
    visualize_coloring(fb_graph, fb_coloring, "Facebook Graph Coloring")

    print("\nFor Email graph:")
    email_chromatic, email_coloring = estimate_chromatic_number(email_graph)
    visualize_coloring(email_graph, email_coloring, "Email Graph Coloring")

    # 2. Subgraph analysis
    print("\n2. SUBGRAPH ANALYSIS")
    print("-------------------")
    fb_critical_subgraphs, fb_communities = find_critical_subgraphs(fb_graph, fb_chromatic)
    email_critical_subgraphs, email_communities = find_critical_subgraphs(email_graph, email_chromatic)

    visualize_critical_subgraphs(fb_graph, fb_communities, fb_critical_subgraphs, "Facebook Graph")
    visualize_critical_subgraphs(email_graph, email_communities, email_critical_subgraphs, "Email Graph")

    # 3. Community structure analysis
    print("\n3. COMMUNITY STRUCTURE ANALYSIS")
    print("---------------------------")
    fb_community_analysis = analyze_community_structure(fb_graph, fb_communities)
    email_community_analysis = analyze_community_structure(email_graph, email_communities)

    visualize_community_metrics(fb_community_analysis, "Facebook")
    visualize_community_metrics(email_community_analysis, "Email")

    # 4. Hajós construction test
    print("\n4. HAJÓS CONSTRUCTION TEST")
    print("--------------------------")
    if fb_critical_subgraphs and email_critical_subgraphs:
        fb_hajos_test = test_hajos_construction(fb_graph, fb_critical_subgraphs)
        email_hajos_test = test_hajos_construction(email_graph, email_critical_subgraphs)
        
        # Make comparison
        compare_graphs_hajos_perspective(fb_critical_subgraphs, email_critical_subgraphs)
    else:
        print("Cannot perform Hajós analysis as no critical subgraphs were found.")
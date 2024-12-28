import os  # For file system operations
import pandas as pd  # For data processing and manipulation
import networkx as nx  # For creating and analyzing graph structures
import plotly.graph_objects as go  # Plotly: For creating interactive visualizations; plotly.graph_objects: For creating low-level plotly graphic objects
from plotly.subplots import make_subplots  # For creating subplots
import plotly.express as px  # For quickly creating statistical charts


def read_parquet_files(directory):
    """
    Read all Parquet files in the specified directory and merge them.
    Functionality: Reads all Parquet files in the specified directory and merges them into a single DataFrame.
    Implementation: Uses os.listdir to traverse the directory, pd.read_parquet to read each file, then uses pd.concat to merge.
    """
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            file_path = os.path.join(directory, filename)
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def clean_dataframe(df):
    """
    Clean the DataFrame by removing invalid rows.
    Functionality: Cleans the DataFrame by removing rows with missing values in 'source' and 'target' columns.
    Implementation: Drops rows where 'source' or 'target' is NaN, converts these columns to string type.
    """
    df = df.dropna(subset=["source", "target"])
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    return df


def create_knowledge_graph(df):
    """
    Create a knowledge graph from the DataFrame.
    Functionality: Creates a directed graph from the DataFrame.
    Implementation: Uses networkx to create a directed graph, iterates over each row in the DataFrame to add edges and attributes.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source = row["source"]
        target = row["target"]
        attributes = {k: v for k, v in row.items() if k not in ["source", "target"]}
        G.add_edge(source, target, **attributes)
    return G


def create_node_link_trace(G, pos):
    """
    Create 3D traces for nodes and edges.
    Implementation: Uses networkx layout information to create Plotly Scatter3d objects for nodes and edges.
    """
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )
    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"Node: {node}<br># of connections: {len(adjacencies)}")
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    return edge_trace, node_trace


def create_edge_label_trace(G, pos, edge_labels):
    """
    Create 3D traces for edge labels.
    Implementation: Calculates the midpoint of each edge and creates a Scatter3d object to display the label.
    """
    return go.Scatter3d(
        x=[
            pos[edge[0]][0] + (pos[edge[1]][0] - pos[edge[0]][0]) / 2
            for edge in edge_labels
        ],
        y=[
            pos[edge[0]][1] + (pos[edge[1]][1] - pos[edge[0]][1]) / 2
            for edge in edge_labels
        ],
        z=[
            pos[edge[0]][2] + (pos[edge[1]][2] - pos[edge[0]][2]) / 2
            for edge in edge_labels
        ],
        mode="text",
        text=list(edge_labels.values()),
        textposition="middle center",
        hoverinfo="none",
    )


def create_degree_distribution(G):
    """
    Create a histogram of node degree distribution.
    Implementation: Uses plotly.express to create a histogram.
    """
    degrees = [d for n, d in G.degree()]
    fig = px.histogram(x=degrees, nbins=20, labels={"x": "Degree", "y": "Count"})
    fig.update_layout(
        title_text="Node Degree Distribution",
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
    )
    return fig


def create_centrality_plot(G):
    """
    Create a box plot of node centrality distribution.
    Implementation: Calculates degree centrality and uses plotly.express to create a box plot.
    """
    centrality = nx.degree_centrality(G)
    centrality_values = list(centrality.values())
    fig = px.box(y=centrality_values, labels={"y": "Centrality"})
    fig.update_layout(
        title_text="Degree Centrality Distribution",
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
    )
    return fig


def visualize_graph_plotly(G):
    """
    Create a comprehensive and optimized layout for an advanced interactive knowledge graph visualization using Plotly.
    Implementation:
        Creates 3D layout
        Generates node and edge traces
        Creates subplots including 3D graph, degree distribution plot, and centrality distribution plot
        Adds interactive buttons and sliders
        Optimizes overall layout
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty. Nothing to visualize.")
        return
    pos = nx.spring_layout(G, dim=3)  # 3D layout
    edge_trace, node_trace = create_node_link_trace(G, pos)
    edge_labels = nx.get_edge_attributes(G, "relation")
    edge_label_trace = create_edge_label_trace(G, pos, edge_labels)
    degree_dist_fig = create_degree_distribution(G)
    centrality_fig = create_centrality_plot(G)
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.7, 0.3],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D Knowledge Graph Code by AI超元域频道",
            "Node Degree Distribution",
            "Degree Centrality Distribution",
        ),
    )
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(edge_label_trace, row=1, col=1)
    fig.add_trace(degree_dist_fig.data[0], row=1, col=2)
    fig.add_trace(centrality_fig.data[0], row=2, col=2)
    # Update 3D layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            aspectmode="cube",
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    )
    # Add buttons for different layouts
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[{"visible": [True, True, True, True, True]}],
                            label="Show All",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": [True, True, False, True, True]}],
                            label="Hide Edge Labels",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": [False, True, False, True, True]}],
                            label="Nodes Only",
                            method="update",
                        ),
                    ]
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )
    # Add slider for node size
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Node Size: "},
                pad={"t": 50},
                steps=[
                    dict(
                        method="update",
                        args=[{"marker.size": [i] * len(G.nodes)}],
                        label=str(i),
                    )
                    for i in range(5, 21, 5)
                ],
            )
        ]
    )
    # Optimize overall layout
    # fig.update_layout(
    #     height=1198,  # Increase overall height
    #     width=2055,  # Increase overall width
    #     title_text="Advanced Interactive Knowledge Graph",
    #     margin=dict(l=10, r=10, t=25, b=10),
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    # )
    fig.show()


def main():
    """
    Main function to coordinate the execution flow of the program.
    Implementation:
        Reads Parquet files
        Cleans data
        Creates knowledge graph
        Prints graph statistics
        Calls visualization function
    """
    directory = "/Users/charlesqin/PycharmProjects/RAGCode/inputs/artifacts"  # Replace with actual directory path
    df = read_parquet_files(directory)
    if df.empty:
        print("No data found in the specified directory.")
        return
    print("Original DataFrame shape:", df.shape)
    print("Original DataFrame columns:", df.columns.tolist())
    print("Original DataFrame head:")
    print(df.head())
    df = clean_dataframe(df)
    print("\nCleaned DataFrame shape:", df.shape)
    print("Cleaned DataFrame head:")
    print(df.head())
    if df.empty:
        print("No valid data remaining after cleaning.")
        return
    G = create_knowledge_graph(df)
    print(f"\nGraph statistics:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        print(
            f"Connected components: {nx.number_connected_components(G.to_undirected())}"
        )
        visualize_graph_plotly(G)
    else:
        print("Graph is empty. Cannot visualize.")


if __name__ == "__main__":
    main()

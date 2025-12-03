from matplotlib import lines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapefile
from sklearn.manifold import TSNE
import torch
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import to_networkx

## This function is to visualize graphs
def display_graph(graphs, num_graphs=1, savename=None):
    print(f"...Plotting {num_graphs} graphs...")
    
    for i in range(num_graphs):
        rand_id = np.random.choice(list(graphs.keys()))
        print(f"The randomly selected TRGCODE is {rand_id}")

        G = to_networkx(graphs[rand_id], to_undirected=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        nx.draw(G, ax=ax, with_labels=True, node_size=300, node_color='skyblue')

        if savename is not None:
            plt.savefig(savename, format='pdf')

        plt.show()

## This function is to plot the loss curves
def plot_loss_curves(losses, train_accuracies, validate_accuracies, savename=None):
    print("...Plotting loss curves...")
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(20, 8))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-', label='Loss')
    plt.title('Loss over Epochs', fontsize=21, fontweight='bold')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g--', label='Training Accuracy')
    plt.plot(epochs, validate_accuracies, 'r--', label='Validation Accuracy')
    plt.legend(fontsize=15)
    plt.ylim(0.5, 1)
    plt.title('Accuracy over Epochs', fontsize=21, fontweight='bold')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.grid(True)

    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename, format='pdf')

    plt.show()

    return

## This function is to plot the t-SNE curve
def plot_tsne(model, graphs, savename=None):
    print("...Plotting t-SNE...")
    
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for graph in graphs.values():
            x = graph.x.float()
            edge_index = graph.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long)
            out = model.conv1(x, edge_index)
            out = out.relu()
            out = model.conv2(out, edge_index)
            graph_embed = global_max_pool(out, batch)
            
            embeddings.append(graph_embed.cpu().numpy())
            labels.append(graph.y.item())

    # Reduce to 2D
    embeddings_2d = TSNE(n_components=2).fit_transform(np.vstack(embeddings))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.title("t-SNE of Graph Embeddings", fontsize=18, fontweight='bold')
    plt.colorbar(label="Class (0=Erratic, 1=Consistent)")

    if savename is not None:
        plt.savefig(savename, format='pdf')

    plt.show()

    return

## This function is to visualize the stable and unstable regions in a geographical map
def display_geo_map(dataframe, y_pred, shapefile_path, ids, transform=False, figsize=(10, 10), savename=None):
    print("...Plotting geographical map...")
    # Initialize shapefile reader
    sf = shapefile.Reader(shapefile_path)

    if (transform == True):
        # UTM Zone 43N → WGS84 lat/lon
        transformer = lambda x, y: Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform(x, y)
    else:
        ## Identity Transformation
        transformer = lambda x, y: (x, y)

    fig, ax = plt.subplots(figsize=figsize)

    for shape in sf.shapes():
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        
        for i in range(len(parts) - 1):
            part_points = points[parts[i]:parts[i + 1]]
            x_proj, y_proj = zip(*part_points)
            lon, lat = transformer(x_proj, y_proj)
            ax.plot(lon, lat, color='black')

    for id in ids:

        if y_pred[id] == 1:
            color = 'red'
            alpha = 0.7
        else:
            color = 'green'
            alpha = 0.5

        plt.scatter(
            dataframe.loc[dataframe['LOC_ID'] == id, 'LONGITUDE'],
            dataframe.loc[dataframe['LOC_ID'] == id, 'LATITUDE'],
            color=color,
            alpha=alpha
        )

    legend_elements = [
        lines.Line2D([0], [0], marker='o', color='w', label='Consistent', markerfacecolor='green', markersize=8, alpha=0.7),
        lines.Line2D([0], [0], marker='o', color='w', label='Erratic', markerfacecolor='red', markersize=8, alpha=0.5)
    ]

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Geographical Map Visualization")
    plt.legend(handles=legend_elements)
    plt.grid(True)

    if savename is not None:
        plt.savefig(savename, format='pdf')

    plt.show()

## This function is to vectorize the plotting process
def display_geo_map_optimized(dataframe, y_pred, shapefile_path, ids, transform=False, figsize=(10, 10), savename=None):
    print("...Plotting geographical map...")
    # Initialize shapefile reader
    sf = shapefile.Reader(shapefile_path)

    if transform:
        transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform
    else:
        transformer = lambda x, y: (x, y)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the shapefile borders once
    for shape in sf.shapes():
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        
        for i in range(len(parts) - 1):
            part_points = points[parts[i]:parts[i + 1]]
            x_proj, y_proj = zip(*part_points)
            lon, lat = transformer(x_proj, y_proj)
            ax.plot(lon, lat, color='black', linewidth=0.5)

    # Vectorized plotting: Filter the dataframe and prepare colors/alphas in one go
    plot_df = dataframe.loc[dataframe['LOC_ID'].isin(ids)].copy()
    plot_df['prediction'] = plot_df['LOC_ID'].map(y_pred)
    
    # Map predictions to colors and alpha values
    plot_df['color'] = plot_df['prediction'].apply(lambda p: 'red' if p == 1 else 'green')
    plot_df['alpha'] = plot_df['prediction'].apply(lambda p: 0.7 if p == 1 else 0.5)
    
    # Perform a single scatter plot call for all points
    plt.scatter(
        plot_df['LONGITUDE'],
        plot_df['LATITUDE'],
        c=plot_df['color'],
        alpha=plot_df['alpha']
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Geographical Map Visualization", fontsize=21, fontweight='bold')

    # Create a legend with appropriate markers
    legend_elements = [
        lines.Line2D([0], [0], marker='o', color='w', label='Erratic', markerfacecolor='green', markersize=8, alpha=0.7),
        lines.Line2D([0], [0], marker='o', color='w', label='Consistent', markerfacecolor='red', markersize=8, alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.grid(True)

    if savename is not None:
        plt.savefig(savename, format='pdf')

    plt.show()

## This function is to plot the gridpoints in a state
def plot_state_grid(dataframe, figsize=(5, 5), marker_size=0.05, marker_color='black', savename=None):
    ## This cell is to verify graphically that all grid points got imported
    transformer = lambda x, y: (x, y)

    fig, ax = plt.subplots(figsize=figsize)

    sf = shapefile.Reader('/home/prad/code/precipitation_gauge_gnn/shapefiles/india/india_updated_state_boundary.shp')

    for shape in sf.shapes():
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        
        for i in range(len(parts) - 1):
            part_points = points[parts[i]:parts[i + 1]]
            x_proj, y_proj = zip(*part_points)
            lon, lat = transformer(x_proj, y_proj)
            ax.plot(lon, lat, color='black')

    # Create a scatter plot
    plt.scatter(dataframe['LONGITUDE'], dataframe['LATITUDE'], s=marker_size, color=marker_color)

    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Location Plot')

    if savename is not None:
        plt.savefig(savename, format='pdf')

    plt.show()

## This function is to plot the gridpoints in a state, binned into 10 groups
def plot_binned_grid(dataframe, figsize=(8, 6), marker_size=0.5, colormap='inferno', savename=None):
    """
    Plots grid points on a map of India from a local shapefile, color-coded into 10 bins
    using a colorbar for the legend.

    Args:
        dataframe (pd.DataFrame): A DataFrame with 'LONGITUDE' and 'LATITUDE' columns,
                                  pre-sorted in descending order of a metric (e.g., variance).
        figsize (tuple): The size of the figure for the plot.
        marker_size (float): The size of the scatter plot markers.
        colormap (string): The colormap to use to indicate the bins
    """
    fig, ax = plt.subplots(figsize=figsize)

    # --- 1. Plot the Map of India (using your local shapefile) ---
    try:
        sf = shapefile.Reader('/home/prad/code/precipitation_gauge_gnn/shapefiles/india/india_updated_state_boundary.shp')
        for shape in sf.shapes():
            points = shape.points
            parts = list(shape.parts) + [len(points)]
            
            for i in range(len(parts) - 1):
                part_points = points[parts[i]:parts[i + 1]]
                lon, lat = zip(*part_points)
                ax.plot(lon, lat, color='gray', linewidth=0.75)
    except shapefile.ShapefileException as e:
        print(f"Error reading shapefile: {e}")
        ax.set_xlim(68, 98)
        ax.set_ylim(6, 38)

    # --- 2. Assign Bin IDs to the DataFrame ---
    # Create a new column to store which of the 10 bins each row belongs to.
    dataframe['bin_id'] = 0
    bins = np.array_split(dataframe, 10)
    for i, bin_df in enumerate(bins):
        # REVERSED ASSIGNMENT: Highest variance (i=0) gets the highest ID (9).
        # This makes the colorbar intuitive.
        dataframe.loc[bin_df.index, 'bin_id'] = 9 - i

    # --- 3. Plot All Points at Once Using a Colormap ---
    # Use the 'inferno' colormap with 10 discrete colors.
    cmap = plt.get_cmap(colormap, 10)
    
    # The 'c' argument maps the bin_id to a color in the colormap.
    scatter = ax.scatter(dataframe['LONGITUDE'], dataframe['LATITUDE'], s=marker_size,
                         c=dataframe['bin_id'], cmap=cmap, alpha=0.8)

    # --- 4. Create and Customize the Colorbar ---
    N = 10  # Your number of bins

    # Define the 11 boundaries for the 10 discrete color blocks
    # (e.g., -0.5 to 0.5, 0.5 to 1.5, ..., 8.5 to 9.5)
    boundaries = np.linspace(-0.5, N - 0.5, N + 1)
    
    # Define the 10 ticks to be at the center of each block (0, 1, 2, ..., 9)
    ticks = np.arange(N)

    # Create the colorbar using the 'boundaries' and 'ticks' arguments
    # This forces the labels to be centered in the blocks.
    cbar = fig.colorbar(scatter, ax=ax, boundaries=boundaries, ticks=ticks)
    
    # --- Set custom labels (Your original logic here is correct) ---
    tick_labels = [f"Top {i*10}-{ (i+1)*10 }%" for i in range(10)]
    tick_labels[0] = "Top 10% (Highest)"
    tick_labels[-1] = "Bottom 10% (Lowest)"
    
    # Reverse the labels to match your reversed bin IDs (bin_id 0 is "Bottom", bin_id 9 is "Top")
    cbar.set_ticklabels(tick_labels[::-1])
    cbar.set_label('Variance Percentile', rotation=270, labelpad=15, fontsize=12)

    # --- 5. Finalize the Plot (Beautification) ---
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title("Geographical Distribution of Rainfall Variance", fontsize=16, fontweight='bold', pad=20, loc='center')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()

    if savename is not None:
        plt.savefig(savename, format='pdf')
    plt.show()
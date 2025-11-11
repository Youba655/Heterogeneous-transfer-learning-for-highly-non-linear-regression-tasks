import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##################################################################################################################################################################
def plot_correlation_matrix(corr_matrix, title="Correlation Matrix", ax=None):
    """
    Plot correlation matrix as a bubble chart with color-coded correlations.
    
    Creates a visualization where:
    - Bubble size represents the absolute correlation strength (larger = stronger correlation)
    - Bubble color represents correlation sign and magnitude (green = positive, red = negative)
    - Uses Red-Yellow-Green colormap for intuitive interpretation
    
    Args:
        corr_matrix (DataFrame): Correlation matrix to visualize (typically from df.corr()).
        title (str): Plot title. Default is "Correlation Matrix".
        ax (matplotlib.axes, optional): Axis object to plot on. If None, creates new figure.
    
    Returns:
        None (displays plot)
    
    Note:
        - Correlation scale is fixed from -1 to 1
        - Bubble sizes scaled by factor of 500 for visibility
        - Diagonal (self-correlation = 1.0) will have largest bubbles
    """
    # Create new figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    num_vars = len(corr_matrix.columns)
    corr_values = corr_matrix.values
    
    # Use absolute values for bubble size scaling
    # Ensures both strong positive and negative correlations have large bubbles
    abs_corr = np.abs(corr_values)

    # Create scatter plot for each correlation value
    # Each (i,j) position gets a bubble sized by |correlation| and colored by sign
    for i in range(num_vars):
        for j in range(num_vars):
            ax.scatter(j, i, 
                       s=abs_corr[i, j] * 500,  # Bubble size: larger = stronger correlation
                       c=corr_values[i, j],      # Color: actual correlation value (with sign)
                       cmap="RdYlGn",            # Red (negative) -> Yellow (zero) -> Green (positive)
                       edgecolors="black",       # Black outline for visibility
                       alpha=0.8,                # Slight transparency
                       vmin=-1, vmax=1)          # Fix color scale: -1 (red) to +1 (green)

    # Set axis ticks and labels
    ax.set_xticks(range(num_vars))
    ax.set_yticks(range(num_vars))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)  # Rotate x-labels for readability
    ax.set_yticklabels(corr_matrix.index)

    # Add colorbar to show correlation scale
    norm = plt.Normalize(vmin=-1, vmax=1)  # Normalize from -1 to 1
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])  # Mark key correlation values
    cbar.set_label("Correlation Coefficient", fontsize=12)

    # Set plot title
    ax.set_title(title, fontsize=14)

##################################################################################################################################################################
def compute_overlap_matrix(files_dict):
    """
    Compute overlap matrix showing number of common samples between datasets.
    
    Creates a symmetric matrix where each cell (i,j) contains the count of samples
    that appear in both dataset i and dataset j. This is useful for analyzing
    dataset diversity and overlap in transfer learning scenarios.
    
    Args:
        files_dict (dict): Dictionary mapping dataset names to their file paths.
                          Example: {'dataset1': 'path/to/data1.csv', 
                                   'dataset2': 'path/to/data2.csv'}
    
    Returns:
        DataFrame: Overlap matrix with dataset names as both index and columns.
                  - Diagonal elements = total samples in each dataset (self-overlap)
                  - Off-diagonal elements = common samples between two datasets
                  - Matrix is symmetric: overlap(i,j) = overlap(j,i)
    """
    # Initialize empty overlap matrix with dataset names as index/columns
    overlap_matrix = pd.DataFrame(index=files_dict.keys(), columns=files_dict.keys(), dtype=int)

    # Compare each pair of datasets (nested loop for all combinations)
    for file1 in files_dict:
        # Load first dataset
        df1 = pd.read_csv(files_dict[file1])
        
        for file2 in files_dict:
            # Load second dataset
            df2 = pd.read_csv(files_dict[file2])
            
            # Find common samples using inner join
            # Inner join keeps only rows that exist in BOTH datasets
            # Matches on all common columns (operating conditions + feed properties)
            common_samples = pd.merge(df1, df2, how='inner')
            
            # Store count of common samples in overlap matrix
            # If file1 == file2, this gives the total number of samples in that dataset
            overlap_matrix.loc[file1, file2] = common_samples.shape[0]
            
    return overlap_matrix



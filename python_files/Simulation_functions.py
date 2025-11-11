import pandas as pd
import os, sys
import numpy as np
import itertools
import seaborn as sns

# Get the current working directory (where the notebook is)
current_dir = os.getcwd()

# Add the parent directory to sys.path (where hck_tools and python_files are)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from python_files.ks_optimized import * # An optimized version of the Kennard-Stone algorithm
from python_files.utils import *
##################################################################################################################################################################
def create_frame(matrix, X_train, nb_feed, columns_names):
    '''
    Create DataFrame by combining operating conditions matrix with feed properties.
    Repeats each feed for all operating conditions and labels them by feed number.
    
    Args:
        matrix (array): Operating conditions matrix.
        X_train (array): Feed properties data.
        nb_feed (int): Number of feeds.
        columns_names (list): List of column names for the resulting DataFrame, including both operating conditions 
                             (e.g., Temperature, LHSV) and feed properties.
    
    Returns:
        DataFrame: Combined dataframe with feed names and all features.
    '''
    repeated_X_train = np.repeat(X_train, len(matrix), axis=0)
    combination_matrix = np.concatenate((np.tile(matrix, (len(X_train), 1)), repeated_X_train), axis=1)
    
    feed_names = []
    for i in range(nb_feed):
        feed_names = feed_names + ["Feed_"+str(i)]*len(matrix)
        
    datdon = pd.DataFrame(combination_matrix, columns=columns_names)
    datdon.insert(0, "Feed_name", feed_names)
    return datdon
##################################################################################################################################################################
def create_frame_final(matrix, nb_pt, nb_pt_feed, columns_names):
    '''
    Create final DataFrame with proper feed naming for selected points.
    Handles cases where total points don't divide evenly by points per feed.
    
    Args:
        matrix (array): Data matrix to be converted to DataFrame.
        nb_pt (int): Total number of points.
        nb_pt_feed (int): Number of points per feed.
        columns_names (list): List of column names for the resulting DataFrame.
    
    Returns:
        DataFrame: Dataframe with feed names and all features.
    '''
    nb_feed = nb_pt//nb_pt_feed + min(nb_pt%nb_pt_feed, 1)
    feed_names = []
    for i in range(nb_feed):
        if nb_pt%nb_pt_feed != 0:
            if i == nb_feed-1:
                m = nb_pt%nb_pt_feed
            else:
                m = nb_pt_feed
        else:
            m = nb_pt_feed
        feed_names = feed_names + ["Feed_"+str(i)]*m
        
    datdon = pd.DataFrame(matrix, columns=columns_names)
    datdon.insert(0, "Feed_name", feed_names)
    return datdon

##################################################################################################################################################################
def Simulation_multiple_ks_opt(n_data, params, nb_pt, nb_pt_feed, delta_var, noise_per, nte=False):
    '''
    Generate multiple simulated data sets using Kennard-Stone sampling.
    
    Creates diverse experimental designs by:
    1. Generating all possible operating conditions and feed combinations
    2. Ranking feeds by diversity using Kennard-Stone algorithm
    3. Selecting representative points for each dataset
    4. Simulating HDN reaction and adding measurement noise
    
    Parameters:
    -----------
    n_data : int
        Number of independent data sets to generate
    params : DataFrame
        Kinetic parameters for HDN model
    nb_pt : int
        Total number of points per dataset
    nb_pt_feed : int
        Number of points per feed
    delta_var : list
        Grid spacing for each variable [T, LHSV, ppH2, Res0, N0, S0, TMP, Tire]
    noise_per : float
        Percentage of measurement noise to add (e.g., 0.05 for ±5% noise)
    nte : bool
        If True, includes tire content for NTE feeds (model 70)
        If False, fossil feeds only (model 24)
    
    Returns:
    --------
    data_final_list : list of DataFrames
        List of n_data simulated datasets with realistic measurement noise
    '''
    # Calculate number of feeds needed based on total points and points per feed
    nb_feed = nb_pt//nb_pt_feed + min(nb_pt%nb_pt_feed, 1)
    
    # Generate operating conditions grid (Temperature, H2 pressure, LHSV)
    T = np.arange(360, 410, delta_var[0], dtype=np.float32)  
    LHSV = np.arange(0.5, 4.25, delta_var[1], dtype=np.float32)  
    ppH2 = np.arange(90, 150, delta_var[2], dtype=np.float32)  
    
    # Generate feed properties grid
    Res0 = np.arange(5, 15.5, delta_var[3], dtype=np.float32)
    N0 = np.arange(500, 3100, delta_var[4], dtype=np.float32)
    S0 = np.arange(0.5, 4, delta_var[5], dtype=np.float32)
    TMP = np.arange(450, 550, delta_var[6], dtype=np.float32)
    Tire = np.arange(0.5, 30.5, delta_var[7], dtype=np.float32)
    
    # Create all combinations of operating conditions
    combinaison1 = list(itertools.product(T, ppH2, LHSV))
    matrix1 = np.array(combinaison1, dtype=np.float32)

    # Generate feed combinations based on fossil vs NTE
    if nte == True:
        matrix2 = np.array(list(itertools.product(Res0, N0, S0, Tire, TMP)), dtype=np.float32)
        columns_names = ["T", "ppH2", "LHSV", "Res0", "N0", "S0", "TMP", "Tire"]
    else:
        matrix2 = np.array(list(itertools.product(Res0, N0, S0, TMP)), dtype=np.float32)
        columns_names = ["T", "ppH2", "LHSV", "Res0", "N0", "S0", "TMP"]

    # Rank feeds by diversity using optimized Kennard-Stone algorithm
    # Uses memory-efficient version since feed space can be very large
    index = ks_sampling_mem(matrix2, n_result=len(matrix2))[0]
    feed_ranked = matrix2[index]
    
    # Select different feeds for each dataset to ensure diversity across datasets
    matrix1_list = []
    X_train2_list = []
    for i in range(n_data):
        # Take top nb_feed most diverse feeds for this dataset
        X_train2_iter = feed_ranked[:nb_feed]
        X_train2_list.append(X_train2_iter)
        # Remove first feed and re-rank remaining feeds for next dataset
        feed_ranked = feed_ranked[1:]
        index = ks_sampling_mem(feed_ranked, n_result=len(feed_ranked))[0]
        feed_ranked = feed_ranked[index]
    
    data_final_list = []
    for i in range(n_data):
        # Create full experimental design combining operating conditions and feeds
        datdon = create_frame(matrix1, X_train2_list[i], nb_feed, columns_names)
        
        # Simulate HDN reaction to get nitrogen outputs
        N_output = list(output_data(datdon, params, nte, N=True))
        
        # Check feasibility: ensure enough points in valid range [5, 500] ppm
        ppm = 0  # Counter for feeds with insufficient valid points
        ppm_comb = 0  # Feed combination counter
        for start in range(0, len(N_output), len(matrix1)):
            end = start + len(combinaison1)
            # Count valid nitrogen values (5-500 ppm) for this feed
            if sum(1 for value in N_output[start:end] if 5 <= value <= 500) <= nb_pt_feed:
                print(f"The combination {ppm_comb} of the dataset {i} has number of points (with output nitrogen between [5, 500] ppm) less than {nb_pt_feed}. \
                      Change delta variation.")
                ppm += 1
            ppm_comb += 1

        # Proceed only if all feeds have sufficient valid points
        if ppm == 0:
            pt = create_list_matrix(N_output, matrix1, X_train2_list[i])
        else:
            print("Change delta variations or parameters")
            continue

        pt_ks = []

        # Final Kennard-Stone selection for each feed
        # Uses classical Kennard-Stone (calculating distance matrix at once) since
        # the sampling space is now much smaller after filtering valid points
        if nb_pt % nb_pt_feed != 0:
            # Handle case where points don't divide evenly among feeds
            for j in range(nb_feed):
                if j != nb_feed - 1:
                    # Regular feeds get nb_pt_feed points
                    X_train, X_test = ks.train_test_split(pt[j], test_size=pt[j].shape[0] - nb_pt_feed)
                else:
                    # Last feed gets remaining points
                    X_train, X_test = ks.train_test_split(pt[j], test_size=pt[j].shape[0] - (nb_pt % nb_pt_feed))
                pt_ks.append(X_train)
        else:
            # Equal distribution: all feeds get nb_pt_feed points
            for j in range(nb_feed):
                X_train, X_test = ks.train_test_split(pt[j], test_size=pt[j].shape[0] - nb_pt_feed)
                pt_ks.append(X_train)

        # Combine all selected points from all feeds
        pt_ks = np.concatenate(pt_ks, axis=0)
        pt_ks = create_frame_final(pt_ks, nb_pt, nb_pt_feed, columns_names)
        
        # Final simulation with realistic measurement noise
        data_final = output_data(pt_ks, params, nte)
        data_final["N_simul"] = add_subtract_random(data_final["N_simul"], noise_per, seed=0)
        data_final_list.append(data_final)

    return data_final_list



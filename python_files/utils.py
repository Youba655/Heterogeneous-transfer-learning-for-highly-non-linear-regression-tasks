import pandas as pd
import numpy as np
import kennard_stone as ks
import torch
import torch.nn as nn

######################################################################## The ODE solver ############################################################################
from python_files.solver import solver  
# A Python ODE solver that accepts DataFrame format for inputs and parameters.
# Operating conditions: T (Temperature in K), ppH2 (H2 partial pressure in bar), 
#                       LHSV (Liquid Hourly Space Velocity in h^-1)
# Feedstock properties: TMP (weighted average simulated distillation in K), 
#                       N_0 (nitrogen in ppm), S_0 (sulfur in mass %), 
#                       Res_0 (resins in mass %)
# Target case only: Tire (tire content for NTE feeds)
# The DataFrame format enables computational efficiency in Bayesian inference (MHwG) since its more rapid than numpy array.
####################################################################################################################################################################
def params_transform(params, nte, bounds):
    '''
    Transform parameter array into DataFrame format required by the ODE solver.
    
    Note: This function is specifically designed for our proprietary ODE solver's 
    DataFrame format requirements. The solver itself does not require the bounds
    (param_min, param_max), but they are included here for:
    1. Documentation and illustration purposes
    2. Later use in Bayesian methods (MCMC) to constrain parameter sampling
    
    If using a different solver, you may need to adapt this function or create 
    your own parameter formatting approach.
    
    The parameter order is defined as follows:
    - Common parameters (both source and target): k0, Ea, m, n, alpha, b, A0, C0, u, l, v
    - Target-only parameter: p (tire effect parameter for NTE feeds)
    
    Args:
        params (array-like): Parameter values in the order specified above.
                            Length should be 11 for source or 12 for target (with p).
        nte (bool): If True, includes the 'p' parameter for target (NTE feeds with tire).
                   If False, uses only the 11 common parameters for source (fossil feeds).
                   Default is True.
        bounds (list of tuples, optional): Parameter bounds as [(min, max), ...].
                                          Length should match number of parameters (11 or 12).
                                          If None, uses default bounds.
                                          Note: These bounds are primarily for constraining 
                                          Bayesian sampling (MCMC), not required by the solver itself.
    
    Returns:
        DataFrame: Parameter DataFrame with columns:
                  - 'param init': Initial parameter values
                  - 'param min': Lower bounds (used in MCMC, not by ODE solver)
                  - 'param max': Upper bounds (used in MCMC, not by ODE solver)
                  - 'Iflag': Boolean flags (all True)
                  - 'description': Parameter names
                  - 'ID': Parameter identifiers
    '''
    # Define parameter names in order: k0, Ea, m, n, alpha, b, A0, C0, u, l, v
    param_names = ['k0', 'Ea', 'm', 'n', 'alpha', 'b', 'A0', 'C0', 'u', 'l', 'v']
    
    # Set default bounds if not provided
    # Note: Bounds are not required by the ODE solver but are used later in 
    # Bayesian methods (MCMC) to constrain parameter sampling space
    if bounds is None:
        if nte:
            # Target case: 12 parameters including 'p'
            bounds = [(0, 1e3), (10000, 80000), (0.3, 10), (0.3, 10), (-10, 0), 
                    (-40000, 0), (0, 10), (-5, 5), (0, 3), (-10, 10), (-10, 10), 
                    (0, 10)]
        else:
            bounds = [(0, 1e3), (10000, 80000), (0.3, 10), (0.3, 10), (-10, 0), 
                    (-40000, 0), (0, 10), (-5, 5), (0, 3), (-10, 10), (-10, 10)]
    
    if nte:
        # Target case: add 'p' parameter for tire effect in NTE feeds
        param_names.append('p')
        param_init = list(params[:12])
    else:
        # Source case: use only the 11 common parameters
        param_init = list(params[:11])
    
    # Extract param_min and param_max from bounds
    # These bounds are stored in the DataFrame for later use in Bayesian methods
    # but are not used by the ODE solver itself
    param_min = [b[0] for b in bounds]
    param_max = [b[1] for b in bounds]
    
    # Create parameter DataFrame with bounds and metadata
    # This specific format (with Iflag, description, ID) is required by our solver
    params_df = pd.DataFrame({
        "param init": param_init,
        "param min": param_min,  # For MCMC/optimization, not ODE solver
        "param max": param_max,  # For MCMC/optimization, not ODE solver
        "Iflag": [True] * len(param_names),
        "description": param_names,
        "ID": param_names
    }, index=param_names)
    
    return params_df

###################################################################################################################################################################
def output_data(data, params, nte, N=False):
    '''
    Simulate HDN reaction using the solver "solver". The solver solves the ODE for each row independently.
    The output nitrogen is named N_simul and concatenated with the dataframe "data".
    
    Args:
        data (DataFrame): Input dataframe containing the source or target data for the HDN reaction 
                         (Temperature, LHSV, ppH2, feed properties, etc.).
        params (DataFrame): Dataframe containing the ODE parameters. Each row corresponds to a set of parameters,
                           and each column represents a specific parameter feature used in the ODE system.
                           Note: The solver uses only the 'param init' column values for simulation. 
                           The parameter bounds (param_min, param_max) are not used in the ODE solving process.
        nte (bool): Boolean flag indicating the case type. True when adding an additional feature in the ODE 
                    (Target case with Tire), False for source case.
        N (bool): If True, returns only the pandas Series containing the nitrogen output. 
                  If False, returns the complete dataset with the output. Default is False.
    
    Returns:
        Series or DataFrame: If N=True, returns pandas Series with estimated nitrogen output (N_simul).
                            If N=False, returns the complete input dataframe with an additional N_simul column.
    
    Note:
        The ODE solver only uses parameter values from the 'param init' column. Parameter bounds 
        (param_min, param_max) are included in the params DataFrame for later use in optimization 
        and Bayesian methods, but are ignored during simulation.
    '''
    data = solver(data, params, nte)
    if N:
        return data["N_simul"]
    else:
        return data
####################################################################################################################################################################
def kennard_stone_split_df(df, feature_columns, test_size):
    """
    Apply Kennard-Stone sampling to split DataFrame while preserving all original columns.
    
    Uses classical Kennard-Stone algorithm (full distance matrix computation) for train/test
    splitting. Returns split DataFrames with all columns intact, plus the original indices
    for traceability.
    
    Args:
        df (DataFrame): Input dataframe to split.
        feature_columns (list): Column names to use for Kennard-Stone distance calculation.
                               Only these features are used to determine sample diversity.
        test_size (int): Number of samples for test set (e.g., 10, 50, 100).
                        Remaining samples go to training set.
    
    Returns:
        tuple: (df_train, df_test, train_indices, test_indices)
               - df_train: Training DataFrame with reset index
               - df_test: Test DataFrame with reset index
               - train_indices: Original row indices of training samples
               - test_indices: Original row indices of test samples
    
    Note:
        The kennard_stone library's train_test_split doesn't return indices directly,
        only the feature arrays. We use array matching as a workaround to recover
        the original DataFrame indices.
    """
    # Extract only the feature columns as numpy array for KS algorithm
    # KS uses these features to calculate sample diversity/distances
    X_features = df[feature_columns].values
    
    # Apply Kennard-Stone split on features
    # Returns numpy arrays of selected train and test samples
    X_train_np, X_test_np = ks.train_test_split(X_features, test_size=test_size)
    
    # Workaround: ks.train_test_split returns arrays but not indices
    # We need indices to split the original DataFrame while keeping all columns
    train_indices = []
    test_indices = []
    
    # Find original DataFrame indices for training set
    # Match each row in X_train_np back to its position in original X_features
    for i, row in enumerate(X_train_np):
        # Find first row in X_features that exactly matches this training sample
        idx = np.where((X_features == row).all(axis=1))[0][0]
        train_indices.append(idx)
    
    # Find original DataFrame indices for test set
    # Match each row in X_test_np back to its position in original X_features
    for i, row in enumerate(X_test_np):
        # Find first row in X_features that exactly matches this test sample
        idx = np.where((X_features == row).all(axis=1))[0][0]
        test_indices.append(idx)
    
    # Split original DataFrame using recovered indices
    # This preserves all columns (features + target + any metadata)
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_test = df.iloc[test_indices].reset_index(drop=True)
    
    return df_train, df_test, train_indices, test_indices

##################################################################################################################################################################
def add_subtract_random(val, percentage, seed=0):
    """
    Add heteroscedastic random noise to values, simulating measurement uncertainty.
    
    Generates noise proportional to the value magnitude (percentage-based), which
    is realistic for analytical measurements where uncertainty scales with concentration.
    Results are clipped to physically valid nitrogen concentration range [5, 500] ppm.
    
    Args:
        val (array-like): Original values (typically nitrogen concentrations in ppm).
        percentage (float): Noise level as a fraction (e.g., 0.05 for 5% noise).
                           Noise will be uniformly distributed in [-percentage*val, +percentage*val].
        seed (int): Random seed for reproducibility. Default is 0.
    
    Returns:
        ndarray: Values with added noise, clipped to [5, 500] ppm range.
    
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random noise uniformly distributed in [-1, 1]
    random_vector = np.random.uniform(-1, 1, size=len(val))
    
    # Apply heteroscedastic noise: noise magnitude scales with value
    # noise = random_vector * percentage * val gives ±percentage variation
    modified_values = val + random_vector * percentage * val
    
    # Clip to valid nitrogen concentration range [5, 500] ppm
    modified_values = np.clip(modified_values, 5, 500)
    
    print("noise added")
    return modified_values
##################################################################################################################################################################
MAE = nn.L1Loss() # The mean absolute error function -> calculate the prediction error on test
##################################################################################################################################################################
def size_fold(train):
    """
    Determine appropriate fold size for cross-validation based on training set size.
    
    Uses adaptive fold sizing to ensure meaningful validation:
    - Small datasets (≤20): 1 fold (leave-one-out style validation)
    - Medium datasets (21-30): 2 folds
    - Large datasets (>30): 5 folds (standard k-fold)
    
    Args:
        train (int): Size of the training dataset.
    
    Returns:
        int: Number of folds for cross-validation.
    
    """
    if train <= 20:
        fold_size = 1  # Very small data: minimal folding
    elif train > 20 and train <= 30:
        fold_size = 2  # Medium data: 2-fold cross-validation
    else:
        fold_size = 5  # Larger data: standard 5-fold cross-validation
    
    return fold_size

##################################################################################################################################################################
def loss_function(params_array, data, bounds, target_column="N_simul"):
    '''
    Calculate relative mean squared error loss for L-BFGS-B (L-BFGS-B_init) optimization.
    
    This function is used in L-BFGS-B and L-BFGS-B_init methods to optimize
    HDN model parameters.
    
    Args:
        params_array (array): Parameter values in order [k0, Ea, m, n, alpha, b, A0, C0, u, l, v, p].
        data (DataFrame): Training data containing:
                         - Operating conditions and feedstock properties
                         - target_column column (target nitrogen values)
    
    Returns:
        float: Relative MSE loss = sum((y_true - y_pred)^2 / y_true).
               Lower values indicate better fit.
    '''
    
    # Transform parameter array to DataFrame format required by solver
    params_df = params_transform(params=params_array, nte=True, bounds=bounds)
    
    # Extract true nitrogen values from data
    y_true = data[target_column]

    # Simulate nitrogen output using proposed parameters
    y_pred = output_data(data, params_df, nte=True, N=True)
    
    
    # Calculate relative MSE: sum((y_true - y_pred)^2 / y_true)
    loss = np.sum((y_true - y_pred)**2 / y_true)
    
    return loss
##################################################################################################################################################################
class CustomLoss(nn.Module):
    """
    Custom PyTorch loss module implementing relative mean squared error.
    This is the neural network (tensor) version of the loss function used in L-BFGS-B
        
    Attributes:
        Inherits from nn.Module with no additional attributes.
    """

    def __init__(self):
        """Initialize the custom loss module."""
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Calculate relative mean squared error loss.
        
        Args:
            y_pred (torch.Tensor): Predicted nitrogen values from neural network.
            y_true (torch.Tensor): True nitrogen values from data.
        
        Returns:
            torch.Tensor: Scalar loss value (relative MSE).
        """
        # Calculate relative MSE: sum((y_pred - y_true)^2 / y_true)
        loss = torch.sum((y_pred - y_true) ** 2 / y_true)
        return loss

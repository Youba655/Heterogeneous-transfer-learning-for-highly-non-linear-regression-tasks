import pandas as pd
import os, sys
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize # used only for L-BFGS-B and L-BFGS-B_init
from dill import dump, load
import time
from time import strftime
import warnings
# Get the current working directory (where the notebook is)
current_dir = os.getcwd()
# Add the parent directory to sys.path (where hck_tools and python_files are)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from .utils import * 

##################################################################################################################################################################
def return_term(data, params):
    '''
    Calculate the thermodynamic driving force term: 1 - reverse_reaction_factor.
    Returns the thermodynamic constraint that must remain positive for the reaction to proceed.
    
    Args:
        data (DataFrame): Input data containing operating conditions and feedstock properties
                         (T, ppH2, TMP, N0, etc.).
        params (array-like): Parameter array in order: [k0, Ea, m, n, alpha, b, A0, C0, u, l, v]
                            This function uses: alpha (index 4), b (index 5), u (index 8), 
                            l (index 9), v (index 10).
    
    Returns:
        array: Thermodynamic driving force (must be > 0 for reaction feasibility).
    '''
    # Gas constant and reference conditions
    R = 1.987  # Gas constant (cal/mol·K)
    T_ref = 375 + 273.15  # Reference temperature (K)
    ppH2_ref = 32.5  # Reference H2 partial pressure (bar)
    TMP_ref = 370 + 273.15  # Reference TMP (K)
    
    # Extract operating conditions and feedstock properties from data
    T = data["T"]
    ppH2 = data["ppH2"]
    TMP = data["TMP"]
    N0 = data["N0"]
    
    # Extract relevant parameters from array: [k0, Ea, m, n, alpha, b, A0, C0, u, l, v]
    u = params[8]  # Reverse reaction amplitude parameter
    b = params[5]  # Reverse reaction activation energy parameter
    alpha = params[4]  # H2 pressure effect exponent
    l = params[9]  # Nitrogen content effect exponent
    v = params[10]  # Temperature (TMP) effect exponent
    
    # Calculate thermodynamic driving force: 1 - reverse_reaction_factor
    out = 1 - u * np.exp(-b/R * (1/(T+273.15) - 1/(T_ref))) * (ppH2/ppH2_ref)**alpha * N0**l * ((TMP+273.15)/TMP_ref)**v
    
    return out
################################################################################################################################################################## 
def f_hard(X, params):
    '''
    Hard constraint to ensure thermodynamic feasibility.
    Rejects parameter sets if the thermodynamic driving force becomes non-positive.
    
    Args:
        X (DataFrame): Input data containing operating conditions and feedstock properties
                      (T, ppH2, TMP, N0, etc.) used to evaluate thermodynamic feasibility.
        params (array-like): Parameter array in order: [k0, Ea, m, n, alpha, b, A0, C0, u, l, v]
                            (or with 'p' for target case).
    
    Returns:
        str or bool: Returns error message string if constraint is violated (thermodynamic driving force <= 0).
                    Returns False if constraint is satisfied (thermodynamic driving force > 0).
    '''
    # Calculate minimum thermodynamic driving force across all data points
    thermo_limit = return_term(X, params).min()
    
    # Check if thermodynamic constraint is violated
    if thermo_limit <= 0:
        # Reject parameter set: thermodynamic driving force must always be positive
        return f"Thermodynamic limit constraint violated: {thermo_limit}"
    else:
        # Accept parameter set: no constraint violation
        return False
####################################################################################################################################################################
def f_var(Y_exp, params):
    '''
    Construct diagonal covariance matrix for likelihood: Cov_ii = sigma * y_i.
    Variance scales with measurement magnitude (heteroscedastic noise model).
    
    Args:
        Y_exp (array-like): Experimental/observed nitrogen measurements (in ppm).
        params (array-like): Parameter array in order: [k0, Ea, m, n, alpha, b, A0, C0, u, l, v, sigma]
                            (or with 'p' before sigma for target case).
                            This function uses params[-1] which is the sigma parameter.
    
    Returns:
        ndarray: Diagonal covariance matrix of shape (n, n) where n = len(Y_exp).
                Diagonal elements are Cov_ii = sigma * y_i (heteroscedastic variance).
    '''
    # Extract sigma parameter (last element in parameter array)
    sigma = params[-1]
    
    # Construct diagonal covariance matrix: Cov_ii = sigma * y_i
    return np.diag([Y_exp[i] * sigma for i in range(len(Y_exp))])
####################################################################################################################################################################
def f_likelyhood(Y_true, Y_pred, VAR_mdl_diag):
    '''
    Calculate log-likelihood using multivariate normal distribution.    
    Args:
        Y_true (array-like): Observed/experimental nitrogen measurements (in ppm).
        Y_pred (array-like): Predicted nitrogen values from the ODE model (in ppm).
                            Serves as the mean of the multivariate normal distribution.
        VAR_mdl_diag (ndarray): Diagonal covariance matrix (n x n) from f_var function.
                               Represents measurement uncertainty structure.
    
    Returns:
        float: Log-likelihood value. Used in MCMC acceptance criterion.
    '''
    out = multivariate_normal.logpdf(Y_true, Y_pred,
                                     VAR_mdl_diag, 
                                     allow_singular=True)
    return out
####################################################################################################################################################################
def adjust_param_step(params_df, Nacc_df, n_last=100, levels=None):
    """
    Adjust parameter step sizes based on acceptance rates to optimize MCMC exploration.
    
    The acceptance rate reflects the balance between exploration and exploitation:
    - Too low (<30%): steps too large, most proposals rejected → reduce step size
    - Too high (>70%): steps too small, limited exploration → increase step size
    - Optimal (30-70%): good balance between acceptance and exploration
    
    Args:
        params_df (DataFrame): Parameter DataFrame containing 'param step' column with current step sizes.
        Nacc_df (DataFrame): Acceptance count DataFrame tracking accepted proposals per parameter.
        n_last (int): Number of recent iterations to consider for acceptance rate calculation. 
                     Default is 100.
        levels (list of lists, optional): Acceptance rate thresholds and adjustment factors.
                                         Format: [[lower%, upper%, factor], ...]
                                         If None, uses default levels optimized for MCMC performance.
    
    Returns:
        str: Log string documenting the adjustments made for each parameter.
    
    """
    # Calculate starting index for the last n_last iterations
    i0 = max(0, Nacc_df.shape[0] - n_last)
    
    # Calculate acceptance rates (%) for each parameter over last n_last iterations
    nacc = Nacc_df.iloc[i0:].sum() / (Nacc_df.shape[0] - i0) * 100
    
    # Initialize log string
    log_str = "-" * 100 + "\n"
    log_str += f"Running Parameter Step Adjustment for iterations # {i0} - {Nacc_df.shape[0]}\n"
    param_step = params_df["param step"].tolist()
    
    # Define default adjustment levels if not provided
    # Format: [lower_bound(%), upper_bound(%), multiplication_factor]
    if levels is None:
        levels = [[-np.inf, 10, 0.25],  # Very low acceptance (<10%): drastically reduce step size
                  [10, 20, 0.5],         # Low acceptance (10-20%): reduce step size
                  [20, 30, 0.8],         # Slightly low (20-30%): slightly reduce step size
                  [70, 80, 1.2],         # Slightly high (70-80%): slightly increase step size
                  [80, 90, 2.0],         # High acceptance (80-90%): increase step size
                  [90, np.inf, 4.0]]     # Very high acceptance (>90%): drastically increase step size
    
    # Adjust step size for each parameter based on its acceptance rate
    for i, (p, pstep) in enumerate(zip(params_df.index, params_df["param step"])):
        m = 1  # Default multiplication factor (no change)
        
        # Find which level range the acceptance rate falls into
        for lower, upper, factor in levels:
            if (nacc[p] >= lower) and nacc[p] < upper:
                m = factor
                break
        
        # Apply adjustment if needed
        if m != 1:
            pstep_new = pstep * m
            log_str += f"\t{p}: Nacc=[{lower:0.1f}% < {nacc[p]:0.1f}% < {upper:0.1f}] step {pstep:0.2e} -> {pstep_new:0.2e}\n"
            param_step[i] = pstep_new
        else:
            # Step size is optimal (acceptance rate in 30-70% range)
            log_str += f"\t{p}: Nacc={nacc[p]:0.1f}% step {pstep:0.2e} OK\n"

    # Update parameter DataFrame with new step sizes
    params_df["param step"] = param_step
    log_str += "-" * 100
    return log_str
####################################################################################################################################################################
def check_bounds(params_new, params_current, i_param, i, upper_bound, lower_bound, 
                 param_names, verbose, log_file):
    """
    Check if proposed parameter value is within defined bounds.
    Rejects parameters that violate upper or lower bound constraints.
    
    Args:
        params_new (list): Proposed parameter values for current MCMC iteration.
        params_current (list): Current parameter values (before proposal).
        i_param (int): Index of the parameter being checked.
        i (int): Current MCMC iteration number (for logging).
        upper_bound (list or None): Upper bounds for each parameter. None if no upper bounds.
        lower_bound (list or None): Lower bounds for each parameter. None if no lower bounds.
        param_names (list): List of parameter names for logging purposes.
        verbose (bool): If True, prints rejection messages to console.
        log_file (str or None): Path to log file. If None, no file logging.
    
    Returns:
        bool: True if parameter violates bounds (reject proposal).
              False if parameter is within bounds (accept for further evaluation).
    """
    # Check upper bound constraint
    if upper_bound is not None:
        if params_new[i_param] > upper_bound[i_param]:
            # Format rejection message with parameter name and values
            out_str = (f"{i}: "
                       + param_names[i_param].ljust(max([len(x) for x in param_names]))
                       + f"  {params_current[i_param]: .4e} -> {params_new[i_param]: .4e} "
                       + " reject on upper bound")
            
            # Log to console if verbose mode enabled
            if verbose:
                print(out_str)
            
            # Log to file if log file specified
            if log_file is not None:
                with open(log_file, "a") as log:
                    log.write(out_str + "\n")
            
            return True  # Reject: upper bound violated
    
    # Check lower bound constraint
    if lower_bound is not None:
        if params_new[i_param] < lower_bound[i_param]:
            # Format rejection message
            out_str = (f"{i}: "
                       + param_names[i_param].ljust(max([len(x) for x in param_names]))
                       + f"  {params_current[i_param]: .4e} -> {params_new[i_param]: .4e} "
                       + " reject on lower bound")
            
            # Log to console if verbose mode enabled
            if verbose:
                print(out_str)
            
            # Log to file if log file specified
            if log_file is not None:
                with open(log_file, "a") as log:
                    log.write(out_str + "\n")
            
            return True  # Reject: lower bound violated
    
    return False  # Accept: parameter within bounds
####################################################################################################################################################################
def check_constraints(X, params_new, params_current, i_param, i, f_hard, 
                     param_names, verbose, log_file):
    """
    Check if proposed parameters satisfy hard physical/thermodynamic constraints.
    Rejects parameters that violate constraints beyond simple bounds.
    
    Hard constraints enforce physical feasibility (e.g., thermodynamic driving force > 0)
    rather than just parameter ranges. These are model-specific requirements.
    
    Args:
        X (DataFrame): Input data (operating conditions and feedstock properties) used
                      to evaluate constraint satisfaction across all data points.
        params_new (list): Proposed parameter values for current MCMC iteration.
        params_current (list): Current parameter values (before proposal).
        i_param (int): Index of the parameter being checked.
        i (int): Current MCMC iteration number (for logging).
        f_hard (function): Hard constraint function that returns False if satisfied,
                          or error message string if violated.
        param_names (list): List of parameter names for logging purposes.
        verbose (bool): If True, prints rejection messages to console.
        log_file (str or None): Path to log file. If None, no file logging.
    
    Returns:
        bool: True if constraints are violated (reject proposal).
              False if constraints are satisfied (accept for further evaluation).
    """
    # Attempt to evaluate hard constraint function
    try:
        constraint_check = f_hard(X, params_new)
    except Exception as err:
        # If f_hard fails (e.g., numerical error), treat as constraint violation
        constraint_check = f"Call to f_hard failed with {err}"
    
    # Check if constraint is violated
    # f_hard returns False if OK, or error message string if violated
    if constraint_check:
        # Format rejection message with constraint violation details
        out_str = (f"{i}: "
                   + param_names[i_param].ljust(max([len(x) for x in param_names]))
                   + f"  {params_current[i_param]: .4e} -> {params_new[i_param]: .4e} "
                   + " reject on hard constraint(s) : " + str(constraint_check))
        
        # Log to console if verbose mode enabled
        if verbose:
            print(out_str)
        
        # Log to file if log file specified
        if log_file is not None:
            with open(log_file, "a") as log:
                log.write(out_str + "\n")
        
        return True  # Reject: hard constraint violated
    
    return False  # Accept: constraints satisfied
####################################################################################################################################################################
def setup_logging(log_file, i, quiet, model_name):
    """
    Initialize or continue logging for MCMC run with appropriate headers.
    Creates a new log file for fresh runs or appends to existing log for continued runs.
    
    Args:
        log_file (str or None): Path to the log file. If None, no logging is performed.
        i (int): Current MCMC iteration number. If 0, creates new log file (fresh start).
                If > 0, appends to existing log file (continuation).
        quiet (bool): If True, suppresses console output about log file creation.
                     If False, prints log file path to console.
        model_name (str): Name of the MCMC model being run (e.g., "MCMC_Source", "MCMC_Target")
                         for identification in log headers.
    
    Returns:
        None
    """
    if log_file is not None:
        # Print log file location to console unless in quiet mode
        if not quiet:
            print("Writing log to file", os.path.abspath(log_file))
        
        if i == 0:
            # Fresh start: create new log file (overwrite if exists)
            log = open(log_file, "w")
            log.write(strftime("%d/%m/%Y %H:%M") + f" Start MCMC iterations for {model_name} model" + "\n")
            log.write("-" * 100 + "\n")
            log.close()
        else:
            # Continuation: append to existing log file
            log = open(log_file, "a")
            log.write("=" * 100 + "\n")  # Use "=" to distinguish continuation sections
            log.write(strftime("%d/%m/%Y %H:%M") + f" Continue MCMC iterations for {model_name} model" + "\n")
            log.write("-" * 100 + "\n")
            log.close()

####################################################################################################################################################################
def save_state(params_estim, iterations, Nacc, savefolder, file_format="csv"):
    """
    Save MCMC state to files for checkpointing and post-processing analysis.
    Saves three key components: parameter history, iteration metrics, and acceptance counts.
    
    Args:
        params_estim (DataFrame): Parameter estimates history. Each row is an iteration,
                                 each column is a parameter value at that iteration.
        iterations (DataFrame): Iteration-level metrics including likelihood (and prior/posterior
                               for target models) at each MCMC iteration.
        Nacc (DataFrame): Acceptance counts tracking how many times each parameter was accepted
                         during the MCMC run. Used for step size adjustment.
        savefolder (str): Directory path where MCMC state files will be saved.
        file_format (str): Output file format. Options:
                          - "csv": Three separate CSV files (human-readable, good for analysis)
                          - "pickle": Three pickle files (Python-specific, preserves data types)
                          - "xlsx" or other Excel formats: Single Excel file with multiple sheets
                          Default is "csv".
    
    Returns:
        None
    
    Raises:
        ValueError: If file_format is not one of 'csv', 'pickle', or valid Excel format.
    """
    if file_format == "csv":
        # Save as three separate CSV files
        params_estim.to_csv(os.path.join(savefolder, "param_estim_MCMC.csv"))
        iterations.to_csv(os.path.join(savefolder, "iterations_MCMC.csv"))
        Nacc.to_csv(os.path.join(savefolder, "Nacc_MCMC.csv"))
    
    elif file_format == "pickle":
        # Save as three separate pickle files (Python binary format)
        params_estim.to_pickle(os.path.join(savefolder, "param_estim_MCMC.pickle"))
        iterations.to_pickle(os.path.join(savefolder, "iterations_MCMC.pickle"))
        Nacc.to_pickle(os.path.join(savefolder, "Nacc_MCMC.pickle"))
    
    elif file_format.startswith("xls"):
        # Save as single Excel file with multiple sheets
        with pd.ExcelWriter(os.path.join(savefolder, "MCMC_state." + file_format)) as xls:
            params_estim.to_excel(xls, sheet_name="params_estim", freeze_panes=(1, 1))
            iterations.to_excel(xls, sheet_name="iterations", freeze_panes=(1, 1))
            Nacc.to_excel(xls, sheet_name="Nacc", freeze_panes=(1, 1))
    
    else:
        raise ValueError(f"Invalid file_format '{file_format}', must be either 'csv', 'pickle', or valid Excel file format (xlsx, xls, xlsm, etc.)")
####################################################################################################################################################################
def accept_reject_step_source(params_new, params_current, likelihood_new, likelihood_current, 
                      i_param, i, param_names, verbose, log_file):
    """
    Metropolis-Hastings accept/reject step for source model (likelihood only, no prior).
    Decides whether to accept or reject proposed parameter based on likelihood improvement.
    
    The Metropolis-Hastings criterion accepts proposals with probability:
        min(1, exp(likelihood_new - likelihood_current))
    
    This means:
    - Always accept if likelihood improves (ratio > 0)
    - Sometimes accept if likelihood worsens (ratio < 0) to enable exploration
    
    Args:
        params_new (list): Proposed parameter values for this iteration.
        params_current (list): Current parameter values (before proposal).
        likelihood_new (float): Log-likelihood for proposed parameters.
        likelihood_current (float): Log-likelihood for current parameters.
        i_param (int): Index of the parameter being modified.
        i (int): Current MCMC iteration number.
        param_names (list): List of parameter names for logging.
        verbose (bool): If True, prints accept/reject decision to console.
        log_file (str or None): Path to log file for recording decisions.
    
    Returns:
        bool: True if proposal is accepted (update parameters).
              False if proposal is rejected (keep current parameters).
    """
    # Calculate log-likelihood ratio (difference in log space = ratio in probability space)
    ratio = likelihood_new - likelihood_current
    
    # Draw random number uniformly from [0, 1] for acceptance criterion
    rdm = np.random.uniform(0, 1)
    
    # Format log string with iteration info, parameter change, and likelihood change
    log_string = (f"{i}: "
                    + param_names[i_param].ljust(max([len(x) for x in param_names]))
                    + f"   {params_current[i_param]: .4e}->{params_new[i_param]: .4e}"
                    + f" L {likelihood_current: .2f}->{likelihood_new: .2f}"
                    + (" accept" if ratio > np.log(rdm) else " reject")
                    )
    
    # Write decision to log file if specified
    if log_file is not None:
        with open(log_file, "a") as log:
            log.write(log_string + "\n")
    
    # Print decision to console if verbose mode enabled
    if verbose:
        print(log_string)
    
    # Metropolis-Hastings acceptance criterion
    # Accept if ratio > log(rdm), which means exp(ratio) > rdm
    # Equivalent to: accept with probability min(1, exp(ratio))
    return ratio > np.log(rdm)
####################################################################################################################################################################
def accept_reject_step_target(params_new, params_current, likelihood_new, likelihood_current,
                             prior_new, prior_current, i_param, i, param_names, verbose, log_file):
    """
    Metropolis-Hastings accept/reject step for target model (with prior from source).
    Decides whether to accept or reject proposed parameter based on posterior improvement.
    
    The Metropolis-Hastings criterion with prior accepts proposals with probability:
        min(1, exp(posterior_new - posterior_current))
    where posterior = likelihood + prior (in log space)
    
    This incorporates knowledge from source domain (prior) while fitting target data (likelihood).
    
    Args:
        params_new (list): Proposed parameter values for this iteration.
        params_current (list): Current parameter values (before proposal).
        likelihood_new (float): Log-likelihood for proposed parameters on target data.
        likelihood_current (float): Log-likelihood for current parameters on target data.
        prior_new (float): Log-prior probability for proposed parameters (from source).
        prior_current (float): Log-prior probability for current parameters (from source).
        i_param (int): Index of the parameter being modified.
        i (int): Current MCMC iteration number.
        param_names (list): List of parameter names for logging.
        verbose (bool): If True, prints accept/reject decision to console.
        log_file (str or None): Path to log file for recording decisions.
    
    Returns:
        bool: True if proposal is accepted (update parameters).
              False if proposal is rejected (keep current parameters).

    """
    # Calculate log-posterior ratio
    # posterior = likelihood + prior (in log space, multiplication in probability space)
    ratio = (likelihood_new + prior_new) - (likelihood_current + prior_current)
    
    # Draw random number uniformly from [0, 1] for acceptance criterion
    rdm = np.random.uniform(0, 1)
    
    # Format log string with iteration info, parameter change, likelihood, and prior changes
    log_string = (f"{i}: "
                    + param_names[i_param].ljust(max([len(x) for x in param_names]))
                    + f"   {params_current[i_param]: .4e}->{params_new[i_param]: .4e}"
                    + f" L {likelihood_current: .2f}->{likelihood_new: .2f}"
                    + f" P {prior_current: .2f}->{prior_new: .2f}"
                    + (" accept" if ratio > np.log(rdm) else " reject")
                    )
    
    # Write decision to log file if specified
    if log_file is not None:
        with open(log_file, "a") as log:
            log.write(log_string + "\n")
    
    # Print decision to console if verbose mode enabled
    if verbose:
        print(log_string)
    
    # Metropolis-Hastings acceptance criterion with posterior
    # Accept if ratio > log(rdm), which means exp(ratio) > rdm
    # Equivalent to: accept with probability min(1, exp(ratio))
    return ratio > np.log(rdm)

####################################################################################################################################################################
def run_mcmc_source(mcmc_obj, init_params=None, iteration=np.inf, max_time=np.inf, verbose=False, 
             quiet=False, save_state_flag=False, log_file="mcmc_source.log", adjust_par_step=False):
    """
    Execute Markov Chain Monte Carlo (MCMC) algorithm for source model parameter estimation.
    Implements Metropolis-Hastings algorithm with adaptive step size adjustment.
    
    Args:
        mcmc_obj (MCMC_Source): MCMC source object containing data, model, and configuration.
        init_params (list or None): Initial parameter values. If None, uses last saved state
                                    or default initialization from params DataFrame.
        iteration (int or inf): Maximum number of MCMC iterations. Default is infinite (run until stopped).
        max_time (float or inf): Maximum run time in minutes. Default is infinite.
        verbose (bool): If True, prints detailed accept/reject decisions for each parameter proposal.
        quiet (bool): If True, suppresses progress output to console (log file still written).
        save_state_flag (bool): If True, saves MCMC state (params, iterations, acceptance counts) 
                               periodically during run.
        log_file (str): Path to log file for recording MCMC progress and decisions.
        adjust_par_step (int or bool): If True, uses default adjustment frequency (100 iterations).
                                      If int, adjusts parameter step sizes every N iterations.
                                      If False, no step size adjustment.
    
    Returns:
        DataFrame: Combined dataframe with iteration metrics (likelihood) and acceptance counts.
    """
    
    # INITIALIZATION: Print run configuration
    if not quiet:
        print("-"*60)
        if pd.to_numeric(iteration) < np.inf:
            print(f"Running up to {iteration} iterations.")
        if pd.to_numeric(max_time) < np.inf:
            print(f"Running iterations for a maximum of {max_time} minutes.")
        if not (pd.to_numeric(iteration) < np.inf) and not (pd.to_numeric(max_time) < np.inf):
            print("No maximum number of iterations or run-time specified, the algorithm will "
                 + " run indefinitely.\n-> Stop iterations with KeyboardInterrupt. <-")
        print("-"*60)

    # Ensure log file path is absolute
    if log_file is not None:
        if not os.path.isabs(log_file):
            log_file = os.path.join(mcmc_obj.savefolder_property(), log_file)

    # Extract parameter bounds and step sizes from configuration
    upper_bound = mcmc_obj._params["param max"].tolist()
    lower_bound = mcmc_obj._params["param min"].tolist()
    step_params = mcmc_obj._params["param step"].tolist()

    # Set default adjustment frequency if True
    if adjust_par_step is True:
        adjust_par_step = 100

    # PARAMETER INITIALIZATION
    i = 0
    if init_params is None:
        # Case 1: No explicit initialization provided
        if mcmc_obj._params_estim.shape[0] == 0:
            # Fresh start: use default initialization
            init_params = mcmc_obj._params["param init"].tolist()
            mcmc_obj._params_estim.loc[0] = init_params
            mcmc_obj._Nacc.loc[i] = 0
        else:
            # Continue from last saved state
            i = mcmc_obj._params_estim.index[-1]
            init_params = mcmc_obj._params_estim.loc[i].tolist()
            if not quiet:
                print(f"Continuing iterations from {i}")
    else:
        # Case 2: Explicit initialization provided - validate shape
        if (np.ndim(init_params) != 1) or (np.size(init_params) != len(mcmc_obj._params["param init"])):
            raise ValueError(f"Wrong shape of init_params argument {np.shape(init_params)}")
        # Reset MCMC state and start fresh
        mcmc_obj.reset()
        init_params = list(init_params)
        mcmc_obj._params_estim.loc[0] = init_params
        mcmc_obj._Nacc.loc[i] = 0
        i = 0

    # Calculate initial likelihood for starting parameters
    mcmc_obj._iterations.loc[i, "likelyhood"] = mcmc_obj.calc_likelyhood(init_params)

    # Setup timing and logging
    t0 = time.time()
    i0 = i  # Store starting iteration for progress tracking
    if not quiet:
        print("="*20 + "Starting Iterations" + "="*20)
    mcmc_obj._setup_logging(log_file, i, quiet)

    # MAIN MCMC LOOP
    try:
        while True:
            # Check time limit
            if (time.time() - t0) / 60 > max_time:
                raise TimeoutError
            
            # Get current state
            params_current = mcmc_obj._params_estim.loc[i].tolist()
            likelihood_current = mcmc_obj._iterations.loc[i, "likelyhood"]
            
            # Increment iteration counter
            i += 1
            
            # Log iteration start
            if log_file is not None:
                with open(log_file, "a") as log:
                    log.write(f"Iteration {i} " + strftime("%d/%m/%Y %H:%M") + "\n")
            
            # Initialize acceptance counter for this iteration
            mcmc_obj._Nacc.loc[i] = 0

            # Calculate current model variance (for likelihood)
            VAR_mdl = mcmc_obj.f_var(mcmc_obj.Y, params_current)

            # LOOP OVER EACH PARAMETER (Gibbs-style update)
            for i_param, step_param in enumerate(step_params):
                # Propose new parameter value
                params_new = params_current.copy()
                # Draw from normal distribution centered at current value
                params_new[i_param] = np.random.normal(params_current[i_param], step_params[i_param])

                # Check if proposed parameter is within bounds
                if mcmc_obj._check_bounds(params_new, params_current, i_param, i, upper_bound, lower_bound, verbose, log_file):
                    continue  # Reject and move to next parameter

                # Check if proposed parameter satisfies hard constraints (e.g., thermodynamics)
                if mcmc_obj._check_constraints(params_new, params_current, i_param, i, verbose, log_file):
                    continue  # Reject and move to next parameter

                # Calculate likelihood for proposed parameters
                likelihood_new, VAR_mdl_new = mcmc_obj._calculate_likelihood(params_new)
                
                # Only proceed if likelihood calculation was successful
                if not pd.isna(likelihood_new):
                    # Metropolis-Hastings accept/reject decision
                    if mcmc_obj._accept_reject_step(params_new, params_current, likelihood_new, likelihood_current, 
                                              i_param, i, verbose, log_file):
                        # ACCEPT: Update current state
                        params_current = params_new.copy()
                        likelihood_current = likelihood_new
                        VAR_mdl = VAR_mdl_new.copy()
                        # Track acceptance
                        mcmc_obj._Nacc.loc[i, "Nacc"] += 1  # Total acceptances this iteration
                        mcmc_obj._Nacc.loc[i, mcmc_obj._params.index[i_param]] += 1  # Per-parameter acceptance

            # ITERATION COMPLETE: Log and save progress
            # Print current state to console
            if not quiet:
                print(i, f"{likelihood_current: .2f}",
                      ", ".join([f"{p: .3e}" for p in params_current]))
            if verbose:
                print("-"*60)
            
            # Write separator to log file
            if log_file is not None:
                with open(log_file, "a") as log:
                    log.write("-"*100 + "\n")
            
            # Store accepted parameters and likelihood
            mcmc_obj._params_estim.loc[i] = params_current
            mcmc_obj._iterations.loc[i, "likelyhood"] = likelihood_current

            # ADAPTIVE STEP SIZE ADJUSTMENT
            if adjust_par_step:
                if i % adjust_par_step == 0:
                    # Adjust step sizes based on recent acceptance rates
                    log_str_adj = mcmc_obj.adjust_param_step_fn(mcmc_obj._params, mcmc_obj._Nacc, adjust_par_step)
                    if log_file is not None:
                        with open(log_file, "a") as log:
                            log.write(log_str_adj + "\n")
                    if not quiet:
                        print(log_str_adj)
                    # Update step sizes for next iterations
                    step_params = mcmc_obj._params["param step"].tolist()
            
            # PERIODIC STATE SAVING
            if save_state_flag:
                mcmc_obj._params.to_csv(os.path.join(mcmc_obj.savefolder_property(), "params.csv"))
                mcmc_obj.save_state_fn(mcmc_obj._params_estim, mcmc_obj._iterations, mcmc_obj._Nacc, 
                                      mcmc_obj.savefolder_property(), "pickle")
            
            # Check iteration limit
            if (i - i0) >= iteration:
                break
    
    # ERROR HANDLING AND CLEANUP
    except KeyboardInterrupt:
        msg = f"Interrupted... returning partial results for {i - i0} iterations in {(time.time() - t0)/60:.2f} minutes."
    except TimeoutError:
        msg = f"Maximum calculation time of {max_time} minutes reached, {i - i0} iterations done."
    except Exception as err:
        msg = "ERROR: " + str(err) + " returning partial results."
        raise err
    else:
        msg = f"Calculation with {i - i0} iterations finished in {(time.time() - t0)/60:.2f} minutes."
    finally:
        # Always print completion message and log
        if not quiet:
            print(msg)
        if log_file is not None:
            with open(log_file, "a") as log:
                log.write("="*100  + "\n" + msg + "\n" + "="*100  + "\n")
    
    # Return combined results: iterations with acceptance counts
    return mcmc_obj._iterations.join(mcmc_obj._Nacc)

####################################################################################################################################################################
def run_mcmc_target(mcmc_obj, init_params=None, iteration=np.inf, max_time=np.inf, verbose=False, 
             quiet=False, save_state_flag=False, log_file="mcmc_target.log", adjust_par_step=False):
    """
    Execute Markov Chain Monte Carlo (MCMC) algorithm for target model parameter estimation with prior.
    Implements Metropolis-Hastings algorithm with prior from source domain (Hierarchical Bayesian Transfer Learning).
    
    This version differs from run_mcmc_source by incorporating prior knowledge from the source domain,
    enabling transfer learning from source to target domain with limited target data.
    
    Args:
        mcmc_obj (MCMC_Target): MCMC target object containing data, model, prior, and configuration.
        init_params (list or None): Initial parameter values. If None, uses last saved state
                                    or default initialization from params DataFrame.
        iteration (int or inf): Maximum number of MCMC iterations. Default is infinite (run until stopped).
        max_time (float or inf): Maximum run time in minutes. Default is infinite.
        verbose (bool): If True, prints detailed accept/reject decisions with likelihood and prior values.
        quiet (bool): If True, suppresses progress output to console (log file still written).
        save_state_flag (bool): If True, saves MCMC state (params, iterations, acceptance counts) 
                               periodically during run.
        log_file (str): Path to log file for recording MCMC progress and decisions.
        adjust_par_step (int or bool): If True, uses default adjustment frequency (100 iterations).
                                      If int, adjusts parameter step sizes every N iterations.
                                      If False, no step size adjustment.
    
    Returns:
        DataFrame: Combined dataframe with iteration metrics (likelihood, prior, posterior) and acceptance counts.
    """
    
    # INITIALIZATION: Print run configuration
    if not quiet:
        print("-"*60)
        if pd.to_numeric(iteration) < np.inf:
            print(f"Running up to {iteration} iterations.")
        if pd.to_numeric(max_time) < np.inf:
            print(f"Running iterations for a maximum of {max_time} minutes.")
        if not (pd.to_numeric(iteration) < np.inf) and not (pd.to_numeric(max_time) < np.inf):
            print("No maximum number of iterations or run-time specified, the algorithm will "
                 + " run indefinitely.\n-> Stop iterations with KeyboardInterrupt. <-")
        print("-"*60)

    # Ensure log file path is absolute
    if log_file is not None:
        if not os.path.isabs(log_file):
            log_file = os.path.join(mcmc_obj.savefolder_property(), log_file)

    # Extract parameter bounds and step sizes from configuration
    upper_bound = mcmc_obj._params["param max"].tolist()
    lower_bound = mcmc_obj._params["param min"].tolist()
    step_params = mcmc_obj._params["param step"].tolist()

    # Set default adjustment frequency if True
    if adjust_par_step is True:
        adjust_par_step = 100

    # PARAMETER INITIALIZATION
    i = 0
    if init_params is None:
        # Case 1: No explicit initialization provided
        if mcmc_obj._params_estim.shape[0] == 0:
            # Fresh start: use default initialization
            init_params = mcmc_obj._params["param init"].tolist()
            mcmc_obj._params_estim.loc[0] = init_params
            mcmc_obj._Nacc.loc[i] = 0
        else:
            # Continue from last saved state
            i = mcmc_obj._params_estim.index[-1]
            init_params = mcmc_obj._params_estim.loc[i].tolist()
            if not quiet:
                print(f"Continuing iterations from {i}")
    else:
        # Case 2: Explicit initialization provided - validate shape
        if (np.ndim(init_params) != 1) or (np.size(init_params) != len(mcmc_obj._params["param init"])):
            raise ValueError(f"Wrong shape of init_params argument {np.shape(init_params)}")
        # Reset MCMC state and start fresh
        mcmc_obj.reset()
        init_params = list(init_params)
        mcmc_obj._params_estim.loc[0] = init_params
        mcmc_obj._Nacc.loc[i] = 0
        i = 0

    # Calculate initial likelihood, prior, and posterior for starting parameters
    mcmc_obj._iterations.loc[i, "likelyhood"] = mcmc_obj.calc_likelyhood(init_params)
    mcmc_obj._iterations.loc[i, "prior"] = mcmc_obj.calc_prior(init_params)
    mcmc_obj._iterations.loc[i, "posterior"] = mcmc_obj._iterations.loc[i, "likelyhood"] + mcmc_obj._iterations.loc[i, "prior"]

    # Setup timing and logging
    t0 = time.time()
    i0 = i  # Store starting iteration for progress tracking
    if not quiet:
        print("="*20 + "Starting Iterations" + "="*20)
    mcmc_obj._setup_logging(log_file, i, quiet)

    # MAIN MCMC LOOP
    try:
        while True:
            # Check time limit
            if (time.time() - t0) / 60 > max_time:
                raise TimeoutError
            
            # Get current state (including prior from source)
            params_current = mcmc_obj._params_estim.loc[i].tolist()
            likelihood_current = mcmc_obj._iterations.loc[i, "likelyhood"]
            prior_current = mcmc_obj._iterations.loc[i, "prior"]
            
            # Increment iteration counter
            i += 1
            
            # Log iteration start
            if log_file is not None:
                with open(log_file, "a") as log:
                    log.write(f"Iteration {i} " + strftime("%d/%m/%Y %H:%M") + "\n")
            
            # Initialize acceptance counter for this iteration
            mcmc_obj._Nacc.loc[i] = 0

            # Calculate current model variance (for likelihood)
            VAR_mdl = mcmc_obj.f_var(mcmc_obj.Y, params_current)

            # ===== LOOP OVER EACH PARAMETER (Gibbs-style update) =====
            for i_param, step_param in enumerate(step_params):
                # Propose new parameter value
                params_new = params_current.copy()
                # Draw from normal distribution centered at current value
                params_new[i_param] = np.random.normal(params_current[i_param], step_params[i_param])

                # Check if proposed parameter is within bounds
                if mcmc_obj._check_bounds(params_new, params_current, i_param, i, upper_bound, lower_bound, verbose, log_file):
                    continue  # Reject and move to next parameter

                # Check if proposed parameter satisfies hard constraints (e.g., thermodynamics)
                if mcmc_obj._check_constraints(params_new, params_current, i_param, i, verbose, log_file):
                    continue  # Reject and move to next parameter

                # Calculate likelihood and prior for proposed parameters
                likelihood_new, VAR_mdl_new = mcmc_obj._calculate_likelihood(params_new)
                prior_new = mcmc_obj.calc_prior(params_new)  # Evaluate prior from source domain
                
                # Only proceed if likelihood calculation was successful
                if not pd.isna(likelihood_new):
                    # Metropolis-Hastings accept/reject decision (with prior)
                    # Uses posterior = likelihood + prior for acceptance criterion
                    if mcmc_obj._accept_reject_step(params_new, params_current, likelihood_new, likelihood_current, 
                                              prior_new, prior_current, i_param, i, verbose, log_file):
                        # ACCEPT: Update current state including prior
                        params_current = params_new.copy()
                        likelihood_current = likelihood_new
                        prior_current = prior_new
                        VAR_mdl = VAR_mdl_new.copy()
                        # Track acceptance
                        mcmc_obj._Nacc.loc[i, "Nacc"] += 1  # Total acceptances this iteration
                        mcmc_obj._Nacc.loc[i, mcmc_obj._params.index[i_param]] += 1  # Per-parameter acceptance

            # ITERATION COMPLETE: Log and save progress
            # Print current state to console (showing both likelihood and prior)
            if not quiet:
                print(i, f"L={likelihood_current: .2f} P={prior_current: .2f}",
                      ", ".join([f"{p: .3e}" for p in params_current]))
            if verbose:
                print("-"*60)
            
            # Write separator to log file
            if log_file is not None:
                with open(log_file, "a") as log:
                    log.write("-"*100 + "\n")
            
            # Store accepted parameters, likelihood, prior, and posterior
            mcmc_obj._params_estim.loc[i] = params_current
            mcmc_obj._iterations.loc[i, "likelyhood"] = likelihood_current
            mcmc_obj._iterations.loc[i, "prior"] = prior_current
            mcmc_obj._iterations.loc[i, "posterior"] = likelihood_current + prior_current  # Log-space addition

            # ADAPTIVE STEP SIZE ADJUSTMENT 
            if adjust_par_step:
                if i % adjust_par_step == 0:
                    # Adjust step sizes based on recent acceptance rates
                    log_str_adj = mcmc_obj.adjust_param_step_fn(mcmc_obj._params, mcmc_obj._Nacc, adjust_par_step)
                    if log_file is not None:
                        with open(log_file, "a") as log:
                            log.write(log_str_adj + "\n")
                    if not quiet:
                        print(log_str_adj)
                    # Update step sizes for next iterations
                    step_params = mcmc_obj._params["param step"].tolist()
            
            # PERIODIC STATE SAVING
            if save_state_flag:
                mcmc_obj._params.to_csv(os.path.join(mcmc_obj.savefolder_property(), "params.csv"))
                mcmc_obj.save_state_fn(mcmc_obj._params_estim, mcmc_obj._iterations, mcmc_obj._Nacc, 
                                      mcmc_obj.savefolder_property(), "pickle")
            
            # Check iteration limit
            if (i - i0) >= iteration:
                break
    
    # ERROR HANDLING AND CLEANUP
    except KeyboardInterrupt:
        msg = f"Interrupted... returning partial results for {i - i0} iterations in {(time.time() - t0)/60:.2f} minutes."
    except TimeoutError:
        msg = f"Maximum calculation time of {max_time} minutes reached, {i - i0} iterations done."
    except Exception as err:
        msg = "ERROR: " + str(err) + " returning partial results."
        raise err
    else:
        msg = f"Calculation with {i - i0} iterations finished in {(time.time() - t0)/60:.2f} minutes."
    finally:
        # Always print completion message and log
        if not quiet:
            print(msg)
        if log_file is not None:
            with open(log_file, "a") as log:
                log.write("="*100  + "\n" + msg + "\n" + "="*100  + "\n")
    
    # Return combined results: iterations (with likelihood, prior, posterior) with acceptance counts
    return mcmc_obj._iterations.join(mcmc_obj._Nacc)
####################################################################################################################################################################
class MCMC_Source():
    """
    MCMC class for Bayesian parameter estimation on source (fossil feed) domain.
    Implements Metropolis-Hastings algorithm for calibrating HDN model parameters using source data.
    
    This class orchestrates the MCMC process by:
    1. Managing parameter bounds, step sizes, and acceptance tracking
    2. Coordinating model evaluation, likelihood calculation, and constraint checking
    3. Executing the MCMC loop with adaptive step size adjustment
    4. Saving and loading MCMC state for checkpointing
    
    The class uses external functions for flexibility, allowing customization of:
    - Acceptance criteria, step size adjustment, bounds/constraint checking
    - Logging, state saving, and the main MCMC loop
    """
    
    def __init__(self, X, Y, params, f_model, f_var, f_likelyhood, f_hard, 
                 adjust_param_step_fn=None, check_bounds_fn=None, check_constraints_fn=None,
                 accept_reject_step_fn=None, setup_logging_fn=None, save_state_fn=None,
                 run_mcmc_fn=None, name="MCMC_Source", savefolder=None):
        """
        Initialize MCMC_Source object for Bayesian parameter estimation.
        
        Args:
            X (DataFrame): Input data with operating conditions and feedstock properties.
            Y (Series): Observed nitrogen measurements (target values).
            params (DataFrame): Parameter configuration with columns:
                               'param init', 'param min', 'param max', 'param step', 'description'
            f_model (function): Model function that simulates nitrogen output (f_N_source).
            f_var (function): Function constructing covariance matrix for likelihood.
            f_likelyhood (function): Log-likelihood calculation function.
            f_hard (function): Hard constraint checker (returns False if OK, error message if violated).
            adjust_param_step_fn (function, optional): Custom step size adjustment function.
            check_bounds_fn (function, optional): Custom bounds checking function.
            check_constraints_fn (function, optional): Custom constraint checking function.
            accept_reject_step_fn (function, optional): Custom accept/reject criterion.
            setup_logging_fn (function, optional): Custom logging setup function.
            save_state_fn (function, optional): Custom state saving function.
            run_mcmc_fn (function, optional): Custom MCMC loop function.
            name (str): Model name for identification in logs. Default "MCMC_Source".
            savefolder (str, optional): Directory for saving results. Default is current directory.
        """
        self.X = X
        self.Y = Y
        self.Y_sim = None
        self.name = name
        self.f_model = f_model
        self.f_var = f_var
        self.f_likelyhood = f_likelyhood
        self.f_hard = f_hard
        self.adjust_param_step_fn = adjust_param_step_fn or adjust_param_step
        self.check_bounds_fn = check_bounds_fn or check_bounds
        self.check_constraints_fn = check_constraints_fn or check_constraints
        self.accept_reject_step_fn = accept_reject_step_fn or accept_reject_step_source
        self.setup_logging_fn = setup_logging_fn or setup_logging
        self.save_state_fn = save_state_fn or save_state
        self.run_mcmc_fn = run_mcmc_fn or run_mcmc_source
        
        # Set parameters
        self.set_params(params)
        
        # Initialize
        self._savefolder = savefolder
        self.reset()

    def set_params(self, params):
        """
        Configure parameter settings including bounds, step sizes, and descriptions.
        Sets default step sizes (10% of initial value) and descriptions if not provided.
        """
        self._params = params.copy()
        if "param step" not in self._params.columns:
            self._params["param step"] = self._params["param init"].abs() / 10
        if "description" not in self._params.columns:
            self._params["description"] = self._params.index

    def savefolder_property(self):
        """Get save folder path (returns current directory if not set)."""
        if self._savefolder is None:
            return os.getcwd()
        else:
            return self._savefolder

    def set_savefolder(self, value):
        """Set save folder for MCMC outputs (creates directory if needed)."""
        if not os.path.exists(value):
            print("creating", value)
            os.makedirs(value, exist_ok=True)
        self._savefolder = value
    
    def reset(self):
        """Reset MCMC state: clears parameter history, iterations, and acceptance counts."""
        self._params_estim = pd.DataFrame(columns=self._params.index)
        self._iterations = pd.DataFrame(columns=["likelyhood"], dtype="float64")
        self._Nacc = pd.DataFrame(columns=["Nacc"] + self._params.index.tolist(), dtype=int)

    def to_pickle(self, fname=None):
        """Save MCMC object to pickle file for later resumption or analysis."""
        if fname is None:
            fname = os.path.join(self.savefolder_property(), "MCMC_Source.pickle")
            print(f"Saving {self.name} model to {fname}")
        with open(fname, "wb") as f:
            dump(self, f)

    @classmethod
    def read_pickle(cls, fname):
        """Load MCMC object from pickle file to resume or analyze a previous run."""
        with open(fname, "rb") as f:
            model = load(f)
        if not isinstance(model, cls):
            raise IOError(f"{fname} is not a valid {str(cls)} pickle file")
        return model

    def _check_bounds(self, params_new, params_current, i_param, i, upper_bound, lower_bound, verbose, log_file):
        """Check parameter bounds using external function"""
        return self.check_bounds_fn(params_new, params_current, i_param, i, upper_bound, lower_bound,
                                   self._params.index, verbose, log_file)

    def _check_constraints(self, params_new, params_current, i_param, i, verbose, log_file):
        """Check hard constraints using external function"""
        return self.check_constraints_fn(self.X, params_new, params_current, i_param, i, self.f_hard,
                                        self._params.index, verbose, log_file)

    def _calculate_likelihood(self, params_new):
        """Calculate likelihood for new parameters"""
        try:
            VAR_mdl_new = self.f_var(self.Y, params_new)
            self.Y_sim = self.f_model(self.X, params_new)
            likelihood_new = self.f_likelyhood(self.Y, self.Y_sim, VAR_mdl_new)
            return likelihood_new, VAR_mdl_new
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as err:
            print(str(err.__class__), str(err), "when calculate new loglikelihood, step ignored")
            return np.nan, None

    def _accept_reject_step(self, params_new, params_current, likelihood_new, likelihood_current, 
                           i_param, i, verbose, log_file):
        """Metropolis-Hastings accept/reject step using external function"""
        return self.accept_reject_step_fn(params_new, params_current, likelihood_new, likelihood_current,
                                         i_param, i, self._params.index, verbose, log_file)

    def _setup_logging(self, log_file, i, quiet):
        """Setup logging using external function"""
        return self.setup_logging_fn(log_file, i, quiet, self.name)

    def run(self, init_params=None, iteration=np.inf, max_time=np.inf, verbose=False, 
            quiet=False, save_state=False, log_file="mcmc_source.log", adjust_par_step=False):
        """Run MCMC using external function"""
        return self.run_mcmc_fn(self, init_params, iteration, max_time, verbose, 
                               quiet, save_state, log_file, adjust_par_step)

    def save_state(self, file_format="csv"):
        """Save MCMC state using external function"""
        self.save_state_fn(self._params_estim, self._iterations, self._Nacc, 
                          self.savefolder_property(), file_format)

    def calc_likelyhood(self, params):
        """
        Calculate log-likelihood for given parameters.
        """
        self.Y_sim = self.f_model(self.X, params)
        return self.f_likelyhood(self.Y, self.Y_sim, self.f_var(self.Y, params))

####################################################################################################################################################################
class MCMC_Target():
    """
    MCMC class for Bayesian parameter estimation on target (NTE feed) domain with prior from source.
    Implements Hierarchical Bayesian Transfer Learning (HBTL) using Metropolis-Hastings algorithm.
    
    This class extends MCMC_Source by incorporating prior knowledge from source domain calibration,
    enabling effective parameter estimation with limited target data through transfer learning.
    
    Key difference from MCMC_Source:
    - Includes prior distribution from source domain (params_prior)
    - Uses posterior = likelihood + prior for acceptance criterion
    - Hyperparameter g controls prior strength vs. likelihood
    - Tracks likelihood, prior, and posterior separately
    """
    
    def __init__(self, X, Y, params, f_model, f_var, f_likelyhood, f_hard, 
                 params_prior=None, g=None,
                 adjust_param_step_fn=None, check_bounds_fn=None, check_constraints_fn=None,
                 accept_reject_step_fn=None, setup_logging_fn=None, save_state_fn=None,
                 run_mcmc_fn=None, name="MCMC_Target", savefolder=None):
        """
        Initialize MCMC_Target object for Bayesian parameter estimation with prior.
        
        Args:
            X (DataFrame): Input data with operating conditions and feedstock properties.
            Y (Series): Observed nitrogen measurements on target domain.
            params (DataFrame): Parameter configuration (param init, min, max, step, description).
            f_model (function): Model function that simulates nitrogen output (f_N_target).
            f_var (function): Function constructing covariance matrix for likelihood.
            f_likelyhood (function): Log-likelihood calculation function.
            f_hard (function): Hard constraint checker.
            params_prior (DataFrame, optional): Prior distribution from source MCMC (parameter samples).
            g (float, optional): Hyperparameter controlling prior strength. Higher g = weaker prior.
                                Default is 1.0. Typically optimized via cross-validation for HBTL.
            adjust_param_step_fn (function, optional): Custom step size adjustment function.
            check_bounds_fn (function, optional): Custom bounds checking function.
            check_constraints_fn (function, optional): Custom constraint checking function.
            accept_reject_step_fn (function, optional): Custom accept/reject criterion (uses prior).
            setup_logging_fn (function, optional): Custom logging setup function.
            save_state_fn (function, optional): Custom state saving function.
            run_mcmc_fn (function, optional): Custom MCMC loop function.
            name (str): Model name for logging. Default "MCMC_Target".
            savefolder (str, optional): Directory for saving results.
        """
        self.X = X
        self.Y = Y
        self.Y_sim = None
        self.name = name
        self.f_model = f_model
        self.f_var = f_var
        self.f_likelyhood = f_likelyhood
        self.f_hard = f_hard
        
        # Prior parameters for transfer learning
        self.params_prior = params_prior
        self.g = g if g is not None else 1.0
        
        # External functions
        self.adjust_param_step_fn = adjust_param_step_fn or adjust_param_step
        self.check_bounds_fn = check_bounds_fn or check_bounds
        self.check_constraints_fn = check_constraints_fn or check_constraints
        self.accept_reject_step_fn = accept_reject_step_fn or accept_reject_step_target
        self.setup_logging_fn = setup_logging_fn or setup_logging
        self.save_state_fn = save_state_fn or save_state
        self.run_mcmc_fn = run_mcmc_fn or run_mcmc_target
        
        # Set parameters
        self.set_params(params)
        
        # Initialize
        self._savefolder = savefolder
        self.reset()

    def set_params(self, params):
        """
        Configure parameter settings including bounds, step sizes, and descriptions.
        Sets default step sizes (10% of initial value) and descriptions if not provided.
        """
        self._params = params.copy()
        if "param step" not in self._params.columns:
            self._params["param step"] = self._params["param init"].abs() / 10
        if "description" not in self._params.columns:
            self._params["description"] = self._params.index

    def savefolder_property(self):
        """Get save folder path (returns current directory if not set)."""
        if self._savefolder is None:
            return os.getcwd()
        else:
            return self._savefolder

    def set_savefolder(self, value):
        """Set save folder for MCMC outputs (creates directory if needed)."""
        if not os.path.exists(value):
            print("creating", value)
            os.makedirs(value, exist_ok=True)
        self._savefolder = value
    
    def reset(self):
        """
        Reset MCMC state: clears parameter history, iterations (likelihood, prior, posterior), 
        and acceptance counts.
        """
        self._params_estim = pd.DataFrame(columns=self._params.index)
        self._iterations = pd.DataFrame(columns=["likelyhood", "prior", "posterior"], dtype="float64")
        self._Nacc = pd.DataFrame(columns=["Nacc"] + self._params.index.tolist(), dtype=int)

    def to_pickle(self, fname=None):
        """Save MCMC object to pickle file for later resumption or analysis."""
        if fname is None:
            fname = os.path.join(self.savefolder_property(), "MCMC_Target.pickle")
            print(f"Saving {self.name} model to {fname}")
        with open(fname, "wb") as f:
            dump(self, f)

    @classmethod
    def read_pickle(cls, fname):
        """Load MCMC object from pickle file to resume or analyze a previous run."""
        with open(fname, "rb") as f:
            model = load(f)
        if not isinstance(model, cls):
            raise IOError(f"{fname} is not a valid {str(cls)} pickle file")
        return model

    def _check_bounds(self, params_new, params_current, i_param, i, upper_bound, lower_bound, verbose, log_file):
        """Check parameter bounds using external function"""
        return self.check_bounds_fn(params_new, params_current, i_param, i, upper_bound, lower_bound,
                                   self._params.index, verbose, log_file)

    def _check_constraints(self, params_new, params_current, i_param, i, verbose, log_file):
        """Check hard constraints using external function"""
        return self.check_constraints_fn(self.X, params_new, params_current, i_param, i, self.f_hard,
                                        self._params.index, verbose, log_file)

    def _calculate_likelihood(self, params_new):
        """Calculate likelihood for new parameters"""
        try:
            VAR_mdl_new = self.f_var(self.Y, params_new)
            self.Y_sim = self.f_model(self.X, params_new)
            likelihood_new = self.f_likelyhood(self.Y, self.Y_sim, VAR_mdl_new)
            return likelihood_new, VAR_mdl_new
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as err:
            print(str(err.__class__), str(err), "when calculate new loglikelihood, step ignored")
            return np.nan, None

    def _accept_reject_step(self, params_new, params_current, likelihood_new, likelihood_current, 
                           prior_new, prior_current, i_param, i, verbose, log_file):
        """Metropolis-Hastings accept/reject step using external function (with prior)"""
        return self.accept_reject_step_fn(params_new, params_current, likelihood_new, likelihood_current,
                                         prior_new, prior_current, i_param, i, self._params.index, verbose, log_file)

    def _setup_logging(self, log_file, i, quiet):
        """Setup logging using external function"""
        return self.setup_logging_fn(log_file, i, quiet, self.name)

    def run(self, init_params=None, iteration=np.inf, max_time=np.inf, verbose=False, 
            quiet=False, save_state=False, log_file="mcmc_target.log", adjust_par_step=False):
        """Run MCMC using external function"""
        return self.run_mcmc_fn(self, init_params, iteration, max_time, verbose, 
                               quiet, save_state, log_file, adjust_par_step)

    def save_state(self, file_format="csv"):
        """Save MCMC state using external function"""
        self.save_state_fn(self._params_estim, self._iterations, self._Nacc, 
                          self.savefolder_property(), file_format)

    def calc_likelyhood(self, params):
        """
        Calculate log-likelihood for given parameters.
        Used for initialization and can be called independently for testing.
        """
        self.Y_sim = self.f_model(self.X, params)
        return self.f_likelyhood(self.Y, self.Y_sim, self.f_var(self.Y, params))
    
    def calc_prior(self, params):
        """
        Calculate log-prior probability from source domain parameter distribution.
        Implements scaled multivariate normal prior for Hierarchical Bayesian Transfer Learning.
        
        The prior is constructed from source MCMC samples (params_prior) as:
        1. Parameter-wise standardization by source posterior std deviation
        2. Multivariate normal with covariance scaled by hyperparameter g
        3. Higher g = weaker prior (more reliance on target data)
        
        Args:
            params (list): Parameter values to evaluate prior probability.
        
        Returns:
            float: Log-prior probability. Returns 0.0 if no prior specified.
        """
        if self.params_prior is None or self.g is None:
            return 0.0
        
        param_names = list(self._params.index)
        
        if hasattr(self.params_prior, 'columns'):
            available_params = [p for p in param_names if p in self.params_prior.columns]
        else:
            available_params = param_names[:self.params_prior.shape[1]]
        
        if len(available_params) == 0:
            return 0.0
        
        param_indices = [i for i, p in enumerate(param_names) if p in available_params]
        params_with_prior = np.array(params)[param_indices]
        
        if hasattr(self.params_prior, 'columns'):
            prior_data = self.params_prior[available_params].values
        else:
            prior_data = self.params_prior[:, :len(available_params)]
            
        param_scale = np.sqrt(np.diag(np.cov(prior_data.T)))
        param_scale = np.where(param_scale == 0, 1, param_scale)
        
        params_scaled = params_with_prior / param_scale
        params_prior_scaled = prior_data / param_scale
        cov_prior_scaled = np.cov(params_prior_scaled.T)
        mean_prior_scaled = np.mean(params_prior_scaled, axis=0)
        
        cov_scaled = cov_prior_scaled * self.g
        
        return multivariate_normal.logpdf(params_scaled, mean=mean_prior_scaled, cov=cov_scaled, allow_singular=True)
    
####################################################################################################################################################################
def run_bayesian_method_unified(method, iteration, fold_dir, base_index, train_size, fold_idx=None,
                               df_train=None, df_val=None, df_test=None, params_init=None, sigma_value=None,
                               g=None, params_prior=None, f_model=None, f_hard=None, f_var=None, f_likelyhood=None,
                               target_column="N_simul", index_name="base_index"):
    """
    Unified function for Bayesian MCMC methods.
    
    Parameters:
    -----------
    method : str
        'MHwG_source': Source data calibration
        'HBTL': Hierarchical Bayesian Transfer Learning (uses prior from source)
        'MHwG': Metropolis-Hastings without prior
        'MHwG_init': MH with source initialization (no prior in likelihood)
    iteration : int
        Number of MCMC iterations (typically 20000)
    fold_dir : str
        Directory to save results
    base_index : int
        Base/seed index
    train_size : int
        Training size (only for target methods)
    fold_idx : int, optional
        Fold index (only for target methods)
    df_train : DataFrame
        Training data
    df_val : DataFrame, optional
        Validation fold data (only for target methods)
    df_test : DataFrame, optional
        Test data (only for target methods)
    params_init : DataFrame
        Initial parameters (already transformed as DataFrame)
    sigma_value : float, optional
        Sigma initialization value (for HBTL and MHwG_init)
    g : int, optional
        Hyperparameter g (only for HBTL)
    params_prior : DataFrame, optional
        Prior parameters from source (only for HBTL)
    f_model : function
        Model function (f_N_source or f_N_target)
    f_hard : function
        Hard constraint function
    f_var : function
        Variance function
    f_likelyhood : function
        Likelihood function
    target_column : str
        Column name for target variable ('N_simul' for simulated, 'Azote_liqTot' for real)
    index_name : str
        Name for index column ('base_index' for simulated, 'seed_index' for real)
    """
    
    os.makedirs(fold_dir, exist_ok=True)
    warnings.filterwarnings('ignore')
    
    if method == 'MHwG_source':
        nte = False
        # SOURCE DATA PROCESSING
        params_mcmc = params_init.copy()
        
        # Add sigma parameter for likelihood covariance: Cov = diag(sigma * y_i)
        # Sigma controls likelihood shape: too small -> over-confident (poor exploration), 
        # too large -> under-confident (data weakly constrains parameters)
        params_mcmc.loc["sigma"] = {"param init": 0.001, "param min": 0, "param max": 3}
        params_mcmc["description"] = params_mcmc.index
        
        # Create MCMC_Source object
        mcmc = MCMC_Source(df_train,
                          df_train[target_column],
                          params=params_mcmc,
                          f_model=f_model,
                          f_hard=f_hard,
                          f_var=f_var,
                          f_likelyhood=f_likelyhood,
                          savefolder=fold_dir)
        
        mcmc.reset()
        
        # Run MCMC
        sys.stdout = open(os.devnull, 'w')
        mcmc.run(adjust_par_step=100, log_file="mcmc_source.log", quiet=True, 
                save_state=True, iteration=iteration)
        sys.stdout = sys.__stdout__
        
        mcmc.save_state()
        
        # Read and clean up files
        name = "param_estim_MCMC_source_" + str(iteration) + "it.csv"
        data = pd.read_csv(fold_dir + "/param_estim_MCMC.csv")
        
        extentions = [".csv", ".pickle", ".log"]
        for file in os.listdir(fold_dir):
            if any(file.endswith(extension) for extension in extentions):
                os.remove(os.path.join(fold_dir, file))
        
        # Save with specific name
        data.to_csv(fold_dir + "/" + name)
    
    else:  # TARGET METHODS: HBTL, MHwG, MHwG_init
        nte = True
        params_mcmc_target = params_init.copy()
        
        # Set sigma based on method
        if method == 'HBTL':
            params_mcmc_target.loc["sigma"] = {"param init": sigma_value, "param min": sigma_value, "param max": sigma_value}
        elif method == 'MHwG':
            params_mcmc_target.loc["sigma"] = {"param init": 0.001, "param min": 0, "param max": 3}
        elif method == 'MHwG_init':
            params_mcmc_target.loc["sigma"] = {"param init": sigma_value, "param min": 0, "param max": 3}
        
        params_mcmc_target["description"] = params_mcmc_target.index
        
        # Create MCMC_Target object based on method
        if method == 'HBTL':
            if g is None or params_prior is None:
                raise ValueError("HBTL requires both 'g' and 'params_prior' parameters")
            mcmc_target = MCMC_Target(df_train, df_train[target_column], 
                                     params=params_mcmc_target, 
                                     f_model=f_model, f_hard=f_hard, f_var=f_var, 
                                     f_likelyhood=f_likelyhood, 
                                     g=g, params_prior=params_prior, 
                                     savefolder=fold_dir)
        else:  # MHwG or MHwG_init
            mcmc_target = MCMC_Target(df_train, df_train[target_column], 
                                     params=params_mcmc_target, 
                                     f_model=f_model, f_hard=f_hard, f_var=f_var, 
                                     f_likelyhood=f_likelyhood, 
                                     savefolder=fold_dir)
        
        mcmc_target.reset()
        
        # Run MCMC
        sys.stdout = open(os.devnull, 'w')
        mcmc_target.run(adjust_par_step=100, log_file="mcmc_target.log", quiet=True, 
                       save_state=True, iteration=iteration)
        sys.stdout = sys.__stdout__
        
        mcmc_target.save_state()
        
        # Read and clean up files
        data = pd.read_csv(fold_dir + "/param_estim_MCMC.csv")
        extentions = [".pickle", ".log", 'csv']
        for file in os.listdir(fold_dir):
            if any(file.endswith(extension) for extension in extentions):
                os.remove(os.path.join(fold_dir, file))
        
        # Parameter estimation from last 1000 iterations
        data_infer = data.copy()
        data_infer = data_infer.iloc[19000:, :]
        data_infer = data_infer.drop(data_infer.columns[12], axis=1)
        params_estim = list(data_infer.mean())
        params_target_estim = params_transform(params_estim, nte)
        
        # Predictions
        y_val = df_val[target_column]
        y_test = df_test[target_column]
        
        y_pred_val = output_data(df_val, params_target_estim, nte=True, N=True)
        y_pred_test = output_data(df_test, params_target_estim, nte=True, N=True)
        
        # Metrics
        met_val = np.mean((y_val - y_pred_val)**2 / y_val)
        mae_test = np.mean(np.abs(y_test - y_pred_test))
        
        # Save results based on method
        if method == 'HBTL':
            columns = [index_name, "train_size", "g", "fold", "val_loss", "MAE_test"]
            frame_results = pd.DataFrame(columns=columns)
            frame_results.loc[0] = [base_index, train_size, g, fold_idx, met_val, mae_test]
        else:  # MHwG or MHwG_init
            columns = [index_name, "train_size", "fold", "val_loss", "MAE_test"]
            frame_results = pd.DataFrame(columns=columns)
            frame_results.loc[0] = [base_index, train_size, fold_idx, met_val, mae_test]
        
        frame_results.to_csv(fold_dir + "/frame_results.csv", index=False, header=True)

####################################################################################################################################################################
def run_lbfgsb_method(method, fold_dir, base_index, train_size, fold_idx,
            df_train, df_val, df_test, params_init, bounds, loss_function,
            target_column="N_simul", index_name="base_index"):
    """
    Unified function for L-BFGS-B optimization methods.
    
    Parameters:
    -----------
    method : str
        'LBFGSB' or 'LBFGSB_init'
    fold_dir : str
        Directory to save results
    base_index : int
        Base/seed index
    train_size : int
        Training size
    fold_idx : int
        Fold index
    df_train : DataFrame
        Training fold data
    df_val : DataFrame
        Validation fold data
    df_test : DataFrame
        Test data
    params_init : list or list of lists
        For LBFGSB: list of 1000 initialization arrays
        For LBFGSB_init: single initialization array
    bounds : list of tuples
        Parameter bounds for optimization
    loss_function : function
        Loss function to minimize
    target_column : str
        Column name for target variable ('N_simul' for simulated, 'Azote_liqTot' for real)
    index_name : str
        Name for index column ('base_index' for simulated, 'seed_index' for real)
    """
    
    os.makedirs(fold_dir, exist_ok=True)
    warnings.filterwarnings('ignore')
    
    y_val = df_val[target_column]
    y_test = df_test[target_column]
    
    if method == 'LBFGSB':
        # LBFGSB: Loop over 1000 initializations
        columns = ["param_index", index_name, "train_size", "fold", "val_loss", "MAE_test"]
        frame_results = pd.DataFrame(columns=columns)
        
        for i in range(len(params_init)):
            # Optimize using L-BFGS-B
            result = minimize(
                loss_function,
                x0=params_init[i],
                args=(df_train, bounds, target_column),  
                method='L-BFGS-B',
                bounds=bounds
            )
            
            print(f"Optimization {i+1} finished. Success: {result.success}")
            
            params_lbfgsb = params_transform(result.x, nte=True, bounds=bounds)
            
            # Predictions always use "N_simul" in output_data
            y_pred_val = output_data(df_val, params_lbfgsb, nte=True, N=True)
            y_pred_test = output_data(df_test, params_lbfgsb, nte=True, N=True)
            
            met_val = np.mean((y_val - y_pred_val)**2 / y_val)
            mae_test = np.mean(np.abs(y_test - y_pred_test))
            
            frame_results.loc[i] = [i, base_index, train_size, fold_idx, met_val, mae_test]
        
        frame_results.to_csv(fold_dir + "/frame_results.csv", index=False, header=True)
    
    elif method == 'LBFGSB_init':
        # LBFGSB_init: Single optimization with source initialization
        columns = [index_name, "train_size", "fold", "val_loss", "MAE_test"]
        frame_results = pd.DataFrame(columns=columns)
        
        # Optimize using L-BFGS-B
        result = minimize(
            loss_function,
            x0=params_init,  # Single initialization
            args=(df_train, target_column),  
            method='L-BFGS-B',
            bounds=bounds
        ) 
        
        print(f"Optimization finished. Success: {result.success}")
        
        params_lbfgsb = params_transform(result.x, nte=True, bounds=bounds)
        
        # Predictions always use "N_simul" in output_data
        y_pred_val = output_data(df_val, params_lbfgsb, nte=True, N=True)
        y_pred_test = output_data(df_test, params_lbfgsb, nte=True, N=True)
        
        met_val = np.mean((y_val - y_pred_val)**2 / y_val)
        mae_test = np.mean(np.abs(y_test - y_pred_test))
        
        frame_results.loc[0] = [base_index, train_size, fold_idx, met_val, mae_test]
        frame_results.to_csv(fold_dir + "/frame_results.csv", index=False, header=True)

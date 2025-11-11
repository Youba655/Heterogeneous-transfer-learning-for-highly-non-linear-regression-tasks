import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
import copy
import optuna # The package used to optimize the hyperparameters of the proposed methods (MLP-based methods) and their competitors
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from python_files.utils import * 
############################################################################################################################################################################################################
class TransferNN(nn.Module):
    '''
    Neural network for transfer learning with configurable hidden layers.
    
    Args:
        input_dim: Number of input features
        hidden_layers_config: List of integers specifying units in each hidden layer
        output_dim: Number of output features
        dtype: Data type for tensors (default: torch.float64)
    '''
    def __init__(self, input_dim, hidden_layers_config, output_dim, dtype=torch.float64):
        super(TransferNN, self).__init__()
        layers = []
        prev_units = input_dim

        for units in hidden_layers_config:
            layer = nn.Linear(prev_units, units, dtype=dtype)
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_units = units

        # Final output layer
        final_layer = nn.Linear(prev_units, output_dim, dtype=dtype)
        layers.append(final_layer)
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass through the network'''
        x = x.to(next(self.parameters()).dtype)
        return self.model(x)

############################################################################################################################################################################################################
def freeze_last_n_layers(net, n_layers):
    '''
    Freeze the last n Linear layers of the network to prevent training.
    
    Args:
        net: Neural network model
        n_layers: Number of Linear layers to freeze from the end
    
    Returns:
        net: Network with frozen layers
    '''
    linear_layers = [module for module in net.modules() if isinstance(module, nn.Linear)]
    
    for i in range(len(linear_layers) - n_layers, len(linear_layers)):
        if i >= 0:
            for param in linear_layers[i].parameters():
                param.requires_grad = False
    return net

############################################################################################################################################################################################################
def freeze_first_n_layers(net, n_layers):
    '''
    Freeze the first n Linear layers of the network to prevent training.
    
    Args:
        net: Neural network model
        n_layers: Number of Linear layers to freeze from the beginning
    
    Returns:
        net: Network with frozen layers
    '''
    linear_layers = [module for module in net.modules() if isinstance(module, nn.Linear)]
         
    for i in range(min(n_layers, len(linear_layers))):
        for param in linear_layers[i].parameters():
            param.requires_grad = False
    return net

############################################################################################################################################################################################################
def train_NN(model, data_loader, X_val, y_val, 
             optimizer, scheduler, criterion, epochs=500):
    '''
    Train neural network with early stopping based on validation loss.
    
    Args:
        model: Neural network model to train
        data_loader: DataLoader for training data
        X_val: Validation input features
        y_val: Validation target values
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        criterion: Loss function
        epochs: Maximum number of training epochs (default: 500)
    
    Returns:
        model: Trained model with best weights loaded
    '''
    patience = 30
    best_val_loss = float('1e100')
    epochs_no_improve = 0
    
    model.train()

    for epoch in range(epochs):
        for target_data, target_labels in data_loader:
            optimizer.zero_grad()
            target_output = model(target_data)
            loss_target = criterion(target_output, target_labels)
            loss_target.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val).item()
        model.train()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    model.load_state_dict(best_model_wts)
    return model

############################################################################################################################################################################################################
def train_phase_DHTLM(source_net, target_net, source_loader, target_loader, 
                optimizer_source, optimizer_target, scheduler_source, scheduler_target, criterion,
                gamma, N_2, X_val_target, y_val_target, epochs=1000):
    '''
    Train Deep Heterogeneous Transfer Learning Model (DHTLM) with two-phase training.
    
    Phase 1: Joint training of source and target networks with L2 regularization on last N_2 layers
    Phase 2: Fine-tune target network with frozen last N_2 layers and frozen source network
    
    Args:
        source_net: Source neural network
        target_net: Target neural network
        source_loader: DataLoader for source domain data
        target_loader: DataLoader for target domain data
        optimizer_source: Optimizer for source network
        optimizer_target: Optimizer for target network
        scheduler_source: Learning rate scheduler for source network
        scheduler_target: Learning rate scheduler for target network
        criterion: Loss function
        gamma: Regularization coefficient for L2 penalty on layer differences
        N_2: Last N_2 layers to regularize and freeze (with the output layers weights also)
        X_val_target: Validation input features for target domain
        y_val_target: Validation target values for target domain
        epochs: Maximum number of training epochs (default: 1000)
    
    Returns:
        source_net: Trained source network with best weights
        target_net: Trained target network with best weights
    '''
    
    # Early stopping parameters
    patience = 30
    best_val_loss = float('1e100')
    epochs_no_improve = 0
    early_stop = False
    
    # Ensure networks are in training mode
    source_net.train()
    target_net.train()

    for epoch in range(epochs):
        # Phase 1: Joint training with regularization
        for (source_data, source_labels), (target_data, target_labels) in zip(source_loader, target_loader):
            # Zero gradients
            optimizer_source.zero_grad()
            optimizer_target.zero_grad()

            # Forward pass
            source_output = source_net(source_data)
            target_output = target_net(target_data)

            # Compute losses
            loss_source = criterion(source_output, source_labels)
            loss_target = criterion(target_output, target_labels)

            # Regularization on last N_2 layers
            source_params = list(source_net.parameters())
            target_params = list(target_net.parameters())

            l2_reg = sum(
                torch.norm(source_params[-(2 * i)] - target_params[-(2 * i)], p=2) ** 2
                + torch.norm(source_params[-(2 * i) + 1] - target_params[-(2 * i) + 1], p=2) ** 2
                for i in range(1, N_2 + 1)
            )

            # Total loss
            total_loss = loss_source + loss_target + (gamma / 2) * l2_reg 
            total_loss.backward()

            # Optimizer steps
            optimizer_source.step()
            optimizer_target.step()
        scheduler_source.step()
        scheduler_target.step()

        # Phase 2: Fine-tune target network with frozen layers
        target_net.train() 
        
        # Freeze source network parameters
        for param in source_net.parameters():
            param.requires_grad = False
        
        # Freeze last N_2 layers of the target network
        target_params = list(target_net.parameters())
        for i, param in enumerate(target_params):
            if i >= len(target_params) - (2 * N_2):
                param.requires_grad = False
                
        # Train only on target data
        for target_data, target_labels in target_loader:
            optimizer_source.zero_grad()
            optimizer_target.zero_grad()

            target_output = target_net(target_data)
            loss_target_phase2 = criterion(target_output, target_labels)

            loss_target_phase2.backward()
            optimizer_target.step()
        
        # Unfreeze all parameters
        for param in source_net.parameters():
            param.requires_grad = True
        
        for param in target_net.parameters():
            param.requires_grad = True
        
        # Validation
        target_net.eval()
        with torch.no_grad():
            outputs_val = target_net(X_val_target)
            val_loss = criterion(outputs_val, y_val_target).item()
        target_net.train()
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            source_model_wts = copy.deepcopy(source_net.state_dict())
            best_model_wts = copy.deepcopy(target_net.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                early_stop = True
                break
    
    source_net.load_state_dict(source_model_wts)
    target_net.load_state_dict(best_model_wts)
    return source_net, target_net
############################################################################################################################################################################################################
def initialize_target_from_source(source_net, target_net):
    '''
    Initialize target network weights from source network with transfer learning.
    
    For the first layer: copies weights for shared features and initializes 
    additional features with Kaiming normal initialization.
    For other layers: directly copies all weights and biases.
    
    Args:
        source_net: Pre-trained source network
        target_net: Target network to initialize
    
    Returns:
        target_net: Initialized target network
    '''
    with torch.no_grad():
        for i, (source_layer, target_layer) in enumerate(zip(source_net.model, target_net.model)):
            if isinstance(source_layer, nn.Linear) and isinstance(target_layer, nn.Linear):
                if i == 0:  # First layer handling (different input size)
                    target_layer.weight[:, :-1].copy_(source_layer.weight)
                    nn.init.kaiming_normal_(target_layer.weight[:, -1:])  # Only extra column
                    target_layer.bias.copy_(source_layer.bias)
                else:
                    target_layer.weight.copy_(source_layer.weight)
                    target_layer.bias.copy_(source_layer.bias)

    return target_net
############################################################################################################################################################################################################
def freeze_first_n_layers_keep_tire(net, n_layers, tire_feature_idx=7):
    """
    Freeze the first n_layers but ALWAYS keep Tire input weights trainable
    
    Args:
        net: Neural network
        n_layers: Number of layers to freeze from the beginning
        tire_feature_idx: Python index of the Tire feature (7, corresponding to the 8th feature)
    """
    linear_layers = [module for module in net.modules() if isinstance(module, nn.Linear)]
    
    for i in range(min(n_layers, len(linear_layers))):
        layer = linear_layers[i]
        
        if i == 0:  # First layer - special handling to keep Tire weights trainable
            # Enable gradients for first layer weights
            layer.weight.requires_grad = True
            
            # Create hook to keep Tire weights trainable and freeze others
            def selective_grad_hook(grad):
                grad_masked = torch.zeros_like(grad)
                grad_masked[:, tire_feature_idx] = grad[:, tire_feature_idx]  # Keep only Tire gradients
                return grad_masked
            
            layer.weight.register_hook(selective_grad_hook)
            layer.bias.requires_grad = False
        else:
            # Freeze all parameters in subsequent layers
            for param in layer.parameters():
                param.requires_grad = False
    
    return net
############################################################################################################################################################################################################
def objective_unified(trial, method, X_train_source, y_train_source, X_val_source, y_val_source, 
                     X_test_source, y_test_source, X_train_target, y_train_target, X_test_target, 
                     y_test_target, criterion, train_dir, base_index, train_size, index_name, device):
    """
    Unified objective function for all methods.
    
    Parameters:
    -----------
    method: str
        One of: 'DHTLM', 'DHTLFT', 'MLP', 'MLP_init', 'FTLL', 'FTLL_tire'
    """
    
    best_cv_loss = float('inf')
    
    # Tune hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    unit_options = [16, 32, 64, 128, 256]
    hidden_layers_config = [
        trial.suggest_categorical(f'hidden_layer_{i}_units', unit_options) 
        for i in range(n_layers)
    ]
    
    # Method-specific hyperparameters
    if method == 'DHTLM':
        gamma_TL = trial.suggest_categorical('gamma_TL', [1, 10, 100, 1000, 10000, 100000])
        lamda_source = trial.suggest_categorical('lamda_source', [0.99, 0.999])
        lamda_target = trial.suggest_categorical('lamda_target', [0.99, 0.999])
        lr_source = trial.suggest_categorical('lr_source', [1e-2, 1e-3, 1e-4])
        lr_target = trial.suggest_categorical('lr_target', [1e-2, 1e-3, 1e-4])
        N_2 = trial.suggest_int('N_2', 1, n_layers)
    elif method in ['DHTLFT', 'MLP_init', 'FTLL', 'FTLL_tire']:
        lamda_target = trial.suggest_categorical('lamda_target', [0.99, 0.999])
        lr_target = trial.suggest_categorical('lr_target', [1e-2, 1e-3, 1e-4])
        if method == 'DHTLFT':
            N_2 = trial.suggest_int('N_2', 1, n_layers)
        elif method in ['FTLL', 'FTLL_tire']:
            N_1 = trial.suggest_int('N_1', 1, n_layers)
    elif method == 'MLP':
        lamda_target = trial.suggest_categorical('lamda_target', [0.99, 0.999])
        lr_target = trial.suggest_categorical('lr_target', [1e-2, 1e-3, 1e-4])
    
    torch.manual_seed(0)
    
    # Train source network if needed (all methods except MLP)
    source_net_init = None
    if method != 'MLP':
        source_net_init = TransferNN(
            input_dim=X_train_source.shape[1], 
            hidden_layers_config=hidden_layers_config, 
            output_dim=1,
            dtype=torch.float64).to(device)
        source_dataset_train_init = TensorDataset(X_train_source, y_train_source)
        source_loader_init = DataLoader(source_dataset_train_init, batch_size=64, shuffle=False)
        optimizer_source_init = optim.Adam(source_net_init.parameters(), lr=0.01)
        scheduler_source_init = optim.lr_scheduler.ExponentialLR(optimizer_source_init, gamma=0.999)
        
        source_net_init = train_NN(source_net_init, source_loader_init, X_val_source, y_val_source, 
                                   optimizer_source_init, scheduler_source_init, criterion, epochs=500)
        
        source_net_init.eval()
        with torch.no_grad():
            source_train_outputs = source_net_init(X_train_source)
            source_val_outputs = source_net_init(X_val_source)
            source_test_outputs = source_net_init(X_test_source)
            source_train_loss = MAE(source_train_outputs, y_train_source).item()
            source_val_loss = MAE(source_val_outputs, y_val_source).item()
            source_test_loss = MAE(source_test_outputs, y_test_source).item()
            print(f"Source init Train MAE = {source_train_loss:.6f}, Source init Val MAE = {source_val_loss:.6f}, Source init Test MAE = {source_test_loss:.6f}")
    
    # Cross-validation setup
    fold_size = size_fold(train_size)
    k_folds = int(len(X_train_target)/fold_size)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    
    list_val_target_score = []
    list_test_target_mae = []
    columns = [index_name, "train_size", "MAE_test"]
    frame_results = pd.DataFrame(columns=columns)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_target)):
        # Get fold data
        X_train_target_fold = X_train_target[train_idx]
        y_train_target_fold = y_train_target[train_idx]
        X_val_target_fold = X_train_target[val_idx]
        y_val_target_fold = y_train_target[val_idx]
        
        # Normalize target data for this fold
        target_scaler = StandardScaler()
        X_train_target_fold_norm = target_scaler.fit_transform(X_train_target_fold)
        X_val_target_fold_norm = target_scaler.transform(X_val_target_fold)
        
        # Convert to tensors
        X_train_target_fold_norm = torch.tensor(X_train_target_fold_norm, dtype=torch.float64).to(device)
        X_val_target_fold_norm = torch.tensor(X_val_target_fold_norm, dtype=torch.float64).to(device)
        y_train_target_fold = torch.tensor(y_train_target_fold, dtype=torch.float64).reshape(-1, 1).to(device)
        y_val_target_fold = torch.tensor(y_val_target_fold, dtype=torch.float64).reshape(-1, 1).to(device)
        
        # DHTLM specific: create source network for this fold
        if method == 'DHTLM':
            source_net = TransferNN(
                input_dim=X_train_source.shape[1], 
                hidden_layers_config=hidden_layers_config, 
                output_dim=1,
                dtype=torch.float64).to(device)
            source_net.load_state_dict(copy.deepcopy(source_net_init.state_dict()))
        
        # Create target network
        target_net = TransferNN(
            input_dim=X_train_target_fold_norm.shape[1], 
            hidden_layers_config=hidden_layers_config, 
            output_dim=1,
            dtype=torch.float64).to(device)
        
        # Initialize target network from source (all methods except MLP)
        if method != 'MLP':
            target_net = initialize_target_from_source(source_net_init, target_net)
        
        # Apply freezing based on method
        if method == 'DHTLFT':
            target_net = freeze_last_n_layers(target_net, N_2)
        elif method == 'FTLL':
            target_net = freeze_first_n_layers(target_net, N_1)
        elif method == 'FTLL_tire':
            target_net = freeze_first_n_layers_keep_tire(target_net, N_1, tire_feature_idx=7)
        
        # Create target data loader
        target_dataset_train = TensorDataset(X_train_target_fold_norm, y_train_target_fold)
        target_loader = DataLoader(target_dataset_train, batch_size=2, shuffle=False)
        
        # Method-specific training
        if method == 'DHTLM':
            # Create source data loader
            source_dataset_train = TensorDataset(X_train_source, y_train_source)
            num_batches_target = len(target_loader)
            batch_size_source = len(source_dataset_train) // num_batches_target
            if len(source_dataset_train) % num_batches_target != 0:
                batch_size_source += 1
            source_loader = DataLoader(source_dataset_train, batch_size=batch_size_source, shuffle=False)
            
            # Optimizers for both networks
            optimizer_source = optim.Adam(source_net.parameters(), lr=lr_source)
            optimizer_target = optim.Adam(target_net.parameters(), lr=lr_target)
            scheduler_source = optim.lr_scheduler.ExponentialLR(optimizer_source, gamma=lamda_source)
            scheduler_target = optim.lr_scheduler.ExponentialLR(optimizer_target, gamma=lamda_target)
            
            # Train transfer learning model
            source_net, target_net = train_phase_DHTLM(source_net, target_net, source_loader, target_loader,
                                                       optimizer_source, optimizer_target, scheduler_source, 
                                                       scheduler_target, criterion, gamma_TL, N_2, 
                                                       X_val_target_fold_norm, y_val_target_fold, epochs=500)
        else:
            # All other methods: train target network only
            optimizer_target = optim.Adam(target_net.parameters(), lr=lr_target)
            scheduler_target = optim.lr_scheduler.ExponentialLR(optimizer_target, gamma=lamda_target)
            
            target_net = train_NN(target_net, target_loader, X_val_target_fold_norm, y_val_target_fold,
                                 optimizer_target, scheduler_target, criterion, epochs=500)
        
        # Evaluate on validation and test sets
        target_net.eval()
        with torch.no_grad():
            X_test_target_norm = target_scaler.transform(X_test_target)
            X_test_target_norm = torch.tensor(X_test_target_norm, dtype=torch.float64).to(device)
            y_test_target_tensor = torch.tensor(y_test_target, dtype=torch.float64).reshape(-1, 1).to(device)
            
            target_outputs_val = target_net(X_val_target_fold_norm)
            target_outputs_test = target_net(X_test_target_norm)
            
            MAE_test = MAE(target_outputs_test, y_test_target_tensor).item()
            criterion_val = criterion(target_outputs_val, y_val_target_fold).item()
            
            list_val_target_score.append(criterion_val)
            list_test_target_mae.append(MAE_test)
    
    # Calculate mean losses
    mean_cv_loss = np.mean(list_val_target_score)
    mean_test_loss = np.mean(list_test_target_mae)
    
    print(f"Trial {trial.number}: CV Loss = {mean_cv_loss:.6f}, Target Test MAE = {mean_test_loss:.6f}")
    
    # Check if this is the best model across all trials
    if mean_cv_loss < best_cv_loss:
        frame_results.loc[0] = [base_index, train_size, mean_test_loss]
        frame_results.to_csv(os.path.join(train_dir, "frame_results.csv"), index=False, header=True)
        best_cv_loss = mean_cv_loss
    
    return mean_cv_loss
############################################################################################################################################################################################################

def run_optimization_unified(method, X_train_source, y_train_source, X_val_source, y_val_source,
                            X_test_source, y_test_source, X_train_target, y_train_target, X_test_target, 
                            y_test_target, criterion, train_dir, base_index, train_size, index_name, device):
    """
    Unified run_optimization function for all methods.
    
    Parameters:
    -----------
    method : str
        One of: 'DHTLM', 'DHTLFT', 'MLP', 'MLP_init', 'FTLL', 'FTLL_tire'
    """
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
    
    def objective_with_data(trial):
        return objective_unified(trial, method, X_train_source, y_train_source, X_val_source, y_val_source,
                                X_test_source, y_test_source, X_train_target, y_train_target, X_test_target, 
                                y_test_target, criterion, train_dir, base_index, train_size, index_name, device)
    
    def early_stopping_callback(study, trial):
        if study.best_trial.number < trial.number - 200:
            study.stop()
    
    study.optimize(objective_with_data, n_trials=500, callbacks=[early_stopping_callback])
    
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print("Best cross-validation loss:", study.best_value)
    
    return study


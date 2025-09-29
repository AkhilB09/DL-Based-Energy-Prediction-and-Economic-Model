# tuner.py

import os
import sys
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
import optuna
from torch.utils.data import DataLoader

# Import all necessary components
from config import SEED, OPTUNA_N_TRIALS, N_CV_SPLITS_FOR_TUNING, TRAIN_RATIO, \
                   INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, BATCH_SIZE, QUICK_TRAIN_EPOCHS 
from utils import SimpleLSTM, TimeSeriesDataset, create_sequences, load_and_prepare_data 


# ==============================================================================
# --- Environment Setup (Must be run first) ---
# ==============================================================================
random.seed(SEED); 
np.random.seed(SEED); 
torch.manual_seed(SEED)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(SEED)
    
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# --- OPTUNA OBJECTIVE FUNCTION ---
# ==============================================================================

def optuna_objective(trial, X_train_r, Y_train_r, X_val_r, Y_val_r):
    """
    Optuna objective function to train and evaluate a SimpleLSTM model.
    """
    # 1. Suggest Hyperparameters
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 192, 256]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }

    # 2. Setup Model, Optimizer, Loss
    model = SimpleLSTM(X_train_r.shape[-1], params['hidden_dim'], OUTPUT_SEQ_LEN, params['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = nn.MSELoss()
    
    train_loader = DataLoader(TimeSeriesDataset(X_train_r, Y_train_r), batch_size=BATCH_SIZE, shuffle=True)

    # 3. Training Loop (Quick Train)
    model.train()
    for _ in range(QUICK_TRAIN_EPOCHS):
        for x_b, y_b in train_loader:
            output = model(x_b.to(device))
            loss = loss_fn(output, y_b.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4. Validation
    val_loader = DataLoader(TimeSeriesDataset(X_val_r, Y_val_r), batch_size=BATCH_SIZE)
    model.eval(); 
    val_loss = 0
    with torch.no_grad():
        for x_v, y_v in val_loader:
             if len(y_v) > 0: 
                 val_loss += loss_fn(model(x_v.to(device)), y_v.to(device)).item()

    return val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

# ==============================================================================
# --- MAIN EXECUTION: FINDING BEST PARAMETERS ---
# ==============================================================================
def main():
    # Load data with lag features (Tuner uses richer features for clustering)
    X_raw, Y_raw, cluster_map, _, _, _, _ = load_and_prepare_data(include_lag_features=True)

    cv_splits_list = list(TimeSeriesSplit(n_splits=N_CV_SPLITS_FOR_TUNING).split(X_raw))
    train_val_indices, _ = cv_splits_list[0] 

    best_params_found = {}

    print(f"\n{'='*70}\n--- Starting Hyperparameter Search for each Cluster ---\n{'='*70}")
    
    def reshape_data_for_training(X_seq, Y_seq): 
        X_reshaped = X_seq.transpose(0,2,1,3).reshape(-1, INPUT_SEQ_LEN, X_seq.shape[-1])
        Y_reshaped = Y_seq.transpose(0,2,1,3).reshape(-1, OUTPUT_SEQ_LEN, 1)
        return X_reshaped, Y_reshaped

    for group_id, station_indices in cluster_map.items():
        if len(station_indices) < 10 and group_id != -1: 
            print(f"\n--- Skipping tuning for Cluster {group_id} (too few stations: {len(station_indices)}) ---")
            continue

        group_name = f"Cluster {group_id}" if group_id != -1 else "Outliers (-1)"
        print(f"\n--- Tuning {group_name} ({len(station_indices)} stations) ---")

        # A. Prepare Data for this specific cluster
        X_group, Y_group = X_raw[:, station_indices, :], Y_raw[:, station_indices, :]
        X_tv, Y_tv = X_group[train_val_indices], Y_group[train_val_indices]

        scaler_X = StandardScaler().fit(X_tv.reshape(-1, X_tv.shape[-1]))
        scaler_Y = StandardScaler().fit(Y_tv.reshape(-1, 1))
        X_tv_s = scaler_X.transform(X_tv.reshape(-1, X_tv.shape[-1])).reshape(X_tv.shape)
        Y_tv_s = scaler_Y.transform(Y_tv.reshape(-1, 1)).reshape(Y_tv.shape)

        train_len = int(len(X_tv_s) * TRAIN_RATIO)
        
        X_train_seq, Y_train_seq = create_sequences(X_tv_s[:train_len], Y_tv_s[:train_len], INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
        X_val_seq, Y_val_seq = create_sequences(X_tv_s[train_len:], Y_tv_s[train_len:], INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)

        X_train_r, Y_train_r = reshape_data_for_training(X_train_seq, Y_train_seq)
        X_val_r, Y_val_r = reshape_data_for_training(X_val_seq, Y_val_seq)
        
        if X_val_r.size == 0:
             print(f"--- Warning: Validation set is empty for {group_name}. Skipping tuning. ---")
             continue


        # B. Run Optuna Study
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: optuna_objective(trial, X_train_r, Y_train_r, X_val_r, Y_val_r),
                       n_trials=OPTUNA_N_TRIALS,
                       n_jobs=-1) 

        best_params = study.best_params
        best_params_found[group_id] = best_params
        print(f"\n✅ Best params for {group_name}: {best_params}")

    print(f"\n\n{'='*70}\n--- HYPERPARAMETER SEARCH COMPLETE ---\n{'='*70}")
    
    print("✅ Suggested Dictionary to update the 'TUNED_PARAMS_PER_CLUSTER' in config.py:")
    
    print("\nTUNED_PARAMS_PER_CLUSTER = {")
    for group_id, params in best_params_found.items():
        rounded_params = {
            'lr': round(params['lr'], 6),
            'hidden_dim': params['hidden_dim'],
            'dropout': round(params['dropout'], 4)
        }
        group_name = "Outliers (-1)" if group_id == -1 else f"Cluster {group_id}"
        print(f"    # {group_name}")
        print(f"    {group_id}: {rounded_params},")
        
    print("\n    # Fallback default parameters for the Global model")
    print("    'default': {'lr': 0.0025, 'hidden_dim': 128, 'dropout': 0.20}")
    print("}")

if __name__ == '__main__':
    main()
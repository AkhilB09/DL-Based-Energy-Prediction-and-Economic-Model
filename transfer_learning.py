# transfer_learning.py

# PURPOSE: Loads the pre-trained 'Global' LSTM model weights, fine-tunes them
#          on individual cluster data, and measures performance.

import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
import joblib
from torch.utils.data import DataLoader

# Import components from project files
from config import SEED, N_CV_SPLITS, BATCH_SIZE, EARLY_STOPPING_PATIENCE, \
                   INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, TUNED_PARAMS_PER_CLUSTER, \
                   MODELS_DIR, FINETUNED_MODELS_DIR, FINETUNED_REPORTS_DIR, TRAIN_RATIO
                   
from utils import SimpleLSTM, TimeSeriesDataset, create_sequences, \
                  calculate_multistep_metrics, load_and_prepare_data, train_final_model as fine_tune_model_base


print("✅ Starting Transfer Learning Experiment Cell (V2)...")


FINETUNE_PARAMS = {'lr': 0.0001, 'epochs': 15}

# Directories (ORIGINAL_MODELS_DIR is defined as MODELS_DIR in config.py)
ORIGINAL_MODELS_DIR = MODELS_DIR

if not os.path.exists(FINETUNED_MODELS_DIR): os.makedirs(FINETUNED_MODELS_DIR)
if not os.path.exists(FINETUNED_REPORTS_DIR): os.makedirs(FINETUNED_REPORTS_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOBAL_MODEL_PARAMS = TUNED_PARAMS_PER_CLUSTER['default']
print(f"Using device: {device}")


def fine_tune_model(model, params, X_train, Y_train, X_val, Y_val, model_path):
    """
    Wrapper to call the base training function with fine-tuning specific params.
    """
    # Use the same logic as train_final_model but with finetuning params/epochs
    params_with_epochs = {**params, 'epochs': params.get('epochs', 15)}
    fine_tune_model_base(
        SimpleLSTM, # Must pass the class for instantiation inside the function
        params_with_epochs, 
        X_train, Y_train, X_val, Y_val, 
        model_path, 
        device
    )

def main():
    # Load data without lag features (MUST match main.py)
    print("\n--- STAGE 1: Re-loading Data and Cluster Definitions ---")
    X_raw, Y_raw, cluster_map, _, _, _, _ = load_and_prepare_data(include_lag_features=False) 
    print(f"Data loaded successfully. X_raw shape: {X_raw.shape}")

    print(f"\n{'='*70}\n--- STAGE 2: Running Transfer Learning (Fine-Tuning) Experiment ---\n{'='*70}")
    cv_splits_list = list(TimeSeriesSplit(n_splits=N_CV_SPLITS).split(X_raw))
    transfer_learn_results = {}

    def reshape_data_for_training(X_seq, Y_seq): 
        X_reshaped = X_seq.transpose(0,2,1,3).reshape(-1, INPUT_SEQ_LEN, X_seq.shape[-1])
        Y_reshaped = Y_seq.transpose(0,2,1,3).reshape(-1, OUTPUT_SEQ_LEN, 1)
        return X_reshaped, Y_reshaped

    for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits_list):
        print(f"\n-- Fine-Tuning Fold {fold_idx + 1}/{N_CV_SPLITS} --")
        test_seq_len = len(test_indices) - INPUT_SEQ_LEN - OUTPUT_SEQ_LEN + 1
        fold_predictions = np.full((test_seq_len, OUTPUT_SEQ_LEN, X_raw.shape[1], 1), np.nan)
        fold_actuals = np.full_like(fold_predictions, np.nan)

        for group_id, station_indices in cluster_map.items():
            if len(station_indices) == 0: continue
            group_name = f"Cluster {group_id}"

            X_group, Y_group = X_raw[:, station_indices, :], Y_raw[:, station_indices, :]
            X_tv, Y_tv = X_group[train_val_indices], Y_group[train_val_indices]
            X_test, Y_test = X_group[test_indices], Y_group[test_indices]

            # Load the pre-calculated scalers 
            try:
                scaler_X_path = os.path.join(ORIGINAL_MODELS_DIR, f"scaler_X_LSTM_(Clustered)_group_{group_id}_fold_{fold_idx}.joblib")
                scaler_Y_path = os.path.join(ORIGINAL_MODELS_DIR, f"scaler_Y_LSTM_(Clustered)_group_{group_id}_fold_{fold_idx}.joblib")
                scaler_X = joblib.load(scaler_X_path)
                scaler_Y = joblib.load(scaler_Y_path)
            except FileNotFoundError:
                print(f"  -> ERROR: Scalers not found for {group_name}. Skipping. (Expected in {ORIGINAL_MODELS_DIR})")
                continue

            # Scale and Sequence Data
            X_tv_s = scaler_X.transform(X_tv.reshape(-1, X_tv.shape[-1])).reshape(X_tv.shape)
            Y_tv_s = scaler_Y.transform(Y_tv.reshape(-1, 1)).reshape(Y_tv.shape)
            X_test_s = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            train_len = int(len(X_tv_s) * TRAIN_RATIO)
            
            X_train_seq, Y_train_seq = create_sequences(X_tv_s[:train_len], Y_tv_s[:train_len], INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
            X_val_seq, Y_val_seq = create_sequences(X_tv_s[train_len:], Y_tv_s[train_len:], INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
            X_test_seq, Y_test_seq_actuals = create_sequences(X_test_s, Y_test, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
            
            X_train_r, Y_train_r = reshape_data_for_training(X_train_seq, Y_train_seq)
            X_val_r, Y_val_r = reshape_data_for_training(X_val_seq, Y_val_seq)
            X_test_r_reshaped, _ = reshape_data_for_training(X_test_seq, Y_test_seq_actuals)

            print(f"  -> Fine-tuning for {group_name}...")
            
            # --- Load Global Pre-trained Weights ---
            num_features = X_train_r.shape[-1]
            model = SimpleLSTM(
                num_features=num_features,
                hidden_dim=GLOBAL_MODEL_PARAMS['hidden_dim'],
                output_seq_len=OUTPUT_SEQ_LEN,
                dropout=GLOBAL_MODEL_PARAMS['dropout']
            ).to(device)

            pretrained_model_path = os.path.join(ORIGINAL_MODELS_DIR, f"LSTM_(Global)_group_Global_fold_{fold_idx}.pth")
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"  -> Loaded pre-trained weights from Global model (Fold {fold_idx+1})")
            
            # --- Set Fine-Tuning Params and Execute ---
            params_for_cluster = TUNED_PARAMS_PER_CLUSTER.get(group_id, TUNED_PARAMS_PER_CLUSTER['default'])
            training_params = {**params_for_cluster, **FINETUNE_PARAMS}
            
            model_path = os.path.join(FINETUNED_MODELS_DIR, f"LSTM_(Transfer_Learned)_group_{group_id}_fold_{fold_idx}.pth")
            full_X_train_r, full_Y_train_r = np.vstack([X_train_r, X_val_r]), np.vstack([Y_train_r, Y_val_r])

            fine_tune_model(model, training_params, full_X_train_r, full_Y_train_r, X_val_r, Y_val_r, model_path)

            # --- Prediction ---
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            test_loader = DataLoader(TimeSeriesDataset(X_test_r_reshaped, X_test_r_reshaped), batch_size=BATCH_SIZE)
            preds_s_list = [model(x_t.to(device)).detach().cpu().numpy() for x_t, _ in test_loader]
            preds_s = np.vstack(preds_s_list)
            
            preds_u = scaler_Y.inverse_transform(preds_s.reshape(-1,1)).reshape(preds_s.shape)
            num_seq, num_stations_in_group = X_test_seq.shape[0], X_test_seq.shape[2]
            preds_reshaped = preds_u.reshape(num_seq, num_stations_in_group, OUTPUT_SEQ_LEN, 1).transpose(0, 2, 1, 3)

            fold_predictions[:, :, station_indices, :] = preds_reshaped
            fold_actuals[:, :, station_indices, :] = Y_test_seq_actuals

        multistep_metrics = calculate_multistep_metrics(fold_actuals, fold_predictions)
        transfer_learn_results[fold_idx + 1] = multistep_metrics
        print(f"\n  -> RESULTS FOR FINE-TUNING FOLD {fold_idx+1}:")
        for item in multistep_metrics:
            print(f"     {item['Horizon']}: RMSE={item['RMSE']:.4f}, MAE={item['MAE']:.4f}")

    # --- FINAL REPORTING ---
    print("\n\n✅ --- Fine-Tuning Experiment Finished Successfully --- ✅")
    
    all_results_list = []
    for fold_num, metrics_list in transfer_learn_results.items():
        for metric_dict in metrics_list:
            row = {'Model': 'LSTM (Transfer Learned)', 'Fold': fold_num, **metric_dict}
            all_results_list.append(row)
    
    results_df = pd.DataFrame(all_results_list)
    summary_df = results_df.groupby(['Model', 'Horizon']).mean().drop(columns='Fold')
    summary_df = summary_df.reindex(['15-min', '30-min', '45-min', '60-min'], level='Horizon')

    report_path = os.path.join(FINETUNED_REPORTS_DIR, '04_transfer_learning_summary.csv')
    summary_df.to_csv(report_path)

    print("\n--- FINAL SUMMARY (Mean Performance of Transfer Learning Models) ---")
    print(summary_df.round(4))

if __name__ == '__main__':
    main()
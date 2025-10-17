# main.py

import os
import sys
import time
import random
import warnings
import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
import joblib

#mport
from config import * from utils import SimpleLSTM, TimeSeriesDataset, create_sequences, load_and_prepare_data, \
                  calculate_multistep_metrics, train_final_model, generate_reports, generate_visualizations

# Environmen
for d in [MODELS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    if not os.path.exists(d): os.makedirs(d)
    
random.seed(SEED); 
np.random.seed(SEED); 
torch.manual_seed(SEED)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(SEED)
    
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    X_raw, Y_raw, cluster_map, static_info_df, time_df, station_profiles, cluster_labels = load_and_prepare_data(include_lag_features=False)

    print(f"\n{'='*70}\n--- STAGE 2: Running Final Training with Tuned Hyperparameters ---\n{'='*70}")

    experiments = [
        {'name': 'LSTM (Global)', 'model_class': SimpleLSTM}, 
        {'name': 'LSTM (Clustered)', 'model_class': SimpleLSTM}
    ]
    
    cv_splits_list = list(TimeSeriesSplit(n_splits=N_CV_SPLITS).split(X_raw))
    all_results, final_fold_predictions = {}, {}

    def reshape_data_for_training(X_seq, Y_seq): 
        X_reshaped = X_seq.transpose(0,2,1,3).reshape(-1, INPUT_SEQ_LEN, X_seq.shape[-1])
        Y_reshaped = Y_seq.transpose(0,2,1,3).reshape(-1, OUTPUT_SEQ_LEN, 1)
        return X_reshaped, Y_reshaped

    for exp in experiments:
        print(f"\n--- Running Final Experiment: {exp['name']} ---")
        fold_results_dict = {}
        use_clustering = "Clustered" in exp['name']

        for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits_list):
            print(f"\n-- Fold {fold_idx + 1}/{N_CV_SPLITS} --")
            
            test_seq_len = len(test_indices) - INPUT_SEQ_LEN - OUTPUT_SEQ_LEN + 1
            num_stations = X_raw.shape[1]
            fold_predictions = np.full((test_seq_len, OUTPUT_SEQ_LEN, num_stations, 1), np.nan)
            fold_actuals = np.full_like(fold_predictions, np.nan)
            
            iterator = cluster_map.items() if use_clustering else [('Global', np.arange(num_stations))]

            for group_id, station_indices in iterator:
                if len(station_indices) == 0: continue
                group_name = f"Cluster {group_id}" if use_clustering and group_id != 'Global' else "Global"

               
                X_group, Y_group = X_raw[:, station_indices, :], Y_raw[:, station_indices, :]
                X_tv, Y_tv = X_group[train_val_indices], Y_group[train_val_indices]
                X_test, Y_test = X_group[test_indices], Y_group[test_indices]
                
                
                scaler_X = joblib.dump(StandardScaler().fit(X_tv.reshape(-1, X_tv.shape[-1])),
                                        os.path.join(MODELS_DIR, f"scaler_X_{exp['name'].replace(' ', '_')}_group_{group_id}_fold_{fold_idx}.joblib"))
                scaler_Y = joblib.dump(StandardScaler().fit(Y_tv.reshape(-1, 1)),
                                        os.path.join(MODELS_DIR, f"scaler_Y_{exp['name'].replace(' ', '_')}_group_{group_id}_fold_{fold_idx}.joblib"))
                
                scaler_X = joblib.load(os.path.join(MODELS_DIR, f"scaler_X_{exp['name'].replace(' ', '_')}_group_{group_id}_fold_{fold_idx}.joblib"))
                scaler_Y = joblib.load(os.path.join(MODELS_DIR, f"scaler_Y_{exp['name'].replace(' ', '_')}_group_{group_id}_fold_{fold_idx}.joblib"))

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

                params_for_group = TUNED_PARAMS_PER_CLUSTER.get(group_id, TUNED_PARAMS_PER_CLUSTER['default'])
                print(f"  -> Using params for {group_name}: {params_for_group}")
                
                model_path = os.path.join(MODELS_DIR, f"{exp['name'].replace(' ', '_')}_group_{group_id}_fold_{fold_idx}.pth")
                
                full_X_train_r = np.vstack([X_train_r, X_val_r])
                full_Y_train_r = np.vstack([Y_train_r, Y_val_r])
                
                train_final_model(exp['model_class'], params_for_group, full_X_train_r, full_Y_train_r, X_val_r, Y_val_r, model_path, device)

                model = exp['model_class'](X_test_r_reshaped.shape[-1], params_for_group['hidden_dim'], OUTPUT_SEQ_LEN, params_for_group['dropout']).to(device)
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
            fold_results_dict[fold_idx + 1] = multistep_metrics
            
            print(f"  -> Fold {fold_idx+1} Metrics:")
            for item in multistep_metrics:
                print(f"    {item['Horizon']}: RMSE={item['RMSE']:.4f}, MAE={item['MAE']:.4f}")

            if fold_idx == N_CV_SPLITS - 1: 
                final_fold_predictions[exp['name']] = {'preds': fold_predictions, 'actuals': fold_actuals}

        all_results[exp['name']] = fold_results_dict

    
    generate_reports(all_results, cluster_map, station_profiles, cluster_labels)
    generate_visualizations(final_fold_predictions, static_info_df, cluster_labels, station_profiles)

    print("\n\n✅ --- Definitive Training Pipeline Finished Successfully --- ✅")

if __name__ == '__main__':

    main()

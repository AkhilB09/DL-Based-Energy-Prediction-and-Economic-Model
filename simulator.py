# simulator.py

import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.optimize import minimize, Bounds

# Import components from local project files
from config import * from utils import SimpleLSTM, load_and_prepare_data, run_economic_simulation, generate_bi_dashboard

print("🚀 Definitive Economic Simulator & BI Dashboard (Fine-Tuned Version) Initialized.")
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="viridis")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_simulation_assets():
    """Loads data, re-creates clusters, and loads fine-tuned models/scalers."""
    print("\n--- Loading Simulation Assets for Fine-Tuned Models ---")

    # Load data without lag features (MUST match training)
    X_raw, Y_raw, cluster_map, static_info_df, _, _, cluster_labels = load_and_prepare_data(include_lag_features=False) 
    print("  ✅ Raw data loaded and features engineered.")

    print("  -> Loading pre-trained FINE-TUNED models and scalers...")
    models, scalers_X, scalers_Y = {}, {}, {}
    last_fold_idx = N_CV_SPLITS - 1
    exp_name = 'LSTM_(Transfer_Learned)'
    
    for group_id in cluster_map.keys():
        params = TUNED_PARAMS_PER_CLUSTER['default']
        model_path = os.path.join(FINETUNED_MODELS_DIR, f"{exp_name}_group_{group_id}_fold_{last_fold_idx}.pth")
        
        # Scalers are loaded from the original directory
        scaler_x_path = os.path.join(MODELS_DIR, f"scaler_X_LSTM_(Clustered)_group_{group_id}_fold_{last_fold_idx}.joblib")
        scaler_y_path = os.path.join(MODELS_DIR, f"scaler_Y_LSTM_(Clustered)_group_{group_id}_fold_{last_fold_idx}.joblib")

        if not all(os.path.exists(p) for p in [model_path, scaler_x_path, scaler_y_path]):
            continue

        model = SimpleLSTM(X_raw.shape[-1], params['hidden_dim'], OUTPUT_SEQ_LEN, params['dropout']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[group_id] = model
        scalers_X[group_id] = joblib.load(scaler_x_path)
        scalers_Y[group_id] = joblib.load(scaler_y_path)

    if not models:
        print(f"\n❌ FATAL ERROR: No models were loaded from '{FINETUNED_MODELS_DIR}'. Check paths and run fine-tuning.")
        sys.exit(1)

    print(f"  ✅ Successfully loaded assets for {len(models)} fine-tuned clusters.")
    return X_raw, Y_raw, models, scalers_X, scalers_Y, cluster_map, static_info_df, cluster_labels

if __name__ == '__main__':
    # Step 1: Load all required data, models, and scalers
    X_raw, Y_raw, models, scalers_X, scalers_Y, cluster_map, static_info_df, cluster_labels = load_simulation_assets()

    # Step 2: Run the core economic simulation
    hourly_df, station_df = run_economic_simulation(
        X_raw, Y_raw, models, scalers_X, scalers_Y, cluster_map
    )

    # Step 3: Generate the final dashboard with the simulation results
    if not hourly_df.empty and not station_df.empty:
        generate_bi_dashboard(hourly_df, station_df, static_info_df, cluster_labels)
    else:
        print("\nCould not generate dashboard due to empty simulation results.")

    print("\n✅ --- Definitive Economic Simulation Finished Successfully ---")
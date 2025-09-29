# utils.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize, Bounds
import joblib
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D

# Import global constants from config
try:
    from config import *
except ImportError:
    print("Error: config.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


# ==============================================================================
# --- MODEL DEFINITION ---
# ==============================================================================

class SimpleLSTM(nn.Module):
    """A basic 2-layer LSTM model for sequence-to-sequence prediction."""
    def __init__(self, num_features, hidden_dim, output_seq_len, dropout):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_seq_len)
        
    def forward(self, x_sequence):
        lstm_out, _ = self.lstm(x_sequence)
        return self.fc(lstm_out[:, -1, :]).unsqueeze(-1)
        # Output shape: (batch_size, output_seq_len, 1)

# ==============================================================================
# --- DATA UTILITIES ---
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """Pytorch Dataset for sequence data."""
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

def create_sequences(input_data, target_data, input_seq_len, output_seq_len):
    """Creates look-back sequences (X) and look-ahead targets (Y)."""
    X_seq, Y_seq, T = [], [], input_data.shape[0]
    for i in range(T - input_seq_len - output_seq_len + 1):
        X_seq.append(input_data[i:i + input_seq_len])
        Y_seq.append(target_data[i + input_seq_len:i + input_seq_len + output_seq_len])
    return np.array(X_seq), np.array(Y_seq)

def load_and_prepare_data(include_lag_features=False):
    """
    Loads all raw data, performs feature engineering (with optional lag), 
    and executes DBSCAN clustering.
    """
    print("\n--- Loading Data & Assembling Features ---")
    try:
        # 1. Load Data
        volume_ts = pd.read_csv(VOLUME_TS_PATH).iloc[:, 1:].values.astype(np.float32)
        duration_ts = pd.read_csv(DURATION_TS_PATH).iloc[:, 1:].values.astype(np.float32)
        occupancy_ts = pd.read_csv(OCCUPANCY_TS_PATH).iloc[:, 1:].values.astype(np.float32)
        price_ts = pd.read_csv(PRICE_TS_PATH).iloc[:, 1:].values.astype(np.float32)
        static_info_df = pd.read_csv(STATIC_INFO_PATH)
        time_df = pd.read_csv(TIME_FEATURES_PATH)
        weather_df = pd.read_excel(WEATHER_XLS_PATH)

        # 2. Time Features (Cyclical)
        time_df['datetime'] = pd.to_datetime(time_df.iloc[:, :6])
        time_df['hour_sin'] = np.sin(2 * np.pi * time_df['datetime'].dt.hour / 24.0)
        time_df['hour_cos'] = np.cos(2 * np.pi * time_df['datetime'].dt.hour / 24.0)
        time_features_ts = time_df[['hour_sin', 'hour_cos']].values.astype(np.float32)

        # 3. Weather Features
        weather_df['timestamp'] = pd.to_datetime(weather_df.iloc[:, 0], dayfirst=True)
        weather_df = weather_df.set_index('timestamp')[['T', 'U']].ffill().bfill()
        weather_ts_np = np.repeat(weather_df.apply(pd.to_numeric, errors='coerce').ffill().bfill().values, 4, axis=0).astype(np.float32)

        # 4. Alignment and Truncation
        all_series = [volume_ts, duration_ts, occupancy_ts, price_ts, time_features_ts, weather_ts_np]
        min_len = min(len(s) for s in all_series)
        (volume_ts, duration_ts, occupancy_ts, price_ts, time_features_ts, weather_ts_np) = [s[:min_len] for s in all_series]
        time_df = time_df[:min_len]

        # 5. Feature Stacking (to create X_raw and Y_raw)
        Y_raw = volume_ts[:, :, np.newaxis] # (Time, Stations, 1)

        dynamic_features = np.stack([volume_ts, duration_ts, occupancy_ts, price_ts], axis=-1)
        static_features = static_info_df[['count', 'CBD', 'dynamic_pricing']].values.astype(np.float32)
        static_features_expanded = np.repeat(static_features[np.newaxis, :, :], min_len, axis=0)
        time_features_expanded = np.repeat(time_features_ts[:, np.newaxis, :], static_info_df.shape[0], axis=1)
        weather_features_expanded = np.repeat(weather_ts_np[:, np.newaxis, :], static_info_df.shape[0], axis=1)
        
        # Base Features
        X_raw_components = [dynamic_features, static_features_expanded, time_features_expanded, weather_features_expanded]

        # Optional Lag/Rolling Features (used only in tuner.py if needed, but not final run)
        if include_lag_features:
            LAG_STEPS = 96
            ROLLING_WINDOW_STEPS = 12
            volume_lagged = np.roll(volume_ts, LAG_STEPS, axis=0); volume_lagged[:LAG_STEPS] = 0
            rolling_mean = pd.DataFrame(volume_ts).rolling(window=ROLLING_WINDOW_STEPS).mean().fillna(0).values
            volume_lagged_expanded = volume_lagged[:, :, np.newaxis]
            rolling_mean_expanded = rolling_mean[:, :, np.newaxis]
            X_raw_components.extend([volume_lagged_expanded, rolling_mean_expanded])
            
        X_raw = np.concatenate(X_raw_components, axis=-1)
        print(f"  -> Feature matrix X_raw created with shape: {X_raw.shape}")

        # 6. Station Clustering (DBSCAN on Rich Profiles)
        print("  -> Building rich station profiles for clustering...")
        volume_df = pd.DataFrame(volume_ts)
        station_profiles = pd.DataFrame(index=volume_df.columns)
        
        station_profiles['mean_demand'] = volume_df.mean()
        station_profiles['std_demand'] = volume_df.std()
        station_profiles['skew_demand'] = volume_df.apply(pd.to_numeric, errors='coerce').skew()
        volume_df.index = time_df['datetime']
        station_profiles['weekday_mean'] = volume_df[volume_df.index.dayofweek < 5].mean()
        station_profiles['weekend_mean'] = volume_df[volume_df.index.dayofweek >= 5].mean()

        static_info_df_local = static_info_df.set_index('station_id') if 'station_id' in static_info_df.columns else static_info_df.copy()
        station_profiles = station_profiles.join(static_info_df_local[['CBD', 'dynamic_pricing']]).fillna(0)

        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             scaled_profiles = StandardScaler().fit_transform(station_profiles.values)
        
        print(f"  -> Running DBSCAN with eps={DBSCAN_EPS}...")
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        cluster_labels_arr = dbscan.fit_predict(scaled_profiles)
        cluster_labels = pd.Series(cluster_labels_arr, name='cluster_id')
        cluster_map = {label: group.index.to_numpy() for label, group in cluster_labels.groupby(cluster_labels)}
        print(f"  -> Clustering complete. Found {len(cluster_map)} groups.")

        return X_raw, Y_raw, cluster_map, static_info_df, time_df, station_profiles, cluster_labels
        
    except Exception as e:
        print(f"Fatal Error during data loading: {e}"); 
        sys.exit(1)

# ==============================================================================
# --- TRAINING AND EVALUATION HELPERS ---
# ==============================================================================

def calculate_multistep_metrics(y_true, y_pred):
    """Calculates RMSE and MAE for each step in the forecast horizon."""
    results = []
    output_seq_len = y_true.shape[1]
    for step in range(output_seq_len):
        horizon_minutes = (step + 1) * 15
        y_true_step = y_true[:, step, :, :].flatten()
        y_pred_step = y_pred[:, step, :, :].flatten()
        valid_mask = np.isfinite(y_true_step) & np.isfinite(y_pred_step)
        
        if not np.any(valid_mask):
            rmse, mae = np.nan, np.nan
        else:
            rmse = np.sqrt(np.mean((y_true_step[valid_mask] - y_pred_step[valid_mask])**2))
            mae = np.mean(np.abs(y_true_step[valid_mask] - y_pred_step[valid_mask]))
            
        results.append({'Horizon': f'{horizon_minutes}-min', 'RMSE': rmse, 'MAE': mae})
    return results

def train_final_model(model_class, params, X_train, Y_train, X_val, Y_val, model_path, device):
    """Trains the model with early stopping and saves the best model state."""
    num_features = X_train.shape[-1]
    model = model_class(num_features, params['hidden_dim'], OUTPUT_SEQ_LEN, params['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TimeSeriesDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, Y_val), batch_size=BATCH_SIZE)
    best_val_loss, patience_counter = float('inf'), 0
    
    for epoch in range(FINAL_TRAIN_EPOCHS):
        model.train()
        for x_b, y_b in train_loader:
            output = model(x_b.to(device)); loss = loss_fn(output, y_b.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
        model.eval(); val_loss = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                if len(y_v) > 0: val_loss += loss_fn(model(x_v.to(device)), y_v.to(device)).item()
        
        if len(val_loader) > 0: val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOPPING_PATIENCE: 
            print(f" (Early stopped at epoch {epoch+1})")
            break

def generate_reports(all_results, cluster_map, station_profiles, cluster_labels):
    """Generates and saves cluster statistics and final performance summaries."""
    print("\n\n" + "="*80 + "\n|  GENERATING ALL REPORTS FOR THE RESEARCH PAPER" + " "*28 + "|\n" + "="*80)
    
    # 1. Cluster Characteristics Report
    report_path = os.path.join(REPORTS_DIR, '01_cluster_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"{'='*80}\n| {'TABLE 1: CLUSTER CHARACTERISTICS & COMPOSITION':^76} |\n{'='*80}\n")
        
        if station_profiles.shape[0] == len(cluster_labels):
            station_profiles_with_cluster = station_profiles.copy()
            station_profiles_with_cluster['cluster'] = cluster_labels.values
            cluster_char_df = station_profiles_with_cluster.groupby('cluster').agg(['mean', 'std'])
            cluster_counts = cluster_labels.value_counts().rename('num_stations')
            cluster_char_df['num_stations'] = cluster_counts
            f.write("Average station profile for each cluster:\n")
            f.write(cluster_char_df.round(2).to_string())
        else: 
            f.write("Error: Mismatch between station profiles and cluster labels.\n")
            
        f.write("\n\nStations per cluster:\n")
        for cluster_id, stations in cluster_map.items():
            name = "Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
            f.write(f"  - {name}: {len(stations)} stations\n")
            
    print(f"-> Saved Cluster Report to {report_path}")

    # 2. Detailed Fold Performance Report (CSV)
    all_results_list = []
    for model_name, fold_data in all_results.items():
        for fold_num, metrics_list in fold_data.items():
            for metric_dict in metrics_list:
                row = {'Model': model_name, 'Fold': fold_num, **metric_dict}
                all_results_list.append(row)
    results_df = pd.DataFrame(all_results_list)

    report_path_fold = os.path.join(REPORTS_DIR, '02_fold_performance.csv')
    pivot_fold = results_df.pivot_table(index=['Model', 'Fold'], columns='Horizon', values=['RMSE', 'MAE'])
    pivot_fold.to_csv(report_path_fold)
    print(f"-> Saved Detailed Fold Performance to {report_path_fold}")

    # 3. Final Summary Report (CSV & Print)
    report_path_summary = os.path.join(REPORTS_DIR, '03_final_summary.csv')
    summary_df = results_df.groupby(['Model', 'Horizon']).mean().drop(columns='Fold')
    summary_df = summary_df.reindex(['15-min', '30-min', '45-min', '60-min'], level='Horizon')
    summary_df.to_csv(report_path_summary)
    
    print("\n--- FINAL SUMMARY (Mean Performance Across Folds) ---")
    print(summary_df.round(4))
    print(f"-> Saved Final Summary to {report_path_summary}")

def generate_visualizations(final_fold_predictions, static_info_df, cluster_labels, station_profiles):
    """Generates and saves the primary comparison and cluster visualizations."""
    print("\n\n" + "="*80 + "\n|  GENERATING ALL VISUALIZATIONS" + " "*41 + "|\n" + "="*80)
    
    # 1. Geographic map of station clusters
    print("Generating Viz 1: Geographic map of station clusters...")
    if 'lon' in static_info_df.columns and 'la' in static_info_df.columns:
        plot_df = static_info_df.copy(); plot_df['cluster'] = cluster_labels.values
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=plot_df, x='lon', y='la', hue='cluster', palette='viridis', s=100, legend='full')
        plt.title('Geographic Distribution of Station Clusters'); plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, '01_geographic_clusters.png'))
        plt.close()
    else: 
        print("  [!] Skipping geo plot: 'lon'/'la' columns not found in static_info_df.")

    # 2. Comparing average demand profiles of clusters
    print("Generating Viz 2: Comparing average demand profiles of clusters...")
    profile_df = station_profiles.copy(); profile_df['cluster'] = cluster_labels.values
    cluster_avg_profiles = profile_df.groupby('cluster').mean().T
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=cluster_avg_profiles, dashes=False, palette='viridis', legend=True)
    plt.title('Average Demand Profile by Cluster')
    plt.xlabel('Profile Feature'); plt.ylabel('Average Value')
    plt.xticks(rotation=45); plt.grid(True, linestyle='--'); plt.legend(title='Cluster'); plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, '02_cluster_demand_profiles.png'))
    plt.close()

    # 3. Forecast vs. Actuals deep dive for example stations
    print("Generating Viz 3: Forecast vs. Actuals deep dive...")
    
    if 'LSTM (Global)' not in final_fold_predictions or 'LSTM (Clustered)' not in final_fold_predictions:
        print("  [!] Skipping deep dive plot: Missing final fold predictions for Global or Clustered model.")
        return

    preds_global = final_fold_predictions['LSTM (Global)']['preds']
    actuals_global = final_fold_predictions['LSTM (Global)']['actuals']
    preds_clustered = final_fold_predictions['LSTM (Clustered)']['preds']
    
    unique_clusters = sorted([c for c in cluster_labels.unique() if np.any(cluster_labels==c)])
    example_stations = [np.where(cluster_labels == c)[0][0] for c in unique_clusters[:3] if np.any(cluster_labels==c)]
    
    for station_idx in example_stations:
        station_cluster = cluster_labels.iloc[station_idx]
        fig, ax = plt.subplots(figsize=(18, 6))
        
        actuals_slice = actuals_global[:, :, station_idx, :].flatten()
        global_pred_slice = preds_global[:, :, station_idx, :].flatten()
        clustered_pred_slice = preds_clustered[:, :, station_idx, :].flatten()
        time_axis = range(len(actuals_slice))
        
        ax.plot(time_axis, actuals_slice, label='Actual Demand', color='black', lw=2)
        ax.plot(time_axis, global_pred_slice, label='Global LSTM', color='red', linestyle='--')
        ax.plot(time_axis, clustered_pred_slice, label='Clustered LSTM', color='blue', linestyle='-.')
        
        ax.set_title(f'Forecast Deep Dive: Station {station_idx} (Cluster {station_cluster})')
        ax.set_ylabel('Demand'); ax.set_xlabel('Time Step in Final Test Fold'); ax.legend(); ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'03_forecast_station_{station_idx}.png'))
        plt.close()
        
    print("-> Saved all reports and visualizations.")

# ==============================================================================
# --- ECONOMIC AND OPTIMIZATION UTILITIES ---
# ==============================================================================

def calculate_heuristic_prices(demands, p_base, p_min, p_max, alpha):
    """Calculates dynamic prices using a simple heuristic based on average demand."""
    d_avg = np.mean(demands, axis=1, keepdims=True)
    d_avg_safe = np.where(d_avg == 0, 1e-9, d_avg)
    return np.clip(p_base + alpha * (demands - d_avg) / d_avg_safe, p_min, p_max)

def calculate_responsive_demand(prices, base_demands, p_base, beta):
    """Models demand response based on price elasticity."""
    price_change_ratio = (prices - p_base) / p_base if p_base > 0 else 0
    return np.maximum(0, base_demands * (1 - beta * price_change_ratio))

def objective_function(prices, base_demands, p_base, beta, cost_per_kwh):
    """Optimization objective: maximize total profit (negative sum of profit)."""
    responsive_demand = calculate_responsive_demand(prices, base_demands, p_base, beta)
    profit = (prices - cost_per_kwh) * responsive_demand
    return -np.sum(profit)

def run_economic_simulation(X_raw, Y_raw, models, scalers_X, scalers_Y, cluster_map):
    """
    Runs a 24-hour simulation comparing heuristic vs. optimized pricing strategies.
    """
    print("\n--- Starting 24-Hour Economic Simulation with Fine-Tuned Models ---")
    
    test_start_idx = int(X_raw.shape[0] * TRAIN_RATIO)
    price_bounds = Bounds([ECON_PMIN] * OUTPUT_SEQ_LEN, [ECON_PMAX] * OUTPUT_SEQ_LEN)

    hourly_results = []
    station_results = np.zeros((X_raw.shape[1], 4))
    
    for hour_step in range(24):
        print(f"  Simulating Hour {hour_step + 1}/24...")
        current_timestep = test_start_idx + hour_step * OUTPUT_SEQ_LEN
        
        if current_timestep + INPUT_SEQ_LEN > X_raw.shape[0]: 
            print("  Simulation stopped: Not enough data for a full look-back window.")
            break
            
        hour_profit_h, hour_profit_o = 0, 0
        hour_demand_h, hour_demand_o = 0, 0
        
        for group_id, station_indices in cluster_map.items():
            if group_id not in models or len(station_indices) == 0: continue
            
            model = models[group_id]
            
            x_input_raw = X_raw[current_timestep : current_timestep + INPUT_SEQ_LEN, station_indices, :]
            x_input_scaled = scalers_X[group_id].transform(x_input_raw.reshape(-1, x_input_raw.shape[-1])).reshape(x_input_raw.shape)
            x_batch = torch.tensor(x_input_scaled.transpose(1, 0, 2), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pred_scaled = model(x_batch)
            
            pred_demands = scalers_Y[group_id].inverse_transform(
                pred_scaled.cpu().numpy().reshape(-1, 1)
            ).reshape(len(station_indices), OUTPUT_SEQ_LEN)
            pred_demands[pred_demands < 0] = 0
            
            # 3. Heuristic Pricing Strategy
            h_prices = calculate_heuristic_prices(pred_demands, ECON_PBASE, ECON_PMIN, ECON_PMAX, ECON_ALPHA)
            h_responsive_demand = calculate_responsive_demand(h_prices, pred_demands, ECON_PBASE, DEMAND_ELASTICITY_BETA)
            h_profit_per_station = np.sum((h_prices - COST_PER_KWH) * h_responsive_demand, axis=1)
            
            hour_profit_h += np.sum(h_profit_per_station)
            hour_demand_h += np.sum(h_responsive_demand)
            station_results[station_indices, 0] += h_profit_per_station
            station_results[station_indices, 2] += np.sum(h_responsive_demand, axis=1)
            
            # 4. Optimized Pricing Strategy
            for i, station_abs_idx in enumerate(station_indices):
                result = minimize(
                    objective_function, x0=h_prices[i], 
                    args=(pred_demands[i], ECON_PBASE, DEMAND_ELASTICITY_BETA, COST_PER_KWH),
                    method='SLSQP', bounds=price_bounds
                )
                
                if result.success:
                    o_prices = result.x
                    o_responsive_demand = calculate_responsive_demand(o_prices, pred_demands[i], ECON_PBASE, DEMAND_ELASTICITY_BETA)
                    profit = np.sum((o_prices - COST_PER_KWH) * o_responsive_demand)
                    hour_profit_o += profit
                    hour_demand_o += np.sum(o_responsive_demand)
                    station_results[station_abs_idx, 1] += profit
                    station_results[station_abs_idx, 3] += np.sum(o_responsive_demand)

        hourly_results.append({
            'Hour': hour_step + 1, 'Heuristic Profit': hour_profit_h, 'Optimized Profit': hour_profit_o,
            'Heuristic Demand': hour_demand_h, 'Optimized Demand': hour_demand_o
        })
        
    print("  ✅ Simulation finished.")
    return pd.DataFrame(hourly_results), pd.DataFrame(station_results, columns=['h_profit', 'o_profit', 'h_demand', 'o_demand'])

def generate_bi_dashboard(hourly_df, station_df, static_info_df, cluster_labels):
    """Generates and displays the final business intelligence visualizations."""
    print(f"\n{'='*70}\n--- Generating Business Intelligence Dashboard ---\n{'='*70}")
    
    # 1. Overall Performance Analysis
    total_h_profit = station_df['h_profit'].sum()
    total_o_profit = station_df['o_profit'].sum()
    improvement = (total_o_profit - total_h_profit) / abs(total_h_profit) * 100 if total_h_profit != 0 else float('inf')
    
    print(f"Total Heuristic Profit: ${total_h_profit:,.2f}")
    print(f"Total Optimized Profit: ${total_o_profit:,.2f}")
    print(f"Overall Profit Improvement: {improvement:.2f}%")
    
    # Hourly Profit Comparison
    plt.figure(figsize=(18, 8))
    x_axis = hourly_df['Hour']
    plt.bar(x_axis - 0.2, hourly_df['Heuristic Profit'], 0.4, label=f'Heuristic Profit (${total_h_profit:,.0f})', color='skyblue')
    plt.bar(x_axis + 0.2, hourly_df['Optimized Profit'], 0.4, label=f'Optimized Profit (${total_o_profit:,.0f})', color='darkviolet')
    plt.ylabel("Total Net Profit ($)"); plt.xlabel("Simulation Hour"); plt.title("24-Hour Profit Simulation: Heuristic vs. Optimized", fontsize=18)
    plt.xticks(x_axis); plt.legend(); plt.grid(axis='y', linestyle='--')
    plt.show()
    
    # Cumulative Profit Plot
    plt.figure(figsize=(18, 8))
    plt.plot(hourly_df['Hour'], hourly_df['Heuristic Profit'].cumsum(), marker='o', linestyle='--', label='Cumulative Heuristic Profit')
    plt.plot(hourly_df['Hour'], hourly_df['Optimized Profit'].cumsum(), marker='o', linestyle='-', label='Cumulative Optimized Profit', lw=3)
    plt.ylabel("Cumulative Net Profit ($)"); plt.xlabel("Simulation Hour"); plt.title("Cumulative Profit Growth Over 24 Hours", fontsize=18)
    plt.xticks(x_axis); plt.legend(); plt.grid(True, which='both', linestyle=':')
    plt.show()

    # 2. Cluster-Level Analysis
    station_df['cluster'] = cluster_labels
    cluster_performance = station_df.groupby('cluster')[['h_profit', 'o_profit']].sum()
    
    # Profit Contribution Pie Chart
    plt.figure(figsize=(12, 12))
    cluster_performance.plot(kind='pie', y='o_profit', autopct='%1.1f%%', startangle=90,
                             title='Optimized Profit Contribution by Cluster',
                             labels=[f'Cluster {c}' if c != -1 else 'Outliers' for c in cluster_performance.index],
                             legend=False, wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.ylabel('')
    plt.show()

    # 3. Geospatial Profit Analysis
    full_station_df = static_info_df.join(station_df)
    
    # 3D Profit Hotspot Map
    print("  -> Plotting 3D profit hotspot map...")
    if 'lon' in full_station_df.columns and 'la' in full_station_df.columns:
        fig = plt.figure(figsize=(18, 15)); ax = fig.add_subplot(projection='3d')
        x, y, z = full_station_df['lon'].values, full_station_df['la'].values, np.zeros_like(full_station_df['o_profit'])
        dx = dy = 0.005; dz = full_station_df['o_profit'].fillna(0)
        colors = plt.cm.plasma((dz - dz.min()) / (dz.max() - dz.min() + 1e-9))
        ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)
        ax.set_title('3D Map of 24-Hour Optimized Profit per Station', fontsize=20)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_zlabel('Total Net Profit ($)')
        plt.show()
    
    # 2D Profit Improvement Map
    print("  -> Plotting 2D profit improvement map...")
    if 'lon' in full_station_df.columns and 'la' in full_station_df.columns:
        full_station_df['profit_lift'] = (full_station_df['o_profit'] - full_station_df['h_profit']) / abs(full_station_df['h_profit']).replace(0, 1) * 100
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        
        # Load and plot district shapefile
        shapefile_path = SHAPEFILE_PATH
        if os.path.exists(shapefile_path):
            try:
                gpd.read_file(shapefile_path).plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)
            except Exception as e:
                print(f"  [!] Warning: Could not plot shapefile. Ensure geopandas is installed and file is valid. Error: {e}")
                
        plt.scatter(x=full_station_df['lon'], y=full_station_df['la'],
                    s=abs(full_station_df['profit_lift']),
                    c=full_station_df['profit_lift'],
                    cmap='coolwarm', alpha=0.8)
        plt.colorbar(label='Profit Improvement (%)')
        plt.title('Geographic Map of Profit Improvement per Station', fontsize=18)
        plt.xlabel('Longitude'); plt.ylabel('Latitude')
        plt.show()
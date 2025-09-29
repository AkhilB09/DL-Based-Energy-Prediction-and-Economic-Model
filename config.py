# config.py

# Data file paths (relative to the expected execution environment)
STATIC_INFO_PATH = 'data/raw/information.csv'
VOLUME_TS_PATH = 'data/raw/volume.csv'
DURATION_TS_PATH = 'data/raw/duration.csv'
OCCUPANCY_TS_PATH = 'data/raw/occupancy.csv'
PRICE_TS_PATH = 'data/raw/price.csv'
TIME_FEATURES_PATH = 'data/raw/time.csv'
WEATHER_XLS_PATH = 'data/raw/SZweather20220619-20220718.xls'
SHAPEFILE_PATH = 'data/raw/SZ_districts.shp' # Note: path adjusted to remove the '(1)' found in a late block

# ==============================================================================
# --- FINAL RUN CONFIGURATION (Training, CV, & Data) ---
# ==============================================================================
SEED = 42

# --- Model & Data Settings ---
TRAIN_RATIO = 0.85
INPUT_SEQ_LEN = 12  # Look-back window (e.g., 3 hours)
OUTPUT_SEQ_LEN = 4  # Forecast horizon (e.g., 1 hour)

# --- Training & Cross-Validation Parameters ---
N_CV_SPLITS = 3
FINAL_TRAIN_EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10

# --- DBSCAN & Clustering Parameters ---
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 5
# NOTE: Lag and Rolling features are INTENTIONALLY omitted for the final run.

# --- File Paths & Output Directories ---
MODELS_DIR = 'final_run_models_no_lag' # Source for pre-trained models
VISUALIZATIONS_DIR = 'final_run_visualizations_no_lag'
REPORTS_DIR = 'final_run_reports_no_lag'
FINETUNED_MODELS_DIR = 'fine_tuned_models' # Source for fine-tuned models
FINETUNED_REPORTS_DIR = 'fine_tuned_reports'


# --- FINAL TUNED HYPERPARAMETERS ---
TUNED_PARAMS_PER_CLUSTER = {
    # Outliers
    -1: {'lr': 0.000336, 'hidden_dim': 192, 'dropout': 0.1444},
    # Cluster 0
    0: {'lr': 0.000217, 'hidden_dim': 128, 'dropout': 0.4887},
    # Cluster 1
    1: {'lr': 0.000554, 'hidden_dim': 256, 'dropout': 0.2926},
    # Cluster 2
    2: {'lr': 0.000694, 'hidden_dim': 128, 'dropout': 0.1852},
    # Cluster 3
    3: {'lr': 0.000263, 'hidden_dim': 192, 'dropout': 0.351},
    # Fallback default parameters for the Global model
    'default': {'lr': 0.0025, 'hidden_dim': 128, 'dropout': 0.20}
}

# ==============================================================================
# --- ECONOMIC SIMULATION AND OPTIMIZATION CONFIGURATION ---
# ==============================================================================
ECON_PBASE = 0.15           # Base price ($/kWh)
ECON_PMIN = 0.05            # Minimum price ($/kWh)
ECON_PMAX = 0.50            # Maximum price ($/kWh)
ECON_ALPHA = 0.5            # Heuristic price sensitivity factor
COST_PER_KWH = 0.08         # Cost of energy ($/kWh)
DEMAND_ELASTICITY_BETA = 0.6 # Demand elasticity factor

# --- Optuna Settings (for tuner.py) ---
OPTUNA_N_TRIALS = 50
N_CV_SPLITS_FOR_TUNING = 3
QUICK_TRAIN_EPOCHS = 15